#!/usr/bin/env python3
"""
Fit both an ODE (2-population sensitive/resistant) and a PDE (trait-structured resistance)
model to longitudinal liquidCNA resistant fraction + CA125, and compute goodness-of-fit.

Input data: Subclonal_ratio_estimates.extended.txt (tab-separated)
Required columns: Patient, Time, context, ratio, ratio_min95, ratio_max95, CA125, Accept_estimate

Models:
  ODE:
    dS/dt = S*(aS*(1 - (S+R)/K)) - u(t)*dS*S
    dR/dt = R*(aR*(1 - (S+R)/K)) - u(t)*dR*R
    r(t)=R/(S+R), CA125 ~ gamma*(S+R)

  PDE (trait-structured):
    ∂n/∂t = n(x)*(g(x)*(1 - N/K) - u(t)*d(x)) + D*∂²n/∂x²
    g(x)=g0*(1 - c*x)  (fitness cost increases with resistance trait x)
    d(x)=d0*(1 - b*x)  (drug kill decreases with resistance)
    r(t)=∫_{x>=x*} n dx / ∫ n dx
    CA125 ~ gamma*N

Treatment u(t):
  If you don't have dosing times, we use context-driven piecewise constant u_c in [0,1],
  chosen by the context of the most recent sample (previous time point).
  u_c values are fitted (one per unique context in that patient's data).

Goodness-of-fit:
  - Neg log-likelihood (Gaussian on logit(ratio) using CI-derived SE, Gaussian on log(CA125))
  - RMSE/MAE on ratio and log(CA125)
  - AIC, BIC (using NLL and number of fitted parameters)

Run:
  python fit_ode_pde.py --data data/liquidCNA_results/Subclonal_ratio_estimates.extended.txt --patient UP0018
"""

import argparse
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Tuple, List

from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp, trapezoid
from scipy.optimize import minimize
from joblib import Parallel, delayed
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
import traceback


# -------------------------- Utilities --------------------------


def logit(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 1e-6, 1 - 1e-6)   # was 1e-9
    return np.log(x / (1 - x))


def invlogit(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def ci95_to_se_logit(r: np.ndarray, r_lo: np.ndarray, r_hi: np.ndarray) -> np.ndarray:
    """Approx SE on logit scale from a 95% interval on ratio scale."""
    y_lo = logit(np.clip(r_lo, 1e-9, 1 - 1e-9))
    y_hi = logit(np.clip(r_hi, 1e-9, 1 - 1e-9))
    se = (y_hi - y_lo) / 3.92
    se = np.clip(se, 1e-2, 5.0)
    return se


def safe_log(x: np.ndarray) -> np.ndarray:
    return np.log(np.clip(x, 1e-12, None))


@dataclass
class PatientData:
    patient: str
    t: np.ndarray               # time (months by default)
    context: np.ndarray         # integer context id per sample
    context_names: List[str]
    ratio: np.ndarray
    se_logit_ratio: np.ndarray
    log_ca125: np.ndarray
    maybe_mask: np.ndarray


def load_patient_data(path: str, patient: str, time_unit: str = "months") -> PatientData:
    df = pd.read_csv(path, sep="\t")
    df = df[df["Accept_estimate"].isin(["yes", "maybe"])].copy()
    df["Time"] = pd.to_numeric(df["Time"], errors="coerce")
    df["CA125"] = pd.to_numeric(df["CA125"], errors="coerce")
    df = df.dropna(subset=["Time", "ratio", "ratio_min95", "ratio_max95", "CA125"]).copy()

    df["Patient"] = df["Patient"].astype(str)
    df = df[df["Patient"] == str(patient)].copy()
    if df.empty:
        raise ValueError(f"No rows found for patient={patient}")

    # rescale time to months (strongly recommended)
    if time_unit == "months":
        df["Time"] = df["Time"] / 30.0
    elif time_unit == "days":
        pass
    else:
        raise ValueError("time_unit must be 'months' or 'days'")

    df = df.sort_values("Time").reset_index(drop=True)

    # contexts
    context_names = df["context"].astype(str).unique().tolist()
    ctx_map = {c: i for i, c in enumerate(context_names)}
    ctx = df["context"].astype(str).map(ctx_map).to_numpy()

    ratio = np.clip(df["ratio"].to_numpy().astype(float), 1e-9, 1 - 1e-9)
    r_lo = np.clip(df["ratio_min95"].to_numpy().astype(float), 1e-9, 1 - 1e-9)
    r_hi = np.clip(df["ratio_max95"].to_numpy().astype(float), 1e-9, 1 - 1e-9)
    se = ci95_to_se_logit(ratio, r_lo, r_hi)

    maybe = (df["Accept_estimate"].to_numpy() == "maybe")
    se = np.where(maybe, se * 2.0, se)

    ca = df["CA125"].to_numpy().astype(float)
    log_ca = safe_log(ca)

    return PatientData(
        patient=str(patient),
        t=df["Time"].to_numpy().astype(float),
        context=ctx.astype(int),
        context_names=context_names,
        ratio=ratio,
        se_logit_ratio=se,
        log_ca125=log_ca,
        maybe_mask=maybe,
    )

def get_patients_with_flag(path: str, flags: List[str]) -> List[str]:
    df = pd.read_csv(path, sep="\t")
    flags = [f.strip() for f in flags]
    df = df[df["Accept_estimate"].isin(flags)]
    return sorted(df["Patient"].astype(str).unique())

def make_u_of_t(t_samples: np.ndarray, ctx_samples: np.ndarray, u_ctx: np.ndarray):
    """
    Piecewise constant u(t) using the context of the most recent observed sample.
    For t < t0 => use context at first sample.
    """
    t_samples = np.asarray(t_samples)
    ctx_samples = np.asarray(ctx_samples)

    def u(t: float) -> float:
        i = np.searchsorted(t_samples, t, side="right") - 1
        if i < 0:
            i = 0
        c = ctx_samples[i]
        return float(np.clip(u_ctx[c], 0.0, 1.0))

    return u


# -------------------------- ODE model --------------------------


def ode_rhs(t, y, pars, u_fun):
    S, R = y
    aS, aR, dS, dR, K = pars
    N = S + R
    g = max(0.0, 1.0 - N / K)
    u = u_fun(t)
    dS_eff = u * dS
    dR_eff = u * dR
    dSdt = S * (aS * g) - dS_eff * S
    dRdt = R * (aR * g) - dR_eff * R
    return [dSdt, dRdt]


def simulate_ode(data: PatientData, theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    theta = [log_aS, log_aR, log_dS, log_dR, log_K, log_N0, logit_r0, log_gamma, log_sigma_ca, u_ctx...]
    """
    C = len(data.context_names)
    assert theta.size == 9 + C

    log_aS, log_aR, log_dS, log_dR, log_K, log_N0, logit_r0, log_gamma, log_sigma_ca = theta[:9]
    u_ctx = invlogit(theta[9:9 + C])  # (0,1)

    aS, aR, dS, dR, K = np.exp([log_aS, log_aR, log_dS, log_dR, log_K])
    N0 = float(np.exp(log_N0))
    r0 = float(invlogit(np.array([logit_r0]))[0])
    gamma = float(np.exp(log_gamma))
    sigma_ca = float(np.exp(log_sigma_ca))

    S0 = N0 * (1 - r0)
    R0 = N0 * r0

    u_fun = make_u_of_t(data.t, data.context, u_ctx)
    t0, t1 = float(data.t[0]), float(data.t[-1])

    sol = solve_ivp(
        fun=lambda t, y: ode_rhs(t, y, (aS, aR, dS, dR, K), u_fun),
        t_span=(t0, t1),
        y0=[S0, R0],
        t_eval=data.t,
        method="LSODA",
        rtol=1e-6,
        atol=1e-9,
        max_step=max(1e-3, (t1 - t0) / 200.0),
    )

    if not sol.success or np.any(~np.isfinite(sol.y)):
        raise RuntimeError("ODE solver failed")

    S = np.clip(sol.y[0], 1e-12, None)
    R = np.clip(sol.y[1], 1e-12, None)
    N = S + R
    r = R / N
    log_ca = safe_log(gamma * N)

    return r, log_ca


def nll_ode(theta: np.ndarray, data: PatientData) -> float:
    try:
        r_hat, logca_hat = simulate_ode(data, theta)
    except Exception:
        return 1e50

    # ratio likelihood on logit scale using CI-derived SE
    y_obs = logit(data.ratio)
    y_hat = logit(r_hat)
    se = data.se_logit_ratio
    nll_ratio = 0.5 * np.sum(((y_obs - y_hat) / se) ** 2 + 2 * np.log(se) + np.log(2 * np.pi))

    # CA125 likelihood on log scale
    sigma_ca = float(np.exp(theta[8]))
    nll_ca = 0.5 * np.sum(((data.log_ca125 - logca_hat) / sigma_ca) ** 2 + 2 * np.log(sigma_ca) + np.log(2 * np.pi))

    return float(nll_ratio + nll_ca)


# -------------------------- PDE model (trait-structured) --------------------------


def laplacian_neumann(M: int, dx: float) -> np.ndarray:
    """Second derivative matrix with Neumann (zero-flux) boundaries."""
    L = np.zeros((M, M))
    for i in range(1, M - 1):
        L[i, i - 1] = 1.0
        L[i, i] = -2.0
        L[i, i + 1] = 1.0

    # Neumann at boundaries via mirrored ghost points:
    # n[-1]=n[1], n[M]=n[M-2]
    L[0, 0] = -2.0
    L[0, 1] = 2.0
    L[M - 1, M - 2] = 2.0
    L[M - 1, M - 1] = -2.0

    return L / (dx ** 2)


def pde_rhs(t, n, x, L, pars, u_fun):
    # pars: g0, c, d0, b, K, D
    g0, c, d0, b, K, D = pars
    n = np.clip(n, 0.0, None)
    N = float(trapezoid(n, x))
    g = g0 * (1.0 - c * x)              # fitness decreases with resistance
    d = d0 * (1.0 - b * x)              # drug kill decreases with resistance
    g = np.clip(g, 0.0, None)
    d = np.clip(d, 0.0, None)

    u = u_fun(t)
    growth = n * (g * (1.0 - N / K) - u * d)
    diff = D * (L @ n)
    dn = growth + diff
    return dn


def init_trait_density(M: int, x: np.ndarray, N0: float, r0: float, x_star: float) -> np.ndarray:
    sens = np.exp(-8.0 * x)
    res = np.exp(-8.0 * (1.0 - x))

    sens_area = float(trapezoid(sens, x))
    res_area = float(trapezoid(res, x))
    sens = sens / max(sens_area, 1e-12)
    res = res / max(res_area, 1e-12)

    n = (1.0 - r0) * sens + r0 * res

    area = float(trapezoid(n, x))
    n *= N0 / max(area, 1e-12)

    mask = x >= x_star
    total = float(trapezoid(n, x))
    resmass = float(trapezoid(n[mask], x[mask]))
    frac = resmass / max(total, 1e-12)

    if frac > 1e-12:
        n[mask] *= (r0 / frac)
        area = float(trapezoid(n, x))
        n *= N0 / max(area, 1e-12)

    n = np.clip(n, 0.0, None)
    return n



def simulate_pde(data: PatientData, theta: np.ndarray, M: int = 81) -> Tuple[np.ndarray, np.ndarray]:
    """
    theta = [log_g0, logit_c, log_d0, logit_b, log_K, log_D, log_N0, logit_r0, logit_xstar,
             log_gamma, log_sigma_ca, u_ctx...]
    """
    C = len(data.context_names)
    assert theta.size == 11 + C

    log_g0, logit_c, log_d0, logit_b, log_K, log_D, log_N0, logit_r0, logit_xstar, log_gamma, log_sigma_ca = theta[:11]
    u_ctx = invlogit(theta[11:11 + C])

    g0 = float(np.exp(log_g0))
    c = float(invlogit(np.array([logit_c]))[0])          # (0,1)
    d0 = float(np.exp(log_d0))
    b = float(invlogit(np.array([logit_b]))[0])          # (0,1)
    K = float(np.exp(log_K))
    D = float(np.exp(log_D))
    N0 = float(np.exp(log_N0))
    r0 = float(invlogit(np.array([logit_r0]))[0])
    x_star = float(invlogit(np.array([logit_xstar]))[0])
    gamma = float(np.exp(log_gamma))
    sigma_ca = float(np.exp(log_sigma_ca))

    x = np.linspace(0.0, 1.0, M)
    dx = x[1] - x[0]
    L = laplacian_neumann(M, dx)

    u_fun = make_u_of_t(data.t, data.context, u_ctx)

    n0 = init_trait_density(M, x, N0=N0, r0=r0, x_star=x_star)

    t0, t1 = float(data.t[0]), float(data.t[-1])

    sol = solve_ivp(
        fun=lambda t, n: pde_rhs(t, n, x, L, (g0, c, d0, b, K, D), u_fun),
        t_span=(t0, t1),
        y0=n0,
        t_eval=data.t,
        method="BDF",
        rtol=1e-6,
        atol=1e-9,
        max_step=max(1e-3, (t1 - t0) / 200.0),
    )

    if not sol.success or np.any(~np.isfinite(sol.y)):
        raise RuntimeError("PDE (method-of-lines) solver failed")

    # Compute r(t) and CA125 proxy from n(t,x)
    r_hat = np.zeros_like(data.t)
    log_ca_hat = np.zeros_like(data.t)

    for i in range(len(data.t)):
        n = np.clip(sol.y[:, i], 0.0, None)
        N = float(trapezoid(n, x))
        mask = x >= x_star
        Rmass = float(trapezoid(n[mask], x[mask]))
        frac = Rmass / max(N, 1e-12)
        r_hat[i] = np.clip(frac, 1e-9, 1 - 1e-9)
        log_ca_hat[i] = float(safe_log(np.array([gamma * N]))[0])

    return r_hat, log_ca_hat


def nll_pde(theta: np.ndarray, data: PatientData, M: int) -> float:
    try:
        r_hat, logca_hat = simulate_pde(data, theta, M=M)
    except Exception:
        return 1e50

    # ratio likelihood on logit scale
    y_obs = logit(data.ratio)
    y_hat = logit(r_hat)
    se = data.se_logit_ratio
    nll_ratio = 0.5 * np.sum(((y_obs - y_hat) / se) ** 2 + 2 * np.log(se) + np.log(2 * np.pi))

    sigma_ca = float(np.exp(theta[10]))
    nll_ca = 0.5 * np.sum(((data.log_ca125 - logca_hat) / sigma_ca) ** 2 + 2 * np.log(sigma_ca) + np.log(2 * np.pi))

    return float(nll_ratio + nll_ca)


# -------------------------- Fit + GOF --------------------------


def gof_metrics(r_obs, r_hat, logca_obs, logca_hat, nll, k_params):
    # ratio metrics on original scale
    rmse_r = float(np.sqrt(np.mean((r_obs - r_hat) ** 2)))
    mae_r = float(np.mean(np.abs(r_obs - r_hat)))

    # CA125 metrics on log scale
    rmse_ca = float(np.sqrt(np.mean((logca_obs - logca_hat) ** 2)))
    mae_ca = float(np.mean(np.abs(logca_obs - logca_hat)))

    n = len(r_obs) + len(logca_obs)
    aic = float(2 * k_params + 2 * nll)
    bic = float(k_params * np.log(max(n, 1)) + 2 * nll)

    return {
        "NLL": float(nll),
        "AIC": aic,
        "BIC": bic,
        "RMSE_ratio": rmse_r,
        "MAE_ratio": mae_r,
        "RMSE_logCA125": rmse_ca,
        "MAE_logCA125": mae_ca,
    }

def multistart_minimize(fun, x0, bounds, n_starts=8, rel_noise=0.3, seed=0, method="L-BFGS-B", maxiter=800):
    rng = np.random.default_rng(seed)
    best = None

    # always include the provided x0 as a start
    starts = [x0.copy()]

    # additional random starts around x0 in parameter space
    lo = np.array([b[0] for b in bounds], dtype=float)
    hi = np.array([b[1] for b in bounds], dtype=float)

    for _ in range(n_starts - 1):
        z = x0.copy()
        # multiplicative noise on transformed params (still in theta-space)
        noise = rng.normal(0.0, rel_noise, size=z.size)
        z = z + noise
        z = np.clip(z, lo, hi)
        starts.append(z)

    for s in starts:
        res = minimize(fun, s, method=method, bounds=bounds, options={"maxiter": maxiter})
        if best is None or res.fun < best.fun:
            best = res

    return best

def fit_ode(data: PatientData) -> Tuple[np.ndarray, Dict]:
    C = len(data.context_names)

    # Initial guesses (months time scale)
    # aS > aR, dS >> dR, big K
    x0 = np.zeros(9 + C)
    x0[0] = np.log(0.5)    # aS
    x0[1] = np.log(0.3)    # aR
    x0[2] = np.log(0.8)    # dS
    x0[3] = np.log(0.05)   # dR
    x0[4] = np.log(1e6)    # K
    x0[5] = np.log(1e4)    # N0
    x0[6] = logit(np.array([np.clip(data.ratio[0], 1e-4, 1-1e-4)]))[0]  # r0
    x0[7] = np.log(np.exp(np.mean(data.log_ca125)) / np.exp(x0[5]))     # gamma rough
    x0[8] = np.log(0.5)    # sigma_ca
    x0[9:] = logit(np.full(C, 0.5))  # u_ctx (0.5)

    # Bounds (on transformed parameters)
    bnds = []
    # logs
    bnds += [(-10, 5)]   # log_aS
    bnds += [(-10, 5)]   # log_aR
    bnds += [(-10, 5)]   # log_dS
    bnds += [(-10, 5)]   # log_dR
    bnds += [(0, 20)]    # log_K
    bnds += [(-5, 30)]   # log_N0
    bnds += [(-10, 10)]  # logit_r0
    bnds += [(-20, 20)]  # log_gamma
    bnds += [(-3, 5)]   # log_sigma_ca
    bnds += [(-10, 10)] * C  # u_ctx logits

    res = multistart_minimize(
        fun=lambda th: nll_ode(th, data),
        x0=x0,
        bounds=bnds,
        n_starts=10,
        rel_noise=0.25,
        seed=hash(data.patient) % (2 ** 32),
        method="L-BFGS-B",
        maxiter=1200,
    )

    theta = res.x
    nll = float(res.fun)

    r_hat, logca_hat = simulate_ode(data, theta)
    metrics = gof_metrics(data.ratio, r_hat, data.log_ca125, logca_hat, nll=nll, k_params=theta.size)

    out = {"success": bool(res.success), "message": res.message, "metrics": metrics}
    return theta, out


def fit_pde(data: PatientData, M: int) -> Tuple[np.ndarray, Dict]:
    C = len(data.context_names)

    x0 = np.zeros(11 + C)
    x0[0] = np.log(0.6)    # g0
    x0[1] = logit(np.array([0.5]))[0]  # c in (0,1)
    x0[2] = np.log(1.0)    # d0
    x0[3] = logit(np.array([0.8]))[0]  # b in (0,1)
    x0[4] = np.log(1e6)    # K
    x0[5] = np.log(1e-3)   # D diffusion
    x0[6] = np.log(1e4)    # N0
    x0[7] = logit(np.array([np.clip(data.ratio[0], 1e-4, 1-1e-4)]))[0]  # r0
    x0[8] = logit(np.array([0.8]))[0]  # x* threshold
    x0[9] = np.log(np.exp(np.mean(data.log_ca125)) / np.exp(x0[6]))     # gamma
    x0[10] = np.log(0.5)   # sigma_ca
    x0[11:] = logit(np.full(C, 0.5))   # u_ctx

    bnds = []
    bnds += [(-10, 5)]     # log_g0
    bnds += [(-10, 10)]    # logit_c
    bnds += [(-10, 5)]     # log_d0
    bnds += [(-10, 10)]    # logit_b
    bnds += [(0, 20)]      # log_K
    bnds += [(-20, 0)]     # log_D
    bnds += [(-5, 30)]     # log_N0
    bnds += [(-10, 10)]    # logit_r0
    bnds += [(-10, 10)]    # logit_xstar
    bnds += [(-20, 20)]    # log_gamma
    bnds += [(-3, 5)]     # log_sigma_ca
    bnds += [(-10, 10)] * C

    res = multistart_minimize(
        fun=lambda th: nll_pde(th, data, M),
        x0=x0,
        bounds=bnds,
        n_starts=8,
        rel_noise=0.25,
        seed=(hash(data.patient) + 12345) % (2 ** 32),
        method="L-BFGS-B",
        maxiter=900,
    )

    theta = res.x
    nll = float(res.fun)

    r_hat, logca_hat = simulate_pde(data, theta, M=M)
    metrics = gof_metrics(data.ratio, r_hat, data.log_ca125, logca_hat, nll=nll, k_params=theta.size)

    out = {"success": bool(res.success), "message": res.message, "metrics": metrics}
    return theta, out


def pretty_print(title: str, context_names: List[str], theta: np.ndarray, metrics: Dict):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    for k, v in metrics.items():
        print(f"{k:>14s}: {v:.6g}")
    print("-" * 80)
    print(f"n_params: {theta.size}")
    print("contexts:", context_names)

def plot_gof_scatter_all(df_points: pd.DataFrame, out_prefix="gof"):
    """
    Creates 4 plots:
      ODE ratio, ODE logCA125, PDE ratio, PDE logCA125
    Labels out-of-95% points with patient IDs.
    """
    for model in sorted(df_points["model"].unique()):
        for var in ["ratio", "logCA125"]:
            sub = df_points[(df_points["model"] == model) & (df_points["var"] == var)].copy()
            if sub.empty:
                continue

            x = sub["obs"].to_numpy()
            y = sub["pred"].to_numpy()

            plt.figure(figsize=(7, 7))
            plt.scatter(x, y, alpha=0.35, s=18)

            # diagonal line
            if var == "ratio":
                lo, hi = 0.0, 1.0
            else:
                lo = float(np.nanmin(np.r_[x, y]))
                hi = float(np.nanmax(np.r_[x, y]))

            plt.plot([lo, hi], [lo, hi])

            plt.xlabel("Observed")
            plt.ylabel("Predicted")
            plt.title(f"{model} GOF scatter: {var} (label points outside 95%)")

            # label out-of-95 points
            out = sub.loc[sub["flag_out95"]].copy()

            # reduce clutter: label each patient at most once per plot (worst point)
            if not out.empty:
                # pick one point per patient (largest absolute deviation on plot scale)
                out["dev"] = np.abs(out["obs"] - out["pred"])
                out_best = out.sort_values("dev", ascending=False).groupby("patient", as_index=False).head(1)

                for _, r in out_best.iterrows():
                    plt.text(r["obs"], r["pred"], str(r["patient"]), fontsize=8)

            plt.tight_layout()
            plt.savefig(f"{out_prefix}_{model}_{var}.png", dpi=200)
            plt.close()


def run_one_patient(patient_id: str, args) -> Dict:
    try:
        data = load_patient_data(args.data, patient_id, time_unit=args.time_unit)

        theta_ode, out_ode = fit_ode(data)
        ode_metrics = {f"ODE_{k}": v for k, v in out_ode["metrics"].items()}

        result = {
            "patient": patient_id,
            "n_samples": int(len(data.t)),
            "n_contexts": int(len(data.context_names)),
            "ode_success": bool(out_ode["success"]),
            "ode_message": str(out_ode["message"]),
            **ode_metrics,
        }

        if not args.no_pde:
            theta_pde, out_pde = fit_pde(data, M=args.pde_grid)
            pde_metrics = {f"PDE_{k}": v for k, v in out_pde["metrics"].items()}
            result.update({
                "pde_success": bool(out_pde["success"]),
                "pde_message": str(out_pde["message"]),
                **pde_metrics,
            })

        return result

    except Exception as e:
        return {
            "patient": patient_id,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }

def fit_and_collect_points(patient_id: str, args) -> List[Dict]:
    """
    Fit models for one patient and return per-timepoint obs/pred rows
    for GOF scatter plots + out-of-95% flags.
    """
    data = load_patient_data(args.data, patient_id, time_unit=args.time_unit)
    rows = []

    # ---- ODE ----
    theta_ode, out_ode = fit_ode(data)
    r_hat_ode, logca_hat_ode = simulate_ode(data, theta_ode)
    sigma_ca_ode = float(np.exp(theta_ode[8]))  # theta[8] is log_sigma_ca in ODE

    y_obs = logit(data.ratio)
    y_hat = logit(r_hat_ode)
    out_ratio = np.abs(y_obs - y_hat) > (1.96 * data.se_logit_ratio)
    out_ca = np.abs(data.log_ca125 - logca_hat_ode) > (1.96 * sigma_ca_ode)

    for i, t in enumerate(data.t):
        rows.append({
            "patient": patient_id,
            "time": float(t),
            "model": "ODE",
            "var": "ratio",
            "obs": float(data.ratio[i]),
            "pred": float(r_hat_ode[i]),
            "flag_out95": bool(out_ratio[i]),
        })
        rows.append({
            "patient": patient_id,
            "time": float(t),
            "model": "ODE",
            "var": "logCA125",
            "obs": float(data.log_ca125[i]),
            "pred": float(logca_hat_ode[i]),
            "flag_out95": bool(out_ca[i]),
        })

    # ---- PDE ----
    if not args.no_pde:
        theta_pde, out_pde = fit_pde(data, M=args.pde_grid)
        r_hat_pde, logca_hat_pde = simulate_pde(data, theta_pde, M=args.pde_grid)
        sigma_ca_pde = float(np.exp(theta_pde[10]))  # theta[10] is log_sigma_ca in PDE

        y_hat2 = logit(r_hat_pde)
        out_ratio2 = np.abs(y_obs - y_hat2) > (1.96 * data.se_logit_ratio)
        out_ca2 = np.abs(data.log_ca125 - logca_hat_pde) > (1.96 * sigma_ca_pde)

        for i, t in enumerate(data.t):
            rows.append({
                "patient": patient_id,
                "time": float(t),
                "model": "PDE",
                "var": "ratio",
                "obs": float(data.ratio[i]),
                "pred": float(r_hat_pde[i]),
                "flag_out95": bool(out_ratio2[i]),
            })
            rows.append({
                "patient": patient_id,
                "time": float(t),
                "model": "PDE",
                "var": "logCA125",
                "obs": float(data.log_ca125[i]),
                "pred": float(logca_hat_pde[i]),
                "flag_out95": bool(out_ca2[i]),
            })

    return rows

# -------------------------- Main --------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to Subclonal_ratio_estimates.extended.txt")
    ap.add_argument("--time_unit", default="months", choices=["months", "days"])
    ap.add_argument("--pde_grid", type=int, default=81)
    ap.add_argument("--no_pde", action="store_true")

    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--patient", help="Run a single patient, e.g. UP0018")
    group.add_argument("--flag", help="Run all patients with Accept_estimate in this comma-separated list, e.g. yes or yes,maybe")

    ap.add_argument("--n_jobs", type=int, default=-1, help="Parallel workers for --flag mode")
    ap.add_argument("--out_csv", default=None, help="Output CSV path for --flag mode")
    ap.add_argument("--out_points", default=None, help="CSV for per-timepoint obs/pred points")
    args = ap.parse_args()

    # ---- single patient mode ----
    if args.patient:
        data = load_patient_data(args.data, args.patient, time_unit=args.time_unit)
        print(f"Patient={data.patient}  N={len(data.t)}  time_unit={args.time_unit}")
        print("Context levels:", data.context_names)

        theta_ode, out_ode = fit_ode(data)
        pretty_print("ODE fit (S/R competition)", data.context_names, theta_ode, out_ode["metrics"])
        print("ODE optimizer:", out_ode["success"], out_ode["message"])

        if not args.no_pde:
            theta_pde, out_pde = fit_pde(data, M=args.pde_grid)
            pretty_print(f"PDE fit (trait-structured, M={args.pde_grid})", data.context_names, theta_pde, out_pde["metrics"])
            print("PDE optimizer:", out_pde["success"], out_pde["message"])

            print("\nModel comparison (lower is better):")
            print(f"  ODE  AIC={out_ode['metrics']['AIC']:.3f}  BIC={out_ode['metrics']['BIC']:.3f}")
            print(f"  PDE  AIC={out_pde['metrics']['AIC']:.3f}  BIC={out_pde['metrics']['BIC']:.3f}")

        print("\nDone.")
        return

    # ---- all patients mode ----
    flags = [x.strip() for x in args.flag.split(",") if x.strip()]
    patients = get_patients_with_flag(args.data, flags=flags)

    with tqdm_joblib(tqdm(total=len(patients), desc="Patients")):
        all_rows_nested = Parallel(n_jobs=args.n_jobs, backend="loky", prefer="processes")(
            delayed(fit_and_collect_points)(pid, args) for pid in patients
        )

    # flatten
    all_rows = [row for rows in all_rows_nested for row in rows]
    df_points = pd.DataFrame(all_rows)

    # save long table (useful for debugging / re-plotting)
    out_points = args.out_points or f"gof_points_flags_{'_'.join(flags)}.csv"
    df_points.to_csv(out_points, index=False)

    # plot
    plot_gof_scatter_all(df_points, out_prefix=f"gof_flags_{'_'.join(flags)}")

    print(f"Saved points: {out_points}")
    print("Saved plots: gof_*.png")


if __name__ == "__main__":
    main()
