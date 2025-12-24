#!/usr/bin/env python3
"""
Fit an ODE (2-population sensitive/resistant)
model to longitudinal liquidCNA resistant fraction + CA125, and compute goodness-of-fit.

Input data: Subclonal_ratio_estimates.extended.txt (tab-separated)
Required columns: Patient, Time, context, ratio, ratio_min95, ratio_max95, CA125, Accept_estimate

Model:
  ODE:
    dS/dt = S*(aS*(1 - (S+R)/K)) - u(t)*dS*S
    dR/dt = R*(aR*(1 - (S+R)/K)) - u(t)*dR*R
    r(t)=R/(S+R), CA125 ~ gamma*(S+R)
    u(t): piecewise constant treatment intensity based on context of most recent sample

"""

import argparse
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Tuple, List
from functools import partial
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from joblib import Parallel, delayed
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
import traceback
import time
import os
from datetime import datetime
_NLL_FAIL = {"ODE": 0}

# Prevent hidden oversubscription from BLAS/OpenMP inside each worker
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


def log(msg: str, patient: str = "-", model: str = "-"):
    ts = datetime.now().strftime("%H:%M:%S")
    pid = os.getpid()
    print(f"[{ts} pid={pid} {patient} {model}] {msg}", flush=True)

class Timer:
    def __init__(self): self.t0 = time.time()
    def s(self): return time.time() - self.t0


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


def load_patient_data(
    path: str,
    patient: str,
    time_unit: str = "months",
    sample_list_path: str = None,
    use_ca125_updated: bool = False,
    drop_failed: bool = False,
    require_panel_sequenced: bool = False,
    require_detected_cna: bool = False,
) -> PatientData:
    df = pd.read_csv(path, sep="\t")
    df = df[df["Accept_estimate"].isin(["yes", "maybe"])].copy()

    # --- optional merge with sample list for updated CA125 + QC flags ---
    # We need a join key. In your extended table the sample id is typically in column "time"
    # (strings like UP0018HHT2 / ...). We'll merge on SampleName <-> time.
    if sample_list_path is not None:
        sl = load_sample_list(sample_list_path)

        # detect the sample-id column in the main df
        # common: "time" (string sample id), while "Time" is numeric
        sample_col = None
        for cand in ["time", "SampleName", "sample", "sample_id"]:
            if cand in df.columns:
                sample_col = cand
                break
        if sample_col is None:
            raise ValueError("Could not find sample id column in extended table (expected 'time' or similar).")

        df[sample_col] = df[sample_col].astype(str)
        sl = sl.rename(columns={"SampleName": sample_col})

        # merge (keep only samples that appear in both)
        df = df.merge(
            sl[[sample_col, "Patient", "CA125_updated", "Failed", "PanelSequenced", "DetectedCNA"]],
            on=[sample_col, "Patient"],
            how="left",
        )

        # apply QC filters if requested
        def is_true(x):
            if pd.isna(x): return False
            if isinstance(x, (bool, np.bool_)): return bool(x)
            s = str(x).strip().lower()
            return s in ["true", "t", "1", "yes", "y"]

        if drop_failed and "Failed" in df.columns:
            df = df[~df["Failed"].apply(is_true)].copy()

        if require_panel_sequenced and "PanelSequenced" in df.columns:
            df = df[df["PanelSequenced"].apply(is_true)].copy()

        if require_detected_cna and "DetectedCNA" in df.columns:
            df = df[df["DetectedCNA"].apply(is_true)].copy()

        if use_ca125_updated and "CA125_updated" in df.columns:
            # replace CA125 where updated exists
            df["CA125"] = np.where(df["CA125_updated"].notna(), df["CA125_updated"], df["CA125"])

    # --- continue with your original parsing ---
    df["Time"] = pd.to_numeric(df["Time"], errors="coerce")
    df["CA125"] = pd.to_numeric(df["CA125"], errors="coerce")
    df = df.dropna(subset=["Time", "ratio", "ratio_min95", "ratio_max95", "CA125"]).copy()

    df["Patient"] = df["Patient"].astype(str)
    df = df[df["Patient"] == str(patient)].copy()
    if df.empty:
        raise ValueError(f"No rows found for patient={patient} after filtering/merging")

    # rescale time
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

    ratio = np.clip(df["ratio"].to_numpy().astype(float), 1e-4, 1 - 1e-4)
    r_lo = np.clip(df["ratio_min95"].to_numpy().astype(float), 1e-4, 1 - 1e-4)
    r_hi = np.clip(df["ratio_max95"].to_numpy().astype(float), 1e-4, 1 - 1e-4)
    se = ci95_to_se_logit(ratio, r_lo, r_hi)

    maybe = (df["Accept_estimate"].to_numpy() == "maybe")
    se = np.where(maybe, se * 2.0, se)

    ca = df["CA125"].to_numpy().astype(float)
    log_ca = safe_log(ca)

    log(f"Loaded N={len(df)} points, contexts={len(context_names)}, "
        f"t=[{df['Time'].min():.3g},{df['Time'].max():.3g}] "
        f"ratio=[{ratio.min():.3g},{ratio.max():.3g}] "
        f"CA125=[{ca.min():.3g},{ca.max():.3g}] "
        f"maybe={maybe.sum()}", patient=str(patient), model="DATA")

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


def load_sample_list(path: str) -> pd.DataFrame:
    """
    OV_patientDNA_sampleList.txt: tab-separated, columns include:
    SampleName, Patient, Context, Time, CA125_updated, Failed, PanelSequenced, DetectedCNA, ...
    """
    df = pd.read_csv(path, sep="\t")
    # normalize column names defensively
    df.columns = [c.strip() for c in df.columns]
    # force string types for keys
    if "SampleName" in df.columns:
        df["SampleName"] = df["SampleName"].astype(str)
    if "Patient" in df.columns:
        df["Patient"] = df["Patient"].astype(str)
    return df


def load_drivers(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]

    if "Patient" not in df.columns:
        raise ValueError(f"No Patient column. Columns={list(df.columns)}")

    # your file uses GeneName
    gene_col = "GeneName" if "GeneName" in df.columns else ("GeneID" if "GeneID" in df.columns else df.columns[0])

    df = df[["Patient", gene_col]].rename(columns={gene_col: "Driver"}).copy()
    df["Patient"] = df["Patient"].astype(str).str.strip()
    df["Driver"] = df["Driver"].astype(str).str.strip()

    g = (df.groupby("Patient")["Driver"]
           .apply(lambda s: ",".join(sorted(set([x for x in s if x and x.lower() != "nan"]))))
           .reset_index())
    g["n_drivers"] = g["Driver"].apply(lambda x: 0 if not x else len(x.split(",")))
    return g

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
    tm = Timer()

    C = len(data.context_names)
    assert theta.size == 10 + C

    log_aS, log_aR, log_dS, log_dR, log_K, log_N0, logit_r0, log_gamma, log_ca0, log_sigma_ca = theta[:10]
    u_ctx = invlogit(theta[9:9 + C])  # (0,1)

    aS = np.exp(log_aS)
    aR = aS * invlogit(np.array([log_aR]))[0]  # forces 0 < aR < aS
    dS = np.exp(log_dS)
    dR = dS * invlogit(np.array([log_dR]))[0]  # forces 0 < dR < dS
    K = np.exp(log_K)

    N0 = float(np.exp(log_N0))
    r0 = float(invlogit(np.array([logit_r0]))[0])
    gamma = float(np.exp(log_gamma))
    ca0 = float(np.exp(log_ca0))

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

    ode_method = "LSODA"
    # log(f"solve_ivp method={ode_method} success={sol.success} nfev={getattr(sol, 'nfev', None)} "
    #     f"njev={getattr(sol, 'njev', None)} nlu={getattr(sol, 'nlu', None)} "
    #     f"status={getattr(sol, 'status', None)} message={getattr(sol, 'message', '')} dt={tm.s():.2f}s",
    #     patient=data.patient, model="ODE")

    if not sol.success or np.any(~np.isfinite(sol.y)):
        raise RuntimeError("ODE solver failed")

    S = np.clip(sol.y[0], 1e-12, None)
    R = np.clip(sol.y[1], 1e-12, None)
    N = S + R
    r = R / N
    log_ca = safe_log(ca0 + gamma * N)

    # log(f"Pred r=[{r.min():.3g},{r.max():.3g}] N=[{N.min():.3g},{N.max():.3g}]",
    #     patient=data.patient, model="ODE")

    return r, log_ca


def nll_ode(theta: np.ndarray, data: PatientData) -> float:
    try:
        r_hat, logca_hat = simulate_ode(data, theta)
    except Exception as e:
        _NLL_FAIL["ODE"] += 1
        if _NLL_FAIL["ODE"] <= 3 or _NLL_FAIL["ODE"] % 50 == 0:
            log(f"NLL solver fail #{_NLL_FAIL['ODE']}: {type(e).__name__}: {e}",
                patient=data.patient, model="ODE")
        return 1e50

    # ratio likelihood on logit scale using CI-derived SE
    y_obs = logit(data.ratio)
    y_hat = logit(r_hat)
    se = data.se_logit_ratio
    nll_ratio = 0.5 * np.sum(((y_obs - y_hat) / se) ** 2 + 2 * np.log(se) + np.log(2 * np.pi))

    # CA125 likelihood on log scale
    sigma_ca = float(np.exp(theta[9]))  # theta[9] is log_sigma_ca in ODE
    nll_ca = 0.5 * np.sum(((data.log_ca125 - logca_hat) / sigma_ca) ** 2 + 2 * np.log(sigma_ca) + np.log(2 * np.pi))

    w_ca = 0.5  # equal weight
    # or
    # w_ca = 2.0  # force CA125 to matter

    return float(nll_ratio + w_ca * nll_ca)


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

def multistart_minimize(
    fun,
    x0,
    bounds,
    n_starts=1,
    rel_noise=0.3,
    seed=0,
    method="L-BFGS-B",
    maxiter=800,
    tag="",
    patient="-",
    n_jobs_starts=1,          # NEW
    starts_backend="threading" # NEW: "threading" recommended here
):
    rng = np.random.default_rng(seed)
    tm_all = Timer()

    lo = np.array([b[0] for b in bounds], dtype=float)
    hi = np.array([b[1] for b in bounds], dtype=float)

    # build starts (always include x0)
    starts = [np.asarray(x0, dtype=float).copy()]
    for _ in range(max(0, n_starts - 1)):
        z = starts[0].copy()
        noise = rng.normal(0.0, rel_noise, size=z.size)
        z = np.clip(z + noise, lo, hi)
        starts.append(z)

    def one_start(i, s):
        tm = Timer()
        try:
            res = minimize(fun, s, method=method, bounds=bounds, options={"maxiter": maxiter})
            val = float(res.fun) if np.isfinite(res.fun) else np.inf
            msg = (
                f"start {i}/{len(starts)} done: fun={val:.3g} "
                f"success={getattr(res, 'success', False)} nit={getattr(res, 'nit', None)} dt={tm.s():.2f}s"
            )
            return (val, res, msg, None)
        except Exception as e:
            msg = f"start {i}/{len(starts)} exception: {type(e).__name__}: {e} dt={tm.s():.2f}s"
            return (np.inf, None, msg, e)

    # Run starts (optionally parallel)
    if n_jobs_starts is None or n_jobs_starts <= 1 or len(starts) == 1:
        results = [one_start(i, s) for i, s in enumerate(starts, 1)]
    else:
        results = Parallel(
            n_jobs=n_jobs_starts,
            backend="loky",
            prefer="processes",
            batch_size=1,
        )(delayed(one_start)(i, s) for i, s in enumerate(starts, 1))

    # pick best
    best_val = np.inf
    best_res = None
    for (val, res, msg, err) in results:
        log(msg + (f" best_so_far={min(best_val, val):.3g}" if np.isfinite(val) else ""),
            patient=patient, model=tag)
        if res is not None and val < best_val:
            best_val, best_res = val, res

    if best_res is None:
        log("multistart failed: no successful starts (all exceptions)", patient=patient, model=tag)
        return minimize(fun, starts[0], method=method, bounds=bounds, options={"maxiter": 1})

    log(f"multistart done: best_fun={best_val:.3g} total_dt={tm_all.s():.2f}s",
        patient=patient, model=tag)
    return best_res

def nll_ode_wrapped(theta, data):
    return nll_ode(theta, data)

def fit_ode(data: PatientData, n_starts=1, rel_noise=0.25, n_jobs_starts=1) -> Tuple[np.ndarray, Dict]:
    C = len(data.context_names)

    # Initial guesses (months time scale)
    # aS > aR, dS >> dR, big K
    x0 = np.zeros(10 + C)
    x0[0] = np.log(0.5)    # aS
    x0[1] = logit(np.array([0.6]))[0]  # means aR ~ 0.6*aS
    x0[2] = np.log(0.8)    # dS
    x0[3] = logit(np.array([0.05]))[0]  # means dR ~ 0.05*dS
    x0[4] = np.log(1e6)    # K
    x0[5] = np.log(1e4)    # N0
    x0[6] = logit(np.array([np.clip(data.ratio[0], 1e-4, 1-1e-4)]))[0]  # r0
    x0[7] = np.log(1e-3)  # gamma (small slope)
    x0[8] = np.mean(data.log_ca125) - 1.0  # log_ca0 baseline
    x0[9] = np.log(0.5)  # sigma_ca
    x0[10:] = logit(np.full(C, 0.5))

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
    bnds += [(-20, 5)]  # log_gamma
    bnds += [(-5, 15)]  # log_ca0  (baseline CA125)
    bnds += [(-3, 5)]  # log_sigma_ca
    bnds += [(-10, 10)] * C

    res = multistart_minimize(
        fun = partial(nll_ode, data=data),
        x0=x0,
        bounds=bnds,
        n_starts=n_starts,
        rel_noise=rel_noise,
        seed=hash(data.patient) % (2 ** 32),
        method="L-BFGS-B",
        maxiter=1200,
        patient=data.patient,
        tag="ODE",
        n_jobs_starts=n_jobs_starts,
        starts_backend="threading",
    )

    theta = res.x
    u_ctx = invlogit(theta[9:9 + C])
    log(f"u_ctx: min={u_ctx.min():.3g} max={u_ctx.max():.3g}", patient=data.patient, model="ODE")

    nll = float(res.fun)

    r_hat, logca_hat = simulate_ode(data, theta)
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
            plt.title(f"{model} GOF scatter: {var}")

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
            plt.savefig(f"{out_prefix}_{model}_{var}.png", dpi=300)
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

        return result

    except Exception as e:
        return {
            "patient": patient_id,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }

def fit_and_collect_points(patient_id: str, args) -> List[Dict]:
    """
    Fit models for one patient and return:
      - per-timepoint obs/pred rows for GOF scatter plots + out-of-95% flags
      - parameter rows (theta) for ODE and PDE in the same long-table format
        (written as var="theta:<name>", pred=<value>)
    """
    data = load_patient_data(
        args.data, patient_id, time_unit=args.time_unit,
        sample_list_path=args.sample_list,
        use_ca125_updated=args.use_ca125_updated,
        drop_failed=args.drop_failed,
        require_panel_sequenced=args.require_panel_sequenced,
        require_detected_cna=args.require_detected_cna,
    )
    rows: List[Dict] = []

    # ---------------- ODE ----------------
    theta_ode, out_ode = fit_ode(
        data,
        n_starts=args.n_starts,
        rel_noise=args.rel_noise,
        n_jobs_starts=args.n_jobs_starts
    )
    # ---- per-patient diagnostics (plots + optional CSVs) ----
    try:
        save_patient_states_plots(
            data=data,
            theta=theta_ode,
            out_dir=os.path.join("per_patient_plots",
                                 f"flags_{'_'.join([x.strip() for x in args.flag.split(',') if x.strip()])}"),
            tag="ODE",
            save_csv=True,
            dpi=300,
        )
    except Exception as e:
        log(f"Diagnostics plot failed: {type(e).__name__}: {e}", patient=data.patient, model="PLOT")

    # ---- STORE ODE THETA ROWS ----
    ode_names = (
            ["log_aS", "logit_aR_over_aS", "log_dS", "logit_dR_over_dS",
             "log_K", "log_N0", "logit_r0", "log_gamma", "log_ca0", "log_sigma_ca"]
            + [f"logit_u_ctx[{c}]" for c in data.context_names]
    )
    for name, val in zip(ode_names, theta_ode):
        rows.append({
            "patient": patient_id,
            "time": np.nan,
            "model": "ODE",
            "var": f"theta:{name}",
            "obs": np.nan,
            "pred": float(val),
            "flag_out95": False,
        })
    # -----------------------------

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

    return rows

def simulate_states(data: PatientData, theta: np.ndarray):
    """
    Returns state trajectories evaluated at data.t:
      S(t), R(t), N(t)=S+R, r(t)=R/N, logCA125_hat(t)
    Uses the same parameterization + LSODA as your simulate_ode().
    """
    C = len(data.context_names)
    assert theta.size == 10 + C

    log_aS, log_aR, log_dS, log_dR, log_K, log_N0, logit_r0, log_gamma, log_ca0, log_sigma_ca = theta[:10]
    u_ctx = invlogit(theta[9:9 + C])

    aS = np.exp(log_aS)
    aR = aS * invlogit(np.array([log_aR]))[0]         # 0 < aR < aS
    dS = np.exp(log_dS)
    dR = dS * invlogit(np.array([log_dR]))[0]         # 0 < dR < dS
    K = np.exp(log_K)

    N0 = float(np.exp(log_N0))
    r0 = float(invlogit(np.array([logit_r0]))[0])
    gamma = float(np.exp(log_gamma))
    ca0 = float(np.exp(log_ca0))

    S0 = N0 * (1.0 - r0)
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
    if (not sol.success) or np.any(~np.isfinite(sol.y)):
        raise RuntimeError("ODE solver failed in simulate_states()")

    S = np.clip(sol.y[0], 1e-12, None)
    R = np.clip(sol.y[1], 1e-12, None)
    N = S + R
    r = R / N
    log_ca = safe_log(ca0 + gamma * N)

    return S, R, N, r, log_ca, u_ctx


def save_patient_states_plots(
    data: PatientData,
    theta: np.ndarray,
    out_dir: str,
    *,
    tag: str = "ODE",
    save_csv: bool = True,
    dpi: int = 300,
):
    """
    Creates per-patient files under out_dir:
      - <patient>_<tag>_states.png        (S, R, N)
      - <patient>_<tag>_fit.png           (ratio obs/pred + logCA125 obs/pred)
      - <patient>_<tag>_u_ctx.csv         (context -> u value)
      - <patient>_<tag>_states.csv        (time, S, R, N, r_hat, logCA_hat, ratio_obs, logCA_obs, context)
    """
    os.makedirs(out_dir, exist_ok=True)
    pid = str(data.patient)

    S, R, N, r_hat, logca_hat, u_ctx = simulate_states(data, theta)

    # ---------- Save CSV ----------
    if save_csv:
        df = pd.DataFrame({
            "patient": pid,
            "time": data.t.astype(float),
            "context_id": data.context.astype(int),
            "context_name": [data.context_names[i] for i in data.context],
            "S": S.astype(float),
            "R": R.astype(float),
            "N": N.astype(float),
            "r_hat": r_hat.astype(float),
            "ratio_obs": data.ratio.astype(float),
            "logCA_hat": logca_hat.astype(float),
            "logCA_obs": data.log_ca125.astype(float),
        })
        df.to_csv(os.path.join(out_dir, f"{pid}_{tag}_states.csv"), index=False)

        dfu = pd.DataFrame({
            "patient": pid,
            "context_name": data.context_names,
            "u_ctx": u_ctx.astype(float),
            "logit_u_ctx": theta[9:9 + len(data.context_names)].astype(float),
        })
        dfu.to_csv(os.path.join(out_dir, f"{pid}_{tag}_u_ctx.csv"), index=False)

    # ---------- Plot 1: states (S, R, N) ----------
    fig = plt.figure(figsize=(10, 6))
    plt.plot(data.t, S, label="S(t)")
    plt.plot(data.t, R, label="R(t)")
    plt.plot(data.t, N, label="N(t)=S+R")
    plt.xlabel(f"Time ({'months' if True else 'time'})")  # keep generic
    plt.ylabel("Population (a.u.)")
    plt.title(f"{pid} {tag}: simulated states")
    plt.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{pid}_{tag}_states.png"), dpi=dpi)
    plt.close(fig)

    # ---------- Plot 2: fit (ratio + logCA) ----------
    fig = plt.figure(figsize=(10, 6))

    # ratio
    plt.plot(data.t, data.ratio, marker="o", linestyle="-", label="ratio obs")
    plt.plot(data.t, r_hat, marker="o", linestyle="--", label="ratio pred")

    # logCA
    plt.plot(data.t, data.log_ca125, marker="s", linestyle="-", label="logCA125 obs")
    plt.plot(data.t, logca_hat, marker="s", linestyle="--", label="logCA125 pred")

    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.title(f"{pid} {tag}: observed vs predicted")
    plt.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{pid}_{tag}_fit.png"), dpi=dpi)
    plt.close(fig)

    return {
        "patient": pid,
        "out_dir": out_dir,
        "states_png": os.path.join(out_dir, f"{pid}_{tag}_states.png"),
        "fit_png": os.path.join(out_dir, f"{pid}_{tag}_fit.png"),
        "states_csv": os.path.join(out_dir, f"{pid}_{tag}_states.csv") if save_csv else None,
        "u_csv": os.path.join(out_dir, f"{pid}_{tag}_u_ctx.csv") if save_csv else None,
    }
# -------------------------- Main --------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to Subclonal_ratio_estimates.extended.txt")
    ap.add_argument("--time_unit", default="months", choices=["months", "days"])
    ap.add_argument("--sample_list", default=None,
                    help="Path to OV_patientDNA_sampleList.txt for CA125_updated + QC flags merge")
    ap.add_argument("--drivers", default=None,
                    help="Path to Drivers_subclonalCNA.txt to annotate patients with subclonal CNA drivers")

    ap.add_argument("--use_ca125_updated", action="store_true",
                    help="Replace CA125 with CA125_updated when sample_list is provided")
    ap.add_argument("--drop_failed", action="store_true", help="Drop samples flagged Failed==TRUE in sample_list")
    ap.add_argument("--require_panel_sequenced", action="store_true",
                    help="Keep only PanelSequenced==TRUE in sample_list")
    ap.add_argument("--require_detected_cna", action="store_true", help="Keep only DetectedCNA==TRUE in sample_list")

    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--patient", help="Run a single patient, e.g. UP0018")
    group.add_argument("--flag", help="Run all patients with Accept_estimate in this comma-separated list, e.g. yes or yes,maybe")

    ap.add_argument("--n_jobs", type=int, default=-1, help="Parallel workers for --flag mode")
    ap.add_argument("--n_starts", type=int, default=8, help="Number of multistart runs per patient")
    ap.add_argument("--n_jobs_patients", type=int, default=-1, help="Parallel workers across patients")
    ap.add_argument("--n_jobs_starts", type=int, default=1,
                    help="Parallel workers for starts within a patient (threads)")
    ap.add_argument("--rel_noise", type=float, default=0.25, help="Relative noise for random starts")

    ap.add_argument("--out_csv", default=None, help="Output CSV path for --flag mode")
    ap.add_argument("--out_points", default=None, help="CSV for per-timepoint obs/pred points")
    args = ap.parse_args()

    # ---- single patient mode ----
    if args.patient is not None:
        data = load_patient_data(
            args.data, args.patient, time_unit=args.time_unit,
            sample_list_path=args.sample_list,
            use_ca125_updated=args.use_ca125_updated,
            drop_failed=args.drop_failed,
            require_panel_sequenced=args.require_panel_sequenced,
            require_detected_cna=args.require_detected_cna,
        )

        print(f"Patient={data.patient}  N={len(data.t)}  time_unit={args.time_unit}")
        print("Context levels:", data.context_names)

        theta_ode, out_ode = fit_ode(data)
        pretty_print("ODE fit (S/R competition)", data.context_names, theta_ode, out_ode["metrics"])
        print("ODE optimizer:", out_ode["success"], out_ode["message"])

        # print ODE theta
        print("\nODE theta (raw):")
        print(theta_ode)

        print("\nODE theta (named):")
        C1 = len(data.context_names)
        names = (
                ["log_g0", "logit_c", "log_d0", "logit_b", "log_K", "log_D", "log_N0", "logit_r0", "logit_xstar",
                 "log_gamma", "log_sigma_ca"]
                + [f"logit_u_ctx[{c}]" for c in data.context_names]
        )
        for k, v in zip(names, theta_ode):
            print(f"{k:>20s} = {v: .6f}")

    # ---- all patients mode ----
    flags = [x.strip() for x in args.flag.split(",") if x.strip()]
    patients = get_patients_with_flag(args.data, flags=flags)
    drivers_df = None
    if args.drivers:
        drivers_df = load_drivers(args.drivers)
        drivers_map = dict(zip(drivers_df["Patient"], drivers_df["Driver"]))
        ndrivers_map = dict(zip(drivers_df["Patient"], drivers_df["n_drivers"]))
    else:
        drivers_map, ndrivers_map = {}, {}

    with tqdm_joblib(tqdm(total=len(patients), desc="Patients")):
        all_rows_nested = Parallel(n_jobs=args.n_jobs_patients, backend="loky", prefer="processes")(
            delayed(fit_and_collect_points)(pid, args) for pid in patients
        )

    # flatten
    all_rows = [row for rows in all_rows_nested for row in rows]
    df_points = pd.DataFrame(all_rows)

    if args.drivers:
        ddf = load_drivers(args.drivers)
        df_points = df_points.merge(ddf, left_on="patient", right_on="Patient", how="left")
        df_points = df_points.drop(columns=["Patient"])
    # save long table (useful for debugging / re-plotting)
    out_points = args.out_points or f"gof_points_flags_{'_'.join(flags)}.csv"
    df_points.to_csv(out_points, index=False)

    log(f"GOF points: rows={len(df_points)} patients={df_points['patient'].nunique()} "
        f"out95={int(df_points['flag_out95'].sum())}",
        patient="ALL", model="GOF")
    g = (df_points.groupby(["patient", "model", "var"])["flag_out95"]
         .mean().reset_index().sort_values("flag_out95", ascending=False))
    log("Worst out95 rates:\n" + g.head(12).to_string(index=False), patient="ALL", model="GOF")

    # plot
    plot_gof_scatter_all(df_points, out_prefix=f"gof_flags_{'_'.join(flags)}")

    print(f"Saved points: {out_points}")
    print("Saved plots: gof_*.png")


if __name__ == "__main__":
    main()
