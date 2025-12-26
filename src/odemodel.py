from __future__ import annotations

from typing import Callable, Tuple, List

import numpy as np
from scipy.integrate import solve_ivp

from .odeio import PatientData
from .utils import invlogit, safe_log


ODE_THETA_BASE_NAMES: List[str] = [
    "log_aS",
    "logit_aR_over_aS",
    "log_dS",
    "logit_dR_over_dS",
    "log_K",
    "log_N0",
    "logit_r0",
    "log_gamma",
    "log_ca0",
    "log_sigma_ca",
]


def ode_theta_names(context_names: list[str]) -> list[str]:
    return ODE_THETA_BASE_NAMES + [f"logit_u_ctx[{c}]" for c in context_names]


def make_u_of_t(t_samples: np.ndarray, ctx_samples: np.ndarray, u_ctx: np.ndarray) -> Callable[[float], float]:
    t_samples = np.asarray(t_samples, float)
    ctx_samples = np.asarray(ctx_samples, int)
    u_ctx = np.asarray(u_ctx, float)

    def u(t: float) -> float:
        i = np.searchsorted(t_samples, t, side="right") - 1
        if i < 0:
            i = 0
        c = int(ctx_samples[i])
        return float(np.clip(u_ctx[c], 0.0, 1.0))

    return u


def ode_rhs(t: float, y: np.ndarray, pars: tuple[float, float, float, float, float], u_fun: Callable[[float], float]):
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


def unpack_theta_ode(data: PatientData, theta: np.ndarray):
    """
    Enforces canonical layout:
      base 10 + C context logits
    """
    theta = np.asarray(theta, float)
    C = len(data.context_names)
    if theta.size != 10 + C:
        raise ValueError(f"theta size mismatch: got {theta.size}, expected {10+C}")

    log_aS, logit_aR_ratio, log_dS, logit_dR_ratio, log_K, log_N0, logit_r0, log_gamma, log_ca0, log_sigma = theta[:10]
    logit_u = theta[10:10 + C]  # ✅ correct
    u_ctx = invlogit(logit_u)

    aS = float(np.exp(log_aS))
    aR = float(aS * invlogit(np.array([logit_aR_ratio]))[0])  # 0<aR<aS
    dS = float(np.exp(log_dS))
    dR = float(dS * invlogit(np.array([logit_dR_ratio]))[0])  # 0<dR<dS
    K = float(np.exp(log_K))
    N0 = float(np.exp(log_N0))
    r0 = float(invlogit(np.array([logit_r0]))[0])
    gamma = float(np.exp(log_gamma))
    ca0 = float(np.exp(log_ca0))
    sigma_ca = float(np.exp(log_sigma))

    S0 = N0 * (1.0 - r0)
    R0 = N0 * r0

    return (aS, aR, dS, dR, K, S0, R0, gamma, ca0, sigma_ca, u_ctx)


def simulate_ode(data: PatientData, theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    aS, aR, dS, dR, K, S0, R0, gamma, ca0, _sigma_ca, u_ctx = unpack_theta_ode(data, theta)

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
        raise RuntimeError("ODE solver failed")

    S = np.clip(sol.y[0], 1e-12, None)
    R = np.clip(sol.y[1], 1e-12, None)
    N = S + R
    r = R / N
    log_ca = safe_log(ca0 + gamma * N)

    return r, log_ca


def simulate_states(data: PatientData, theta: np.ndarray):
    """
    Returns state trajectories evaluated at data.t:
      S(t), R(t), N(t), r(t), logCA_hat(t), u_ctx
    """
    aS, aR, dS, dR, K, S0, R0, gamma, ca0, _sigma_ca, u_ctx = unpack_theta_ode(data, theta)
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
