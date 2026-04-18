# SPDX-FileCopyrightText: 2025 Abhinav Mishra
# SPDX-License-Identifier: MIT
# tumorfits/pdesolve.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numba import njit

from .metrics import nll_ratio_ca
from .odeio import PatientData
from .pdemodel import PDEConfig, get_treatment_value


# ----------------------------- numba kernels -----------------------------

@njit(cache=True, fastmath=True, nogil=True)
def _u_piecewise(tt: float, t_samples: np.ndarray, ctx_samples: np.ndarray, u_ctx: np.ndarray) -> float:
    lo = 0
    hi = t_samples.shape[0]
    while lo < hi:
        mid = (lo + hi) // 2
        if tt < t_samples[mid]:
            hi = mid
        else:
            lo = mid + 1
    i = lo - 1
    if i < 0:
        i = 0
    c = int(ctx_samples[i])
    u = float(u_ctx[c])
    if u < 0.0:
        u = 0.0
    elif u > 1.0:
        u = 1.0
    return u


@njit(cache=True, fastmath=True, nogil=True)
def _reaction_step_inplace(
    s_vals: np.ndarray,
    r_vals: np.ndarray,
    dt_step: float,
    aS: float,
    aR: float,
    dS: float,
    dR: float,
    K: float,
    u_val: float,
    outS: np.ndarray,
    outR: np.ndarray,
) -> None:
    n = s_vals.shape[0]
    for i in range(n):
        S = float(s_vals[i])
        R = float(r_vals[i])
        N = S + R

        g = 1.0 - N / K
        if g < 0.0:
            g = 0.0

        dSdt = (aS * g) * S - (u_val * dS) * S
        dRdt = (aR * g) * R - (u_val * dR) * R

        ns = S + dt_step * dSdt
        nr = R + dt_step * dRdt

        if ns < 0.0:
            ns = 0.0
        if nr < 0.0:
            nr = 0.0

        outS[i] = ns
        outR[i] = nr


@njit(cache=True, fastmath=True, nogil=True)
def _observables_from_stacks(
    S_stack: np.ndarray,  # (T, X)
    R_stack: np.ndarray,  # (T, X)
    dx: float,
    gamma: float,
    ca0: float,
    ratio_out: np.ndarray,  # (T,)
    logca_out: np.ndarray,  # (T,)
) -> None:
    T = S_stack.shape[0]
    X = S_stack.shape[1]
    for ti in range(T):
        totalS = 0.0
        totalR = 0.0
        for xi in range(X):
            s = float(S_stack[ti, xi])
            r = float(R_stack[ti, xi])
            if s < 0.0:
                s = 0.0
            if r < 0.0:
                r = 0.0
            totalS += s
            totalR += r

        totalS *= dx
        totalR *= dx
        N = totalS + totalR
        if N < 1e-12:
            N = 1e-12

        ratio_out[ti] = totalR / N
        ca = ca0 + gamma * N
        if ca < 1e-12:
            ca = 1e-12
        logca_out[ti] = np.log(ca)


# ----------------------------- cached fenics system -----------------------------

@dataclass
class _PDECacheEntry:
    # core fenics objects
    domain: Any
    V: Any
    S: Any
    R: Any
    S_prev: Any
    R_prev: Any

    # rhs forms
    rhs_form_S: Any
    rhs_form_R: Any

    # petsc objects
    A_S: Any
    A_R: Any
    ksp_S: Any
    ksp_R: Any
    b_S: Any
    b_R: Any

    # optional history helpers
    xgeom_sorted: np.ndarray | None
    sorted_idx: np.ndarray | None


# key: (comm_id, L, n_cells, dt, DS, DR)
_PDE_SYSTEM_CACHE: dict[tuple[int, float, int, float, float, float], _PDECacheEntry] = {}


def _comm_key(comm: Any) -> int:
    # stable-ish key for a communicator; for MPI.COMM_SELF/COMM_WORLD this is fine.
    try:
        return int(comm.py2f())  # mpi4py Comm -> Fortran handle
    except Exception:
        return id(comm)


def _get_or_build_system(comm: Any, cfg: PDEConfig, return_history: bool) -> _PDECacheEntry:
    # Local imports so non-PDE users don't need FEniCS installed.
    from dolfinx import fem, mesh
    from dolfinx.fem import petsc as dx_petsc
    from petsc4py import PETSc
    import ufl
    import basix.ufl

    L = float(cfg.L)
    n_cells = int(cfg.n_cells)
    dt = float(cfg.dt)
    DS = float(cfg.DS)
    DR = float(cfg.DR)

    key = (_comm_key(comm), L, n_cells, dt, DS, DR)

    entry = _PDE_SYSTEM_CACHE.get(key, None)
    if entry is not None:
        # if someone now asks for history but cached entry didn't store geometry, enrich it lazily
        if return_history and entry.sorted_idx is None:
            xgeom = entry.domain.geometry.x[:, 0]
            sidx = np.argsort(xgeom)
            entry.xgeom_sorted = xgeom[sidx].copy()
            entry.sorted_idx = sidx
        return entry

    domain = mesh.create_interval(comm, n_cells, [0.0, L])
    element = basix.ufl.element("Lagrange", domain.topology.cell_name(), 1)
    V = fem.functionspace(domain, element)

    S = fem.Function(V)
    R = fem.Function(V)
    S_prev = fem.Function(V)
    R_prev = fem.Function(V)

    u_trial = ufl.TrialFunction(V)
    v_test = ufl.TestFunction(V)

    a_S = (
        ufl.inner(u_trial, v_test) * ufl.dx
        + dt * DS * ufl.inner(ufl.grad(u_trial), ufl.grad(v_test)) * ufl.dx
    )
    a_R = (
        ufl.inner(u_trial, v_test) * ufl.dx
        + dt * DR * ufl.inner(ufl.grad(u_trial), ufl.grad(v_test)) * ufl.dx
    )

    # IMPORTANT: rhs uses S_prev/R_prev symbols, so forms must be built after those Functions exist
    L_S_form = ufl.inner(S_prev, v_test) * ufl.dx
    L_R_form = ufl.inner(R_prev, v_test) * ufl.dx

    rhs_form_S = fem.form(L_S_form)
    rhs_form_R = fem.form(L_R_form)

    A_S = dx_petsc.assemble_matrix(fem.form(a_S)); A_S.assemble()
    A_R = dx_petsc.assemble_matrix(fem.form(a_R)); A_R.assemble()

    ksp_S = PETSc.KSP().create(domain.comm)
    ksp_S.setOperators(A_S)
    ksp_S.setType("cg")
    ksp_S.getPC().setType("jacobi")
    ksp_S.setFromOptions()

    ksp_R = PETSc.KSP().create(domain.comm)
    ksp_R.setOperators(A_R)
    ksp_R.setType("cg")
    ksp_R.getPC().setType("jacobi")
    ksp_R.setFromOptions()

    b_S = A_S.createVecRight()
    b_R = A_R.createVecRight()

    xgeom_sorted = None
    sorted_idx = None
    if return_history:
        xgeom = domain.geometry.x[:, 0]
        sorted_idx = np.argsort(xgeom)
        xgeom_sorted = xgeom[sorted_idx].copy()

    entry = _PDECacheEntry(
        domain=domain, V=V,
        S=S, R=R, S_prev=S_prev, R_prev=R_prev,
        rhs_form_S=rhs_form_S, rhs_form_R=rhs_form_R,
        A_S=A_S, A_R=A_R,
        ksp_S=ksp_S, ksp_R=ksp_R,
        b_S=b_S, b_R=b_R,
        xgeom_sorted=xgeom_sorted, sorted_idx=sorted_idx,
    )
    _PDE_SYSTEM_CACHE[key] = entry
    return entry


# ----------------------------- main solver (cached) -----------------------------

def solve_pde(
    params: list[float] | np.ndarray,
    cfg: PDEConfig,
    data: PatientData,
    *,
    comm: Any = None,
    return_history: bool = False,
) -> tuple[float, dict | None, pd.DataFrame | None, dict | None]:
    """
    Same behavior as your current solve_pde(), but caches the FEniCS/PETSc system
    (mesh/V/matrices/KSP/vecs) across repeated objective evaluations.
    """
    from mpi4py import MPI
    from dolfinx.fem import petsc as dx_petsc  # assembly helper

    if comm is None:
        comm = MPI.COMM_SELF

    # ---- params + sanity ----
    aS, aR, dS, dR, K = map(float, params)
    if (aS <= 0) or (aR <= 0) or (dS < 0) or (dR < 0) or (K <= 1e-6):
        return 1e12, None, None, None

    # ---- pull cached system ----
    sys = _get_or_build_system(comm, cfg, return_history)
    S = sys.S
    R = sys.R
    S_prev = sys.S_prev
    R_prev = sys.R_prev

    # ---- initial conditions from first observation ----
    ratio0 = float(np.asarray(data.ratio, dtype=float)[0])
    ratio0 = float(np.clip(ratio0, 1e-9, 1.0 - 1e-9))

    if hasattr(data, "ca125") and getattr(data, "ca125") is not None:
        ca125_0 = float(np.asarray(getattr(data, "ca125"), dtype=float)[0])
    else:
        ca125_0 = float(np.exp(np.asarray(data.log_ca125, dtype=float)[0]))

    gamma = float(cfg.gamma)
    ca0 = float(cfg.ca0)

    N0 = max((ca125_0 - ca0) / max(gamma, 1e-12), 1e-12)
    S0_val = (1.0 - ratio0) * N0
    R0_val = ratio0 * N0

    # reset state every call (since cached Functions are reused)
    S.interpolate(lambda x: np.full(x.shape[1], S0_val, dtype=float))
    R.interpolate(lambda x: np.full(x.shape[1], R0_val, dtype=float))
    S_prev.x.array[:] = S.x.array[:]
    R_prev.x.array[:] = R.x.array[:]

    # ---- time axis: shift so first sample is 0 ----
    t_raw = np.asarray(data.t, dtype=float)
    t0 = float(t_raw[0])
    t_samples = (t_raw - t0).astype(np.float64)
    obs_times = t_samples.tolist()
    if len(obs_times) == 0:
        return 1e12, None, None, None
    T_total = float(obs_times[-1])
    if T_total < 0.0:
        return 1e12, None, None, None

    dt = float(cfg.dt)
    num_steps = int(np.ceil(T_total / dt)) if T_total > 0 else 0

    # ---- treatment schedule (fast) ----
    t_samples_arr = np.asarray(t_samples, dtype=np.float64)
    ctx_samples_arr = np.asarray(data.context, dtype=np.int64)

    if cfg.u_ctx is None:
        def u_val_at(tt: float) -> float:
            return float(get_treatment_value(tt))
    else:
        u_ctx_arr = np.asarray(cfg.u_ctx, dtype=np.float64)
        u_ctx_arr = np.clip(u_ctx_arr, 0.0, 1.0)

        def u_val_at(tt: float) -> float:
            return float(_u_piecewise(float(tt), t_samples_arr, ctx_samples_arr, u_ctx_arr))

    # ---- stepping + storage ----
    obs_idx = 0
    states_at_obs: list[tuple[np.ndarray, np.ndarray]] = []

    hist: dict | None = None
    next_hist_t = 0.0
    hist_dt = (T_total / 200.0) if (return_history and T_total > 0.0) else 0.0
    if return_history:
        hist = {"x": sys.xgeom_sorted.copy(), "t": [], "S": [], "R": []}

    t = 0.0
    for _ in range(num_steps + 1):
        if obs_idx < len(obs_times) and abs(t - obs_times[obs_idx]) <= 0.5 * dt:
            states_at_obs.append((S.x.array.copy(), R.x.array.copy()))
            obs_idx += 1
            if obs_idx >= len(obs_times):
                break

        if return_history and hist is not None and sys.sorted_idx is not None and hist_dt > 0.0:
            if t >= next_hist_t or len(hist["t"]) == 0:
                sidx = sys.sorted_idx
                hist["t"].append(float(t))
                hist["S"].append(S.x.array[sidx].copy())
                hist["R"].append(R.x.array[sidx].copy())
                next_hist_t = t + hist_dt

        t_next = min(t + dt, T_total)
        dt_step = float(t_next - t)
        if dt_step <= 0.0:
            t = t_next
            continue

        u_val = float(u_val_at(t_next))

        _reaction_step_inplace(
            S.x.array, R.x.array,
            dt_step,
            float(aS), float(aR), float(dS), float(dR), float(K),
            u_val,
            S_prev.x.array, R_prev.x.array,
        )

        # implicit diffusion using cached matrices + solvers + vecs
        with sys.b_S.localForm() as loc:
            loc.set(0.0)
        dx_petsc.assemble_vector(sys.b_S, sys.rhs_form_S)
        sys.ksp_S.solve(sys.b_S, S.x.petsc_vec)

        with sys.b_R.localForm() as loc:
            loc.set(0.0)
        dx_petsc.assemble_vector(sys.b_R, sys.rhs_form_R)
        sys.ksp_R.solve(sys.b_R, R.x.petsc_vec)

        # constraints + divergence check
        S.x.array[S.x.array < 0] = 0.0
        R.x.array[R.x.array < 0] = 0.0
        if np.any(~np.isfinite(S.x.array)) or np.any(~np.isfinite(R.x.array)):
            return 1e12, None, None, None

        t = t_next

    if len(states_at_obs) != len(obs_times):
        return 1e12, None, None, None

    # ---- observables (JIT reduction) ----
    L = float(cfg.L)
    n_cells = int(cfg.n_cells)
    dx = float(L / n_cells)

    S_stack = np.asarray([sr[0] for sr in states_at_obs], dtype=np.float64)
    R_stack = np.asarray([sr[1] for sr in states_at_obs], dtype=np.float64)

    ratio_hat = np.empty(S_stack.shape[0], dtype=np.float64)
    logca_hat = np.empty(S_stack.shape[0], dtype=np.float64)
    _observables_from_stacks(S_stack, R_stack, dx, float(gamma), float(ca0), ratio_hat, logca_hat)

    # ---- likelihood ----
    ratio_obs = np.asarray(data.ratio, dtype=np.float64)
    se_logit = np.asarray(data.se_logit_ratio, dtype=np.float64)
    logca_obs = np.asarray(data.log_ca125, dtype=np.float64)

    nll = nll_ratio_ca(
        ratio_obs=ratio_obs,
        se_logit_ratio=se_logit,
        logca_obs=logca_obs,
        ratio_hat=ratio_hat,
        logca_hat=logca_hat,
        sigma_ca=float(cfg.sigma_ca),
        w_ca=float(cfg.w_ca),
    )

    stats = {
        "rmse_ratio": float(np.sqrt(np.mean((ratio_obs - ratio_hat) ** 2))),
        "rmse_ca": float(np.sqrt(np.mean((logca_obs - logca_hat) ** 2))),
    }

    df_traj = pd.DataFrame(
        {
            "time": t_raw,
            "ratio_obs": ratio_obs,
            "ratio_pred": ratio_hat,
            "logca_obs": logca_obs,
            "logca_pred": logca_hat,
            "patient": str(data.patient),
        }
    )

    if return_history and hist is not None:
        hist["t"] = np.asarray(hist["t"], dtype=np.float64) + t0
        hist["S"] = np.asarray(hist["S"], dtype=np.float64)
        hist["R"] = np.asarray(hist["R"], dtype=np.float64)

    return float(nll), stats, df_traj, hist
