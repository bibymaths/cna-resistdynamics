from __future__ import annotations

from typing import Any, Tuple

import numpy as np
import pandas as pd

from .metrics import nll_ratio_ca
from .odeio import PatientData
from .odemodel import make_u_of_t
from .pdegrid import pde_observables_from_grid
from .pdemodel import PDEConfig, get_treatment_value



def solve_pde(
    params: list[float] | np.ndarray,
    cfg: PDEConfig,
    data: PatientData,
    *,
    comm: Any = None,
    return_history: bool = False,
) -> tuple[float, dict | None, pd.DataFrame | None, dict | None]:
    """
    Solve 1D reaction-diffusion PDE (operator splitting):
      - Explicit reaction (growth/death)
      - Implicit diffusion (Backward Euler with CG+Jacobi)

    Treatment:
      - if cfg.u_ctx is provided: u(t) is piecewise-constant (ODE parity) based on most recent observed context
      - else: uses get_treatment_value(t) (currently constant 1.0)

    Returns:
      (nll, stats, df_traj, history)
    """
    # Local imports so non-PDE users don't need FEniCS installed.
    from mpi4py import MPI
    from dolfinx import fem, mesh
    from petsc4py import PETSc  # PETSc.KSP lives here
    from dolfinx.fem import petsc as dx_petsc  # dolfinx assembly helpers live here
    import ufl
    import basix.ufl

    if comm is None:
        comm = MPI.COMM_SELF

    # ---- params + basic sanity ----
    aS, aR, dS, dR, K = map(float, params)
    if (aS <= 0) or (aR <= 0) or (dS < 0) or (dR < 0) or (K <= 1e-6):
        return 1e12, None, None, None

    # ---- initial conditions from first observation ----
    r0 = float(np.asarray(data.ratio)[0])
    r0 = float(np.clip(r0, 1e-9, 1.0 - 1e-9))

    # Need CA125 at t0 to invert N0. Prefer raw CA125 if present; fallback to exp(log_ca125).
    if hasattr(data, "ca125") and getattr(data, "ca125") is not None:
        ca125_0 = float(np.asarray(data.ca125)[0])
    else:
        ca125_0 = float(np.exp(np.asarray(data.log_ca125)[0]))

    N0 = max((ca125_0 - float(cfg.ca0)) / max(float(cfg.gamma), 1e-12), 1e-12)
    S0_val = (1.0 - r0) * N0
    R0_val = r0 * N0

    # ---- discretization ----
    L = float(cfg.L)
    n_cells = int(cfg.n_cells)
    dt = float(cfg.dt)
    if dt <= 0:
        raise ValueError("cfg.dt must be > 0")

    domain = mesh.create_interval(comm, n_cells, [0.0, L])
    element = basix.ufl.element("Lagrange", domain.topology.cell_name(), 1)
    V = fem.functionspace(domain, element)

    S = fem.Function(V)
    R = fem.Function(V)
    S_prev = fem.Function(V)
    R_prev = fem.Function(V)

    S.interpolate(lambda x: np.full(x.shape[1], S0_val, dtype=float))
    R.interpolate(lambda x: np.full(x.shape[1], R0_val, dtype=float))
    S_prev.x.array[:] = S.x.array[:]
    R_prev.x.array[:] = R.x.array[:]

    # ---- time axis: shift so first sample is 0 ----
    t0 = float(np.asarray(data.t, float)[0])
    t_samples = np.asarray(data.t, float) - t0
    obs_times = t_samples.tolist()
    T_total = float(obs_times[-1])

    num_steps = int(np.ceil(T_total / dt))

    # ---- treatment u(t): build ONCE ----
    if cfg.u_ctx is None:
        # get_treatment_value(t) -> float (constant 1.0 by default)
        def u_fun(tt: float) -> float:
            return float(get_treatment_value(tt))
    else:
        u_ctx = np.asarray(cfg.u_ctx, float)
        u_ctx = np.clip(u_ctx, 0.0, 1.0)
        # piecewise-constant schedule by most recent observed context (ODE parity)
        u_fun = make_u_of_t(t_samples, np.asarray(data.context, int), u_ctx)

    # ---- diffusion forms (Backward Euler) ----
    u_trial = ufl.TrialFunction(V)
    v_test = ufl.TestFunction(V)

    a_S = (
        ufl.inner(u_trial, v_test) * ufl.dx
        + dt * float(cfg.DS) * ufl.inner(ufl.grad(u_trial), ufl.grad(v_test)) * ufl.dx
    )
    a_R = (
        ufl.inner(u_trial, v_test) * ufl.dx
        + dt * float(cfg.DR) * ufl.inner(ufl.grad(u_trial), ufl.grad(v_test)) * ufl.dx
    )

    L_S_form = ufl.inner(S_prev, v_test) * ufl.dx
    L_R_form = ufl.inner(R_prev, v_test) * ufl.dx

    rhs_form_S = fem.form(L_S_form)
    rhs_form_R = fem.form(L_R_form)

    A_S = dx_petsc.assemble_matrix(fem.form(a_S))
    A_S.assemble()
    A_R = dx_petsc.assemble_matrix(fem.form(a_R))
    A_R.assemble()

    solver_S = PETSc.KSP().create(domain.comm)
    solver_S.setOperators(A_S)
    solver_S.setType("cg")
    solver_S.getPC().setType("jacobi")
    solver_S.setFromOptions()

    solver_R = PETSc.KSP().create(domain.comm)
    solver_R.setOperators(A_R)
    solver_R.setType("cg")
    solver_R.getPC().setType("jacobi")
    solver_R.setFromOptions()

    b_S = A_S.createVecRight()
    b_R = A_R.createVecRight()

    # ---- storage at obs times ----
    obs_idx = 0
    states_at_obs: list[tuple[np.ndarray, np.ndarray]] = []

    # ---- optional heatmap history ----
    hist: dict | None = None
    sorted_idx: np.ndarray | None = None
    next_hist_t = 0.0
    hist_dt = (T_total / 200.0) if (return_history and T_total > 0) else 0.0

    if return_history:
        xgeom = domain.geometry.x[:, 0]
        sorted_idx = np.argsort(xgeom)
        hist = {"x": xgeom[sorted_idx].copy(), "t": [], "S": [], "R": []}

    # ---- time stepping ----
    t = 0.0
    for _ in range(num_steps + 1):
        # record at observation times
        if obs_idx < len(obs_times) and abs(t - obs_times[obs_idx]) <= 0.5 * dt:
            states_at_obs.append((S.x.array.copy(), R.x.array.copy()))
            obs_idx += 1
            if obs_idx >= len(obs_times):
                break

        # heatmap history (sparse)
        if return_history and hist is not None and sorted_idx is not None and hist_dt > 0:
            if t >= next_hist_t or len(hist["t"]) == 0:
                hist["t"].append(float(t))
                hist["S"].append(S.x.array[sorted_idx].copy())
                hist["R"].append(R.x.array[sorted_idx].copy())
                next_hist_t = t + hist_dt

        # advance time
        t_next = min(t + dt, T_total)
        u_val = float(u_fun(t_next))

        s_vals = S.x.array
        r_vals = R.x.array
        n_vals = s_vals + r_vals

        # ODE parity logistic clamp: g = max(0, 1 - N/K)
        g = np.maximum(0.0, 1.0 - n_vals / K)

        # reaction (explicit Euler)
        growth_s = aS * s_vals * g
        growth_r = aR * r_vals * g
        death_s = u_val * dS * s_vals
        death_r = u_val * dR * r_vals

        S_prev.x.array[:] = s_vals + (t_next - t) * (growth_s - death_s)
        R_prev.x.array[:] = r_vals + (t_next - t) * (growth_r - death_r)

        # diffusion (implicit)
        with b_S.localForm() as loc:
            loc.set(0.0)
        dx_petsc.assemble_vector(b_S, rhs_form_S)
        solver_S.solve(b_S, S.x.petsc_vec)

        with b_R.localForm() as loc:
            loc.set(0.0)
        dx_petsc.assemble_vector(b_R, rhs_form_R)
        solver_R.solve(b_R, R.x.petsc_vec)

        # constraints
        S.x.array[S.x.array < 0] = 0.0
        R.x.array[R.x.array < 0] = 0.0

        if np.any(~np.isfinite(S.x.array)) or np.any(~np.isfinite(R.x.array)):
            return 1e12, None, None, None

        t = t_next

    if len(states_at_obs) != len(obs_times):
        return 1e12, None, None, None

    # ---- observables at obs times ----
    dx = L / n_cells
    ratio_hat: list[float] = []
    logca_hat: list[float] = []
    for S_vals, R_vals in states_at_obs:
        _, _, r_pred, lc_pred = pde_observables_from_grid(
            S_vals, R_vals, dx, gamma=float(cfg.gamma), ca0=float(cfg.ca0)
        )
        ratio_hat.append(float(r_pred))
        logca_hat.append(float(lc_pred))

    ratio_hat_arr = np.asarray(ratio_hat, float)
    logca_hat_arr = np.asarray(logca_hat, float)

    # ---- likelihood ----
    nll = nll_ratio_ca(
        ratio_obs=np.asarray(data.ratio, float),
        se_logit_ratio=np.asarray(data.se_logit_ratio, float),
        logca_obs=np.asarray(data.log_ca125, float),
        ratio_hat=ratio_hat_arr,
        logca_hat=logca_hat_arr,
        sigma_ca=float(cfg.sigma_ca),
        w_ca=float(cfg.w_ca),
    )

    stats = {
        "rmse_ratio": float(np.sqrt(np.mean((np.asarray(data.ratio, float) - ratio_hat_arr) ** 2))),
        "rmse_ca": float(np.sqrt(np.mean((np.asarray(data.log_ca125, float) - logca_hat_arr) ** 2))),
    }

    df_traj = pd.DataFrame(
        {
            "time": np.asarray(data.t, float),
            "ratio_obs": np.asarray(data.ratio, float),
            "ratio_pred": ratio_hat_arr,
            "logca_obs": np.asarray(data.log_ca125, float),
            "logca_pred": logca_hat_arr,
            "patient": str(data.patient),
        }
    )

    if return_history and hist is not None:
        hist["t"] = np.asarray(hist["t"], float) + t0
        hist["S"] = np.asarray(hist["S"], float)
        hist["R"] = np.asarray(hist["R"], float)

    return float(nll), stats, df_traj, hist