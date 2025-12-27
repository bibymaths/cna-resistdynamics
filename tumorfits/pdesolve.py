# tumorfits/pdesolve.py
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .metrics import nll_ratio_ca
from .odeio import PatientData
from .odemodel import make_u_of_t
from .pdemodel import PDEConfig, get_treatment_value

try:
    from numba import njit
except Exception:
    njit = None

# ----------------------------- numba kernels (optional) -----------------------------

if njit is not None:
    @njit(cache=True)
    def _u_piecewise(tt: float, t_samples: np.ndarray, ctx_samples: np.ndarray, u_ctx: np.ndarray) -> float:
        """
        Piecewise constant u(t) using 'most recent observed sample' context.
        Equivalent to: i = searchsorted(t_samples, tt, side="right") - 1
        """
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


    @njit(cache=True)
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
        """
        Explicit Euler reaction step for each spatial DoF:
          dS/dt = aS*S*max(0,1-(S+R)/K) - u*dS*S
          dR/dt = aR*R*max(0,1-(S+R)/K) - u*dR*R
        Writes new reaction-updated fields to outS/outR (typically S_prev/R_prev arrays).
        Applies non-negativity clamp.
        """
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


    @njit(cache=True)
    def _observables_from_stacks(
            S_stack: np.ndarray,  # (T, X)
            R_stack: np.ndarray,  # (T, X)
            dx: float,
            gamma: float,
            ca0: float,
            ratio_out: np.ndarray,  # (T,)
            logca_out: np.ndarray,  # (T,)
    ) -> None:
        """
        For each time slice t:
          total_S = sum(max(S,0))*dx
          total_R = sum(max(R,0))*dx
          ratio = total_R / (total_S+total_R)
          logCA = log(max(ca0 + gamma*(total_S+total_R), 1e-12))
        """
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


# ----------------------------- main solver -----------------------------

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
    history is optional and includes sparse arrays for heatmaps if requested.
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

    # ---- params + sanity ----
    aS, aR, dS, dR, K = map(float, params)
    if (aS <= 0) or (aR <= 0) or (dS < 0) or (dR < 0) or (K <= 1e-6):
        return 1e12, None, None, None

    # ---- initial conditions from first observation ----
    ratio0 = float(np.asarray(data.ratio, dtype=float)[0])
    ratio0 = float(np.clip(ratio0, 1e-9, 1.0 - 1e-9))

    # prefer raw CA125 if present; fallback to exp(log)
    ca125_0: float
    if hasattr(data, "ca125") and getattr(data, "ca125") is not None:
        ca125_0 = float(np.asarray(getattr(data, "ca125"), dtype=float)[0])
    else:
        ca125_0 = float(np.exp(np.asarray(data.log_ca125, dtype=float)[0]))

    gamma = float(cfg.gamma)
    ca0 = float(cfg.ca0)

    N0 = max((ca125_0 - ca0) / max(gamma, 1e-12), 1e-12)
    S0_val = (1.0 - ratio0) * N0
    R0_val = ratio0 * N0

    # ---- discretization ----
    L = float(cfg.L)
    n_cells = int(cfg.n_cells)
    dt = float(cfg.dt)
    if dt <= 0.0:
        raise ValueError("cfg.dt must be > 0")
    if n_cells <= 1:
        raise ValueError("cfg.n_cells must be > 1")
    if L <= 0.0:
        raise ValueError("cfg.L must be > 0")

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
    t_raw = np.asarray(data.t, dtype=float)
    t0 = float(t_raw[0])
    t_samples = (t_raw - t0).astype(np.float64)
    obs_times = t_samples.tolist()
    if len(obs_times) == 0:
        return 1e12, None, None, None
    T_total = float(obs_times[-1])
    if T_total < 0.0:
        return 1e12, None, None, None

    num_steps = int(np.ceil(T_total / dt)) if T_total > 0 else 0

    # ---- treatment u(t): build once ----
    t_samples_arr = np.asarray(t_samples, dtype=np.float64)
    ctx_samples_arr = np.asarray(data.context, dtype=np.int64)

    if cfg.u_ctx is None:
        # cannot JIT arbitrary python schedule
        def u_val_at(tt: float) -> float:
            return float(get_treatment_value(tt))
    else:
        u_ctx_arr = np.asarray(cfg.u_ctx, dtype=np.float64)
        u_ctx_arr = np.clip(u_ctx_arr, 0.0, 1.0)

        if njit is not None:
            def u_val_at(tt: float) -> float:
                return float(_u_piecewise(float(tt), t_samples_arr, ctx_samples_arr, u_ctx_arr))
        else:
            u_fun_py = make_u_of_t(t_samples_arr, ctx_samples_arr, u_ctx_arr)

            def u_val_at(tt: float) -> float:
                return float(u_fun_py(float(tt)))

    # ---- diffusion forms (Backward Euler) ----
    u_trial = ufl.TrialFunction(V)
    v_test = ufl.TestFunction(V)

    DS = float(cfg.DS)
    DR = float(cfg.DR)

    a_S = (
            ufl.inner(u_trial, v_test) * ufl.dx
            + dt * DS * ufl.inner(ufl.grad(u_trial), ufl.grad(v_test)) * ufl.dx
    )
    a_R = (
            ufl.inner(u_trial, v_test) * ufl.dx
            + dt * DR * ufl.inner(ufl.grad(u_trial), ufl.grad(v_test)) * ufl.dx
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
    hist_dt = (T_total / 200.0) if (return_history and T_total > 0.0) else 0.0

    if return_history:
        xgeom = domain.geometry.x[:, 0]
        sorted_idx = np.argsort(xgeom)
        hist = {"x": xgeom[sorted_idx].copy(), "t": [], "S": [], "R": []}

    # ---- time stepping ----
    t = 0.0
    for _ in range(num_steps + 1):
        # record at observation times (within half-step tolerance)
        if obs_idx < len(obs_times) and abs(t - obs_times[obs_idx]) <= 0.5 * dt:
            states_at_obs.append((S.x.array.copy(), R.x.array.copy()))
            obs_idx += 1
            if obs_idx >= len(obs_times):
                break

        # heatmap history (sparse)
        if return_history and hist is not None and sorted_idx is not None and hist_dt > 0.0:
            if t >= next_hist_t or len(hist["t"]) == 0:
                hist["t"].append(float(t))
                hist["S"].append(S.x.array[sorted_idx].copy())
                hist["R"].append(R.x.array[sorted_idx].copy())
                next_hist_t = t + hist_dt

        # advance time
        t_next = min(t + dt, T_total)
        dt_step = float(t_next - t)
        if dt_step <= 0.0:
            t = t_next
            continue

        u_val = float(u_val_at(t_next))

        s_vals = S.x.array
        r_vals = R.x.array

        # reaction (explicit Euler) -> write into S_prev/R_prev arrays
        if njit is not None:
            _reaction_step_inplace(
                s_vals, r_vals,
                dt_step,
                float(aS), float(aR), float(dS), float(dR), float(K),
                u_val,
                S_prev.x.array, R_prev.x.array,
            )
        else:
            n_vals = s_vals + r_vals
            g_vec = np.maximum(0.0, 1.0 - n_vals / K)
            S_prev.x.array[:] = s_vals + dt_step * ((aS * g_vec) * s_vals - (u_val * dS) * s_vals)
            R_prev.x.array[:] = r_vals + dt_step * ((aR * g_vec) * r_vals - (u_val * dR) * r_vals)
            S_prev.x.array[S_prev.x.array < 0] = 0.0
            R_prev.x.array[R_prev.x.array < 0] = 0.0

        # diffusion (implicit): solve A * new = reaction_result
        with b_S.localForm() as loc:
            loc.set(0.0)
        dx_petsc.assemble_vector(b_S, rhs_form_S)
        solver_S.solve(b_S, S.x.petsc_vec)

        with b_R.localForm() as loc:
            loc.set(0.0)
        dx_petsc.assemble_vector(b_R, rhs_form_R)
        solver_R.solve(b_R, R.x.petsc_vec)

        # constraints + divergence check
        S.x.array[S.x.array < 0] = 0.0
        R.x.array[R.x.array < 0] = 0.0

        if np.any(~np.isfinite(S.x.array)) or np.any(~np.isfinite(R.x.array)):
            return 1e12, None, None, None

        t = t_next

    if len(states_at_obs) != len(obs_times):
        return 1e12, None, None, None

    # ---- observables at obs times ----
    dx = float(L / n_cells)

    # stack once: (T, X). This also lets us JIT the reduction.
    S_stack = np.asarray([sr[0] for sr in states_at_obs], dtype=np.float64)
    R_stack = np.asarray([sr[1] for sr in states_at_obs], dtype=np.float64)

    ratio_hat_arr = np.empty(S_stack.shape[0], dtype=np.float64)
    logca_hat_arr = np.empty(S_stack.shape[0], dtype=np.float64)

    if njit is not None:
        _observables_from_stacks(
            S_stack, R_stack,
            dx,
            float(gamma),
            float(ca0),
            ratio_hat_arr,
            logca_hat_arr,
        )
    else:
        for i in range(S_stack.shape[0]):
            S_vals = S_stack[i]
            R_vals = R_stack[i]
            totalS = float(np.sum(np.clip(S_vals, 0.0, None)) * dx)
            totalR = float(np.sum(np.clip(R_vals, 0.0, None)) * dx)
            N = max(totalS + totalR, 1e-12)
            ratio_hat_arr[i] = totalR / N
            logca_hat_arr[i] = np.log(max(ca0 + gamma * N, 1e-12))

    # ---- likelihood ----
    nll = nll_ratio_ca(
        ratio_obs=np.asarray(data.ratio, dtype=np.float64),
        se_logit_ratio=np.asarray(data.se_logit_ratio, dtype=np.float64),
        logca_obs=np.asarray(data.log_ca125, dtype=np.float64),
        ratio_hat=ratio_hat_arr,
        logca_hat=logca_hat_arr,
        sigma_ca=float(cfg.sigma_ca),
        w_ca=float(cfg.w_ca),
    )

    # ---- stats ----
    ratio_obs_arr = np.asarray(data.ratio, dtype=np.float64)
    logca_obs_arr = np.asarray(data.log_ca125, dtype=np.float64)

    stats = {
        "rmse_ratio": float(np.sqrt(np.mean((ratio_obs_arr - ratio_hat_arr) ** 2))),
        "rmse_ca": float(np.sqrt(np.mean((logca_obs_arr - logca_hat_arr) ** 2))),
    }

    # ---- output table ----
    df_traj = pd.DataFrame(
        {
            "time": t_raw,
            "ratio_obs": ratio_obs_arr,
            "ratio_pred": ratio_hat_arr,
            "logca_obs": logca_obs_arr,
            "logca_pred": logca_hat_arr,
            "patient": str(data.patient),
        }
    )

    # ---- history ----
    if return_history and hist is not None:
        hist["t"] = np.asarray(hist["t"], dtype=np.float64) + t0
        hist["S"] = np.asarray(hist["S"], dtype=np.float64)
        hist["R"] = np.asarray(hist["R"], dtype=np.float64)

    return float(nll), stats, df_traj, hist
