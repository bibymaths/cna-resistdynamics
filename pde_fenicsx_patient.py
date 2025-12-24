#!/usr/bin/env python3
import argparse
import numpy as np

from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem, mesh
import ufl

from pde_common import load_patient_data, pde_observables_from_grid, nll_ratio_ca

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--patient", required=True)
    ap.add_argument("--time_unit", default="months", choices=["months", "days"])
    ap.add_argument("--sample_list", default=None)
    ap.add_argument("--use_ca125_updated", action="store_true")
    ap.add_argument("--drop_failed", action="store_true")
    ap.add_argument("--require_panel_sequenced", action="store_true")
    ap.add_argument("--require_detected_cna", action="store_true")

    ap.add_argument("--L", type=float, default=1.0)
    ap.add_argument("--n_cells", type=int, default=200)
    ap.add_argument("--dt", type=float, default=1e-3)

    ap.add_argument("--aS", type=float, default=0.5)
    ap.add_argument("--aR", type=float, default=0.3)
    ap.add_argument("--dS", type=float, default=0.4)
    ap.add_argument("--dR", type=float, default=0.1)
    ap.add_argument("--K", type=float, default=1.0)
    ap.add_argument("--DS", type=float, default=1e-2)
    ap.add_argument("--DR", type=float, default=1e-2)

    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--ca0", type=float, default=0.0)
    ap.add_argument("--sigma_ca", type=float, default=0.5)
    ap.add_argument("--w_ca", type=float, default=1.0)
    args = ap.parse_args()

    data = load_patient_data(
        args.data, args.patient,
        time_unit=args.time_unit,
        sample_list_path=args.sample_list,
        use_ca125_updated=args.use_ca125_updated,
        drop_failed=args.drop_failed,
        require_panel_sequenced=args.require_panel_sequenced,
        require_detected_cna=args.require_detected_cna,
    )

    r0 = float(data.ratio[0])
    ca125_0 = float(data.ca125[0])
    N0 = max((ca125_0 - args.ca0) / max(args.gamma, 1e-12), 1e-12)
    S0_val = (1.0 - r0) * N0
    R0_val = r0 * N0

    L = float(args.L)
    n_cells = int(args.n_cells)
    dt = float(args.dt)
    T_total = float(np.max(data.t))
    num_steps = int(np.ceil(T_total / dt))

    domain = mesh.create_interval(MPI.COMM_WORLD, n_cells, [0.0, L])
    V = fem.FunctionSpace(domain, ("CG", 1))

    S = fem.Function(V)
    R = fem.Function(V)
    S_prev = fem.Function(V)
    R_prev = fem.Function(V)

    S.interpolate(lambda x: np.full(x.shape[1], S0_val))
    R.interpolate(lambda x: np.full(x.shape[1], R0_val))
    S_prev.x.array[:] = S.x.array[:]
    R_prev.x.array[:] = R.x.array[:]

    # implicit diffusion step: (u_new - u_prev)*v dx + dt*D*grad(u_new)grad(v) dx = 0
    u_trial = ufl.TrialFunction(V)
    v_test = ufl.TestFunction(V)

    a_S = ufl.inner(u_trial, v_test) * ufl.dx + dt * args.DS * ufl.inner(ufl.grad(u_trial), ufl.grad(v_test)) * ufl.dx
    a_R = ufl.inner(u_trial, v_test) * ufl.dx + dt * args.DR * ufl.inner(ufl.grad(u_trial), ufl.grad(v_test)) * ufl.dx

    L_S = ufl.inner(S_prev, v_test) * ufl.dx
    L_R = ufl.inner(R_prev, v_test) * ufl.dx

    A_S = fem.petsc.assemble_matrix(fem.form(a_S)); A_S.assemble()
    A_R = fem.petsc.assemble_matrix(fem.form(a_R)); A_R.assemble()

    solver_S = PETSc.KSP().create(domain.comm)
    solver_S.setOperators(A_S)
    solver_S.setType(PETSc.KSP.Type.CG)
    solver_S.getPC().setType(PETSc.PC.Type.JACOBI)

    solver_R = PETSc.KSP().create(domain.comm)
    solver_R.setOperators(A_R)
    solver_R.setType(PETSc.KSP.Type.CG)
    solver_R.getPC().setType(PETSc.PC.Type.JACOBI)

    rhs_form_S = fem.form(L_S)
    rhs_form_R = fem.form(L_R)
    b_S = fem.petsc.create_vector(rhs_form_S)
    b_R = fem.petsc.create_vector(rhs_form_R)

    def u_of_t(_t):
        return 1.0

    obs_times = list(map(float, data.t))
    obs_idx = 0
    states_at_obs = []

    # approximate dx for integral (uniform)
    dx = L / n_cells

    t = 0.0
    if domain.comm.rank == 0:
        print(f"[FEniCSx] patient={data.patient} T={T_total:.4g} steps={num_steps} dt={dt}")

    for _ in range(num_steps + 1):
        # snapshot
        if obs_idx < len(obs_times) and abs(t - obs_times[obs_idx]) <= 0.5 * dt:
            states_at_obs.append((S.x.array.copy(), R.x.array.copy()))
            obs_idx += 1

        if obs_idx >= len(obs_times):
            break

        t += dt
        u_val = u_of_t(t)

        # reaction (explicit Euler on dofs)
        s_vals = S.x.array
        r_vals = R.x.array
        n_vals = s_vals + r_vals

        growth_s = args.aS * s_vals * (1.0 - n_vals / args.K)
        growth_r = args.aR * r_vals * (1.0 - n_vals / args.K)
        death_s = u_val * args.dS * s_vals
        death_r = u_val * args.dR * r_vals

        S_prev.x.array[:] = s_vals + dt * (growth_s - death_s)
        R_prev.x.array[:] = r_vals + dt * (growth_r - death_r)

        # diffusion (implicit solve)
        with b_S.localForm() as loc: loc.set(0.0)
        fem.petsc.assemble_vector(b_S, rhs_form_S)
        solver_S.solve(b_S, S.vector)

        with b_R.localForm() as loc: loc.set(0.0)
        fem.petsc.assemble_vector(b_R, rhs_form_R)
        solver_R.solve(b_R, R.vector)

        # non-negativity
        S.x.array[S.x.array < 0] = 0.0
        R.x.array[R.x.array < 0] = 0.0

    if len(states_at_obs) != len(data.t):
        raise RuntimeError(f"Captured {len(states_at_obs)} obs snapshots but expected {len(data.t)}. "
                           f"Reduce dt or loosen snapshot tolerance.")

    ratio_hat, logca_hat = [], []
    for S_vals, R_vals in states_at_obs:
        # NOTE: using dof sum as integral proxy; for CG1 this is a crude approx but ok for baseline
        _, _, r_pred, lc_pred = pde_observables_from_grid(S_vals, R_vals, dx, gamma=args.gamma, ca0=args.ca0)
        ratio_hat.append(r_pred)
        logca_hat.append(lc_pred)

    ratio_hat = np.asarray(ratio_hat)
    logca_hat = np.asarray(logca_hat)

    nll = nll_ratio_ca(
        ratio_obs=data.ratio,
        se_logit_ratio=data.se_logit_ratio,
        logca_obs=data.log_ca125,
        ratio_hat=ratio_hat,
        logca_hat=logca_hat,
        sigma_ca=args.sigma_ca,
        w_ca=args.w_ca,
    )

    if domain.comm.rank == 0:
        print("\n--- FEniCSx NLL ---")
        print(f"patient={data.patient}")
        print(f"NLL={nll:.6g}")
        print(f"RMSE_ratio={np.sqrt(np.mean((data.ratio - ratio_hat)**2)):.6g}")
        print(f"RMSE_logCA={np.sqrt(np.mean((data.log_ca125 - logca_hat)**2)):.6g}")

if __name__ == "__main__":
    main()
