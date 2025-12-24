#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd
import sys

from matplotlib import pyplot as plt
from scipy.optimize import minimize

from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem, mesh
from dolfinx.fem import petsc
import ufl
import basix.ufl

# Assumed custom module
from pde_common import load_patient_data, pde_observables_from_grid, nll_ratio_ca


def get_treatment_value(t, treatment_schedule=None):
    """
    Returns the treatment intensity u(t) at time t.
    Modify this logic if you have specific start/stop times in 'data'.
    """
    # Placeholder: Currently constant ON.
    # In the future, you can look up 't' in treatment_schedule to return 0.0 or 1.0
    return 1.0


def solve_pde(params, args, data, comm):
    """
    Sets up and solves the PDE for a given set of parameters.
    Returns:
        nll (float): Negative Log Likelihood
        stats (dict): RMSE and other metrics
        df_traj (DataFrame): Detailed time-series of predictions vs observations
    """
    # 1. Unpack Parameters
    # Order matches the initial_guess construction in main()
    aS, aR, dS, dR, K = params

    # Penalize non-physical parameters heavily
    if aS < 0 or aR < 0 or dS < 0 or dR < 0 or K < 1e-4:
        return 1e12, None, None

    # 2. Derived Initial Conditions
    r0 = float(data.ratio[0])
    ca125_0 = float(data.ca125[0])
    N0 = max((ca125_0 - args.ca0) / max(args.gamma, 1e-12), 1e-12)
    S0_val = (1.0 - r0) * N0
    R0_val = r0 * N0

    # 3. Mesh & Function Spaces
    # Note: Creating mesh inside the loop is slightly inefficient for fitting,
    # but safe for dolfinx garbage collection.
    domain = mesh.create_interval(comm, args.n_cells, [0.0, args.L])
    element = basix.ufl.element("Lagrange", domain.topology.cell_name(), 1)
    V = fem.functionspace(domain, element)

    # 4. Define Functions
    S = fem.Function(V)
    R = fem.Function(V)
    S_prev = fem.Function(V)
    R_prev = fem.Function(V)

    # Interpolate ICs
    S.interpolate(lambda x: np.full(x.shape[1], S0_val))
    R.interpolate(lambda x: np.full(x.shape[1], R0_val))
    S_prev.x.array[:] = S.x.array[:]
    R_prev.x.array[:] = R.x.array[:]

    # 5. Variational Forms (Diffusion - Implicit Euler)
    u_trial = ufl.TrialFunction(V)
    v_test = ufl.TestFunction(V)
    dt = args.dt

    # Weak forms
    a_S = ufl.inner(u_trial, v_test) * ufl.dx + dt * args.DS * ufl.inner(ufl.grad(u_trial), ufl.grad(v_test)) * ufl.dx
    a_R = ufl.inner(u_trial, v_test) * ufl.dx + dt * args.DR * ufl.inner(ufl.grad(u_trial), ufl.grad(v_test)) * ufl.dx

    L_S = ufl.inner(S_prev, v_test) * ufl.dx
    L_R = ufl.inner(R_prev, v_test) * ufl.dx

    rhs_form_S = fem.form(L_S)
    rhs_form_R = fem.form(L_R)

    # Assemble Matrix A (Constant in time)
    A_S = petsc.assemble_matrix(fem.form(a_S));
    A_S.assemble()
    A_R = petsc.assemble_matrix(fem.form(a_R));
    A_R.assemble()

    # Create Solvers
    solver_S = PETSc.KSP().create(domain.comm)
    solver_S.setOperators(A_S)
    solver_S.setType(PETSc.KSP.Type.CG)
    solver_S.getPC().setType(PETSc.PC.Type.JACOBI)

    solver_R = PETSc.KSP().create(domain.comm)
    solver_R.setOperators(A_R)
    solver_R.setType(PETSc.KSP.Type.CG)
    solver_R.getPC().setType(PETSc.PC.Type.JACOBI)

    # Create RHS Vectors
    b_S = A_S.createVecRight()
    b_R = A_R.createVecRight()

    # 6. Time Loop Setup
    obs_times = list(map(float, data.t))
    obs_idx = 0
    states_at_obs = []

    T_total = float(np.max(data.t))
    num_steps = int(np.ceil(T_total / dt))
    t = 0.0

    # 7. Run Simulation
    for _ in range(num_steps + 1):
        # A. Snapshot
        if obs_idx < len(obs_times) and abs(t - obs_times[obs_idx]) <= 0.5 * dt:
            states_at_obs.append((S.x.array.copy(), R.x.array.copy()))
            obs_idx += 1

        if obs_idx >= len(obs_times):
            break

        t += dt
        u_val = get_treatment_value(t)

        # B. Reaction Step (Explicit)
        s_vals = S.x.array
        r_vals = R.x.array
        n_vals = s_vals + r_vals

        growth_s = aS * s_vals * (1.0 - n_vals / K)
        growth_r = aR * r_vals * (1.0 - n_vals / K)
        death_s = u_val * dS * s_vals
        death_r = u_val * dR * r_vals

        # Update "Previous" state with reaction result
        S_prev.x.array[:] = s_vals + dt * (growth_s - death_s)
        R_prev.x.array[:] = r_vals + dt * (growth_r - death_r)

        # C. Diffusion Step (Implicit Solve)
        # Update RHS vector b with reacted state
        with b_S.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b_S, rhs_form_S)
        solver_S.solve(b_S, S.x.petsc_vec)

        with b_R.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b_R, rhs_form_R)
        solver_R.solve(b_R, R.x.petsc_vec)

        # D. Non-negativity constraint
        S.x.array[S.x.array < 0] = 0.0
        R.x.array[R.x.array < 0] = 0.0

        # E. Divergence Check
        if np.any(np.isnan(S.x.array)) or np.any(np.isnan(R.x.array)):
            return 1e12, None, None

    # 8. Compute Likelihood
    if len(states_at_obs) != len(data.t):
        return 1e12, None, None  # Simulation ended early

    dx = args.L / args.n_cells
    ratio_hat, logca_hat = [], []

    for S_vals, R_vals in states_at_obs:
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

    # Compute Statistics
    rmse_ratio = np.sqrt(np.mean((data.ratio - ratio_hat) ** 2))
    rmse_ca = np.sqrt(np.mean((data.log_ca125 - logca_hat) ** 2))

    stats = {"rmse_ratio": rmse_ratio, "rmse_ca": rmse_ca}

    # Create Trajectory Dataframe
    df_traj = pd.DataFrame({
        "time": obs_times,
        "ratio_obs": data.ratio,
        "ratio_pred": ratio_hat,
        "logca_obs": data.log_ca125,
        "logca_pred": logca_hat,
        "patient": args.patient
    })

    return nll, stats, df_traj


def main():
    ap = argparse.ArgumentParser(description="FEniCSx 1D Tumor PDE Solver & Fitter")

    # --- Data Arguments ---
    g_data = ap.add_argument_group("Data Loading")
    g_data.add_argument("--data", required=True, help="Path to patient ratio estimates")
    g_data.add_argument("--patient", required=True, help="Patient ID or 'ALL'")
    g_data.add_argument("--sample_list", default=None)
    g_data.add_argument("--use_ca125_updated", action="store_true")
    g_data.add_argument("--time_unit", default="months", choices=["months", "days"])
    g_data.add_argument("--drop_failed", action="store_true")
    g_data.add_argument("--require_panel_sequenced", action="store_true")
    g_data.add_argument("--require_detected_cna", action="store_true")

    # --- PDE Physics Arguments ---
    g_phys = ap.add_argument_group("PDE Physics & Grid")
    g_phys.add_argument("--L", type=float, default=1.0, help="Domain length")
    g_phys.add_argument("--n_cells", type=int, default=200, help="Number of spatial cells")
    g_phys.add_argument("--dt", type=float, default=1e-3, help="Time step size")

    # --- Parameter Initial Guesses / Fixed Values ---
    g_params = ap.add_argument_group("Model Parameters (Initial Guess or Fixed)")
    g_params.add_argument("--aS", type=float, default=0.5)
    g_params.add_argument("--aR", type=float, default=0.3)
    g_params.add_argument("--dS", type=float, default=0.4)
    g_params.add_argument("--dR", type=float, default=0.1)
    g_params.add_argument("--K", type=float, default=1.0)
    g_params.add_argument("--DS", type=float, default=1e-2, help="Diffusion Sensitive")
    g_params.add_argument("--DR", type=float, default=1e-2, help="Diffusion Resistant")

    # --- Observation Model ---
    g_obs = ap.add_argument_group("Observation Model")
    g_obs.add_argument("--gamma", type=float, default=1.0)
    g_obs.add_argument("--ca0", type=float, default=0.0)
    g_obs.add_argument("--sigma_ca", type=float, default=0.5)
    g_obs.add_argument("--w_ca", type=float, default=1.0)

    # --- Execution Mode ---
    ap.add_argument("--fit", action="store_true", help="Run optimizer to fit data")

    args = ap.parse_args()
    comm = MPI.COMM_WORLD

    # Define list of patients to process
    if args.patient == "ALL":
        # Hardcoded list based on your file structure
        patients_to_run = ["UP0018", "UP0042", "UP0053", "UP0055", "UP0056"]
    else:
        patients_to_run = [args.patient]

    # --- MAIN PROCESSING LOOP ---
    for p_id in patients_to_run:
        if comm.rank == 0:
            print(f"\n{'=' * 60}")
            print(f"PROCESSING PATIENT: {p_id}")
            print(f"{'=' * 60}")

        try:
            # 1. Load Data for Specific Patient
            # Note: We create a temporary 'args' view or just pass p_id explicitly if needed
            # but load_patient_data expects the ID string.
            data = load_patient_data(
                args.data, p_id,
                time_unit=args.time_unit,
                sample_list_path=args.sample_list,
                use_ca125_updated=args.use_ca125_updated,
                drop_failed=args.drop_failed,
                require_panel_sequenced=args.require_panel_sequenced,
                require_detected_cna=args.require_detected_cna,
            )

            if comm.rank == 0:
                print(f"Time: {np.max(data.t):.2f} {args.time_unit} | Grid: {args.n_cells} cells")

            # 2. Reset Parameters for each patient (Initial Guess)
            current_params = [args.aS, args.aR, args.dS, args.dR, args.K]

            # 3. Fitting Logic
            if args.fit:
                if comm.rank == 0:
                    print(f"Running Optimization (Nelder-Mead)...")

                def objective(p):
                    # We pass the specific 'data' object for this patient
                    nll, _, _ = solve_pde(p, args, data, comm)
                    # Optional: Print progress only on rank 0
                    # if comm.rank == 0: print(f"Iter: {p} -> {nll:.2f}")
                    return nll

                res = minimize(objective, current_params, method='Nelder-Mead',
                               options={'maxiter': 50, 'disp': (comm.rank == 0)})

                current_params = res.x
                if comm.rank == 0:
                    print(f"Best Params for {p_id}: {current_params}")

            # 4. Final Run & Output
            if comm.rank == 0:
                print(f"Generating final output for {p_id}...")

            nll, stats, df_traj = solve_pde(current_params, args, data, comm)

            if comm.rank == 0:
                print(f"Results {p_id} -> NLL: {nll:.4f}, RMSE Ratio: {stats['rmse_ratio']:.4f}")

                # Save CSV
                out_csv = f"pde_trajectories_{p_id}.csv"
                df_traj.to_csv(out_csv, index=False)
                print(f"Saved CSV: {out_csv}")

                # 5. Plotting (Matplotlib)
                # We do this inside the loop so we get one plot per patient
                df = df_traj
                fig, ax = plt.subplots(1, 2, figsize=(14, 6))

                # Plot 1: Resistant Fraction
                ax[0].plot(df['time'], df['ratio_obs'], 'ko', label='Observed')
                ax[0].plot(df['time'], df['ratio_pred'], 'r-', linewidth=2, label='PDE Fit')
                ax[0].set_title(f"{p_id}: Resistant Fraction")
                ax[0].set_ylim(-0.05, 1.05)
                ax[0].legend()

                # Plot 2: Tumor Size
                ax[1].plot(df['time'], df['logca_obs'], 'ko', label='Observed')
                ax[1].plot(df['time'], df['logca_pred'], 'b-', linewidth=2, label='PDE Fit')
                ax[1].set_title(f"{p_id}: Tumor Size (Log CA125)")
                ax[1].legend()

                plt.tight_layout()
                plot_filename = f"{p_id}_fit.png"
                plt.savefig(plot_filename, dpi=150)
                plt.close(fig)  # Close memory
                print(f"Saved Plot: {plot_filename}\n")

        except Exception as e:
            if comm.rank == 0:
                print(f"ERROR processing patient {p_id}: {e}")
                # traceback.print_exc() # Uncomment if you want full stack trace
            continue  # Move to next patient


if __name__ == "__main__":
    main()