#!/usr/bin/env python3
"""
FEniCSx 1D Tumor PDE Solver & Fitter (Multi-Patient / Multi-Start Optimized)

This script solves a Reaction-Diffusion PDE for tumor growth (Sensitive vs Resistant cells)
on a 1D domain. It can:
1. Load patient clinical data (Resistance Ratio & Tumor Size).
2. Load pre-fitted ODE parameters as a starting point.
3. Fit the PDE parameters (aS, aR, dS, dR, K) to the data using parallelized optimization.
4. Output trajectory CSVs and fit plots.

Usage:
    python3 pde_solver.py --data ... --ode_results ... --patient ALL --fit --n_starts 5
"""

import argparse
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import minimize
from scipy.special import expit as invlogit  # Sigmoid function for probability conversion

# --- Parallel Processing Imports ---
from joblib import Parallel, delayed  # For multi-start optimization
from mpi4py import MPI  # For FEniCSx mesh communication

# --- FEniCSx (Finite Element) Imports ---
from petsc4py import PETSc
from dolfinx import fem, mesh
from dolfinx.fem import petsc
import ufl
import basix.ufl

# --- Custom Modules ---
# Ensure pde_common is in your python path or folder
from pde_common import load_patient_data, pde_observables_from_grid, nll_ratio_ca


# ==============================================================================
# 1. PARAMETER LOADING & TRANSFORMATION
# ==============================================================================

def load_fitted_parameters(csv_path):
    """
    Parses the ODE fitting results CSV to extract physical parameters.

    The ODE script outputs parameters in a 'Log' or 'Logit' transformed space
    to ensure they stay positive or bounded (0-1). This function reverses that
    transformation to get the actual physical values (e.g., growth rate per day).

    Args:
        csv_path (str): Path to 'gof_points_*.csv'

    Returns:
        dict: {patient_id: [aS, aR, dS, dR, K]}
    """
    if not csv_path or not os.path.exists(csv_path):
        print(f"   [Warn] ODE results file not found: {csv_path}. Using defaults.")
        return {}

    print(f"Loading parameters from: {csv_path}")
    df = pd.read_csv(csv_path)

    # Normalize column names (strip whitespace, lowercase)
    df.columns = [c.strip().lower() for c in df.columns]

    # Detect which column holds the parameter value ('pred' or 'value')
    val_col = 'pred' if 'pred' in df.columns else 'value'
    if val_col not in df.columns:
        print("   [Error] Could not identify value column (pred/value) in CSV.")
        return {}

    params_dict = {}
    patients = df['patient'].unique()

    for pid in patients:
        sub = df[df['patient'] == pid]

        # Helper to safely get a single parameter value by its name
        def get_theta(var_name):
            row = sub[sub['var'] == var_name]
            if not row.empty:
                return float(row[val_col].values[0])
            return None

        # Extract Raw (Transformed) Parameters
        log_aS = get_theta('theta:log_aS')
        logit_aR_ratio = get_theta('theta:logit_aR_over_aS')
        log_dS = get_theta('theta:log_dS')
        logit_dR_ratio = get_theta('theta:logit_dR_over_dS')
        log_K = get_theta('theta:log_K')

        # Skip if any required parameter is missing
        if any(x is None for x in [log_aS, logit_aR_ratio, log_dS, logit_dR_ratio, log_K]):
            continue

        # --- INVERSE TRANSFORMATION (Math -> Biology) ---
        # aS = exp(log_aS) -> Positive growth rate
        aS = np.exp(log_aS)
        # aR is defined as a fraction of aS (Resistant cells usually grow slower)
        aR = invlogit(logit_aR_ratio) * aS

        # dS = exp(log_dS) -> Positive death rate
        dS = np.exp(log_dS)
        # dR is defined as a fraction of dS (Resistant cells die less)
        dR = invlogit(logit_dR_ratio) * dS

        K = np.exp(log_K)  # Carrying capacity

        params_dict[pid] = [aS, aR, dS, dR, K]

    print(f"   Loaded parameters for {len(params_dict)} patients.")
    return params_dict


def get_treatment_value(t, treatment_schedule=None):
    """
    Returns the treatment intensity u(t) at time t.
    Currently returns 1.0 (Constant Treatment).
    Future expansion: Implement a lookup in 'treatment_schedule' for Adaptive Therapy.
    """
    return 1.0


# ==============================================================================
# 2. PDE SOLVER KERNEL (FEniCSx)
# ==============================================================================

def solve_pde(params, args, data, comm):
    """
    Core Physics Engine: Solves the Reaction-Diffusion PDE.

    Equations:
      dS/dt = div(D_S * grad(S)) + aS*S*(1 - N/K) - dS*S
      dR/dt = div(D_R * grad(R)) + aR*R*(1 - N/K) - dR*R

    Args:
        params (list): [aS, aR, dS, dR, K]
        args (Namespace): Configuration (grid size, dt, diffusion coeffs)
        data (PatientData): Observed clinical data for initial conditions and time steps
        comm (MPI.Comm): MPI communicator (usually COMM_SELF or COMM_WORLD)

    Returns:
        nll (float): Negative Log Likelihood (Error metric)
        stats (dict): RMSE for ratio and tumor size
        df_traj (DataFrame): Time-series of simulation results
    """
    # 1. Unpack Parameters
    aS, aR, dS, dR, K = params

    # --- Constraints check ---
    # Return infinity error if parameters are non-physical (negative) or K is too small
    if aS < 0 or aR < 0 or dS < 0 or dR < 0 or K < 1e-4:
        return 1e12, None, None

    # 2. Initial Conditions (Derived from first data point)
    # We assume the first observation (t=0) represents the initial state.
    r0 = float(data.ratio[0])  # Initial resistant fraction
    ca125_0 = float(data.ca125[0])  # Initial tumor size proxy

    # Invert CA125 model: CA125 = gamma * N + ca0  =>  N = (CA125 - ca0) / gamma
    N0 = max((ca125_0 - args.ca0) / max(args.gamma, 1e-12), 1e-12)
    S0_val = (1.0 - r0) * N0
    R0_val = r0 * N0

    # 3. Mesh Generation (1D Interval)
    # We create a new mesh for every run to ensure clean state
    domain = mesh.create_interval(comm, args.n_cells, [0.0, args.L])
    element = basix.ufl.element("Lagrange", domain.topology.cell_name(), 1)
    V = fem.functionspace(domain, element)

    # 4. Function Definitions
    S = fem.Function(V);
    R = fem.Function(V)  # Current Step
    S_prev = fem.Function(V);
    R_prev = fem.Function(V)  # Previous Step

    # Set homogeneous initial conditions across the domain
    S.interpolate(lambda x: np.full(x.shape[1], S0_val))
    R.interpolate(lambda x: np.full(x.shape[1], R0_val))
    S_prev.x.array[:] = S.x.array[:]
    R_prev.x.array[:] = R.x.array[:]

    # 5. Variational Formulation (Weak Form)
    # Method: Implicit Euler (Backward Euler) for stability with Diffusion
    dt = args.dt
    u_trial, v_test = ufl.TrialFunction(V), ufl.TestFunction(V)

    # Define the Bilinear Forms (Left Hand Side)
    # term 1: Mass matrix (u * v)
    # term 2: Stiffness matrix (Diffusion: grad(u) * grad(v))
    a_S = ufl.inner(u_trial, v_test) * ufl.dx + dt * args.DS * ufl.inner(ufl.grad(u_trial), ufl.grad(v_test)) * ufl.dx
    a_R = ufl.inner(u_trial, v_test) * ufl.dx + dt * args.DR * ufl.inner(ufl.grad(u_trial), ufl.grad(v_test)) * ufl.dx

    # Define Linear Forms (Right Hand Side) - Updated in loop
    L_S = ufl.inner(S_prev, v_test) * ufl.dx
    L_R = ufl.inner(R_prev, v_test) * ufl.dx

    # Pre-assemble the constant matrices (Optimization for speed)
    rhs_form_S = fem.form(L_S)
    rhs_form_R = fem.form(L_R)

    A_S = petsc.assemble_matrix(fem.form(a_S));
    A_S.assemble()
    A_R = petsc.assemble_matrix(fem.form(a_R));
    A_R.assemble()

    # Setup Linear Solvers (Krylov Subspace Methods)
    # CG (Conjugate Gradient) is fast for symmetric positive-definite matrices
    solver_S = PETSc.KSP().create(domain.comm)
    solver_S.setOperators(A_S);
    solver_S.setType("cg");
    solver_S.getPC().setType("jacobi")

    solver_R = PETSc.KSP().create(domain.comm)
    solver_R.setOperators(A_R);
    solver_R.setType("cg");
    solver_R.getPC().setType("jacobi")

    b_S = A_S.createVecRight()
    b_R = A_R.createVecRight()

    # 6. Time Stepping Loop
    obs_times = list(map(float, data.t))
    obs_idx = 0
    states_at_obs = []

    T_total = float(np.max(data.t))
    num_steps = int(np.ceil(T_total / dt))
    t = 0.0

    for _ in range(num_steps + 1):
        # A. Check if we need to record data at this time point
        if obs_idx < len(obs_times) and abs(t - obs_times[obs_idx]) <= 0.5 * dt:
            states_at_obs.append((S.x.array.copy(), R.x.array.copy()))
            obs_idx += 1

        if obs_idx >= len(obs_times): break  # Stop if we passed all data points

        t += dt
        u_val = get_treatment_value(t)

        s_vals = S.x.array;
        r_vals = R.x.array
        n_vals = s_vals + r_vals

        # B. Reaction Step (Explicit Euler)
        # We calculate growth/death first, then diffuse. This is Operator Splitting (simple version).
        growth_s = aS * s_vals * (1.0 - n_vals / K)
        growth_r = aR * r_vals * (1.0 - n_vals / K)
        death_s = u_val * dS * s_vals
        death_r = u_val * dR * r_vals

        # The "source" term for the diffusion step is the result of the reaction step
        S_prev.x.array[:] = s_vals + dt * (growth_s - death_s)
        R_prev.x.array[:] = r_vals + dt * (growth_r - death_r)

        # C. Diffusion Step (Implicit Solve)
        # Solve A * u_new = u_reaction
        with b_S.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b_S, rhs_form_S)
        solver_S.solve(b_S, S.x.petsc_vec)

        with b_R.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b_R, rhs_form_R)
        solver_R.solve(b_R, R.x.petsc_vec)

        # D. Biology Constraints
        # Cell counts cannot be negative
        S.x.array[S.x.array < 0] = 0.0
        R.x.array[R.x.array < 0] = 0.0

        # Divergence Check (Numerical Explosion)
        if np.any(np.isnan(S.x.array)) or np.any(np.isnan(R.x.array)):
            return 1e12, None, None

    # 7. Compute Likelihood / Error
    if len(states_at_obs) != len(data.t):
        return 1e12, None, None  # Sim ended early

    dx = args.L / args.n_cells
    ratio_hat, logca_hat = [], []

    # Calculate observable metrics from the spatial fields
    for S_vals, R_vals in states_at_obs:
        # pde_observables_from_grid integrates the field to get total N and Ratio
        _, _, r_pred, lc_pred = pde_observables_from_grid(S_vals, R_vals, dx, gamma=args.gamma, ca0=args.ca0)
        ratio_hat.append(r_pred)
        logca_hat.append(lc_pred)

    ratio_hat = np.asarray(ratio_hat)
    logca_hat = np.asarray(logca_hat)

    # Compute Negative Log Likelihood (NLL)
    nll = nll_ratio_ca(
        ratio_obs=data.ratio,
        se_logit_ratio=data.se_logit_ratio,
        logca_obs=data.log_ca125,
        ratio_hat=ratio_hat,
        logca_hat=logca_hat,
        sigma_ca=args.sigma_ca,
        w_ca=args.w_ca,
    )

    stats = {
        "rmse_ratio": np.sqrt(np.mean((data.ratio - ratio_hat) ** 2)),
        "rmse_ca": np.sqrt(np.mean((data.log_ca125 - logca_hat) ** 2))
    }

    df_traj = pd.DataFrame({
        "time": obs_times,
        "ratio_obs": data.ratio,
        "ratio_pred": ratio_hat,
        "logca_obs": data.log_ca125,
        "logca_pred": logca_hat,
        "patient": args.patient
    })

    return nll, stats, df_traj


# ==============================================================================
# 3. OPTIMIZATION WRAPPERS (Multi-Start)
# ==============================================================================

def run_single_optimization_start(seed, initial_params, args, data):
    """
    Runs one instance of the optimizer with a randomized starting point.
    Used by joblib for parallel execution.
    """
    np.random.seed(seed)

    # Add noise to the initial parameters to explore the space
    # params: [aS, aR, dS, dR, K]
    # We apply multiplicative noise (e.g. +/- 20%)
    noise = np.random.uniform(0.8, 1.2, size=len(initial_params))
    randomized_start = np.array(initial_params) * noise

    # Define objective function for this thread
    # Note: We must create a new COMM_SELF for each thread if using MPI inside threads
    # However, FEniCSx usually handles this if we don't explicitly pass COMM_WORLD.
    # Here we pass MPI.COMM_SELF to ensure isolation.
    comm = MPI.COMM_SELF

    def objective(p):
        return solve_pde(p, args, data, comm)[0]  # Return just NLL

    # Run Optimizer
    # Nelder-Mead is chosen because the PDE landscape is noisy and gradients are unavailable
    res = minimize(objective, randomized_start, method='Nelder-Mead',
                   options={'maxiter': 100, 'disp': False})

    return res.fun, res.x


def run_multi_start_optimization(base_params, args, data, n_starts=5):
    """
    Orchestrates parallel optimization runs using Joblib.
    """
    print(f"   Running {n_starts} optimization starts in parallel...")

    # joblib.Parallel spawns multiple Python processes
    # n_jobs=-1 uses all available cores
    results = Parallel(n_jobs=-1)(
        delayed(run_single_optimization_start)(i, base_params, args, data)
        for i in range(n_starts)
    )

    # Sort results by lowest NLL (best fit)
    results.sort(key=lambda x: x[0])
    best_nll, best_params = results[0]

    print(f"   Best NLL: {best_nll:.4f}")
    return best_params


# ==============================================================================
# 4. MAIN ENTRY POINT
# ==============================================================================

def main():
    ap = argparse.ArgumentParser(description="FEniCSx 1D Tumor PDE Solver & Fitter")

    # Data Args
    ap.add_argument("--data", required=True, help="Path to clinical data CSV/TXT")
    ap.add_argument("--patient", required=True, help="Patient ID or 'ALL'")
    ap.add_argument("--sample_list", default=None)
    ap.add_argument("--use_ca125_updated", action="store_true")
    ap.add_argument("--time_unit", default="months")
    ap.add_argument("--drop_failed", action="store_true")
    ap.add_argument("--require_panel_sequenced", action="store_true")
    ap.add_argument("--require_detected_cna", action="store_true")
    ap.add_argument("--ode_results", required=True, help="Path to pre-fitted ODE params")

    # Physics Args
    ap.add_argument("--L", type=float, default=1.0, help="Domain Size")
    ap.add_argument("--n_cells", type=int, default=200, help="Grid Resolution")
    ap.add_argument("--dt", type=float, default=1e-3, help="Time Step")

    # Initial Guess Defaults (if ODE results missing)
    ap.add_argument("--aS", type=float, default=0.5)
    ap.add_argument("--aR", type=float, default=0.3)
    ap.add_argument("--dS", type=float, default=0.4)
    ap.add_argument("--dR", type=float, default=0.1)
    ap.add_argument("--K", type=float, default=1.0)
    ap.add_argument("--DS", type=float, default=1e-2, help="Diff Coeff Sensitive")
    ap.add_argument("--DR", type=float, default=1e-2, help="Diff Coeff Resistant")

    # Observation Model Args
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--ca0", type=float, default=0.0)
    ap.add_argument("--sigma_ca", type=float, default=0.5)
    ap.add_argument("--w_ca", type=float, default=1.0)

    # Workflow Args
    ap.add_argument("--fit", action="store_true", help="Run optimization loop")
    ap.add_argument("--n_starts", type=int, default=10, help="Number of restarts for optimizer")

    args = ap.parse_args()
    comm = MPI.COMM_WORLD  # Main thread communicator

    # 1. Load Outputs from ODE Step
    patient_params_map = load_fitted_parameters(args.ode_results)

    # 2. Determine Patient List
    if args.patient == "ALL":
        if patient_params_map:
            patients_to_run = list(patient_params_map.keys())
        else:
            # Fallback list if map is empty
            patients_to_run = ["UP0018", "UP0042", "UP0053", "UP0055", "UP0056"]
    else:
        patients_to_run = [args.patient]

    # 3. Create Output Directory
    pde_res_dir = "results_pde_model"
    os.makedirs(pde_res_dir, exist_ok=True)

    # 4. Processing Loop
    for p_id in patients_to_run:
        if comm.rank == 0:
            print(f"\n{'=' * 60}")
            print(f"PROCESSING PATIENT: {p_id}")
            print(f"{'=' * 60}")

        try:
            # A. Load Patient Data
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
                print(f"   Time Range: 0 to {np.max(data.t):.2f} {args.time_unit}")

            # B. Get Initial Parameter Guess
            if p_id in patient_params_map:
                current_params = patient_params_map[p_id]  # List [aS, aR, dS, dR, K]
            else:
                current_params = [args.aS, args.aR, args.dS, args.dR, args.K]

            # C. Run Optimization (Fit)
            if args.fit:
                if args.n_starts > 1:
                    # Parallel Multi-Start
                    current_params = run_multi_start_optimization(current_params, args, data, n_starts=args.n_starts)
                else:
                    # Single Thread Standard
                    if comm.rank == 0: print(f"   Running Single-Start Optimization...")

                    def objective(p):
                        return solve_pde(p, args, data, MPI.COMM_SELF)[0]

                    res = minimize(objective, current_params, method='Nelder-Mead',
                                   options={'maxiter': 100, 'disp': False})
                    current_params = res.x

                if comm.rank == 0:
                    print(f"   Best Params Found: {np.round(current_params, 4)}")

            # D. Final Simulation (Generate Trajectory)
            nll, stats, df_traj = solve_pde(current_params, args, data, MPI.COMM_SELF)

            if comm.rank == 0:
                print(f"   Final Results -> NLL: {nll:.4f}, RMSE Ratio: {stats['rmse_ratio']:.4f}")

                # Save Data
                out_csv = f"pde_trajectories_{p_id}.csv"
                df_traj.to_csv(os.path.join(pde_res_dir, out_csv), index=False)
                print(f"   Saved CSV: {os.path.join(pde_res_dir, out_csv)}")

                # Save Plot
                fig, ax = plt.subplots(1, 2, figsize=(14, 6))

                # Plot Ratio
                ax[0].plot(df_traj['time'], df_traj['ratio_obs'], 'ko', label='Observed')
                ax[0].plot(df_traj['time'], df_traj['ratio_pred'], 'r-', linewidth=2, label='PDE Fit')
                ax[0].set_title(f"{p_id}: Resistant Fraction")
                ax[0].set_ylim(-0.05, 1.05)
                ax[0].legend()

                # Plot Size
                ax[1].plot(df_traj['time'], df_traj['logca_obs'], 'ko', label='Observed')
                ax[1].plot(df_traj['time'], df_traj['logca_pred'], 'b-', linewidth=2, label='PDE Fit')
                ax[1].set_title(f"{p_id}: Tumor Size (Log CA125)")
                ax[1].legend()

                plt.tight_layout()
                plot_path = os.path.join(pde_res_dir, f"{p_id}_fit.png")
                plt.savefig(plot_path, dpi=150)
                plt.close(fig)
                print(f"   Saved Plot: {plot_path}\n")

        except Exception as e:
            if comm.rank == 0:
                print(f"   [Error] Failed on {p_id}: {e}")
                # import traceback; traceback.print_exc()
            continue


if __name__ == "__main__":
    main()