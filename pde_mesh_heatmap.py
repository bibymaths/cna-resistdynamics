#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpi4py import MPI
from dolfinx import fem, mesh
from dolfinx.fem import petsc
import ufl
import basix.ufl
import sys
import os

from pde_common import load_patient_data


def inverse_logit(x):
    """Converts logit(p) back to p (0 to 1)."""
    return 1.0 / (1.0 + np.exp(-x))


def load_fitted_parameters(csv_path):
    """
    Parses the ODE fitting results CSV to extract physical parameters.
    Handles log/logit conversions automatically.

    Expected columns in CSV: 'patient', 'var', 'pred' (or value)
    """
    if not csv_path or not os.path.exists(csv_path):
        print(f"   [Warn] ODE results file not found: {csv_path}. Using defaults.")
        return {}

    print(f"Loading parameters from: {csv_path}")
    df = pd.read_csv(csv_path)

    # Clean up column names just in case
    df.columns = [c.strip().lower() for c in df.columns]

    # Identify the value column (usually 'pred' or 'obs' or 'value')
    # Based on your previous snippet, the fitted theta value is in 'pred'
    val_col = 'pred' if 'pred' in df.columns else 'value'
    if val_col not in df.columns:
        # Fallback: try finding a numeric column that isn't time
        print("   [Error] Could not identify value column (pred/value) in CSV.")
        return {}

    params_dict = {}

    # Get unique patients in the CSV
    patients = df['patient'].unique()

    for pid in patients:
        sub = df[df['patient'] == pid]

        # Helper to grab a specific theta value
        def get_theta(var_name):
            row = sub[sub['var'] == var_name]
            if not row.empty:
                return float(row[val_col].values[0])
            return None

        # Extract Raw Thetas
        log_aS = get_theta('theta:log_aS')
        logit_aR_ratio = get_theta('theta:logit_aR_over_aS')
        log_dS = get_theta('theta:log_dS')
        logit_dR_ratio = get_theta('theta:logit_dR_over_dS')
        log_K = get_theta('theta:log_K')

        # Check if we have enough data to reconstruct params
        if any(x is None for x in [log_aS, logit_aR_ratio, log_dS, logit_dR_ratio, log_K]):
            continue

        # Mathematical Transformation (Inverse Log/Logit)
        aS = np.exp(log_aS)
        aR_ratio = inverse_logit(logit_aR_ratio)
        aR = aR_ratio * aS

        dS = np.exp(log_dS)
        dR_ratio = inverse_logit(logit_dR_ratio)
        dR = dR_ratio * dS

        K = np.exp(log_K)

        params_dict[pid] = {
            "aS": aS, "aR": aR,
            "dS": dS, "dR": dR,
            "K": K
        }

    print(f"   Loaded parameters for {len(params_dict)} patients.")
    return params_dict


def run_simulation(args, data, params):
    """
    Runs the FEniCSx simulation using specific parameters for this patient.
    """
    comm = MPI.COMM_WORLD

    # Unpack specific parameters for this patient, or use CLI defaults
    aS = params.get("aS", args.aS)
    aR = params.get("aR", args.aR)
    dS = params.get("dS", args.dS)
    dR = params.get("dR", args.dR)
    K = params.get("K", args.K)

    # Mesh
    domain = mesh.create_interval(comm, args.n_cells, [0.0, args.L])
    element = basix.ufl.element("Lagrange", domain.topology.cell_name(), 1)
    V = fem.functionspace(domain, element)

    S = fem.Function(V);
    R = fem.Function(V)
    S_prev = fem.Function(V);
    R_prev = fem.Function(V)

    # Initial Conditions
    r0 = float(data.ratio[0])
    ca125_0 = float(data.ca125[0])
    N0 = max((ca125_0 - args.ca0) / max(args.gamma, 1e-12), 1e-12)
    S0_val = (1.0 - r0) * N0
    R0_val = r0 * N0

    S.interpolate(lambda x: np.full(x.shape[1], S0_val))
    R.interpolate(lambda x: np.full(x.shape[1], R0_val))
    S_prev.x.array[:] = S.x.array[:]
    R_prev.x.array[:] = R.x.array[:]

    # Forms
    dt = args.dt
    u_trial, v_test = ufl.TrialFunction(V), ufl.TestFunction(V)

    # Diffusion + Time Stepping
    a_S = ufl.inner(u_trial, v_test) * ufl.dx + dt * args.DS * ufl.inner(ufl.grad(u_trial), ufl.grad(v_test)) * ufl.dx
    a_R = ufl.inner(u_trial, v_test) * ufl.dx + dt * args.DR * ufl.inner(ufl.grad(u_trial), ufl.grad(v_test)) * ufl.dx
    L_S = ufl.inner(S_prev, v_test) * ufl.dx
    L_R = ufl.inner(R_prev, v_test) * ufl.dx

    rhs_S = fem.form(L_S);
    rhs_R = fem.form(L_R)
    A_S = petsc.assemble_matrix(fem.form(a_S));
    A_S.assemble()
    A_R = petsc.assemble_matrix(fem.form(a_R));
    A_R.assemble()
    b_S = A_S.createVecRight();
    b_R = A_R.createVecRight()

    solver_S = petsc.PETSc.KSP().create(comm)
    solver_S.setOperators(A_S);
    solver_S.setType("cg");
    solver_S.getPC().setType("jacobi")
    solver_R = petsc.PETSc.KSP().create(comm)
    solver_R.setOperators(A_R);
    solver_R.setType("cg");
    solver_R.getPC().setType("jacobi")

    # Storage
    geometry = domain.geometry.x[:, 0]
    sorted_idx = np.argsort(geometry)
    x_coords = geometry[sorted_idx]

    T_total = float(np.max(data.t))
    num_steps = int(np.ceil(T_total / dt))
    save_interval = max(1, num_steps // 200)

    time_points = []
    S_hist = [];
    R_hist = []

    if comm.rank == 0:
        print(f"   [Sim] Steps={num_steps}, aS={aS:.3f}, aR={aR:.3f}, K={K:.0f}")

    for i in range(num_steps + 1):
        # Explicit Euler for Reaction
        s_vals = S.x.array
        r_vals = R.x.array
        n_vals = s_vals + r_vals

        growth_s = aS * s_vals * (1.0 - n_vals / K)
        growth_r = aR * r_vals * (1.0 - n_vals / K)
        death_s = 1.0 * dS * s_vals  # u(t)=1
        death_r = 1.0 * dR * r_vals

        S_prev.x.array[:] = s_vals + dt * (growth_s - death_s)
        R_prev.x.array[:] = r_vals + dt * (growth_r - death_r)

        # Implicit for Diffusion
        with b_S.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b_S, rhs_S)
        solver_S.solve(b_S, S.x.petsc_vec)

        with b_R.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b_R, rhs_R)
        solver_R.solve(b_R, R.x.petsc_vec)

        S.x.array[S.x.array < 0] = 0.0
        R.x.array[R.x.array < 0] = 0.0

        if i % save_interval == 0:
            time_points.append(i * dt)
            S_hist.append(S.x.array[sorted_idx].copy())
            R_hist.append(R.x.array[sorted_idx].copy())

    return x_coords, np.array(time_points), np.array(S_hist), np.array(R_hist)


def plot_heatmaps(x, t, S_mat, R_mat, patient_id):
    # Calculate Matrices
    Total_Density = S_mat + R_mat
    # Avoid div by zero
    Resistant_Fraction = np.divide(R_mat, Total_Density, out=np.zeros_like(R_mat), where=Total_Density > 1e-9)

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Total Density
    im1 = ax[0].imshow(Total_Density, aspect='auto', origin='lower', cmap='viridis',
                       extent=[x.min(), x.max(), t.min(), t.max()])
    ax[0].set_title(f"Patient {patient_id}: Total Tumor Density", fontsize=14)
    ax[0].set_xlabel("Space (x)")
    ax[0].set_ylabel("Time (Months)")
    fig.colorbar(im1, ax=ax[0], label="Cell Density")

    # Plot 2: Resistant Fraction
    im2 = ax[1].imshow(Resistant_Fraction, aspect='auto', origin='lower', cmap='inferno',
                       extent=[x.min(), x.max(), t.min(), t.max()], vmin=0, vmax=1)
    ax[1].set_title(f"Resistant Fraction Evolution", fontsize=14)
    ax[1].set_xlabel("Space (x)")
    fig.colorbar(im2, ax=ax[1], label="Fraction (R / N)")

    outfile = f"heatmap_{patient_id}.png"
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close(fig)
    print(f"   [Plot] Saved {outfile}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--patient", required=True, help="Patient ID or 'ALL'")
    ap.add_argument("--sample_list", default=None)
    ap.add_argument("--use_ca125_updated", action="store_true")

    # NEW ARGUMENT for ODE Results
    ap.add_argument("--ode_results", required=True, help="Path to ode_model.py output CSV (gof_points...)")

    # Physics Params (Defaults)
    ap.add_argument("--L", type=float, default=1.0)
    ap.add_argument("--n_cells", type=int, default=100)
    ap.add_argument("--dt", type=float, default=1e-3)
    # Fallbacks
    ap.add_argument("--aS", type=float, default=0.5)
    ap.add_argument("--aR", type=float, default=0.3)
    ap.add_argument("--dS", type=float, default=0.4)
    ap.add_argument("--dR", type=float, default=0.1)
    ap.add_argument("--K", type=float, default=1.0)
    ap.add_argument("--DS", type=float, default=1e-2)
    ap.add_argument("--DR", type=float, default=1e-2)

    # Obs Params
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--ca0", type=float, default=0.0)

    args = ap.parse_args()
    comm = MPI.COMM_WORLD

    # 1. Auto-Load Parameters from ODE Results
    patient_params_map = load_fitted_parameters(args.ode_results)

    # Determine list of patients
    if args.patient == "ALL":
        # If map is loaded, use keys from there, otherwise use hardcoded fallback
        if patient_params_map:
            patients_to_run = list(patient_params_map.keys())
        else:
            patients_to_run = ["UP0018", "UP0042", "UP0053", "UP0055", "UP0056"]
    else:
        patients_to_run = [args.patient]

    for p_id in patients_to_run:
        if comm.rank == 0:
            print(f"\nProcessing Patient: {p_id}")

        try:
            # 2. Load Data
            data = load_patient_data(args.data, p_id, sample_list_path=args.sample_list,
                                     use_ca125_updated=args.use_ca125_updated)

            # 3. Look up Params
            p_params = patient_params_map.get(p_id, {})
            if not p_params and comm.rank == 0:
                print(f"   [Warn] No fitted params found for {p_id}. Using defaults.")

            # 4. Run Simulation
            x, t, S, R = run_simulation(args, data, p_params)

            # 5. Plot
            if comm.rank == 0:
                plot_heatmaps(x, t, S, R, p_id)

        except Exception as e:
            if comm.rank == 0:
                print(f"   [Error] Skipping {p_id}: {e}")
            continue


if __name__ == "__main__":
    main()