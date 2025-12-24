#!/usr/bin/env python3
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
from dolfinx import fem, mesh
from dolfinx.fem import petsc
import ufl
import basix.ufl

from pde_common import load_patient_data


def run_simulation(args, data):
    # --- Identical Physics Engine to your main script ---
    comm = MPI.COMM_WORLD
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
    S0_val = (1.0 - r0) * N0;
    R0_val = r0 * N0

    S.interpolate(lambda x: np.full(x.shape[1], S0_val))
    R.interpolate(lambda x: np.full(x.shape[1], R0_val))
    S_prev.x.array[:] = S.x.array[:];
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

    t = 0.0
    print(f"Running Matplotlib Viz Sim: T={T_total:.1f}, Steps={num_steps}")

    for i in range(num_steps + 1):
        t += dt
        s_vals = S.x.array;
        r_vals = R.x.array
        n_vals = s_vals + r_vals

        # Reaction
        growth_s = args.aS * s_vals * (1.0 - n_vals / args.K)
        growth_r = args.aR * r_vals * (1.0 - n_vals / args.K)
        death_s = 1.0 * args.dS * s_vals
        death_r = 1.0 * args.dR * r_vals

        S_prev.x.array[:] = s_vals + dt * (growth_s - death_s)
        R_prev.x.array[:] = r_vals + dt * (growth_r - death_r)

        # Diffusion
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
            time_points.append(t)
            S_hist.append(S.x.array[sorted_idx].copy())
            R_hist.append(R.x.array[sorted_idx].copy())

    return x_coords, np.array(time_points), np.array(S_hist), np.array(R_hist)


def plot_heatmaps(x, t, S_mat, R_mat, patient_id):
    print("Generating Matplotlib Heatmaps...")

    # Calculate Matrices
    Total_Density = S_mat + R_mat
    Resistant_Fraction = R_mat / (Total_Density + 1e-9)

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Total Density
    # Use aspect='auto' so time (0-25) and space (0-1) scale to fill the square
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
    print(f"Saved visualization to {outfile}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--patient", required=True)
    ap.add_argument("--sample_list", default=None)
    ap.add_argument("--use_ca125_updated", action="store_true")

    # Physics Params
    ap.add_argument("--L", type=float, default=1.0)
    ap.add_argument("--n_cells", type=int, default=100)
    ap.add_argument("--dt", type=float, default=1e-3)
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

    data = load_patient_data(args.data, args.patient, sample_list_path=args.sample_list,
                             use_ca125_updated=args.use_ca125_updated)
    x, t, S, R = run_simulation(args, data)
    plot_heatmaps(x, t, S, R, args.patient)


if __name__ == "__main__":
    main()