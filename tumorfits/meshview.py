# SPDX-FileCopyrightText: 2025 Abhinav Mishra
# SPDX-License-Identifier: MIT
"""
tumorfits.meshview
==================
2-D FEniCS reaction–diffusion simulation and PyVista visualisation routines.

These functions reproduce the workflow originally in ``pde_mesh_view.ipynb``:

1. Run a 2-D unit-square finite-element reaction–diffusion simulation using
   patient-specific growth/death parameters and a treatment duration inferred
   from clinical data.
2. Generate three off-screen PyVista visualisations per patient:
   - Resistance zone map  (``<pid>_resistance_zones.png``)
   - Growth streamlines   (``<pid>_streamlines.png``)
   - Drug-efficacy map    (``<pid>_drug_efficacy.png``)

Usage via CLI:
    tumorfits mesh-view \\
        --data  data/liquidCNA_results/Subclonal_ratio_estimates.extended.txt \\
        --ode-points ode_gof_points.csv \\
        --out-dir results_pde_model \\
        [--patient UP0018] \\
        [--sample-list data/OV_patientDNA_sampleList.txt]

Usage as library:
    from tumorfits.meshview import run_mesh_view_pipeline
    run_mesh_view_pipeline(patient_db, patient_data_map, out_dir="results_pde_model")
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from .timelog import get_logger
from .utils import ensure_dir, invlogit

_log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Parameter loading helpers
# ---------------------------------------------------------------------------


def load_all_patient_params(csv_path: str) -> dict[str, dict[str, float]]:
    """
    Parse the ODE results long-table CSV (produced by ``tumorfits ode``) and
    return a dict mapping patient ID → physical parameter dict.

    The physical parameters extracted are:
    ``aS, aR, dS, dR, K`` (growth rates, death rates, carrying capacity) plus
    diffusion defaults ``DS=0.01, DR=0.01``.

    Parameters
    ----------
    csv_path:
        Path to ``ode_gof_points*.csv``.

    Returns
    -------
    dict[str, dict[str, float]]
    """
    import pandas as pd

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"ODE results CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]
    val_col = "pred" if "pred" in df.columns else "value"

    params_dict: dict[str, dict[str, float]] = {}

    for pid in df["patient"].unique():
        sub = df[df["patient"] == pid]

        def _get(vname: str, _sub: object = sub, _vc: str = val_col) -> float | None:
            row = _sub[_sub["var"] == vname]  # type: ignore[index]
            return float(row[_vc].values[0]) if not row.empty else None  # type: ignore[index]

        log_aS = _get("theta:log_aS")
        logit_aR = _get("theta:logit_aR_over_aS")
        log_dS = _get("theta:log_dS")
        logit_dR = _get("theta:logit_dR_over_dS")
        log_K = _get("theta:log_K")

        if any(v is None for v in [log_aS, logit_aR, log_dS, logit_dR, log_K]):
            _log.warning("Missing theta params for %s; skipping.", pid)
            continue

        aS = float(np.exp(log_aS))  # type: ignore[arg-type]
        aR = aS * float(invlogit(np.asarray(logit_aR, float)))
        dS = float(np.exp(log_dS))  # type: ignore[arg-type]
        dR = dS * float(invlogit(np.asarray(logit_dR, float)))
        K = float(np.exp(log_K))  # type: ignore[arg-type]

        params_dict[pid] = {
            "aS": aS,
            "aR": aR,
            "dS": dS,
            "dR": dR,
            "K": K,
            "DS": 0.01,
            "DR": 0.01,
        }

    return params_dict


# ---------------------------------------------------------------------------
# 2-D FEniCS simulation
# ---------------------------------------------------------------------------


def run_cancer_simulation_2d(
    params: dict[str, float],
    t_max: float,
    *,
    nx: int = 50,
    ny: int = 50,
    dt: float = 0.5,
) -> tuple[Any, Any, Any]:
    """
    Solve a 2-D reaction–diffusion model on a unit-square FEniCS mesh.

    The model implements operator-split time integration:
    reaction step (explicit Euler) followed by diffusion step (implicit/FEM).

    State variables
    ---------------
    S(x, t) : density of sensitive cells
    R(x, t) : density of resistant cells

    Equations
    ---------
    ∂S/∂t = aS·S·(1 − N/K) − dS·S + DS·∇²S
    ∂R/∂t = aR·R·(1 − N/K) − dR·R + DR·∇²R
    N = S + R

    Initial conditions
    ------------------
    Gaussian blobs centred at (0.5, 0.5):
      S(x,0) = 0.8 · K · 0.1 · exp(−|x − 0.5|²/0.05)
      R(x,0) = 0.2 · K · 0.1 · exp(−|x − 0.5|²/0.05)

    Parameters
    ----------
    params:
        Dict with keys ``aS, aR, dS, dR, K, DS, DR``.
    t_max:
        Total simulation time (months).
    nx, ny:
        Mesh resolution.
    dt:
        Time-step (months).

    Returns
    -------
    (mesh, S_function, R_function)
    """
    import basix.ufl
    import ufl
    from dolfinx.fem import Function, functionspace
    from dolfinx.fem import petsc as fem_petsc
    from dolfinx.mesh import CellType, create_unit_square
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    msh = create_unit_square(comm, nx, ny, CellType.quadrilateral)

    element = basix.ufl.element("Lagrange", msh.topology.cell_name(), 1)
    V = functionspace(msh, element)

    S = Function(V)
    R = Function(V)
    S_prev = Function(V)
    R_prev = Function(V)

    K = params["K"]
    DS = params["DS"]
    DR = params["DR"]

    def _init_blob(x: np.ndarray) -> np.ndarray:
        r2 = (x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2
        return K * 0.1 * np.exp(-r2 / 0.05)

    S.interpolate(lambda x: 0.8 * _init_blob(x))
    R.interpolate(lambda x: 0.2 * _init_blob(x))
    S_prev.x.array[:] = S.x.array[:]
    R_prev.x.array[:] = R.x.array[:]

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
    L_S = ufl.inner(S_prev, v_test) * ufl.dx
    L_R = ufl.inner(R_prev, v_test) * ufl.dx

    opts = {"ksp_type": "preonly", "pc_type": "lu"}
    problem_S = fem_petsc.LinearProblem(
        a_S, L_S, u=S, petsc_options=opts, petsc_options_prefix="solver_S_"
    )
    problem_R = fem_petsc.LinearProblem(
        a_R, L_R, u=R, petsc_options=opts, petsc_options_prefix="solver_R_"
    )

    num_steps = max(1, int(np.ceil(t_max / dt)))
    _log.info("2-D FEniCS: %d time steps (dt=%.2f, T=%.1f)", num_steps, dt, t_max)

    aS, aR = params["aS"], params["aR"]
    dS, dR = params["dS"], params["dR"]

    for _ in range(num_steps):
        s_arr = S.x.array
        r_arr = R.x.array
        n_arr = s_arr + r_arr

        S_prev.x.array[:] = s_arr + dt * (aS * s_arr * (1 - n_arr / K) - dS * s_arr)
        R_prev.x.array[:] = r_arr + dt * (aR * r_arr * (1 - n_arr / K) - dR * r_arr)

        problem_S.solve()
        problem_R.solve()

        S.x.array[S.x.array < 0] = 0.0
        R.x.array[R.x.array < 0] = 0.0

    return msh, S, R


# ---------------------------------------------------------------------------
# PyVista visualisation helpers
# ---------------------------------------------------------------------------


def plot_resistance_zones(
    msh: Any,
    S: Any,
    R: Any,
    pid: str,
    out_dir: str,
) -> str:
    """
    Render a 3-D resistance-zone map as an off-screen PNG.

    Saves ``<out_dir>/<pid>_resistance_zones.png``.

    Parameters
    ----------
    msh: dolfinx Mesh
    S, R: dolfinx Functions (sensitive and resistant cell densities)
    pid: patient identifier string
    out_dir: output directory (must exist)

    Returns
    -------
    Path to the saved PNG.
    """
    import dolfinx.plot
    import pyvista as pv

    s_vals = S.x.array
    r_vals = R.x.array
    total = s_vals + r_vals
    ratio = np.divide(r_vals, total, out=np.zeros_like(r_vals), where=total > 1e-9)

    cells, types, x = dolfinx.plot.vtk_mesh(msh)
    grid = pv.UnstructuredGrid(cells, types, x)
    grid.point_data["Resistant Fraction"] = ratio
    grid.point_data["Density"] = total

    pl = pv.Plotter(off_screen=True, window_size=[1024, 768])
    pl.add_text(f"{pid}: Resistance Map", font_size=12)

    max_dens = float(np.max(total))
    if max_dens < 1e-3:
        pl.add_text("(Tumor too small)", font_size=10, position="upper_left")
        pl.add_mesh(grid, style="wireframe", color="grey", opacity=0.3)
    else:
        tumor_body = grid.threshold(max_dens * 0.01, scalars="Density")
        pl.add_mesh(
            tumor_body,
            scalars="Resistant Fraction",
            cmap="inferno",
            clim=[0, 1],
            show_scalar_bar=True,
        )
        pl.add_mesh(grid, style="wireframe", color="grey", opacity=0.1)

    filepath = os.path.join(out_dir, f"{pid}_resistance_zones.png")
    pl.screenshot(filepath)
    pl.close()
    _log.info("Saved resistance zones: %s", filepath)
    return filepath


def plot_growth_streamlines(
    msh: Any,
    S: Any,
    R: Any,
    pid: str,
    out_dir: str,
) -> str:
    """
    Render cell-density gradient streamlines as an off-screen PNG.

    Saves ``<out_dir>/<pid>_streamlines.png``.
    """
    import dolfinx
    import dolfinx.fem
    import pyvista as pv

    V = S.function_space
    N_fn = dolfinx.fem.Function(V)
    N_fn.x.array[:] = S.x.array[:] + R.x.array[:]

    cells, types, x = dolfinx.plot.vtk_mesh(V)
    grid = pv.UnstructuredGrid(cells, types, x)
    grid.point_data["Density"] = N_fn.x.array

    max_dens = float(np.max(N_fn.x.array))
    pl = pv.Plotter(off_screen=True, window_size=[1024, 768])
    pl.add_text(f"{pid}: Growth Streamlines", font_size=12)

    if max_dens < 1e-3:
        _log.warning("%s: tumor too small for streamlines.", pid)
        pl.add_mesh(grid, style="wireframe", color="grey", opacity=0.3)
    else:
        tumor_vol = grid.threshold(max_dens * 0.1)
        radius = 0.1
        if tumor_vol.n_points > 0:
            bounds = tumor_vol.bounds
            radius = (bounds[1] - bounds[0]) / 3.0

        grad = grid.compute_derivative(scalars="Density")
        grad.set_active_vectors("gradient")
        stream = grad.streamlines(
            "gradient",
            source_center=(0.5, 0.5, 0.0),
            source_radius=radius,
            n_points=150,
            max_time=10.0,
            integration_direction="both",
        )

        pl.add_mesh(grid, style="wireframe", opacity=0.1, color="grey")
        if stream.n_points > 0:
            pl.add_mesh(stream.tube(radius=0.003), color="red")

        arrows = grad.glyph(
            orient="gradient",
            scale="gradient",
            factor=0.1,
            tolerance=0.01,
        )
        if arrows.n_points > 0:
            pl.add_mesh(arrows, color="blue", opacity=0.6)

    filepath = os.path.join(out_dir, f"{pid}_streamlines.png")
    pl.screenshot(filepath)
    pl.close()
    _log.info("Saved streamlines: %s", filepath)
    return filepath


def plot_drug_efficacy(
    msh: Any,
    S: Any,
    R: Any,
    pid: str,
    params: dict[str, float],
    out_dir: str,
) -> str:
    """
    Render a drug kill-rate overlay as an off-screen PNG.

    Saves ``<out_dir>/<pid>_drug_efficacy.png``.
    """
    import dolfinx
    import pyvista as pv

    V = S.function_space
    cells, types, x = dolfinx.plot.vtk_mesh(V)
    grid = pv.UnstructuredGrid(cells, types, x)

    s_vals = S.x.array
    r_vals = R.x.array
    grid.point_data["Density"] = s_vals + r_vals
    grid.point_data["Kill Rate"] = params["dS"] * s_vals

    centers = grid.points
    vectors = centers - np.array([0.5, 0.5, 0.0])
    grid["Vectors"] = vectors

    arrows = grid.glyph(
        orient="Vectors",
        scale="Density",
        geom=pv.Arrow(),
        factor=0.1,
        tolerance=0.02,
    )

    pl = pv.Plotter(off_screen=True, window_size=[1024, 768])
    pl.add_text(f"{pid}: Drug Efficacy (Red = High Kill)", font_size=12)
    if arrows.n_points > 0:
        pl.add_mesh(arrows, scalars="Kill Rate", cmap="coolwarm", show_scalar_bar=True)
    pl.add_mesh(grid, style="wireframe", color="grey", opacity=0.1)

    filepath = os.path.join(out_dir, f"{pid}_drug_efficacy.png")
    pl.screenshot(filepath)
    pl.close()
    _log.info("Saved drug efficacy: %s", filepath)
    return filepath


# ---------------------------------------------------------------------------
# High-level pipeline runner
# ---------------------------------------------------------------------------


def run_mesh_view_pipeline(
    patient_db: dict[str, dict[str, float]],
    patient_data_map: dict[str, Any],
    out_dir: str = "results_pde_model",
    *,
    nx: int = 50,
    ny: int = 50,
    dt: float = 0.5,
) -> dict[str, list[str]]:
    """
    Run the full mesh-view pipeline for all patients in *patient_db*.

    Parameters
    ----------
    patient_db:
        Mapping patient_id → physical params dict (from :func:`load_all_patient_params`).
    patient_data_map:
        Mapping patient_id → :class:`tumorfits.odeio.PatientData`.
    out_dir:
        Directory where PNG files are written.

    Returns
    -------
    dict mapping patient_id → list of written PNG paths.
    """
    ensure_dir(out_dir)
    outputs: dict[str, list[str]] = {}

    for pid, params in patient_db.items():
        _log.info("=" * 50)
        _log.info("Processing patient: %s", pid)
        _log.info("aS=%.4f  K=%.0f", params["aS"], params["K"])

        data = patient_data_map.get(pid)
        if data is None:
            _log.warning("No PatientData for %s; skipping.", pid)
            continue

        t_max = float(np.max(data.t))
        msh, S_field, R_field = run_cancer_simulation_2d(params, t_max, nx=nx, ny=ny, dt=dt)

        pngs = [
            plot_resistance_zones(msh, S_field, R_field, pid, out_dir),
            plot_growth_streamlines(msh, S_field, R_field, pid, out_dir),
            plot_drug_efficacy(msh, S_field, R_field, pid, params, out_dir),
        ]
        outputs[pid] = pngs

    return outputs
