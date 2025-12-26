from __future__ import annotations

import os
import numpy as np

from .odeio import load_patient_data
from .pdeio import load_ode_physical_params_map
from .pdemodel import PDEConfig
from .pdesolve import solve_pde
from .pdeplotio import plot_heatmaps
from .utils import ensure_dir


def run_pde_heatmap(
    *,
    data_path: str,
    ode_points_csv: str,
    patient: str,
    cfg: PDEConfig,
    out_dir: str = "results_pde_model",
    time_unit: str = "months",
    sample_list: str | None = None,
):
    out_dir = ensure_dir(out_dir)
    data = load_patient_data(data_path, patient, time_unit=time_unit, sample_list_path=sample_list)

    pmap = load_ode_physical_params_map(ode_points_csv)
    params = np.asarray(pmap.get(patient, [0.5, 0.3, 0.4, 0.1, 1.0]), float)
    # params = np.asarray(pmap.get(patient, [cfg.gamma, cfg.gamma * 0.6, 0.4, 0.1, 1.0]), float)  # fallback

    nll, stats, df, hist = solve_pde(params, cfg, data, comm=None, return_history=True)
    if hist is None:
        raise RuntimeError("No history returned (return_history=True failed).")

    x = hist["x"]
    t = hist["t"]
    S = hist["S"]
    R = hist["R"]

    out_png = os.path.join(out_dir, f"heatmap_{patient}.png")
    plot_heatmaps(x, t, S, R, out_png, title=patient)
    return out_png
