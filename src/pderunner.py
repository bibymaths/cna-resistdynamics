from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from .odeio import load_patient_data
from .pdeio import load_ode_physical_params_map, load_u_ctx_from_ode_points
from .pdemodel import PDEConfig
from .pdefit import multistart_fit_pde
from .pdesolve import solve_pde
from .pdeplotio import plot_pde_fit
from .timelog import get_logger
from .utils import ensure_dir


def run_pde_for_patient(
    *,
    data_path: str,
    ode_points_csv: str,
    patient: str,
    cfg: PDEConfig,
    time_unit: str = "months",
    sample_list: str | None = None,
    use_ca125_updated: bool = False,
    drop_failed: bool = False,
    require_panel_sequenced: bool = False,
    require_detected_cna: bool = False,
    out_dir: str = "results_pde_model",
    do_fit: bool = True,
) -> tuple[float, dict | None, pd.DataFrame | None]:
    logger = get_logger("tumorfit.pde")

    out_dir = ensure_dir(out_dir)
    data = load_patient_data(
        data_path, patient,
        time_unit=time_unit,
        sample_list_path=sample_list,
        use_ca125_updated=use_ca125_updated,
        drop_failed=drop_failed,
        require_panel_sequenced=require_panel_sequenced,
        require_detected_cna=require_detected_cna,
    )
    cfg.u_ctx = load_u_ctx_from_ode_points(ode_points_csv, patient, data.context_names)
    pmap = load_ode_physical_params_map(ode_points_csv)
    base = pmap.get(patient, [0.5, 0.3, 0.4, 0.1, 1.0])
    logger.info(f"{patient}: PDE base params from ODE? {'yes' if patient in pmap else 'no'} base={np.round(base,4)}")

    params = np.asarray(base, float)
    if do_fit:
        params = multistart_fit_pde(params, cfg, data)

    nll, stats, df_traj, _ = solve_pde(params, cfg, data, comm=None, return_history=False)
    if df_traj is not None:
        out_csv = os.path.join(out_dir, f"pde_trajectory_{patient}.csv")
        df_traj.to_csv(out_csv, index=False)
        out_png = os.path.join(out_dir, f"pde_fit_{patient}.png")
        plot_pde_fit(df_traj, out_png, title=patient)
        logger.info(f"{patient}: saved {out_csv} and {out_png}")

    return nll, stats, df_traj
