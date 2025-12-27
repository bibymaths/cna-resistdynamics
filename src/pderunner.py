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

# src/pderunner.py  (append)
def patients_from_ode_points(ode_points_csv: str) -> list[str]:
    df = pd.read_csv(ode_points_csv, usecols=["patient"])
    return sorted(df["patient"].astype(str).unique().tolist())


def run_pde_cohort(
    *,
    data_path: str,
    ode_points_csv: str,
    cfg: PDEConfig,
    time_unit: str = "months",
    sample_list: str | None = None,
    use_ca125_updated: bool = False,
    drop_failed: bool = False,
    require_panel_sequenced: bool = False,
    require_detected_cna: bool = False,
    out_dir: str = "results_pde_model",
    do_fit: bool = True,
    patients: list[str] | None = None,
) -> pd.DataFrame:
    logger = get_logger("tumorfit.pde.cohort")
    out_dir = ensure_dir(out_dir)

    if patients is None:
        patients = patients_from_ode_points(ode_points_csv)

    rows = []
    for pid in patients:
        nll, stats, df_traj = run_pde_for_patient(
            data_path=data_path,
            ode_points_csv=ode_points_csv,
            patient=pid,
            cfg=cfg,
            time_unit=time_unit,
            sample_list=sample_list,
            use_ca125_updated=use_ca125_updated,
            drop_failed=drop_failed,
            require_panel_sequenced=require_panel_sequenced,
            require_detected_cna=require_detected_cna,
            out_dir=out_dir,
            do_fit=do_fit,
        )
        rows.append({
            "patient": pid,
            "nll": float(nll),
            **(stats or {}),
        })

    df = pd.DataFrame(rows)
    out_sum = os.path.join(out_dir, "pde_summary.csv")
    df.to_csv(out_sum, index=False)
    logger.info(f"Saved PDE summary: {out_sum} rows={len(df)}")
    return df


def run_pde_cli(args) -> int:
    cfg = PDEConfig(
        L=args.L,
        n_cells=args.n_cells,
        dt=args.dt,
        DS=args.DS,
        DR=args.DR,
        gamma=args.gamma,
        ca0=args.ca0,
        sigma_ca=args.sigma_ca,
        w_ca=args.w_ca,
        maxiter=args.maxiter,
        n_starts=args.n_starts,
        n_jobs_starts=args.n_jobs_starts
    )

    if args.patient != "ALL":
        run_pde_for_patient(
            data_path=args.data,
            ode_points_csv=args.ode_points,
            patient=args.patient,
            cfg=cfg,
            time_unit=args.time_unit,
            sample_list=args.sample_list,
            use_ca125_updated=args.use_ca125_updated,
            drop_failed=args.drop_failed,
            require_panel_sequenced=args.require_panel_sequenced,
            require_detected_cna=args.require_detected_cna,
            out_dir=args.out_dir,
            do_fit=args.fit,
        )
        return 0

    run_pde_cohort(
        data_path=args.data,
        ode_points_csv=args.ode_points,
        cfg=cfg,
        time_unit=args.time_unit,
        sample_list=args.sample_list,
        use_ca125_updated=args.use_ca125_updated,
        drop_failed=args.drop_failed,
        require_panel_sequenced=args.require_panel_sequenced,
        require_detected_cna=args.require_detected_cna,
        out_dir=args.out_dir,
        do_fit=args.fit,
    )
    return 0
