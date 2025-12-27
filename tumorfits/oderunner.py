from __future__ import annotations

import os
from dataclasses import dataclass
from functools import partial

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .metrics import gof_metrics, nll_ratio_ca
from .odefit import multistart_minimize
from .odeio import PatientData, get_patients_with_flag, load_patient_data
from .odemodel import simulate_ode, ode_theta_names
from .odeplotio import save_patient_states_plots, plot_gof_scatter_all
from .timelog import get_logger
from .utils import invlogit, logit, ensure_dir


@dataclass
class ODEFitConfig:
    n_starts: int = 8
    rel_noise: float = 0.25
    n_jobs_patients: int = -1
    n_jobs_starts: int = 1
    maxiter: int = 1200
    w_ca: float = 0.5


def nll_ode(theta: np.ndarray, data: PatientData, *, w_ca: float = 0.5) -> float:
    try:
        r_hat, logca_hat = simulate_ode(data, theta)
    except Exception:
        return 1e50

    sigma_ca = float(np.exp(theta[9]))  # ✅ canonical
    return nll_ratio_ca(
        ratio_obs=data.ratio,
        se_logit_ratio=data.se_logit_ratio,
        logca_obs=data.log_ca125,
        ratio_hat=r_hat,
        logca_hat=logca_hat,
        sigma_ca=sigma_ca,
        w_ca=w_ca,
    )


def initial_theta_and_bounds(data: PatientData) -> tuple[np.ndarray, list[tuple[float, float]]]:
    C = len(data.context_names)

    x0 = np.zeros(10 + C, dtype=float)
    x0[0] = np.log(0.5)  # log_aS
    x0[1] = logit(np.array([0.6]))[0]  # logit_aR_over_aS
    x0[2] = np.log(0.8)  # log_dS
    x0[3] = logit(np.array([0.05]))[0]  # logit_dR_over_dS
    x0[4] = np.log(1e6)  # log_K
    x0[5] = np.log(1e4)  # log_N0
    x0[6] = logit(np.array([np.clip(data.ratio[0], 1e-4, 1 - 1e-4)]))[0]  # logit_r0
    x0[7] = np.log(1e-3)  # log_gamma
    x0[8] = np.mean(data.log_ca125) - 1.0  # log_ca0
    x0[9] = np.log(0.5)  # log_sigma_ca
    x0[10:] = logit(np.full(C, 0.5))  # logit_u_ctx[...]

    bnds: list[tuple[float, float]] = []
    bnds += [(-10, 5)]  # log_aS
    bnds += [(-10, 10)]  # logit_aR_over_aS
    bnds += [(-10, 5)]  # log_dS
    bnds += [(-10, 10)]  # logit_dR_over_dS
    bnds += [(0, 20)]  # log_K
    bnds += [(-5, 30)]  # log_N0
    bnds += [(-10, 10)]  # logit_r0
    bnds += [(-20, 5)]  # log_gamma
    bnds += [(-5, 15)]  # log_ca0
    bnds += [(-3, 5)]  # log_sigma_ca
    bnds += [(-10, 10)] * C  # logit_u_ctx
    return x0, bnds


def fit_ode(data: PatientData, cfg: ODEFitConfig) -> tuple[np.ndarray, dict]:
    logger = get_logger("tumorfit.ode")
    x0, bnds = initial_theta_and_bounds(data)
    res = multistart_minimize(
        fun=partial(nll_ode, data=data, w_ca=cfg.w_ca),
        x0=x0,
        bounds=bnds,
        n_starts=cfg.n_starts,
        rel_noise=cfg.rel_noise,
        seed=hash(data.patient) % (2 ** 32),
        method="L-BFGS-B",
        maxiter=cfg.maxiter,
        n_jobs_starts=cfg.n_jobs_starts,
        logger_name="tumorfit.ode.fit",
    )

    theta = res.x
    C = len(data.context_names)
    u_ctx = invlogit(theta[10:10 + C])
    logger.info(f"{data.patient}: u_ctx min={u_ctx.min():.3g} max={u_ctx.max():.3g}")

    r_hat, logca_hat = simulate_ode(data, theta)
    metrics = gof_metrics(data.ratio, r_hat, data.log_ca125, logca_hat, nll=float(res.fun), k_params=int(theta.size))
    out = {"success": bool(res.success), "message": res.message, "metrics": metrics}
    return theta, out


def fit_and_collect_points(
        patient_id: str,
        *,
        data_path: str,
        time_unit: str,
        sample_list: str | None,
        use_ca125_updated: bool,
        drop_failed: bool,
        require_panel_sequenced: bool,
        require_detected_cna: bool,
        cfg: ODEFitConfig,
        diag_dir: str | None = None,
) -> list[dict]:
    data = load_patient_data(
        data_path, patient_id,
        time_unit=time_unit,
        sample_list_path=sample_list,
        use_ca125_updated=use_ca125_updated,
        drop_failed=drop_failed,
        require_panel_sequenced=require_panel_sequenced,
        require_detected_cna=require_detected_cna,
    )
    rows: list[dict] = []

    theta, out = fit_ode(data, cfg)

    # diagnostics (optional)
    if diag_dir:
        try:
            save_patient_states_plots(
                data=data,
                theta=theta,
                out_dir=os.path.join(diag_dir, f"patient_{patient_id}"),
                tag="ODE",
                save_csv=True,
                dpi=300,
            )
        except Exception:
            pass

    # store theta rows (long format)
    for name, val in zip(ode_theta_names(data.context_names), theta):
        rows.append({
            "patient": patient_id,
            "time": np.nan,
            "model": "ODE",
            "var": f"theta:{name}",
            "obs": np.nan,
            "pred": float(val),
            "flag_out95": False,
        })

    # per-timepoint obs/pred rows
    r_hat, logca_hat = simulate_ode(data, theta)
    sigma_ca = float(np.exp(theta[9]))  # ✅ canonical

    # out-of-95% flags
    from .utils import logit as _logit  # local to avoid circular
    y_obs = _logit(data.ratio)
    y_hat = _logit(r_hat)
    out_ratio = np.abs(y_obs - y_hat) > (1.96 * data.se_logit_ratio)
    out_ca = np.abs(data.log_ca125 - logca_hat) > (1.96 * sigma_ca)

    for i, t in enumerate(data.t):
        rows.append({
            "patient": patient_id,
            "time": float(t),
            "model": "ODE",
            "var": "ratio",
            "obs": float(data.ratio[i]),
            "pred": float(r_hat[i]),
            "flag_out95": bool(out_ratio[i]),
        })
        rows.append({
            "patient": patient_id,
            "time": float(t),
            "model": "ODE",
            "var": "logCA125",
            "obs": float(data.log_ca125[i]),
            "pred": float(logca_hat[i]),
            "flag_out95": bool(out_ca[i]),
        })

    return rows


def fit_ode_cohort(
        *,
        data_path: str,
        flags: list[str],
        time_unit: str = "months",
        sample_list: str | None = None,
        use_ca125_updated: bool = False,
        drop_failed: bool = False,
        require_panel_sequenced: bool = False,
        require_detected_cna: bool = False,
        cfg: ODEFitConfig = ODEFitConfig(),
        out_points_csv: str = "ode_gof_points.csv",
        diag_dir: str | None = None,
) -> pd.DataFrame:
    logger = get_logger("tumorfit.ode.cohort")
    patients = get_patients_with_flag(data_path, flags=flags)
    logger.info(f"ODE cohort: patients={len(patients)} flags={flags}")

    if diag_dir:
        diag_dir = ensure_dir(diag_dir)

    nested = Parallel(n_jobs=cfg.n_jobs_patients, backend="loky", prefer="processes")(
        delayed(fit_and_collect_points)(
            pid,
            data_path=data_path,
            time_unit=time_unit,
            sample_list=sample_list,
            use_ca125_updated=use_ca125_updated,
            drop_failed=drop_failed,
            require_panel_sequenced=require_panel_sequenced,
            require_detected_cna=require_detected_cna,
            cfg=cfg,
            diag_dir=diag_dir,
        )
        for pid in patients
    )

    rows = [r for group in nested for r in group]
    df = pd.DataFrame(rows)
    df.to_csv(out_points_csv, index=False)
    logger.info(f"Saved ODE points: {out_points_csv} rows={len(df)} patients={df['patient'].nunique()}")
    return df


def fit_ode_single(
        *,
        data_path: str,
        patient: str,
        time_unit: str = "months",
        sample_list: str | None = None,
        use_ca125_updated: bool = False,
        drop_failed: bool = False,
        require_panel_sequenced: bool = False,
        require_detected_cna: bool = False,
        cfg: ODEFitConfig = ODEFitConfig(),
        out_points_csv: str = "ode_points_single.csv",
        diag_dir: str | None = None,
) -> pd.DataFrame:
    if diag_dir:
        diag_dir = ensure_dir(diag_dir)

    rows = fit_and_collect_points(
        patient,
        data_path=data_path,
        time_unit=time_unit,
        sample_list=sample_list,
        use_ca125_updated=use_ca125_updated,
        drop_failed=drop_failed,
        require_panel_sequenced=require_panel_sequenced,
        require_detected_cna=require_detected_cna,
        cfg=cfg,
        diag_dir=diag_dir,
    )
    df = pd.DataFrame(rows)
    df.to_csv(out_points_csv, index=False)
    get_logger("tumorfit.ode.single").info(f"Saved ODE points: {out_points_csv} rows={len(df)}")
    return df


def run_ode_cli(args) -> int:
    cfg = ODEFitConfig(
        n_starts=args.n_starts,
        rel_noise=args.rel_noise,
        n_jobs_patients=args.n_jobs_patients,
        n_jobs_starts=args.n_jobs_starts,
        maxiter=args.maxiter,
        w_ca=args.w_ca,
    )

    if args.patient:
        fit_ode_single(
            data_path=args.data,
            patient=args.patient,
            time_unit=args.time_unit,
            sample_list=args.sample_list,
            use_ca125_updated=args.use_ca125_updated,
            drop_failed=args.drop_failed,
            require_panel_sequenced=args.require_panel_sequenced,
            require_detected_cna=args.require_detected_cna,
            cfg=cfg,
            out_points_csv=args.out_points,
            diag_dir=args.diag_dir,
        )
        return 0

    # cohort
    flags = [x.strip() for x in args.flag.split(",") if x.strip()]
    df = fit_ode_cohort(
        data_path=args.data,
        flags=flags,
        time_unit=args.time_unit,
        sample_list=args.sample_list,
        use_ca125_updated=args.use_ca125_updated,
        drop_failed=args.drop_failed,
        require_panel_sequenced=args.require_panel_sequenced,
        require_detected_cna=args.require_detected_cna,
        cfg=cfg,
        out_points_csv=args.out_points,
        diag_dir=args.diag_dir,
    )
    plot_gof_scatter_all(df, out_prefix="goodness_of_fit")
    return 0
