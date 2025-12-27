from __future__ import annotations

import argparse
import os

from .odeplotio import plot_gof_scatter_all
from .oderunner import fit_ode_cohort, ODEFitConfig
from .pdemodel import PDEConfig
from .pderunner import run_pde_for_patient
from .simpde import run_pde_heatmap
from .timelog import get_logger
from .utils import set_thread_env, as_list, ensure_dir


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="tumorfit")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # ODE cohort
    ode = sub.add_parser("ode", help="Fit ODE for cohort and save GOF points CSV")
    ode.add_argument("--data", required=True)
    ode.add_argument("--flags", required=True, help="comma separated, e.g. yes or yes,maybe")
    ode.add_argument("--time_unit", default="months", choices=["months", "days"])
    ode.add_argument("--sample_list", default=None)
    ode.add_argument("--use_ca125_updated", action="store_true")
    ode.add_argument("--drop_failed", action="store_true")
    ode.add_argument("--require_panel_sequenced", action="store_true")
    ode.add_argument("--require_detected_cna", action="store_true")

    ode.add_argument("--n_starts", type=int, default=8)
    ode.add_argument("--rel_noise", type=float, default=0.25)
    ode.add_argument("--n_jobs_patients", type=int, default=-1)
    ode.add_argument("--n_jobs_starts", type=int, default=1)
    ode.add_argument("--maxiter", type=int, default=1200)
    ode.add_argument("--w_ca", type=float, default=0.5)

    ode.add_argument("--out_points", default="ode_gof_points.csv")
    ode.add_argument("--diag_dir", default="results_ode_model/per_patient_plots")
    ode.add_argument("--scatter_prefix", default="gof")

    # PDE single patient
    pde = sub.add_parser("pde", help="Run PDE for a patient (optionally fit) using ODE params as start")
    pde.add_argument("--data", required=True)
    pde.add_argument("--ode_points", required=True)
    pde.add_argument("--patient", required=True)
    pde.add_argument("--fit", action="store_true")
    pde.add_argument("--out_dir", default="results_pde_model")

    pde.add_argument("--time_unit", default="months", choices=["months", "days"])
    pde.add_argument("--sample_list", default=None)
    pde.add_argument("--use_ca125_updated", action="store_true")
    pde.add_argument("--drop_failed", action="store_true")
    pde.add_argument("--require_panel_sequenced", action="store_true")
    pde.add_argument("--require_detected_cna", action="store_true")

    # PDE config knobs
    pde.add_argument("--L", type=float, default=1.0)
    pde.add_argument("--n_cells", type=int, default=200)
    pde.add_argument("--dt", type=float, default=1e-3)
    pde.add_argument("--DS", type=float, default=1e-2)
    pde.add_argument("--DR", type=float, default=1e-2)

    pde.add_argument("--gamma", type=float, default=1.0)
    pde.add_argument("--ca0", type=float, default=0.0)
    pde.add_argument("--sigma_ca", type=float, default=0.5)
    pde.add_argument("--w_ca", type=float, default=1.0)

    pde.add_argument("--maxiter", type=int, default=150)
    pde.add_argument("--n_starts", type=int, default=10)
    pde.add_argument("--n_jobs_starts", type=int, default=-1)

    # Heatmap
    hm = sub.add_parser("heatmap", help="Generate PDE heatmap for a patient (no fit; uses ODE params)")
    hm.add_argument("--data", required=True)
    hm.add_argument("--ode_points", required=True)
    hm.add_argument("--patient", required=True)
    hm.add_argument("--out_dir", default="results_pde_model")
    hm.add_argument("--time_unit", default="months", choices=["months", "days"])
    hm.add_argument("--sample_list", default=None)

    hm.add_argument("--L", type=float, default=1.0)
    hm.add_argument("--n_cells", type=int, default=100)
    hm.add_argument("--dt", type=float, default=1e-3)
    hm.add_argument("--DS", type=float, default=1e-2)
    hm.add_argument("--DR", type=float, default=1e-2)
    hm.add_argument("--gamma", type=float, default=1.0)
    hm.add_argument("--ca0", type=float, default=0.0)
    hm.add_argument("--sigma_ca", type=float, default=0.5)
    hm.add_argument("--w_ca", type=float, default=1.0)

    # Full pipeline
    full = sub.add_parser("full", help="Run ODE cohort -> then PDE for one patient")
    full.add_argument("--data", required=True)
    full.add_argument("--flags", required=True)
    full.add_argument("--patient", required=True)
    full.add_argument("--out_root", default="results_pipeline")
    full.add_argument("--fit_pde", action="store_true")

    return ap


def main():
    set_thread_env(1)
    logger = get_logger("tumorfit.main")

    ap = build_parser()
    args = ap.parse_args()

    if args.cmd == "ode":
        cfg = ODEFitConfig(
            n_starts=args.n_starts,
            rel_noise=args.rel_noise,
            n_jobs_patients=args.n_jobs_patients,
            n_jobs_starts=args.n_jobs_starts,
            maxiter=args.maxiter,
            w_ca=args.w_ca,
        )
        df = fit_ode_cohort(
            data_path=args.data,
            flags=as_list(args.flags),
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
        plot_gof_scatter_all(df, out_prefix=args.scatter_prefix)
        logger.info("ODE done.")
        return

    if args.cmd == "pde":
        cfg = PDEConfig(
            L=args.L, n_cells=args.n_cells, dt=args.dt,
            DS=args.DS, DR=args.DR,
            gamma=args.gamma, ca0=args.ca0,
            sigma_ca=args.sigma_ca, w_ca=args.w_ca,
            maxiter=args.maxiter, n_starts=args.n_starts, n_jobs_starts=args.n_jobs_starts
        )
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
            do_fit=bool(args.fit),
        )
        logger.info("PDE done.")
        return

    if args.cmd == "heatmap":
        cfg = PDEConfig(
            L=args.L, n_cells=args.n_cells, dt=args.dt,
            DS=args.DS, DR=args.DR,
            gamma=args.gamma, ca0=args.ca0,
            sigma_ca=args.sigma_ca, w_ca=args.w_ca,
        )
        out = run_pde_heatmap(
            data_path=args.data,
            ode_points_csv=args.ode_points,
            patient=args.patient,
            cfg=cfg,
            out_dir=args.out_dir,
            time_unit=args.time_unit,
            sample_list=args.sample_list,
        )
        logger.info(f"Saved heatmap: {out}")
        return

    if args.cmd == "full":
        root = ensure_dir(args.out_root)
        ode_points = os.path.join(root, "ode_gof_points.csv")
        diag_dir = os.path.join(root, "ode_diag")
        pde_dir = os.path.join(root, "pde")

        cfg = ODEFitConfig()
        df = fit_ode_cohort(
            data_path=args.data,
            flags=as_list(args.flags),
            out_points_csv=ode_points,
            diag_dir=diag_dir,
            cfg=cfg,
        )
        plot_gof_scatter_all(df, out_prefix=os.path.join(root, "gof"))

        pdecfg = PDEConfig()
        run_pde_for_patient(
            data_path=args.data,
            ode_points_csv=ode_points,
            patient=args.patient,
            cfg=pdecfg,
            out_dir=pde_dir,
            do_fit=bool(args.fit_pde),
        )
        logger.info("FULL pipeline done.")
        return


if __name__ == "__main__":
    main()
