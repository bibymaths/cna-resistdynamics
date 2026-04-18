# SPDX-FileCopyrightText: 2025 Abhinav Mishra
# SPDX-License-Identifier: MIT
"""
tumorfits.cli
=============
Unified command-line interface for the tumorfits package.

Subcommands
-----------
extract-data  Extract patient CSVs from raw .RData files.
ode           Fit the ODE resistance model to patient data.
pde           Run / fit the PDE reaction-diffusion model.
heatmap       Generate PDE space-time heatmaps (no fitting required).
mesh-view     Run 2-D FEniCS simulation and produce PyVista visualisations.
"""
from __future__ import annotations

import argparse
import sys

from .oderunner import run_ode_cli
from .pderunner import run_pde_cli


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tumorfits",
        description="ODE → PDE tumour resistance fitting pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  tumorfits extract-data --data-root data/ --out-dir data/patient_data\n"
            "  tumorfits ode --data data/liquidCNA_results/Subclonal_ratio_estimates.extended.txt "
            "--flag yes,maybe --out_points ode_gof_points.csv\n"
            "  tumorfits pde --data ... --ode_points ode_gof_points.csv --patient UP0018\n"
            "  tumorfits heatmap --data ... --ode_points ode_gof_points.csv --patient UP0018\n"
            "  tumorfits mesh-view --data ... --ode-points ode_gof_points.csv\n"
        ),
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # ------------------------------------------------------------------ #
    # extract-data                                                         #
    # ------------------------------------------------------------------ #
    ed = sub.add_parser(
        "extract-data",
        help="Extract patient CSVs from raw .RData files",
        description=(
            "Walk a data directory tree, read every .RData file using pyreadr, "
            "identify the patient ID (UP####) from the filename, and write each "
            "DataFrame object to <out-dir>/<patient_id>/<object_name>.csv."
        ),
    )
    ed.add_argument(
        "--data-root",
        default="data",
        help="Root directory containing .RData files (default: data/)",
    )
    ed.add_argument(
        "--out-dir",
        default="data/patient_data",
        help="Output directory for patient CSV sub-folders (default: data/patient_data/)",
    )

    def _run_extract(args: argparse.Namespace) -> int:
        from .dataio import export_all_patient_data

        written = export_all_patient_data(args.data_root, args.out_dir)
        total = sum(len(v) for v in written.values())
        print(f"Extracted {total} CSV files for {len(written)} patients → {args.out_dir}")
        return 0

    ed.set_defaults(func=_run_extract)

    # ------------------------------------------------------------------ #
    # ode                                                                  #
    # ------------------------------------------------------------------ #
    ode = sub.add_parser("ode", help="Fit ODE model; write long-table CSV + (optional) diagnostics")
    ode.add_argument("--data", required=True, help="Subclonal_ratio_estimates.extended.txt")
    ode.add_argument("--time_unit", default="months", choices=["months", "days"])
    ode.add_argument("--sample_list", default=None)
    ode.add_argument("--use_ca125_updated", action="store_true")
    ode.add_argument("--drop_failed", action="store_true")
    ode.add_argument("--require_panel_sequenced", action="store_true")
    ode.add_argument("--require_detected_cna", action="store_true")

    g = ode.add_mutually_exclusive_group(required=True)
    g.add_argument("--patient", help="Single patient id, e.g. UP0018")
    g.add_argument("--flag", help="Comma-separated Accept_estimate values, e.g. yes or yes,maybe")

    ode.add_argument("--out_points", default="ode_gof_points.csv")
    ode.add_argument("--diag_dir", default="results_ODE", help="If set: per-patient plots/CSVs")
    ode.add_argument("--n_starts", type=int, default=8)
    ode.add_argument("--rel_noise", type=float, default=0.25)
    ode.add_argument("--n_jobs_patients", type=int, default=-1)
    ode.add_argument("--n_jobs_starts", type=int, default=1)
    ode.add_argument("--maxiter", type=int, default=1200)
    ode.add_argument("--w_ca", type=float, default=0.5)
    ode.set_defaults(func=run_ode_cli)

    # ------------------------------------------------------------------ #
    # pde                                                                  #
    # ------------------------------------------------------------------ #
    pde = sub.add_parser("pde", help="Run/fit PDE using ODE long-table CSV (theta rows)")
    pde.add_argument("--data", required=True)
    pde.add_argument("--ode_points", required=True, help="ODE long-table CSV from `tumorfits ode`")
    pde.add_argument("--time_unit", default="months", choices=["months", "days"])
    pde.add_argument("--sample_list", default=None)
    pde.add_argument("--use_ca125_updated", action="store_true")
    pde.add_argument("--drop_failed", action="store_true")
    pde.add_argument("--require_panel_sequenced", action="store_true")
    pde.add_argument("--require_detected_cna", action="store_true")
    pde.add_argument("--patient", default="ALL", help="Patient id or ALL")
    pde.add_argument("--out_dir", default="results_pde_model")
    pde.add_argument("--fit", action="store_true")
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
    pde.add_argument("--maxfev", type=int, default=10000)
    pde.add_argument("--n_starts", type=int, default=10)
    pde.add_argument("--n_jobs_starts", type=int, default=-1)
    pde.set_defaults(func=run_pde_cli)

    # ------------------------------------------------------------------ #
    # heatmap                                                              #
    # ------------------------------------------------------------------ #
    hm = sub.add_parser(
        "heatmap",
        help="Generate PDE space-time heatmaps (no fitting required)",
    )
    hm.add_argument("--data", required=True)
    hm.add_argument("--ode_points", required=True)
    hm.add_argument("--time_unit", default="months", choices=["months", "days"])
    hm.add_argument("--sample_list", default=None)
    hm.add_argument("--patient", required=True)
    hm.add_argument("--out_dir", default="results_pde_model")
    hm.add_argument("--L", type=float, default=1.0)
    hm.add_argument("--n_cells", type=int, default=100)
    hm.add_argument("--dt", type=float, default=1e-3)
    hm.add_argument("--DS", type=float, default=1e-2)
    hm.add_argument("--DR", type=float, default=1e-2)
    hm.add_argument("--gamma", type=float, default=1.0)
    hm.add_argument("--ca0", type=float, default=0.0)
    hm.add_argument("--sigma_ca", type=float, default=0.5)
    hm.add_argument("--w_ca", type=float, default=1.0)

    def _run_heatmap(args: argparse.Namespace) -> int:
        from .pdemodel import PDEConfig
        from .simpde import run_pde_heatmap

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
        print(f"Heatmap saved: {out}")
        return 0

    hm.set_defaults(func=_run_heatmap)

    # ------------------------------------------------------------------ #
    # mesh-view                                                            #
    # ------------------------------------------------------------------ #
    mv = sub.add_parser(
        "mesh-view",
        help="Run 2-D FEniCS simulation and produce PyVista visualisations",
        description=(
            "Runs a 2-D reaction–diffusion simulation on a unit-square FEniCS mesh "
            "for each patient in the ODE results table, then saves three off-screen "
            "PyVista PNG plots per patient: resistance zones, growth streamlines, "
            "and drug-efficacy maps."
        ),
    )
    mv.add_argument("--data", required=True, help="Subclonal_ratio_estimates.extended.txt")
    mv.add_argument(
        "--ode-points", dest="ode_points", required=True,
        help="ODE long-table CSV from `tumorfits ode`",
    )
    mv.add_argument("--out-dir", dest="out_dir", default="results_pde_model")
    mv.add_argument(
        "--patient", default="ALL",
        help="Patient ID or ALL (default: ALL)",
    )
    mv.add_argument(
        "--sample-list", dest="sample_list", default=None,
        help="Optional path to OV_patientDNA_sampleList.txt for QC filtering",
    )
    mv.add_argument("--time_unit", default="months", choices=["months", "days"])
    mv.add_argument("--use_ca125_updated", action="store_true")
    mv.add_argument("--drop_failed", action="store_true")
    mv.add_argument("--nx", type=int, default=50, help="FEniCS mesh x-resolution (default: 50)")
    mv.add_argument("--ny", type=int, default=50, help="FEniCS mesh y-resolution (default: 50)")
    mv.add_argument("--dt", type=float, default=0.5, help="Time-step in months (default: 0.5)")

    def _run_mesh_view(args: argparse.Namespace) -> int:
        from .meshview import load_all_patient_params, run_mesh_view_pipeline
        from .odeio import load_patient_data

        patient_db = load_all_patient_params(args.ode_points)

        if args.patient != "ALL":
            patient_db = {k: v for k, v in patient_db.items() if k == args.patient}

        if not patient_db:
            print("No patients found in ODE results; nothing to visualise.", file=sys.stderr)
            return 1

        patient_data_map = {}
        for pid in patient_db:
            try:
                patient_data_map[pid] = load_patient_data(
                    args.data, pid,
                    time_unit=args.time_unit,
                    sample_list_path=args.sample_list,
                    use_ca125_updated=args.use_ca125_updated,
                    drop_failed=args.drop_failed,
                )
            except Exception as exc:
                print(f"Warning: could not load data for {pid}: {exc}", file=sys.stderr)

        outputs = run_mesh_view_pipeline(
            patient_db,
            patient_data_map,
            out_dir=args.out_dir,
            nx=args.nx,
            ny=args.ny,
            dt=args.dt,
        )
        total = sum(len(v) for v in outputs.values())
        print(f"Generated {total} PNG files for {len(outputs)} patients → {args.out_dir}")
        return 0

    mv.set_defaults(func=_run_mesh_view)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
