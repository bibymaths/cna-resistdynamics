
from __future__ import annotations

import argparse

from .oderunner import run_ode_cli
from .pderunner import run_pde_cli


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tumorfits", description="ODE -> PDE tumor fitting pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)

    # ---------------- ODE ----------------
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
    ode.add_argument("--diag_dir", default=None, help="If set: per-patient plots/CSVs")

    # fit config
    ode.add_argument("--n_starts", type=int, default=8)
    ode.add_argument("--rel_noise", type=float, default=0.25)
    ode.add_argument("--n_jobs_patients", type=int, default=-1)
    ode.add_argument("--n_jobs_starts", type=int, default=1)
    ode.add_argument("--maxiter", type=int, default=1200)
    ode.add_argument("--w_ca", type=float, default=0.5)

    ode.set_defaults(func=run_ode_cli)

    # ---------------- PDE ----------------
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

    # PDEConfig knobs
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

    pde.set_defaults(func=run_pde_cli)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
