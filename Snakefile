# SPDX-FileCopyrightText: 2025 Abhinav Mishra
# SPDX-License-Identifier: MIT
"""
Snakemake workflow for the tumorfits ODE → PDE resistance-dynamics pipeline.

All parameters are loaded from config.yaml.  Run the full pipeline with:

    snakemake --cores all --configfile config.yaml

Or run individual targets, e.g.:

    snakemake --cores all results/ode_gof_points.csv

Dry-run:
    snakemake -n --configfile config.yaml
"""

configfile: "config.yaml"

# ── Derived path aliases ──────────────────────────────────────────────────────
DATA            = config["data"]["subclonal_ratios"]
SAMPLE_LIST     = config["data"]["sample_list"]
PATIENT_DATA    = config["data"]["patient_data_dir"]
ODE_POINTS      = config["ode"]["out_points"]
ODE_DIAG        = config["ode"]["diag_dir"]
PDE_OUT         = config["pde"]["out_dir"]
HM_OUT          = config["heatmap"]["out_dir"]
MV_OUT          = config["mesh_view"]["out_dir"]

FLAGS           = config["cohort"]["flags"]
TIME_UNIT       = config["cohort"]["time_unit"]


# ── Helper: boolean flag → CLI option ─────────────────────────────────────────
def _flag(key, opt):
    return opt if config["cohort"].get(key, False) else ""


# ── Default target ────────────────────────────────────────────────────────────
rule all:
    """Run the complete ODE → PDE pipeline."""
    input:
        ODE_POINTS,
        directory(PDE_OUT),
        directory(HM_OUT),


# ── 1. Extract patient CSVs from .RData files ─────────────────────────────────
rule extract_data:
    """
    Convert raw .RData files (QDNAseq + liquidCNA outputs) into per-patient
    CSV archives under data/patient_data/.
    """
    input:
        rdata=directory(config["data"]["root"])
    output:
        directory(PATIENT_DATA)
    log:
        "logs/extract_data.log"
    shell:
        """
        tumorfits extract-data \
            --data-root {input.rdata} \
            --out-dir   {output} \
        2>&1 | tee {log}
        """


# ── 2. ODE cohort fit ─────────────────────────────────────────────────────────
rule ode_fit:
    """
    Fit the ODE resistance model to all patients matching the cohort flags.
    Produces a long-table CSV (one row per parameter per patient).
    """
    input:
        data=DATA
    output:
        ODE_POINTS
    log:
        "logs/ode_fit.log"
    params:
        flags         = FLAGS,
        time_unit     = TIME_UNIT,
        sample_list   = SAMPLE_LIST,
        diag_dir      = ODE_DIAG,
        n_starts      = config["ode"]["n_starts"],
        rel_noise     = config["ode"]["rel_noise"],
        maxiter       = config["ode"]["maxiter"],
        w_ca          = config["ode"]["w_ca"],
        n_jobs_p      = config["ode"]["n_jobs_patients"],
        n_jobs_s      = config["ode"]["n_jobs_starts"],
        use_ca125     = _flag("use_ca125_updated", "--use_ca125_updated"),
        drop_failed   = _flag("drop_failed", "--drop_failed"),
        req_panel     = _flag("require_panel_sequenced", "--require_panel_sequenced"),
        req_cna       = _flag("require_detected_cna", "--require_detected_cna"),
    shell:
        """
        tumorfits ode \
            --data            {input.data} \
            --flag            {params.flags} \
            --time_unit       {params.time_unit} \
            --sample_list     {params.sample_list} \
            --out_points      {output} \
            --diag_dir        {params.diag_dir} \
            --n_starts        {params.n_starts} \
            --rel_noise       {params.rel_noise} \
            --maxiter         {params.maxiter} \
            --w_ca            {params.w_ca} \
            --n_jobs_patients {params.n_jobs_p} \
            --n_jobs_starts   {params.n_jobs_s} \
            {params.use_ca125} {params.drop_failed} \
            {params.req_panel} {params.req_cna} \
        2>&1 | tee {log}
        """


# ── 3. PDE cohort run ─────────────────────────────────────────────────────────
rule pde_run:
    """
    Run (or fit) the 1-D PDE reaction–diffusion model for all patients using
    the ODE parameter estimates as starting points.
    """
    input:
        data       = DATA,
        ode_points = ODE_POINTS,
    output:
        directory(PDE_OUT)
    log:
        "logs/pde_run.log"
    params:
        p   = config["pde"],
        fit = "--fit" if config["pde"].get("fit", False) else "",
    shell:
        """
        tumorfits pde \
            --data        {input.data} \
            --ode_points  {input.ode_points} \
            --time_unit   {TIME_UNIT} \
            --sample_list {SAMPLE_LIST} \
            --patient     ALL \
            --out_dir     {output} \
            --L           {params.p[L]} \
            --n_cells     {params.p[n_cells]} \
            --dt          {params.p[dt]} \
            --DS          {params.p[DS]} \
            --DR          {params.p[DR]} \
            --gamma       {params.p[gamma]} \
            --ca0         {params.p[ca0]} \
            --sigma_ca    {params.p[sigma_ca]} \
            --w_ca        {params.p[w_ca]} \
            --maxiter     {params.p[maxiter]} \
            --maxfev      {params.p[maxfev]} \
            --n_starts    {params.p[n_starts]} \
            --n_jobs_starts {params.p[n_jobs_starts]} \
            {params.fit} \
        2>&1 | tee {log}
        """


# ── 4. Heatmaps for all patients ──────────────────────────────────────────────
rule heatmaps:
    """
    Generate PDE space-time heatmaps for every patient.
    """
    input:
        data       = DATA,
        ode_points = ODE_POINTS,
    output:
        directory(HM_OUT)
    log:
        "logs/heatmaps.log"
    params:
        h = config["heatmap"],
        patients = ["UP0018", "UP0042", "UP0053", "UP0055", "UP0056"],
    run:
        import os
        os.makedirs(output[0], exist_ok=True)
        for pid in params.patients:
            shell(
                f"tumorfits heatmap "
                f"--data {input.data} "
                f"--ode_points {input.ode_points} "
                f"--patient {pid} "
                f"--out_dir {output[0]} "
                f"--L {params.h['L']} "
                f"--n_cells {params.h['n_cells']} "
                f"--dt {params.h['dt']} "
                f"--DS {params.h['DS']} "
                f"--DR {params.h['DR']} "
                f"--gamma {params.h['gamma']} "
                f"--ca0 {params.h['ca0']} "
                f"--sigma_ca {params.h['sigma_ca']} "
                f"--w_ca {params.h['w_ca']} "
                f"2>&1 | tee -a {log}"
            )


# ── 5. Mesh visualisation ─────────────────────────────────────────────────────
rule mesh_view:
    """
    Run the 2-D FEniCS reaction–diffusion simulation and generate PyVista
    visualisations (resistance zones, streamlines, drug efficacy) per patient.
    """
    input:
        data       = DATA,
        ode_points = ODE_POINTS,
    output:
        directory(MV_OUT)
    log:
        "logs/mesh_view.log"
    params:
        mv = config["mesh_view"],
    shell:
        """
        tumorfits mesh-view \
            --data        {input.data} \
            --ode-points  {input.ode_points} \
            --out-dir     {output} \
            --patient     {params.mv[patient]} \
            --sample-list {SAMPLE_LIST} \
            --nx          {params.mv[nx]} \
            --ny          {params.mv[ny]} \
            --dt          {params.mv[dt]} \
        2>&1 | tee {log}
        """


# ── Clean ─────────────────────────────────────────────────────────────────────
rule clean:
    """Remove all generated results and logs."""
    shell:
        "rm -rf results/ logs/"
