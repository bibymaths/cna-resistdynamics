<!-- SPDX-FileCopyrightText: 2025 Abhinav Mishra -->
<!-- SPDX-License-Identifier: MIT -->

# Quick Start

This guide walks through the complete pipeline from raw `.RData` inputs to PDE
model outputs, using the five ovarian cancer patients included in the repository.

## 1. Extract patient data

Convert the raw liquidCNA `.RData` files into per-patient CSV archives:

```bash
tumorfits extract-data \
    --data-root data/ \
    --out-dir   data/patient_data
```

This walks `data/CNA_tables/` and `data/liquidCNA_results/`, reads each
`.RData` file, and writes every `data.frame` object as a CSV to
`data/patient_data/<patient_id>/`.

## 2. Fit the ODE model

Fit the well-mixed ODE model to all patients with `Accept_estimate = yes` or
`maybe`:

```bash
tumorfits ode \
    --data       data/liquidCNA_results/Subclonal_ratio_estimates.extended.txt \
    --flag       yes,maybe \
    --sample_list data/OV_patientDNA_sampleList.txt \
    --use_ca125_updated \
    --drop_failed \
    --out_points ode_gof_points.csv \
    --diag_dir   results/ode_diag \
    --n_starts   8 \
    --maxiter    1200
```

The output `ode_gof_points.csv` is a long-table CSV with one row per parameter
per patient.  It is the required input for all subsequent steps.

## 3. Generate PDE heatmaps

Visualise the 1-D reaction–diffusion dynamics without re-fitting:

```bash
tumorfits heatmap \
    --data       data/liquidCNA_results/Subclonal_ratio_estimates.extended.txt \
    --ode_points ode_gof_points.csv \
    --patient    UP0018 \
    --out_dir    results/heatmaps
```

## 4. Run the PDE model

Run the full 1-D PDE model (optionally fitting diffusion coefficients):

```bash
tumorfits pde \
    --data        data/liquidCNA_results/Subclonal_ratio_estimates.extended.txt \
    --ode_points  ode_gof_points.csv \
    --patient     ALL \
    --out_dir     results/pde \
    --n_cells     200 \
    --DS          0.01 \
    --DR          0.01
```

## 5. Generate mesh visualisations

Run the 2-D FEniCS simulation and produce PyVista PNG visualisations:

```bash
tumorfits mesh-view \
    --data       data/liquidCNA_results/Subclonal_ratio_estimates.extended.txt \
    --ode-points ode_gof_points.csv \
    --out-dir    results/mesh_view \
    --patient    ALL
```

## Run the full pipeline with Snakemake

All steps above are wired into a Snakemake workflow driven by `config.yaml`.
Edit `config.yaml` to customise paths and parameters, then run:

```bash
snakemake --cores all --configfile config.yaml
```

Dry-run first:

```bash
snakemake -n --configfile config.yaml
```
