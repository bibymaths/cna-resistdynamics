<!-- SPDX-FileCopyrightText: 2025 Abhinav Mishra -->
<!-- SPDX-License-Identifier: MIT -->

# CLI Reference

All functionality is exposed via the `tumorfits` command.

```
tumorfits <subcommand> [options]
```

Run `tumorfits --help` or `tumorfits <subcommand> --help` for the full option
listing.

---

## tumorfits extract-data

Extract patient CSVs from raw `.RData` files.

```
tumorfits extract-data [--data-root DIR] [--out-dir DIR]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--data-root` | `data` | Root directory containing `.RData` files |
| `--out-dir` | `data/patient_data` | Output directory |

**Example:**
```bash
tumorfits extract-data \
    --data-root data/ \
    --out-dir   data/patient_data
```

---

## tumorfits ode

Fit the ODE model to one patient or a cohort.

```
tumorfits ode --data FILE (--patient ID | --flag FLAGS) [options]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--data` | *(required)* | `Subclonal_ratio_estimates.extended.txt` |
| `--patient` | — | Single patient ID (mutually exclusive with `--flag`) |
| `--flag` | — | Comma-separated `Accept_estimate` values, e.g. `yes,maybe` |
| `--out_points` | `ode_gof_points.csv` | Output long-table CSV |
| `--diag_dir` | `results_ODE` | Per-patient diagnostics directory |
| `--time_unit` | `months` | `months` or `days` |
| `--n_starts` | `8` | Number of optimiser restarts |
| `--maxiter` | `1200` | Max iterations per restart |
| `--w_ca` | `0.5` | Weight for CA125 term in likelihood |
| `--n_jobs_patients` | `-1` | Parallel workers for patients |
| `--sample_list` | `None` | Path to `OV_patientDNA_sampleList.txt` |
| `--use_ca125_updated` | `false` | Use revised CA125 values |
| `--drop_failed` | `false` | Exclude failed QC samples |

---

## tumorfits pde

Run or fit the 1-D PDE reaction–diffusion model.

```
tumorfits pde --data FILE --ode_points FILE [--patient ID|ALL] [options]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--data` | *(required)* | Subclonal ratios file |
| `--ode_points` | *(required)* | ODE long-table CSV from `tumorfits ode` |
| `--patient` | `ALL` | Patient ID or `ALL` |
| `--out_dir` | `results_pde_model` | Output directory |
| `--fit` | `false` | Re-fit diffusion coefficients |
| `--L` | `1.0` | Domain length |
| `--n_cells` | `200` | Number of FEM cells |
| `--dt` | `0.001` | Time step |
| `--DS`, `--DR` | `0.01` | Diffusion coefficients |
| `--n_starts` | `10` | Optimiser restarts (if `--fit`) |
| `--maxiter` | `150` | Max iterations per restart |

---

## tumorfits heatmap

Generate space-time heatmaps of S(x,t) and R(x,t) without re-fitting.

```
tumorfits heatmap --data FILE --ode_points FILE --patient ID [options]
```

Produces a single PNG file `heatmap_<patient>.png` in `--out_dir`.

| Option | Default | Description |
|--------|---------|-------------|
| `--data` | *(required)* | Subclonal ratios file |
| `--ode_points` | *(required)* | ODE long-table CSV |
| `--patient` | *(required)* | Patient ID |
| `--out_dir` | `results_pde_model` | Output directory |
| `--n_cells` | `100` | Spatial resolution |
| `--L` | `1.0` | Domain length |

---

## tumorfits mesh-view

Run a 2-D FEniCS simulation and produce PyVista visualisations.

```
tumorfits mesh-view --data FILE --ode-points FILE [options]
```

Saves three PNG files per patient: resistance zones, streamlines, drug efficacy.

| Option | Default | Description |
|--------|---------|-------------|
| `--data` | *(required)* | Subclonal ratios file |
| `--ode-points` | *(required)* | ODE long-table CSV |
| `--out-dir` | `results_pde_model` | Output directory |
| `--patient` | `ALL` | Patient ID or `ALL` |
| `--nx`, `--ny` | `50` | FEniCS mesh resolution |
| `--dt` | `0.5` | Time step (months) |
| `--sample-list` | `None` | QC metadata file |
