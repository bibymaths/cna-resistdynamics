<img src="assets/logo.svg" alt="CNA-ResistDynamics" width="140" />

![Python 3.11](https://img.shields.io/badge/python-3.11-blue)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

**CNA-ResistDynamics** is a research software package that infers the temporal
evolution of chemotherapy resistance in ovarian cancer from longitudinal
liquid-biopsy data.

It uses two complementary dynamical models:

- **ODE model** — well-mixed population dynamics of sensitive (S) and resistant (R)
  tumour cells under periodic cytotoxic treatment
- **PDE model** — 1-D reaction–diffusion system that adds spatial structure to the
  resistance dynamics

Resistance is measured via *subclonal CNA fractions* estimated by the
[liquidCNA](https://github.com/McGranahanLab/liquidCNA) algorithm, combined with
CA125 serum protein as a tumour-burden proxy.

---

## Key features

- Likelihood-based multi-start optimisation (L-BFGS-B for ODE, Powell for PDE)
- Numba-JIT hot paths for ODE simulation and negative log-likelihood evaluation
- Operator-splitting FEniCS/dolfinx 1-D PDE solver with PETSc system caching
- Optional PyMC Bayesian posteriors (SMC / Metropolis)
- Sobol sensitivity analysis via SALib
- 2-D FEniCS + PyVista mesh visualisation pipeline
- Unified `tumorfits` CLI covering every workflow step
- Snakemake workflow with `config.yaml` as the single source of truth

---

## Quick navigation

<div class="grid cards" markdown>

- :material-rocket-launch: **[Quick Start](quickstart.md)** — Run your first fit in minutes
- :material-flask: **[Mathematical Model](model.md)** — ODE and PDE equations
- :material-database: **[Data](data.md)** — Data formats and preparation
- :material-console: **[CLI Reference](cli.md)** — All commands and options
- :material-api: **[API Reference](api.md)** — Python package documentation

</div>

---

## Citation

If you use this software, please cite:

> Hockings et al. (2025). *Subclonal copy-number alterations drive resistance to
> platinum-based chemotherapy in ovarian cancer.*
> Cancer Research. DOI: [10.1158/0008-5472.CAN-25-0351](https://doi.org/10.1158/0008-5472.CAN-25-0351)
