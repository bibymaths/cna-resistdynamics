---
hide:
  - navigation
  - toc
---

<!-- SPDX-FileCopyrightText: 2025 Abhinav Mishra -->
<!-- SPDX-License-Identifier: MIT -->

<div align="center">
  <img src="https://raw.githubusercontent.com/bibymaths/cna-resistdynamics/main/docs/assets/logo.svg"
       alt="CNA-ResistDynamics" width="180" style="margin-bottom: 1rem;" />

  <h1 style="font-size: 2.5rem; font-weight: 800; letter-spacing: -1px; margin-bottom: 0.25rem;">
    CNA-ResistDynamics
  </h1>

  <p style="font-size: 1.15rem; color: var(--md-default-fg-color--light); margin-bottom: 1.5rem;">
    Dynamical modelling of chemotherapy resistance from liquidCNA measurements
  </p>

  <p>
    <a href="https://github.com/bibymaths/cna-resistdynamics/actions/workflows/tests.yml">
      <img alt="tests" src="https://github.com/bibymaths/cna-resistdynamics/actions/workflows/tests.yml/badge.svg" />
    </a>
    <a href="https://github.com/bibymaths/cna-resistdynamics/actions/workflows/docs.yml">
      <img alt="docs" src="https://github.com/bibymaths/cna-resistdynamics/actions/workflows/docs.yml/badge.svg" />
    </a>
    <img alt="python" src="https://img.shields.io/badge/python-3.11-blue" />
    <img alt="license" src="https://img.shields.io/badge/license-MIT-green" />
  </p>
</div>

---

## Overview

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
