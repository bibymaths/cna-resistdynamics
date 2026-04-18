<!-- SPDX-FileCopyrightText: 2025 Abhinav Mishra -->
<!-- SPDX-License-Identifier: MIT -->

# Installation

## Requirements

- Python 3.11
- FEniCS/dolfinx 0.10 (for the PDE solver and mesh-view pipeline)
- MPI (for dolfinx parallelism)

## Option 1 — Conda (recommended)

The complete environment including FEniCS, PETSc, and MPI can be installed via
Conda. This is the most reliable route because FEniCS requires native compiled
libraries that are difficult to install with pip alone.

```bash
conda env create -f environment.yml
conda activate cna-resist-dynamics
pip install -e .
```

Verify the installation:

```bash
tumorfits --help
```

## Option 2 — uv (Python-only dependencies)

If you do not need the PDE/FEniCS features, you can install the pure-Python
stack with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
# or for editable install:
uv pip install -e ".[dev,docs]"
```

!!! warning "FEniCS not available via pip/uv"
    The `tumorfits pde`, `tumorfits heatmap`, and `tumorfits mesh-view`
    subcommands require `dolfinx`, which must be installed via Conda or a
    pre-built Docker image.  Use the Conda route for the full pipeline.

## Option 3 — Docker

A pre-built Docker image is available for reproducible execution without a
local Conda/FEniCS installation:

```bash
docker compose up
```

See the [Dockerfile](https://github.com/bibymaths/cna-resistdynamics/blob/main/Dockerfile)
for details.

## Development install

```bash
conda env create -f environment.yml
conda activate cna-resist-dynamics
pip install -e ".[dev,docs]"
pre-commit install
```

## Verify

```bash
tumorfits --help
tumorfits ode --help
tumorfits pde --help
```
