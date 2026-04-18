<!-- SPDX-FileCopyrightText: 2025 Abhinav Mishra -->
<!-- SPDX-License-Identifier: MIT -->

# Contributing to CNA-ResistDynamics

Thank you for your interest in contributing.

## Development environment

```bash
git clone https://github.com/bibymaths/cna-resistdynamics.git
cd cna-resistdynamics
conda env create -f environment.yml
conda activate cna-resist-dynamics
pip install -e ".[dev,docs]"
pre-commit install
```

## Running tests

```bash
pytest tests/ -v --tb=short
```

With coverage:

```bash
pytest tests/ --cov=tumorfits --cov-report=term-missing
```

## Code style

This project uses [ruff](https://docs.astral.sh/ruff/):

```bash
ruff check tumorfits/ tests/
ruff format tumorfits/ tests/
```

## Pre-commit hooks

After running `pre-commit install`, hooks run automatically on each commit.
You can also run them manually:

```bash
pre-commit run --all-files
```

## Submitting a pull request

1. Fork the repository and create a feature branch from `main`.
2. Write tests for any new functionality.
3. Ensure `pytest` passes and coverage does not decrease.
4. Ensure `ruff check` produces no errors.
5. Update docstrings and documentation as needed.
6. Open a PR with a clear title and description.

## Adding support for a new cohort or cancer type

See the [Data documentation](docs/data.md#adapting-for-a-different-dataset)
for a step-by-step guide.

## Reporting bugs

File an issue at https://github.com/bibymaths/cna-resistdynamics/issues with:
- Python version and environment (Conda/uv)
- Full error traceback
- Minimal reproducing example
