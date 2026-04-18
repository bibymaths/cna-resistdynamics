# Contributing

Thank you for your interest in contributing to CNA-ResistDynamics.

## Development setup

```bash
git clone https://github.com/bibymaths/cna-resistdynamics.git
cd cna-resistdynamics
conda env create -f environment.yml
conda activate cna-resist-dynamics
pip install -e ".[dev,docs]"
pre-commit install
```

## Code style

This project uses [ruff](https://docs.astral.sh/ruff/) for linting and
formatting.

```bash
ruff check tumorfits/
ruff format tumorfits/
```

## Tests

```bash
pytest tests/ -v --tb=short
pytest tests/ -v --cov=tumorfits --cov-report=term-missing
```

## Pre-commit hooks

Pre-commit is configured to run ruff, trailing-whitespace, YAML/JSON validators,
and notebook output checks.  After installing (`pre-commit install`) they run
automatically on every commit.

## Submitting changes

1. Fork the repository and create a feature branch.
2. Make your changes, ensuring all tests pass and coverage does not drop.
3. Add or update docstrings for any new public functions.
4. Open a pull request with a clear description of the change.

## Reporting issues

Please file bug reports and feature requests on
[GitHub Issues](https://github.com/bibymaths/cna-resistdynamics/issues).
Include:
- Operating system and Python/Conda version
- Full error traceback
- Minimal reproducing example

## Adding a new patient cohort

See [Data — Adapting for a different dataset](data.md#adapting-for-a-different-dataset).
