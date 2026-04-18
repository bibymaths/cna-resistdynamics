# SPDX-FileCopyrightText: 2025 Abhinav Mishra
# SPDX-License-Identifier: MIT
"""Tests for config.yaml structure and Snakemake integration."""

from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"
SNAKEFILE_PATH = Path(__file__).parent.parent / "Snakefile"


def test_config_yaml_exists():
    assert CONFIG_PATH.exists(), "config.yaml must exist"


def test_config_yaml_valid():
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    assert isinstance(cfg, dict)


def test_config_has_required_keys():
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    for key in ("data", "cohort", "ode", "pde", "heatmap", "mesh_view"):
        assert key in cfg, f"Missing top-level key: {key}"


def test_config_data_section():
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    data = cfg["data"]
    assert "subclonal_ratios" in data
    assert "patient_data_dir" in data


def test_config_ode_section():
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    ode = cfg["ode"]
    assert "n_starts" in ode
    assert isinstance(ode["n_starts"], int)
    assert ode["n_starts"] > 0


def test_snakefile_exists():
    assert SNAKEFILE_PATH.exists(), "Snakefile must exist"
