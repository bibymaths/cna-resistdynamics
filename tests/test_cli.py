# SPDX-FileCopyrightText: 2025 Abhinav Mishra
# SPDX-License-Identifier: MIT
"""Tests for the tumorfits CLI."""

import subprocess
import sys


def run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "tumorfits.cli", *args],
        capture_output=True,
        text=True,
    )


def test_help_top_level():
    result = run_cli("--help")
    assert result.returncode == 0
    assert "tumorfits" in result.stdout.lower()


def test_help_ode():
    result = run_cli("ode", "--help")
    assert result.returncode == 0
    assert "--data" in result.stdout


def test_help_pde():
    result = run_cli("pde", "--help")
    assert result.returncode == 0
    assert "--ode_points" in result.stdout


def test_help_extract_data():
    result = run_cli("extract-data", "--help")
    assert result.returncode == 0
    assert "--data-root" in result.stdout


def test_help_heatmap():
    result = run_cli("heatmap", "--help")
    assert result.returncode == 0
    assert "--patient" in result.stdout


def test_help_mesh_view():
    result = run_cli("mesh-view", "--help")
    assert result.returncode == 0
    assert "--ode-points" in result.stdout


def test_extract_data_no_rdata(tmp_path):
    """extract-data on an empty directory should succeed with zero files."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tumorfits.cli",
            "extract-data",
            "--data-root",
            str(tmp_path),
            "--out-dir",
            str(tmp_path / "patient_data"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "0" in result.stdout or "patient" in result.stdout.lower()
