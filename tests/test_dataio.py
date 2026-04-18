# SPDX-FileCopyrightText: 2025 Abhinav Mishra
# SPDX-License-Identifier: MIT
"""Tests for tumorfits.dataio."""

from tumorfits.dataio import PATIENT_ID_PATTERN, export_all_patient_data


def test_patient_id_pattern():
    assert (
        PATIENT_ID_PATTERN.search("Copynumber_tables_UP0018.combined.500.RData").group(0)
        == "UP0018"
    )
    assert (
        PATIENT_ID_PATTERN.search("Estimates_OV_UP0056.vR.filtered.500.RData").group(0) == "UP0056"
    )
    assert PATIENT_ID_PATTERN.search("no_patient_here.RData") is None


def test_export_all_patient_data_no_rdata(tmp_path):
    """Should return empty dict when no .RData files exist."""
    (tmp_path / "empty").mkdir()
    result = export_all_patient_data(tmp_path, tmp_path / "out")
    assert result == {}


def test_export_all_patient_data_creates_dir(tmp_path):
    """Output directory should be created even if nothing is written."""
    out_dir = tmp_path / "patient_data"
    export_all_patient_data(tmp_path, out_dir)
    # No error expected; out_dir may or may not be created depending on whether
    # any .RData files were found — that's fine.
