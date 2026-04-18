# SPDX-FileCopyrightText: 2025 Abhinav Mishra
# SPDX-License-Identifier: MIT
"""
tumorfits.dataio
================
Utilities for extracting patient data from .RData files produced by the
liquidCNA pipeline and exporting them as per-patient CSV archives.

The liquidCNA R objects stored in the .RData files include:
  - bins.df, cn.df, seg.df  (from QDNAseq copy-number tables)
  - pHat.df, seg.df.corr, seg.av.corr, seg.plot, final.medians, cutOff
    (from the liquidCNA subclonal ratio estimation step)

Usage via CLI:
    tumorfits extract-data --data-root data/ --out-dir data/patient_data

Usage as library:
    from tumorfits.dataio import export_all_patient_data
    export_all_patient_data("data/", "data/patient_data")
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import pandas as pd
import pyreadr

from .timelog import get_logger

_log = get_logger(__name__)

#: Regex pattern used to identify patient IDs (e.g. UP0018) in file names.
PATIENT_ID_PATTERN = re.compile(r"UP\d{4}")


def export_all_patient_data(
    root_dir: str | os.PathLike,
    out_dir: str | os.PathLike = "data/patient_data",
) -> dict[str, list[str]]:
    """
    Recursively locate ``.RData`` files under *root_dir*, infer the patient ID
    from the filename, and write every ``pd.DataFrame`` object found inside
    each file as a CSV under *out_dir*/<patient_id>/<object_name>.csv.

    Parameters
    ----------
    root_dir:
        Directory tree to search.  Typically ``data/``.
    out_dir:
        Base output directory.  Per-patient sub-directories are created
        automatically.  Defaults to ``data/patient_data``.

    Returns
    -------
    dict mapping patient_id → list of written CSV paths.
    """
    root_dir = Path(root_dir)
    out_dir = Path(out_dir)
    written: dict[str, list[str]] = {}

    for file_path in sorted(root_dir.rglob("*.RData")):
        match = PATIENT_ID_PATTERN.search(file_path.name)
        patient_id = match.group(0) if match else "Unknown_Patient"

        patient_dir = out_dir / patient_id
        patient_dir.mkdir(parents=True, exist_ok=True)

        _log.info("Processing %s  (patient=%s)", file_path.name, patient_id)

        try:
            result = pyreadr.read_r(str(file_path))
        except Exception as exc:
            _log.warning("Error reading %s: %s", file_path, exc)
            continue

        for obj_name, obj in result.items():
            if not isinstance(obj, pd.DataFrame):
                _log.debug("Skipped '%s': not a DataFrame", obj_name)
                continue

            csv_path = patient_dir / f"{obj_name}.csv"
            obj.to_csv(csv_path, index=True)
            _log.info("  -> %s", csv_path)
            written.setdefault(patient_id, []).append(str(csv_path))

    return written
