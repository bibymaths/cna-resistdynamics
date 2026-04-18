# SPDX-FileCopyrightText: 2025 Abhinav Mishra
# SPDX-License-Identifier: MIT
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from .utils import safe_log, ci95_to_se_logit


@dataclass
class PatientData:
    patient: str
    t: np.ndarray
    context: np.ndarray
    context_names: List[str]
    ratio: np.ndarray
    se_logit_ratio: np.ndarray
    ca125: np.ndarray
    log_ca125: np.ndarray
    maybe_mask: np.ndarray


def load_sample_list(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    df.columns = [c.strip() for c in df.columns]
    if "SampleName" in df.columns:
        df["SampleName"] = df["SampleName"].astype(str)
    if "Patient" in df.columns:
        df["Patient"] = df["Patient"].astype(str)
    return df


def _is_true(x) -> bool:
    if pd.isna(x):
        return False
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    s = str(x).strip().lower()
    return s in ["true", "t", "1", "yes", "y"]


def load_patient_data(
        path: str,
        patient: str,
        *,
        time_unit: str = "months",
        sample_list_path: Optional[str] = None,
        use_ca125_updated: bool = False,
        drop_failed: bool = False,
        require_panel_sequenced: bool = False,
        require_detected_cna: bool = False,
        accept_flags: Tuple[str, ...] = ("yes", "maybe"),
) -> PatientData:
    """
    Loads Subclonal_ratio_estimates.extended.txt (tab-separated)
    Required columns:
      Patient, Time, context, ratio, ratio_min95, ratio_max95, CA125, Accept_estimate
    Optional merge with OV_patientDNA_sampleList.txt for CA125_updated + QC filters.
    """
    df = pd.read_csv(path, sep="\t")
    df = df[df["Accept_estimate"].isin(list(accept_flags))].copy()

    if sample_list_path is not None:
        sl = load_sample_list(sample_list_path)

        sample_col = None
        for cand in ["time", "SampleName", "sample", "sample_id"]:
            if cand in df.columns:
                sample_col = cand
                break
        if sample_col is None:
            raise ValueError("Could not find sample id column in extended table (expected 'time' or similar).")

        df[sample_col] = df[sample_col].astype(str)
        sl = sl.rename(columns={"SampleName": sample_col})

        keep_cols = [sample_col, "Patient"]
        for c in ["CA125_updated", "Failed", "PanelSequenced", "DetectedCNA"]:
            if c in sl.columns:
                keep_cols.append(c)

        df = df.merge(sl[keep_cols], on=[sample_col, "Patient"], how="left")

        if drop_failed and "Failed" in df.columns:
            df = df[~df["Failed"].apply(_is_true)].copy()
        if require_panel_sequenced and "PanelSequenced" in df.columns:
            df = df[df["PanelSequenced"].apply(_is_true)].copy()
        if require_detected_cna and "DetectedCNA" in df.columns:
            df = df[df["DetectedCNA"].apply(_is_true)].copy()
        if use_ca125_updated and "CA125_updated" in df.columns:
            df["CA125"] = np.where(df["CA125_updated"].notna(), df["CA125_updated"], df["CA125"])

    # numeric parse
    for c in ["Time", "CA125", "ratio", "ratio_min95", "ratio_max95"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["Time", "ratio", "ratio_min95", "ratio_max95", "CA125"]).copy()
    df["Patient"] = df["Patient"].astype(str)
    df = df[df["Patient"] == str(patient)].copy()
    if df.empty:
        raise ValueError(f"No rows found for patient={patient} after filtering/merging.")

    if time_unit == "months":
        df["Time"] = df["Time"] / 30.0
    elif time_unit == "days":
        pass
    else:
        raise ValueError("time_unit must be 'months' or 'days'")

    df = df.sort_values("Time").reset_index(drop=True)

    context_names = df["context"].astype(str).unique().tolist()
    ctx_map = {c: i for i, c in enumerate(context_names)}
    ctx = df["context"].astype(str).map(ctx_map).to_numpy().astype(int)

    ratio = np.clip(df["ratio"].to_numpy().astype(float), 1e-4, 1 - 1e-4)
    r_lo = np.clip(df["ratio_min95"].to_numpy().astype(float), 1e-4, 1 - 1e-4)
    r_hi = np.clip(df["ratio_max95"].to_numpy().astype(float), 1e-4, 1 - 1e-4)
    se = ci95_to_se_logit(ratio, r_lo, r_hi)

    maybe = (df["Accept_estimate"].to_numpy() == "maybe")
    se = np.where(maybe, se * 2.0, se)

    ca = df["CA125"].to_numpy().astype(float)
    log_ca = safe_log(ca)

    return PatientData(
        patient=str(patient),
        t=df["Time"].to_numpy().astype(float),
        context=ctx,
        context_names=context_names,
        ratio=ratio,
        se_logit_ratio=se,
        ca125=ca,
        log_ca125=log_ca,
        maybe_mask=maybe,
    )


def get_patients_with_flag(path: str, flags: list[str]) -> list[str]:
    df = pd.read_csv(path, sep="\t")
    flags = [f.strip() for f in flags]
    df = df[df["Accept_estimate"].isin(flags)]
    return sorted(df["Patient"].astype(str).unique())


def load_drivers(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    if "Patient" not in df.columns:
        raise ValueError(f"No Patient column. Columns={list(df.columns)}")

    gene_col = "GeneName" if "GeneName" in df.columns else ("GeneID" if "GeneID" in df.columns else df.columns[0])
    df = df[["Patient", gene_col]].rename(columns={gene_col: "Driver"}).copy()
    df["Patient"] = df["Patient"].astype(str).str.strip()
    df["Driver"] = df["Driver"].astype(str).str.strip()

    g = (df.groupby("Patient")["Driver"]
         .apply(lambda s: ",".join(sorted(set([x for x in s if x and x.lower() != "nan"]))))
         .reset_index())
    g["n_drivers"] = g["Driver"].apply(lambda x: 0 if not x else len(x.split(",")))
    return g
