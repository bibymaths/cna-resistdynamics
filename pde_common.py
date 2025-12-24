# pde_common.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple, Callable
import numpy as np
import pandas as pd


# ---------------------- math helpers ----------------------

def safe_log(x: np.ndarray) -> np.ndarray:
    return np.log(np.clip(x, 1e-12, None))

def logit(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 1e-6, 1 - 1e-6)
    return np.log(x / (1 - x))

def invlogit(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))

def ci95_to_se_logit(r: np.ndarray, r_lo: np.ndarray, r_hi: np.ndarray) -> np.ndarray:
    """Approx SE on logit scale from a 95% interval on ratio scale."""
    y_lo = logit(np.clip(r_lo, 1e-9, 1 - 1e-9))
    y_hi = logit(np.clip(r_hi, 1e-9, 1 - 1e-9))
    se = (y_hi - y_lo) / 3.92
    return np.clip(se, 1e-2, 5.0)


# ---------------------- data model ----------------------

@dataclass
class PatientData:
    patient: str
    t: np.ndarray               # time in chosen units (months or days)
    context: np.ndarray         # integer context id per sample
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
    Loads Subclonal_ratio_estimates.extended.txt and returns patient-level arrays.
    Optionally merges OV_patientDNA_sampleList.txt for CA125_updated + QC filters.
    """
    df = pd.read_csv(path, sep="\t")
    df = df[df["Accept_estimate"].isin(list(accept_flags))].copy()

    # --- optional merge with sample list for updated CA125 + QC flags ---
    if sample_list_path is not None:
        sl = load_sample_list(sample_list_path)

        # detect sample id column in df (often "time" is sample name, while "Time" is numeric)
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

        def is_true(x) -> bool:
            if pd.isna(x): return False
            if isinstance(x, (bool, np.bool_)): return bool(x)
            s = str(x).strip().lower()
            return s in ["true", "t", "1", "yes", "y"]

        if drop_failed and "Failed" in df.columns:
            df = df[~df["Failed"].apply(is_true)].copy()

        if require_panel_sequenced and "PanelSequenced" in df.columns:
            df = df[df["PanelSequenced"].apply(is_true)].copy()

        if require_detected_cna and "DetectedCNA" in df.columns:
            df = df[df["DetectedCNA"].apply(is_true)].copy()

        if use_ca125_updated and "CA125_updated" in df.columns:
            df["CA125"] = np.where(df["CA125_updated"].notna(), df["CA125_updated"], df["CA125"])

    # --- parse numeric columns and filter to patient ---
    df["Time"] = pd.to_numeric(df["Time"], errors="coerce")
    df["CA125"] = pd.to_numeric(df["CA125"], errors="coerce")
    df["ratio"] = pd.to_numeric(df["ratio"], errors="coerce")
    df["ratio_min95"] = pd.to_numeric(df["ratio_min95"], errors="coerce")
    df["ratio_max95"] = pd.to_numeric(df["ratio_max95"], errors="coerce")

    df = df.dropna(subset=["Time", "ratio", "ratio_min95", "ratio_max95", "CA125"]).copy()
    df["Patient"] = df["Patient"].astype(str)
    df = df[df["Patient"] == str(patient)].copy()
    if df.empty:
        raise ValueError(f"No rows found for patient={patient} after filtering/merging.")

    # rescale time
    if time_unit == "months":
        df["Time"] = df["Time"] / 30.0
    elif time_unit == "days":
        pass
    else:
        raise ValueError("time_unit must be 'months' or 'days'")

    df = df.sort_values("Time").reset_index(drop=True)

    # contexts -> ids
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


# ---------------------- therapy schedule helper ----------------------

def make_u_of_t(t_samples: np.ndarray, ctx_samples: np.ndarray, u_ctx: np.ndarray) -> Callable[[float], float]:
    """
    Piecewise constant u(t) using context of most recent observed sample.
    For t < t0 => use context at first sample.
    """
    t_samples = np.asarray(t_samples)
    ctx_samples = np.asarray(ctx_samples)

    def u(t: float) -> float:
        i = np.searchsorted(t_samples, t, side="right") - 1
        if i < 0: i = 0
        c = int(ctx_samples[i])
        return float(np.clip(u_ctx[c], 0.0, 1.0))

    return u


# ---------------------- observables from PDE grid ----------------------

def integrate_1d(vals: np.ndarray, dx: float) -> float:
    """Simple Riemann sum integral over 1D grid."""
    return float(np.sum(vals) * dx)

def pde_observables_from_grid(
    S_vals: np.ndarray,
    R_vals: np.ndarray,
    dx: float,
    *,
    gamma: float = 1.0,
    ca0: float = 0.0,
) -> Tuple[float, float, float, float]:
    """
    Given S(x), R(x) arrays at time t:
      returns (total_S, total_R, ratio_hat, logCA_hat)
    where
      total_S = ∫ S dx
      total_R = ∫ R dx
      ratio_hat = total_R / (total_S + total_R)
      logCA_hat = log(ca0 + gamma*(total_S+total_R))
    """
    S_vals = np.clip(np.asarray(S_vals, float), 0.0, None)
    R_vals = np.clip(np.asarray(R_vals, float), 0.0, None)
    total_S = integrate_1d(S_vals, dx)
    total_R = integrate_1d(R_vals, dx)
    total_N = max(total_S + total_R, 1e-12)
    ratio_hat = total_R / total_N
    logCA_hat = float(np.log(max(ca0 + gamma * total_N, 1e-12)))
    return total_S, total_R, float(ratio_hat), float(logCA_hat)


# ---------------------- common loss (matches your ODE NLL style) ----------------------

def nll_ratio_ca(
    *,
    ratio_obs: np.ndarray,
    se_logit_ratio: np.ndarray,
    logca_obs: np.ndarray,
    ratio_hat: np.ndarray,
    logca_hat: np.ndarray,
    sigma_ca: float,
    w_ca: float = 1.0,
) -> float:
    """
    Negative log-likelihood:
      - ratio on logit scale with per-timepoint SE from CI
      - CA125 on log scale with shared sigma_ca
    """
    ratio_obs = np.asarray(ratio_obs, float)
    se_logit_ratio = np.asarray(se_logit_ratio, float)
    logca_obs = np.asarray(logca_obs, float)
    ratio_hat = np.asarray(ratio_hat, float)
    logca_hat = np.asarray(logca_hat, float)

    y_obs = logit(ratio_obs)
    y_hat = logit(np.clip(ratio_hat, 1e-6, 1 - 1e-6))
    se = np.clip(se_logit_ratio, 1e-3, 1e3)

    nll_ratio = 0.5 * np.sum(((y_obs - y_hat) / se) ** 2 + 2 * np.log(se) + np.log(2 * np.pi))

    sigma_ca = float(max(sigma_ca, 1e-6))
    nll_ca = 0.5 * np.sum(((logca_obs - logca_hat) / sigma_ca) ** 2 + 2 * np.log(sigma_ca) + np.log(2 * np.pi))

    return float(nll_ratio + w_ca * nll_ca)
