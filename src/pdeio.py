from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from .utils import invlogit


def load_ode_long_theta(ode_points_csv: str, patient: str, context_names: list[str]) -> np.ndarray:
    df = pd.read_csv(ode_points_csv)
    sub = df[(df["patient"].astype(str) == str(patient)) & (df["var"].astype(str).str.startswith("theta:"))].copy()
    if sub.empty:
        raise ValueError(f"No theta rows found for patient={patient} in {ode_points_csv}")

    sub["var_norm"] = sub["var"].astype(str).str.strip().str.lower()

    base = [
        "log_aS","logit_aR_over_aS","log_dS","logit_dR_over_dS","log_K",
        "log_N0","logit_r0","log_gamma","log_ca0","log_sigma_ca",
    ]
    full = base + [f"logit_u_ctx[{c}]" for c in context_names]

    val_map = {str(r["var"]).split("theta:", 1)[1].strip().lower(): float(r["pred"]) for _, r in sub.iterrows()}

    theta = []
    missing = []
    for name in full:
        key = name.strip().lower()
        if key not in val_map:
            missing.append(name)
            theta.append(np.nan)
        else:
            theta.append(val_map[key])

    if missing:
        raise ValueError(f"Missing theta entries for {patient}: {missing}")

    return np.asarray(theta, float)

def load_ode_physical_params_map(ode_points_csv: str) -> Dict[str, list[float]]:
    """
    Returns {patient: [aS, aR, dS, dR, K]} from ODE long-table CSV.
    Works with var names like 'theta:log_aS' etc.
    """
    if not ode_points_csv or not os.path.exists(ode_points_csv):
        return {}

    df = pd.read_csv(ode_points_csv)
    if not {"patient", "var", "pred"}.issubset(df.columns):
        raise ValueError("ODE CSV must contain columns: patient, var, pred")

    # normalize var strings for robust matching
    df["var_norm"] = df["var"].astype(str).str.strip().str.lower()
    out: Dict[str, list[float]] = {}

    def get(sub: pd.DataFrame, name: str):
        key = f"theta:{name}".lower()
        row = sub[sub["var_norm"] == key]
        if row.empty:
            return None
        return float(row["pred"].values[0])

    for pid in df["patient"].astype(str).unique():
        sub = df[(df["patient"].astype(str) == pid) & (df["var_norm"].str.startswith("theta:"))].copy()
        if sub.empty:
            continue

        log_aS = get(sub, "log_aS")
        logit_aR = get(sub, "logit_aR_over_aS")
        log_dS = get(sub, "log_dS")
        logit_dR = get(sub, "logit_dR_over_dS")
        log_K = get(sub, "log_K")

        if any(v is None for v in [log_aS, logit_aR, log_dS, logit_dR, log_K]):
            continue

        aS = float(np.exp(log_aS))
        aR = float(invlogit(np.array([logit_aR]))[0] * 1.0) * aS  # ratio * aS
        dS = float(np.exp(log_dS))
        dR = float(invlogit(np.array([logit_dR]))[0] * 1.0) * dS  # ratio * dS
        K = float(np.exp(log_K))

        out[pid] = [aS, aR, dS, dR, K]

    return out

def load_u_ctx_from_ode_points(ode_points_csv: str, patient: str, context_names: List[str]) -> np.ndarray:
    """
    Returns u_ctx array in the SAME ORDER as context_names.
    Reads rows like: var = 'theta:logit_u_ctx[<context_name>]' and uses invlogit(pred).
    """
    df = pd.read_csv(ode_points_csv)
    sub = df[df["patient"].astype(str) == str(patient)].copy()
    if sub.empty:
        raise ValueError(f"No rows for patient={patient} in {ode_points_csv}")

    sub["var_norm"] = sub["var"].astype(str).str.strip().str.lower()

    def find_logit_for_ctx(c: str) -> float:
        key = f"theta:logit_u_ctx[{c}]".lower()
        row = sub[sub["var_norm"] == key]
        if row.empty:
            raise ValueError(f"Missing {key} for patient={patient}")
        return float(row["pred"].values[0])

    logits = np.array([find_logit_for_ctx(c) for c in context_names], dtype=float)
    u_ctx = 1.0 / (1.0 + np.exp(-logits))
    return np.clip(u_ctx, 0.0, 1.0)