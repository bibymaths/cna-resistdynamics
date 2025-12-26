from __future__ import annotations

import pandas as pd
import numpy as np

from .odeio import load_patient_data
from .odemodel import simulate_states
from .pdeio import load_ode_long_theta  # reuse ODE-theta loader


def simulate_ode_from_saved_theta(
    *,
    data_path: str,
    ode_points_csv: str,
    patient: str,
    time_unit: str = "months",
    sample_list: str | None = None,
) -> pd.DataFrame:
    """
    Load theta for a patient from the long-table ODE points CSV and simulate states.
    Returns states dataframe.
    """
    data = load_patient_data(data_path, patient, time_unit=time_unit, sample_list_path=sample_list)
    theta = load_ode_long_theta(ode_points_csv, patient, context_names=data.context_names)
    S, R, N, r, logca, _u = simulate_states(data, theta)
    return pd.DataFrame({
        "patient": patient,
        "time": data.t,
        "S": S,
        "R": R,
        "N": N,
        "ratio_pred": r,
        "logca_pred": logca,
        "ratio_obs": data.ratio,
        "logca_obs": data.log_ca125,
    })
