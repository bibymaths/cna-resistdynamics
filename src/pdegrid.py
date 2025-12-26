from __future__ import annotations

from typing import Tuple
import numpy as np


def integrate_1d(vals: np.ndarray, dx: float) -> float:
    return float(np.sum(np.asarray(vals, float)) * float(dx))


def pde_observables_from_grid(
    S_vals: np.ndarray,
    R_vals: np.ndarray,
    dx: float,
    *,
    gamma: float = 1.0,
    ca0: float = 0.0,
) -> Tuple[float, float, float, float]:
    S_vals = np.clip(np.asarray(S_vals, float), 0.0, None)
    R_vals = np.clip(np.asarray(R_vals, float), 0.0, None)

    total_S = integrate_1d(S_vals, dx)
    total_R = integrate_1d(R_vals, dx)
    total_N = max(total_S + total_R, 1e-12)
    ratio_hat = total_R / total_N
    logCA_hat = float(np.log(max(ca0 + gamma * total_N, 1e-12)))
    return total_S, total_R, float(ratio_hat), float(logCA_hat)
