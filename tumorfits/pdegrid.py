# SPDX-FileCopyrightText: 2025 Abhinav Mishra
# SPDX-License-Identifier: MIT
# tumorfits/pdegrid.py
from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True, fastmath=True, nogil=True)
def _pde_obs_jit(S_vals: np.ndarray, R_vals: np.ndarray, dx: float, gamma: float, ca0: float):
    total_S = 0.0
    total_R = 0.0

    n = S_vals.shape[0]
    for i in range(n):
        s = S_vals[i]
        r = R_vals[i]
        if s < 0.0:
            s = 0.0
        if r < 0.0:
            r = 0.0
        total_S += s
        total_R += r

    total_S *= dx
    total_R *= dx

    total_N = total_S + total_R
    if total_N < 1e-12:
        total_N = 1e-12

    ratio_hat = total_R / total_N

    ca = ca0 + gamma * total_N
    if ca < 1e-12:
        ca = 1e-12
    logCA_hat = np.log(ca)

    return total_S, total_R, ratio_hat, logCA_hat


def integrate_1d(vals: np.ndarray, dx: float) -> float:
    """Same signature; optionally accelerated."""
    v = np.asarray(vals, dtype=np.float64)
    if njit is not None:
        return float(np.sum(np.clip(v, 0.0, np.inf)) * float(dx))
    return float(np.sum(v) * float(dx))


def pde_observables_from_grid(
    S_vals: np.ndarray,
    R_vals: np.ndarray,
    dx: float,
    *,
    gamma: float = 1.0,
    ca0: float = 0.0,
) -> tuple[float, float, float, float]:
    """
    Same signature as your current code (keyword-only gamma/ca0).
    Returns (total_S, total_R, ratio_hat, logCA_hat).
    """
    S = np.asarray(S_vals, dtype=np.float64)
    R = np.asarray(R_vals, dtype=np.float64)
    dx = float(dx)
    g = float(gamma)
    c0 = float(ca0)

    if S.shape != R.shape:
        raise ValueError("pde_observables_from_grid: S_vals and R_vals must have same shape")

    if njit is not None:
        tS, tR, r, lc = _pde_obs_jit(S, R, dx, g, c0)
        return float(tS), float(tR), float(r), float(lc)

    # numpy fallback
    S2 = np.clip(S, 0.0, None)
    R2 = np.clip(R, 0.0, None)
    total_S = float(np.sum(S2) * dx)
    total_R = float(np.sum(R2) * dx)
    total_N = max(total_S + total_R, 1e-12)
    ratio_hat = float(total_R / total_N)
    logCA_hat = float(np.log(max(c0 + g * total_N, 1e-12)))
    return total_S, total_R, ratio_hat, logCA_hat
