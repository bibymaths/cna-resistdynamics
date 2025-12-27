from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class PDEConfig:
    # domain / discretization
    L: float = 1.0
    n_cells: int = 200
    dt: float = 1e-3

    # diffusion
    DS: float = 1e-2
    DR: float = 1e-2

    # observation model (CA125 ≈ ca0 + gamma * ∫(S+R) dx)
    gamma: float = 1.0
    ca0: float = 0.0
    sigma_ca: float = 0.5
    w_ca: float = 1.0
    u_ctx: Optional[np.ndarray] = None

    # fitting
    maxiter: int = 150
    maxfev: int = 500
    n_starts: int = 10
    n_jobs_starts: int = -1


def get_treatment_value(t: float) -> float:
    """
    Placeholder: constant therapy intensity u(t)=1.
    Replace with context-based schedule if you want parity with ODE later.
    """
    return 1.0
