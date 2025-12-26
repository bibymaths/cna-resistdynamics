from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from joblib import Parallel, delayed
from scipy.optimize import minimize

from .odeio import PatientData
from .pdemodel import PDEConfig
from .pdesolve import solve_pde


def run_single_start(seed: int, base_params: np.ndarray, cfg: PDEConfig, data: PatientData) -> tuple[float, np.ndarray]:
    rng = np.random.default_rng(seed)
    noise = rng.uniform(0.8, 1.2, size=base_params.size)
    x0 = base_params * noise

    def obj(p):
        return solve_pde(p, cfg, data, comm=None, return_history=False)[0]

    res = minimize(obj, x0, method="Nelder-Mead", options={"maxiter": cfg.maxiter, "disp": False})
    return float(res.fun), np.asarray(res.x, float)


def multistart_fit_pde(base_params: list[float] | np.ndarray, cfg: PDEConfig, data: PatientData) -> np.ndarray:
    base = np.asarray(base_params, float)
    if cfg.n_starts <= 1:
        val, x = run_single_start(0, base, cfg, data)
        return x

    results = Parallel(n_jobs=cfg.n_jobs_starts)(
        delayed(run_single_start)(i, base, cfg, data) for i in range(cfg.n_starts)
    )
    results.sort(key=lambda t: t[0])
    return results[0][1]
