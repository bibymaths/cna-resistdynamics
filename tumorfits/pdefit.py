from __future__ import annotations

import numpy as np
from joblib import Parallel, delayed
from scipy.optimize import minimize
from .timelog import get_logger
from .odeio import PatientData
from .pdemodel import PDEConfig
from .pdesolve import solve_pde


def run_single_start(seed: int, base_params: np.ndarray, cfg: PDEConfig, data: PatientData) -> tuple[float, np.ndarray]:
    logger = get_logger("tumorfit.pde")
    rng = np.random.default_rng(seed)

    noise = rng.uniform(0.8, 1.2, size=base_params.size)
    x0 = np.asarray(base_params, float) * noise

    # avoid logging inside obj() for speed
    def obj(p: np.ndarray) -> float:
        return float(solve_pde(p, cfg, data, comm=None, return_history=False)[0])

    start_id = seed + 1
    logger.info(f"{data.patient}: PDE start {start_id}/{cfg.n_starts} x0={np.round(x0, 4)}")

    res = minimize(
        obj,
        x0,
        method="Powell",
        options={"maxiter": int(cfg.maxiter),
                 "maxfev": int(cfg.maxfev),
                 "disp": False},
    )

    logger.info(
        f"{data.patient}: PDE start {start_id}/{cfg.n_starts} done "
        f"fun={float(res.fun):.4g} success={bool(getattr(res,'success', False))} "
        f"nfev={getattr(res,'nfev', None)}"
    )

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
