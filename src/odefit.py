from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Callable, Sequence

import numpy as np
from joblib import Parallel, delayed
from scipy.optimize import minimize

from .timelog import Timer, get_logger


@dataclass
class MultiStartResult:
    x: np.ndarray
    fun: float
    success: bool
    message: str
    nit: int | None = None


def multistart_minimize(
    fun: Callable[[np.ndarray], float],
    x0: np.ndarray,
    bounds: Sequence[tuple[float, float]],
    *,
    n_starts: int = 1,
    rel_noise: float = 0.3,
    seed: int = 0,
    method: str = "L-BFGS-B",
    maxiter: int = 800,
    n_jobs_starts: int = 1,
    logger_name: str = "tumorfit.fit",
) -> MultiStartResult:
    """
    Generic multi-start wrapper around scipy.optimize.minimize.
    """
    logger = get_logger(logger_name)
    rng = np.random.default_rng(seed)
    tm_all = Timer()

    lo = np.array([b[0] for b in bounds], dtype=float)
    hi = np.array([b[1] for b in bounds], dtype=float)

    starts = [np.asarray(x0, float).copy()]
    for _ in range(max(0, n_starts - 1)):
        z = starts[0].copy()
        noise = rng.normal(0.0, rel_noise, size=z.size)
        z = np.clip(z + noise, lo, hi)
        starts.append(z)

    def one_start(i: int, s: np.ndarray):
        tm = Timer()
        try:
            res = minimize(fun, s, method=method, bounds=bounds, options={"maxiter": maxiter})
            val = float(res.fun) if np.isfinite(res.fun) else np.inf
            logger.info(f"start {i}/{len(starts)} fun={val:.4g} success={bool(res.success)} nit={getattr(res,'nit',None)} dt={tm.s():.2f}s")
            return val, res
        except Exception as e:
            logger.warning(f"start {i}/{len(starts)} exception: {type(e).__name__}: {e} dt={tm.s():.2f}s")
            return np.inf, None

    if n_jobs_starts is None or n_jobs_starts <= 1 or len(starts) == 1:
        results = [one_start(i, s) for i, s in enumerate(starts, 1)]
    else:
        results = Parallel(n_jobs=n_jobs_starts, backend="loky", prefer="processes", batch_size=1)(
            delayed(one_start)(i, s) for i, s in enumerate(starts, 1)
        )

    best_val = np.inf
    best_res = None
    for val, res in results:
        if res is not None and val < best_val:
            best_val, best_res = val, res

    if best_res is None:
        return MultiStartResult(x=np.asarray(x0, float), fun=float("inf"), success=False, message="all_starts_failed")

    logger.info(f"multistart best_fun={best_val:.4g} total_dt={tm_all.s():.2f}s")
    return MultiStartResult(
        x=np.asarray(best_res.x, float),
        fun=float(best_res.fun),
        success=bool(best_res.success),
        message=str(best_res.message),
        nit=getattr(best_res, "nit", None),
    )
