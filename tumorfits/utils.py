from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import numpy as np


def set_thread_env(max_threads: int = 1) -> None:
    """
    Prevent oversubscription from BLAS/OpenMP inside parallel workers.
    Call once at startup (main entry).
    """
    v = str(int(max_threads))
    os.environ.setdefault("OMP_NUM_THREADS", v)
    os.environ.setdefault("OPENBLAS_NUM_THREADS", v)
    os.environ.setdefault("MKL_NUM_THREADS", v)
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", v)
    os.environ.setdefault("NUMEXPR_NUM_THREADS", v)


def ensure_dir(path: str | os.PathLike) -> str:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return str(p)


def safe_log(x: np.ndarray) -> np.ndarray:
    return np.log(np.clip(np.asarray(x, float), 1e-12, None))


def logit(x: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(x, float), 1e-6, 1 - 1e-6)
    return np.log(x / (1 - x))


def invlogit(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, float)
    return 1.0 / (1.0 + np.exp(-z))


def ci95_to_se_logit(r: np.ndarray, r_lo: np.ndarray, r_hi: np.ndarray) -> np.ndarray:
    """
    Approx SE on logit scale from a 95% interval on ratio scale.
    """
    y_lo = logit(np.clip(r_lo, 1e-9, 1 - 1e-9))
    y_hi = logit(np.clip(r_hi, 1e-9, 1 - 1e-9))
    se = (y_hi - y_lo) / 3.92
    return np.clip(se, 1e-2, 5.0)


def clip01(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return np.clip(np.asarray(x, float), eps, 1 - eps)


def as_list(x: str | Iterable[str] | None) -> list[str]:
    if x is None:
        return []
    if isinstance(x, str):
        return [s.strip() for s in x.split(",") if s.strip()]
    return [str(s).strip() for s in x if str(s).strip()]
