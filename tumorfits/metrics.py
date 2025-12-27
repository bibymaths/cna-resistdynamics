from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True, fastmath=True, nogil=True)
def _logit_scalar(x: float) -> float:
    eps = 1e-6
    if x < eps:
        x = eps
    elif x > 1.0 - eps:
        x = 1.0 - eps
    return np.log(x / (1.0 - x))


@njit(cache=True, fastmath=True, nogil=True)
def _nll_ratio_ca_jit(
        ratio_obs: np.ndarray,
        se_logit_ratio: np.ndarray,
        logca_obs: np.ndarray,
        ratio_hat: np.ndarray,
        logca_hat: np.ndarray,
        sigma_ca: float,
        w_ca: float,
) -> float:
    n = ratio_obs.shape[0]

    # ratio part (logit scale)
    nll_ratio = 0.0
    for i in range(n):
        y_obs = _logit_scalar(ratio_obs[i])
        y_hat = _logit_scalar(ratio_hat[i])

        se = se_logit_ratio[i]
        if se < 1e-6:
            se = 1e-6

        z = (y_obs - y_hat) / se
        nll_ratio += 0.5 * (z * z + 2.0 * np.log(se) + np.log(2.0 * np.pi))

    # ca part (log scale)
    if sigma_ca < 1e-6:
        sigma_ca = 1e-6

    nll_ca = 0.0
    for i in range(n):
        z = (logca_obs[i] - logca_hat[i]) / sigma_ca
        nll_ca += 0.5 * (z * z + 2.0 * np.log(sigma_ca) + np.log(2.0 * np.pi))

    return nll_ratio + w_ca * nll_ca


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
    sigma_ca = float(sigma_ca)
    w_ca = float(w_ca)

    return float(
        _nll_ratio_ca_jit(
            ratio_obs,
            se_logit_ratio,
            logca_obs,
            ratio_hat,
            logca_hat,
            sigma_ca,
            w_ca,
        )
    )


def gof_metrics(
        r_obs: np.ndarray,
        r_hat: np.ndarray,
        logca_obs: np.ndarray,
        logca_hat: np.ndarray,
        nll: float,
        k_params: int,
) -> dict[str, float]:
    r_obs = np.asarray(r_obs, float)
    r_hat = np.asarray(r_hat, float)
    logca_obs = np.asarray(logca_obs, float)
    logca_hat = np.asarray(logca_hat, float)

    rmse_r = float(np.sqrt(np.mean((r_obs - r_hat) ** 2)))
    mae_r = float(np.mean(np.abs(r_obs - r_hat)))

    rmse_ca = float(np.sqrt(np.mean((logca_obs - logca_hat) ** 2)))
    mae_ca = float(np.mean(np.abs(logca_obs - logca_hat)))

    n = int(r_obs.size + logca_obs.size)
    aic = float(2 * k_params + 2 * nll)
    bic = float(k_params * np.log(max(n, 1)) + 2 * nll)

    return {
        "NLL": float(nll),
        "AIC": aic,
        "BIC": bic,
        "RMSE_ratio": rmse_r,
        "MAE_ratio": mae_r,
        "RMSE_logCA125": rmse_ca,
        "MAE_logCA125": mae_ca,
    }
