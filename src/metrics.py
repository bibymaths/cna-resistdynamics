from __future__ import annotations

import numpy as np

from .utils import logit, clip01


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

    y_obs = logit(clip01(ratio_obs))
    y_hat = logit(clip01(ratio_hat))
    se = np.clip(se_logit_ratio, 1e-3, 1e3)

    nll_ratio = 0.5 * np.sum(((y_obs - y_hat) / se) ** 2 + 2 * np.log(se) + np.log(2 * np.pi))

    sigma_ca = float(max(sigma_ca, 1e-6))
    nll_ca = 0.5 * np.sum(((logca_obs - logca_hat) / sigma_ca) ** 2 + 2 * np.log(sigma_ca) + np.log(2 * np.pi))

    return float(nll_ratio + w_ca * nll_ca)


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
