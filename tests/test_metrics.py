# SPDX-FileCopyrightText: 2025 Abhinav Mishra
# SPDX-License-Identifier: MIT
"""Tests for tumorfits.metrics."""

import numpy as np
import pytest
from tumorfits.metrics import gof_metrics


def test_gof_metrics_shape():
    r_obs = np.array([0.2, 0.5, 0.8])
    r_hat = np.array([0.21, 0.48, 0.79])
    logca_obs = np.log(np.array([100.0, 200.0, 150.0]))
    logca_hat = np.log(np.array([101.0, 198.0, 152.0]))
    result = gof_metrics(r_obs, r_hat, logca_obs, logca_hat, nll=5.0, k_params=5)
    assert "NLL" in result
    assert "AIC" in result
    assert "BIC" in result
    assert "RMSE_ratio" in result
    assert "MAE_ratio" in result


def test_gof_metrics_perfect_fit():
    r_obs = np.array([0.3, 0.5, 0.7])
    logca_obs = np.log(np.array([100.0, 200.0, 150.0]))
    result = gof_metrics(r_obs, r_obs, logca_obs, logca_obs, nll=0.0, k_params=5)
    assert result["RMSE_ratio"] == pytest.approx(0.0, abs=1e-9)
    assert result["MAE_ratio"] == pytest.approx(0.0, abs=1e-9)
