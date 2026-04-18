# SPDX-FileCopyrightText: 2025 Abhinav Mishra
# SPDX-License-Identifier: MIT
"""Tests for tumorfits.metrics."""
import numpy as np
import pytest

from tumorfits.metrics import gof_metrics


def test_gof_metrics_shape():
    obs = np.array([0.2, 0.5, 0.8])
    pred = np.array([0.21, 0.48, 0.79])
    se = np.array([0.05, 0.05, 0.05])
    result = gof_metrics(obs, pred, se, n_params=5, label="test")
    assert "NLL" in result
    assert "AIC" in result
    assert "BIC" in result
    assert "RMSE" in result
    assert "MAE" in result


def test_gof_metrics_perfect_fit():
    obs = np.array([0.3, 0.5, 0.7])
    se = np.array([0.05, 0.05, 0.05])
    result = gof_metrics(obs, obs, se, n_params=5)
    assert result["RMSE"] == pytest.approx(0.0, abs=1e-9)
    assert result["MAE"] == pytest.approx(0.0, abs=1e-9)
