# SPDX-FileCopyrightText: 2025 Abhinav Mishra
# SPDX-License-Identifier: MIT
"""Regression tests for the ODE model."""
import numpy as np
import pytest

# Guard: skip if numba not available (CI without heavy deps)
pytest.importorskip("numba")

from tumorfits.odeio import PatientData
from tumorfits.odemodel import simulate_ode, unpack_theta_ode
from tumorfits.utils import logit, safe_log


def _make_patient_data(n_ctx: int = 2) -> PatientData:
    """Build a minimal PatientData for testing."""
    t = np.linspace(0.0, 12.0, 13)
    context_names = [f"ctx{i}" for i in range(n_ctx)]
    context = np.zeros(len(t), dtype=np.int32)
    ratio = np.linspace(0.1, 0.6, len(t))
    se = np.full(len(t), 0.05)
    ca125 = np.full(len(t), 100.0)
    log_ca = np.log(ca125)
    return PatientData(
        patient="TEST",
        t=t,
        context=context,
        context_names=context_names,
        ratio=ratio,
        se_logit_ratio=se,
        ca125=ca125,
        log_ca125=log_ca,
        maybe_mask=np.zeros(len(t), dtype=bool),
    )


def _make_theta(n_ctx: int = 2) -> np.ndarray:
    """Build a valid theta vector (10 base + n_ctx context logits)."""
    # base: log_aS, logit_aR_ratio, log_dS, logit_dR_ratio, log_K,
    #       log_N0, logit_r0, log_gamma, log_ca0, log_sigma_ca
    base = np.array([
        safe_log(np.array([0.5]))[0],    # log_aS
        logit(np.array([0.6]))[0],        # logit_aR_over_aS
        safe_log(np.array([0.3]))[0],    # log_dS
        logit(np.array([0.5]))[0],        # logit_dR_over_dS
        safe_log(np.array([1e6]))[0],    # log_K
        safe_log(np.array([1e4]))[0],    # log_N0
        logit(np.array([0.1]))[0],        # logit_r0
        safe_log(np.array([1e-4]))[0],   # log_gamma
        safe_log(np.array([10.0]))[0],   # log_ca0
        safe_log(np.array([0.1]))[0],    # log_sigma_ca
    ])
    ctx_logits = np.zeros(n_ctx)  # u_ctx ≈ 0.5 for all contexts
    return np.concatenate([base, ctx_logits])


def test_simulate_ode_output_shape():
    data = _make_patient_data(n_ctx=2)
    theta = _make_theta(n_ctx=2)
    r, log_ca = simulate_ode(data, theta)
    assert r.shape == (len(data.t),), "r should have shape (n_times,)"
    assert log_ca.shape == (len(data.t),), "log_ca should have shape (n_times,)"


def test_simulate_ode_positive():
    data = _make_patient_data(n_ctx=2)
    theta = _make_theta(n_ctx=2)
    r, log_ca = simulate_ode(data, theta)
    assert np.all(r >= 0) and np.all(r <= 1), "Resistant fraction must be in [0, 1]"
    assert np.all(np.isfinite(log_ca)), "log_ca must be finite"


def test_unpack_theta_ode_shapes():
    data = _make_patient_data(n_ctx=2)
    theta = _make_theta(n_ctx=2)
    aS, aR, dS, dR, K, S0, R0, gamma, ca0, sigma_ca, u_ctx = unpack_theta_ode(data, theta)
    assert aS > 0 and aR > 0 and dS > 0 and dR > 0 and K > 0
    assert S0 >= 0 and R0 >= 0
    assert len(u_ctx) == 2
