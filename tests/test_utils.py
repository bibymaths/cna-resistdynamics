# SPDX-FileCopyrightText: 2025 Abhinav Mishra
# SPDX-License-Identifier: MIT
"""Tests for tumorfits.utils."""
import numpy as np
import pytest

from tumorfits.utils import (
    as_list,
    ci95_to_se_logit,
    ensure_dir,
    invlogit,
    logit,
    safe_log,
)


def test_logit_invlogit_roundtrip():
    x = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    assert np.allclose(invlogit(logit(x)), x, atol=1e-6)


def test_logit_clipping():
    # Should not raise even at boundaries
    result = logit(np.array([0.0, 1.0]))
    assert np.all(np.isfinite(result))


def test_invlogit_range():
    z = np.array([-10.0, 0.0, 10.0])
    result = invlogit(z)
    assert np.all(result >= 0) and np.all(result <= 1)


def test_safe_log_no_nan():
    x = np.array([0.0, 1e-20, 1.0, 100.0])
    result = safe_log(x)
    assert np.all(np.isfinite(result))


def test_ci95_to_se_logit_positive():
    r = np.array([0.5])
    r_lo = np.array([0.4])
    r_hi = np.array([0.6])
    se = ci95_to_se_logit(r, r_lo, r_hi)
    assert np.all(se > 0)
    assert np.all(np.isfinite(se))


def test_as_list_string():
    assert as_list("yes,maybe") == ["yes", "maybe"]


def test_as_list_none():
    assert as_list(None) == []


def test_as_list_single():
    assert as_list("yes") == ["yes"]


def test_ensure_dir(tmp_path):
    target = tmp_path / "a" / "b" / "c"
    result = ensure_dir(target)
    assert target.exists()
    assert result == str(target)
