"""
Tests for utility functions.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from utils import (
    compute_cost,
    compute_delta_cost,
    mdl_penalty,
    safe_correlation,
    normalize_signal
)


def test_compute_cost_l2():
    """Test L2 (Gaussian) cost computation."""
    # Constant signal: cost should be near zero
    signal = np.ones(100) * 5
    cost = compute_cost(signal, model='l2')
    assert cost < 1.0, f"Constant signal should have near-zero cost, got {cost}"

    # Variable signal: cost should be positive
    signal_variable = np.random.randn(100)
    cost_variable = compute_cost(signal_variable, model='l2')
    assert cost_variable > 0, "Variable signal should have positive cost"


def test_compute_cost_l1():
    """Test L1 (robust) cost computation."""
    signal = np.array([1, 2, 3, 4, 5])
    cost_l1 = compute_cost(signal, model='l1')
    assert cost_l1 > 0, "L1 cost should be positive"

    # L1 should be more robust to outliers than L2
    signal_outlier = np.array([1, 2, 3, 4, 100])  # One large outlier
    cost_l1_outlier = compute_cost(signal_outlier, model='l1')
    cost_l2_outlier = compute_cost(signal_outlier, model='l2')

    # L1 should be less affected than L2
    assert cost_l1_outlier < cost_l2_outlier, "L1 should be more robust to outliers"


def test_compute_cost_ar1():
    """Test AR(1) cost computation."""
    # AR(1) signal: x_t = phi * x_{t-1} + eps
    phi = 0.8
    signal = np.zeros(100)
    signal[0] = np.random.randn()
    for t in range(1, 100):
        signal[t] = phi * signal[t-1] + np.random.randn() * 0.1

    cost_ar1 = compute_cost(signal, model='ar1')
    assert cost_ar1 >= 0, "AR(1) cost should be non-negative"


def test_compute_delta_cost():
    """Test cost reduction from splitting."""
    # Signal with clear changepoint at 100
    signal = np.concatenate([
        np.ones(100) * 0,
        np.ones(100) * 5
    ])

    # Splitting at true changepoint should reduce cost (negative ΔC)
    delta_C_true = compute_delta_cost(signal, 100, model='l2')
    assert delta_C_true < 0, f"Splitting at true seam should reduce cost: ΔC={delta_C_true}"

    # Splitting at wrong position should increase cost (positive ΔC)
    delta_C_wrong = compute_delta_cost(signal, 50, model='l2')
    assert delta_C_wrong > delta_C_true, "Wrong split should have higher cost"


def test_mdl_penalty():
    """Test MDL penalty computation."""
    T = 1000
    d = 1

    # More changepoints = higher penalty
    pen_1 = mdl_penalty(k=1, T=T, d=d)
    pen_5 = mdl_penalty(k=5, T=T, d=d)
    pen_10 = mdl_penalty(k=10, T=T, d=d)

    assert pen_5 > pen_1, "More changepoints should have higher penalty"
    assert pen_10 > pen_5, "Penalty should scale with k"

    # Longer signal = higher penalty (log T factor)
    pen_T1000 = mdl_penalty(k=5, T=1000, d=1)
    pen_T10000 = mdl_penalty(k=5, T=10000, d=1)

    assert pen_T10000 > pen_T1000, "Longer signal should have higher penalty"


def test_safe_correlation():
    """Test correlation with NaN handling."""
    # Perfect positive correlation
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([2, 4, 6, 8, 10])
    corr = safe_correlation(x, y)
    assert abs(corr - 1.0) < 0.01, f"Expected corr ≈ 1.0, got {corr}"

    # Perfect negative correlation
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([5, 4, 3, 2, 1])
    corr = safe_correlation(x, y)
    assert abs(corr - (-1.0)) < 0.01, f"Expected corr ≈ -1.0, got {corr}"

    # With NaN values
    x = np.array([1, 2, np.nan, 4, 5])
    y = np.array([2, 4, 6, 8, 10])
    corr = safe_correlation(x, y)
    # Should handle NaN gracefully
    assert not np.isnan(corr), "Correlation should not be NaN"


def test_normalize_signal():
    """Test z-score normalization."""
    signal = np.array([1, 2, 3, 4, 5])
    normalized = normalize_signal(signal)

    # Mean should be near zero
    assert abs(np.mean(normalized)) < 1e-10, f"Mean should be ~0, got {np.mean(normalized)}"
    # Std should be near 1
    assert abs(np.std(normalized) - 1.0) < 0.01, f"Std should be ~1, got {np.std(normalized)}"


def test_edge_case_empty_signal():
    """Test behavior on empty signal."""
    signal = np.array([])
    cost = compute_cost(signal, model='l2')
    assert cost == 0.0, "Empty signal should have zero cost"


def test_edge_case_single_value():
    """Test behavior on constant signal."""
    signal = np.ones(100) * 42
    cost = compute_cost(signal, model='l2')
    assert cost < 1e-10, "Constant signal should have near-zero cost"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
