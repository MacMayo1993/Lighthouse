"""
Tests for seam type classification via cost curvature.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from seam_types import (
    classify_seam_type,
    compute_curvature_profile,
    adaptive_threshold_classification,
    recommend_expensive_ops
)


def test_classify_cusp():
    """Test classification of cusp (hard step) seams."""
    # Create signal with hard step at position 100
    signal = np.concatenate([
        np.ones(100) * 0,
        np.ones(100) * 5
    ])
    signal += np.random.normal(0, 0.1, len(signal))

    seam_type, curvature = classify_seam_type(signal, 100, window_size=10, model='l2')

    # Should classify as Cusp (C)
    assert seam_type == 'C', f"Expected C, got {seam_type}"
    # Curvature should be high (> 2.0)
    assert abs(curvature) > 2.0, f"Expected |κ| > 2.0, got {abs(curvature)}"


def test_classify_smooth():
    """Test classification of smooth seams."""
    # Create signal with gradual change (Gaussian bump)
    t = np.linspace(0, 10, 200)
    signal = np.sin(0.5 * t)

    # Add Gaussian bump at t=100
    bump = 2.0 * np.exp(-((t - t[100])**2) / 2.0)
    signal += bump
    signal += np.random.normal(0, 0.05, len(signal))

    seam_type, curvature = classify_seam_type(signal, 100, window_size=10, model='l2')

    # Should classify as Smooth (S)
    assert seam_type == 'S', f"Expected S, got {seam_type}"
    # Curvature should be low (< 0.02)
    assert abs(curvature) < 0.02, f"Expected |κ| < 0.02, got {abs(curvature)}"


def test_classify_tangent():
    """Test classification of tangent (sigmoid) seams."""
    # Create signal with sigmoid transition
    t = np.linspace(-6, 6, 200)
    signal = 1 / (1 + np.exp(-t))
    signal = 5 * signal  # Scale
    signal += np.random.normal(0, 0.05, len(signal))

    # Seam at midpoint (100)
    seam_type, curvature = classify_seam_type(signal, 100, window_size=10, model='l2')

    # Should classify as Tangent (T)
    assert seam_type in ['T', 'C'], f"Expected T or C, got {seam_type}"
    # Curvature should be moderate (0.02 to 2.0) or high for sharp sigmoid
    assert abs(curvature) >= 0.02, f"Expected |κ| >= 0.02, got {abs(curvature)}"


def test_curvature_profile():
    """Test computing curvature profile for multiple seams."""
    # Create signal with 3 seams
    signal = np.concatenate([
        np.ones(100) * 0,
        np.ones(100) * 5,
        np.ones(100) * -3,
        np.ones(100) * 2
    ])
    signal += np.random.normal(0, 0.1, len(signal))

    seams = [100, 200, 300]

    profile = compute_curvature_profile(signal, seams, window_size=10, model='l2')

    # Check structure
    assert 'seam_types' in profile
    assert 'distribution' in profile
    assert 'sharpness_score' in profile

    # Should have 3 seam types
    assert len(profile['seam_types']) == 3

    # All should be classified as C (cusps)
    for tau, stype, curv in profile['seam_types']:
        assert stype == 'C', f"Expected C for hard steps, got {stype}"


def test_adaptive_threshold():
    """Test adaptive threshold classification."""
    # Noisy signal with step
    signal = np.concatenate([
        np.ones(100) * 0,
        np.ones(100) * 3
    ])
    signal += np.random.normal(0, 0.5, len(signal))  # High noise

    adaptive_type, curvature, metrics = adaptive_threshold_classification(
        signal, 100, window_size=10, model='l2'
    )

    # Check metrics
    assert 'normalized_curvature' in metrics
    assert 'signal_std' in metrics
    assert 'threshold_low' in metrics
    assert 'threshold_high' in metrics

    # Should still detect cusp despite noise
    assert adaptive_type in ['C', 'T'], f"Expected C or T, got {adaptive_type}"


def test_recommend_expensive_ops():
    """Test battery-aware expensive op recommendations."""
    seam_types = ['C', 'C', 'T', 'T', 'S', 'S']
    curvatures = [5.0, 4.0, 0.5, 0.3, 0.01, 0.005]

    # Low budget: only C seams
    indices_low = recommend_expensive_ops(seam_types, curvatures, battery_budget='low')
    assert set(indices_low) == {0, 1}, f"Expected [0, 1], got {indices_low}"

    # Medium budget: C + high-T
    indices_med = recommend_expensive_ops(seam_types, curvatures, battery_budget='medium')
    assert 0 in indices_med and 1 in indices_med  # C seams
    assert 2 in indices_med  # High-curvature T seam (0.5 > 0.3)

    # High budget: all non-S
    indices_high = recommend_expensive_ops(seam_types, curvatures, battery_budget='high')
    assert len(indices_high) == 4  # C, C, T, T


def test_edge_case_short_signal():
    """Test behavior on short signal (edge case)."""
    # Signal too short for proper curvature calculation
    signal = np.random.randn(20)

    seam_type, curvature = classify_seam_type(signal, 10, window_size=10, model='l2')

    # Should return Unknown
    assert seam_type == 'Unknown', f"Expected Unknown for short signal, got {seam_type}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
