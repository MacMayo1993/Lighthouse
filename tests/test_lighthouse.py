"""
Tests for Lighthouse local refinement.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from lighthouse import (
    refine_seam,
    refine_seam_multi_scale,
    compute_seam_sharpness,
    filter_weak_seams
)


def test_refine_seam_basic():
    """Test basic seam refinement."""
    # Create signal with seam at 100, but PELT detects at 95
    signal = np.concatenate([
        np.ones(100) * 0,
        np.ones(100) * 5
    ])
    signal += np.random.normal(0, 0.1, len(signal))

    # Lighthouse should refine to near 100
    tau_refined, confidence = refine_seam(signal, 95, window_size=10, method='delta_cost')

    # Should refine to within ±5 samples
    assert abs(tau_refined - 100) <= 5, f"Expected ~100, got {tau_refined}"
    # Confidence should be > 1.0 (peak is sharper than mean)
    assert confidence > 1.0, f"Expected confidence > 1.0, got {confidence}"


def test_refine_seam_methods():
    """Test different refinement methods."""
    signal = np.concatenate([
        np.ones(100) * 0,
        np.ones(100) * 5
    ])
    signal += np.random.normal(0, 0.1, len(signal))

    methods = ['delta_cost', 'entropy', 'kl_divergence']

    for method in methods:
        tau_refined, confidence = refine_seam(signal, 95, window_size=10, method=method)

        # All methods should find seam near 100
        assert 90 <= tau_refined <= 110, f"Method {method} failed: got {tau_refined}"
        # Confidence should be positive
        assert confidence > 0, f"Method {method}: confidence {confidence} should be > 0"


def test_multi_scale_refinement():
    """Test multi-scale refinement."""
    signal = np.concatenate([
        np.ones(100) * 0,
        np.ones(100) * 5
    ])
    signal += np.random.normal(0, 0.1, len(signal))

    tau_refined, confidence = refine_seam_multi_scale(
        signal, 95, scales=[5, 10, 20]
    )

    # Consensus should be near 100
    assert abs(tau_refined - 100) <= 5, f"Expected ~100, got {tau_refined}"


def test_compute_sharpness():
    """Test sharpness computation."""
    # Sharp transition
    signal_sharp = np.concatenate([
        np.ones(100) * 0,
        np.ones(100) * 10  # Large jump
    ])
    signal_sharp += np.random.normal(0, 0.1, len(signal_sharp))

    sharpness_sharp = compute_seam_sharpness(signal_sharp, 100, window_size=10)

    # Gradual transition
    t = np.linspace(-6, 6, 200)
    signal_gradual = 1 / (1 + np.exp(-t))
    signal_gradual += np.random.normal(0, 0.1, len(signal_gradual))

    sharpness_gradual = compute_seam_sharpness(signal_gradual, 100, window_size=10)

    # Sharp should have higher sharpness
    assert sharpness_sharp > sharpness_gradual, \
        f"Sharp ({sharpness_sharp}) should exceed gradual ({sharpness_gradual})"


def test_filter_weak_seams():
    """Test filtering of weak seams."""
    # Signal with one strong seam (100) and one weak seam (50)
    signal = np.concatenate([
        np.ones(50) * 0,
        np.ones(50) * 0.5,  # Weak transition
        np.ones(100) * 5    # Strong transition
    ])
    signal += np.random.normal(0, 0.1, len(signal))

    seams = [50, 100]

    # Filter with sharpness method
    filtered = filter_weak_seams(signal, seams, min_confidence=2.0, method='sharpness')

    # Should keep strong seam (100), possibly filter weak (50)
    assert 100 in filtered, "Strong seam should be retained"
    # Weak seam might be filtered depending on threshold
    assert len(filtered) <= 2


def test_edge_case_boundary_seam():
    """Test refinement near signal boundaries."""
    signal = np.concatenate([
        np.ones(100) * 0,
        np.ones(100) * 5
    ])
    signal += np.random.normal(0, 0.1, len(signal))

    # Try to refine seam near start (should handle gracefully)
    tau_refined, confidence = refine_seam(signal, 5, window_size=10, method='delta_cost')

    # Should return something reasonable (not crash)
    assert 0 <= tau_refined < len(signal)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
