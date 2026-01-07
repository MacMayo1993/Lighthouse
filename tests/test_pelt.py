"""
Tests for PELT changepoint detection.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from lighthouse.pelt import detect_seams, detect_seams_with_mdl


def test_detect_seams_simple():
    """Test PELT on simple two-segment signal."""
    # Create signal with clear changepoint at 100
    signal = np.concatenate([
        np.random.normal(0, 1, 100),
        np.random.normal(5, 1, 100)
    ])

    seams = detect_seams(signal, penalty=10.0, model='l2', min_size=5)

    # Should detect seam near 100
    assert len(seams) > 0, "Should detect at least one seam"
    assert any(85 < s < 115 for s in seams), "Should detect seam near 100"


def test_detect_seams_no_change():
    """Test PELT on constant signal (no changepoints)."""
    signal = np.random.normal(0, 0.1, 200)

    seams = detect_seams(signal, penalty=50.0, model='l2')

    # High penalty should prevent spurious detections
    assert len(seams) <= 1, "Constant signal should have few/no seams"


def test_detect_seams_multiple():
    """Test PELT on multi-segment signal."""
    # Three segments with different means
    signal = np.concatenate([
        np.random.normal(0, 1, 50),
        np.random.normal(5, 1, 50),
        np.random.normal(-3, 1, 50)
    ])

    seams = detect_seams(signal, penalty=5.0, model='l2', min_size=10)

    # Should detect 2 seams (at ~50 and ~100)
    assert len(seams) >= 2, "Should detect multiple seams"


def test_detect_seams_mdl():
    """Test MDL auto-tuning."""
    signal = np.concatenate([
        np.random.normal(0, 1, 100),
        np.random.normal(5, 1, 100)
    ])

    seams = detect_seams_with_mdl(signal, model='l2', min_size=5, k_max=10)

    # Should find optimal seams via MDL
    assert isinstance(seams, list), "Should return list of seams"


def test_cost_models():
    """Test different cost models."""
    signal = np.concatenate([
        np.random.normal(0, 1, 50),
        np.random.normal(3, 1, 50)
    ])

    for model in ['l2', 'l1']:
        seams = detect_seams(signal, penalty=10.0, model=model, min_size=5)
        assert isinstance(seams, list), f"Model {model} should return list"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
