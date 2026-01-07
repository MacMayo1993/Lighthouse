"""
Integration tests for full pipeline.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add src to path for package import
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from lighthouse.pipeline import run_pipeline


def test_pipeline_basic():
    """Test basic pipeline execution."""
    # Simple two-segment signal
    signal = np.concatenate([
        np.random.normal(0, 1, 100),
        np.random.normal(5, 1, 100)
    ])

    results = run_pipeline(
        signal,
        penalty=10.0,
        window_size=10,
        model='l2',
        compute_antipodes=False,  # Skip expensive ops for quick test
        filter_weak=True
    )

    # Check output structure
    assert 'seams' in results
    assert 'results' in results
    assert 'summary' in results

    # Check summary
    summary = results['summary']
    assert 'num_seams_pelt' in summary
    assert 'num_seams_refined' in summary


def test_pipeline_empty_signal():
    """Test pipeline on short signal (edge case)."""
    signal = np.random.randn(10)

    results = run_pipeline(signal, penalty=10.0)

    # Should handle gracefully
    assert 'seams' in results
    assert len(results['seams']) == 0  # Too short for meaningful seams


def test_pipeline_with_antipodes():
    """Test pipeline with antipodal pairing enabled."""
    # Create signal with polarity flip
    base = np.sin(np.linspace(0, 2*np.pi, 50))
    signal = np.concatenate([base, -base])

    results = run_pipeline(
        signal,
        penalty=5.0,
        compute_antipodes=True,
        battery_budget='high'
    )

    # Should detect seam and antipodal pair
    assert 'compression' in results
    if results['compression'] is not None:
        assert 'num_antipodes' in results['compression']


def test_pipeline_seam_types():
    """Test seam type classification."""
    # Signal with cusp (hard step)
    signal = np.concatenate([
        np.ones(50) * 0,
        np.ones(50) * 5
    ])

    results = run_pipeline(signal, penalty=5.0, window_size=5)

    # Check seam types in results
    if not results['results'].empty:
        assert 'seam_type' in results['results'].columns
        # Hard step should be classified as C (cusp)
        types = results['results']['seam_type'].values
        assert any(t in ['C', 'T'] for t in types), "Should detect sharp seam"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
