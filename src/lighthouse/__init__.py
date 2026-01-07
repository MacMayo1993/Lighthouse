"""
PELT-Lighthouse-Antipodes Pipeline

A computationally efficient pipeline for seam detection in time-series signals.

Main Components:
- PELT: Sparse changepoint detection (O(T) average)
- Lighthouse: Local refinement via entropy/novelty peaks
- Antipodes: Symmetry extraction (value & time involutions)
- Seam Types: S/T/C classification via cost-curvature

Usage:
    from lighthouse import run_pipeline
    results = run_pipeline(signal, penalty=10.0)
"""

__version__ = '0.1.0'

# Main pipeline
from .pipeline import run_pipeline, run_pipeline_streaming, analyze_seam_distribution

# Individual components
from .pelt import detect_seams, detect_seams_with_mdl
from .lighthouse import refine_seam, filter_weak_seams
from .antipodes import fit_antipode, detect_antipodal_pairs
from .seam_types import classify_seam_type, recommend_expensive_ops

__all__ = [
    'run_pipeline',
    'run_pipeline_streaming',
    'analyze_seam_distribution',
    'detect_seams',
    'detect_seams_with_mdl',
    'refine_seam',
    'filter_weak_seams',
    'fit_antipode',
    'detect_antipodal_pairs',
    'classify_seam_type',
    'recommend_expensive_ops',
]
