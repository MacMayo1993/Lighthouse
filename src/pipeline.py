"""
Main Pipeline: PELT → Lighthouse → Antipodes → Seam Classification.

Integrates all components for end-to-end seam detection and analysis.
"""

import numpy as np
from typing import List, Dict, Optional
import pandas as pd

from pelt import detect_seams, detect_seams_with_mdl
from lighthouse import refine_seam, filter_weak_seams, compute_seam_sharpness
from antipodes import fit_antipode, detect_antipodal_pairs, compute_symmetry_compression
from seam_types import classify_seam_type, compute_curvature_profile, recommend_expensive_ops


def run_pipeline(signal: np.ndarray,
                penalty: float = 10.0,
                window_size: int = 10,
                model: str = 'l2',
                min_size: int = 2,
                jump: int = 1,
                auto_mdl: bool = False,
                filter_weak: bool = True,
                min_confidence: float = 1.5,
                compute_antipodes: bool = True,
                battery_budget: str = 'medium') -> Dict:
    """
    Full PELT-Lighthouse-Antipodes pipeline.

    Args:
        signal: Input time-series (1D array)
        penalty: PELT penalty (ignored if auto_mdl=True)
        window_size: Lighthouse refinement window
        model: Cost model ('l2', 'l1', 'ar1')
        min_size: Minimum segment size
        jump: PELT stride (higher = faster but coarser)
        auto_mdl: Auto-tune penalty via MDL (slower but principled)
        filter_weak: Filter spurious seams via lighthouse
        min_confidence: Minimum confidence for filtering
        compute_antipodes: Run antipodal pairing (expensive)
        battery_budget: 'low', 'medium', 'high' for expensive ops

    Returns:
        Dictionary with:
            - seams: List of refined seam positions
            - results: DataFrame with detailed per-seam analysis
            - summary: Aggregate statistics
            - compression: Symmetry compression analysis (if compute_antipodes=True)

    Notes:
        Typical workflow:
        1. PELT detects coarse seams (cheap, O(T))
        2. Lighthouse refines locally (moderate, O(k*w))
        3. Weak seams filtered (cheap)
        4. Seam types classified via curvature (moderate, O(k))
        5. Antipodes fitted (expensive, O(k*m)) where m = segment length
        6. Expensive ops triggered only on high-value seams (battery-aware)
    """
    T = len(signal)

    # Step 1: PELT detection
    if auto_mdl:
        seams_pelt = detect_seams_with_mdl(signal, model=model, min_size=min_size)
    else:
        seams_pelt = detect_seams(signal, penalty=penalty, model=model, min_size=min_size, jump=jump)

    if not seams_pelt:
        return {
            'seams': [],
            'results': pd.DataFrame(),
            'summary': {'num_seams': 0, 'pipeline_stage': 'pelt_empty'},
            'compression': None
        }

    # Step 2: Lighthouse refinement
    seams_refined = []
    confidences = []

    for tau in seams_pelt:
        tau_refined, confidence = refine_seam(signal, tau, window_size=window_size, method='delta_cost')
        seams_refined.append(tau_refined)
        confidences.append(confidence)

    # Step 3: Filter weak seams
    if filter_weak:
        # Combine PELT candidates with confidences for filtering
        seam_conf_pairs = list(zip(seams_refined, confidences))
        filtered_pairs = [(s, c) for s, c in seam_conf_pairs if c >= min_confidence]

        if filtered_pairs:
            seams_refined, confidences = zip(*filtered_pairs)
            seams_refined = list(seams_refined)
            confidences = list(confidences)
        else:
            seams_refined = []
            confidences = []

    if not seams_refined:
        return {
            'seams': [],
            'results': pd.DataFrame(),
            'summary': {'num_seams': 0, 'pipeline_stage': 'lighthouse_filtered_all'},
            'compression': None
        }

    # Step 4: Seam type classification via curvature
    curvature_profile = compute_curvature_profile(
        signal, seams_refined, window_size=window_size, model=model
    )

    seam_types = [st[1] for st in curvature_profile['seam_types']]  # Extract type strings
    curvatures = [st[2] for st in curvature_profile['seam_types']]  # Extract curvature values

    # Step 5: Antipodal pairing (optional, expensive)
    antipodal_pairs = None
    compression_stats = None

    if compute_antipodes:
        # Only compute for seams that pass battery budget
        expensive_indices = recommend_expensive_ops(seam_types, curvatures, battery_budget)

        # Full antipodal detection
        antipodal_pairs = detect_antipodal_pairs(signal, seams_refined, min_corr=0.6)

        # Compression analysis
        compression_stats = compute_symmetry_compression(signal, seams_refined, antipodal_pairs)

    # Step 6: Build results DataFrame
    results = []

    for i, tau_pelt in enumerate(seams_pelt):
        # Match with refined seam (might be filtered)
        if i < len(seams_refined):
            tau_refined = seams_refined[i]
            confidence = confidences[i]
            seam_type = seam_types[i]
            curvature = curvatures[i]
            offset = tau_refined - tau_pelt

            # Find antipode info if available
            antipode_info = None
            if antipodal_pairs:
                for pair in antipodal_pairs:
                    if pair['seam_idx'] == i:
                        antipode_info = pair['antipode']
                        break

            antipode_corr = antipode_info['corr'] if antipode_info else 0.0
            antipode_type = antipode_info['type'] if antipode_info else 'none'
            mdl_savings = antipode_info.get('mdl_savings', 0.0) if antipode_info else 0.0

            sharpness = compute_seam_sharpness(signal, tau_refined, window_size=window_size)

            results.append({
                'tau_pelt': tau_pelt,
                'tau_lighthouse': tau_refined,
                'offset': offset,
                'seam_type': seam_type,
                'curvature': curvature,
                'sharpness': sharpness,
                'confidence': confidence,
                'antipodal_corr': antipode_corr,
                'antipode_type': antipode_type,
                'mdl_savings': mdl_savings,
                'filtered': False
            })
        else:
            # Seam was filtered
            results.append({
                'tau_pelt': tau_pelt,
                'tau_lighthouse': tau_pelt,
                'offset': 0,
                'seam_type': 'Unknown',
                'curvature': 0.0,
                'sharpness': 0.0,
                'confidence': 0.0,
                'antipodal_corr': 0.0,
                'antipode_type': 'none',
                'mdl_savings': 0.0,
                'filtered': True
            })

    results_df = pd.DataFrame(results)

    # Summary statistics
    summary = {
        'num_seams_pelt': len(seams_pelt),
        'num_seams_refined': len(seams_refined),
        'num_seams_filtered': len(seams_pelt) - len(seams_refined),
        'seam_distribution': curvature_profile['distribution'],
        'mean_curvature': np.mean(curvatures) if curvatures else 0.0,
        'mean_confidence': np.mean(confidences) if confidences else 0.0,
        'signal_length': T,
        'model': model,
        'pipeline_stage': 'complete'
    }

    return {
        'seams': seams_refined,
        'results': results_df,
        'summary': summary,
        'compression': compression_stats
    }


def run_pipeline_streaming(signal_chunk: np.ndarray,
                          history: Optional[np.ndarray] = None,
                          penalty: float = 10.0,
                          window_size: int = 10,
                          model: str = 'l2') -> Dict:
    """
    Streaming variant for online processing (e.g., wearable devices).

    Args:
        signal_chunk: New signal chunk
        history: Previous signal context
        penalty: PELT penalty
        window_size: Lighthouse window
        model: Cost model

    Returns:
        Seams detected in current chunk only

    Notes:
        - Maintains small context from history
        - Returns only seams in new chunk (relative indices)
        - Suitable for real-time battery-constrained devices
    """
    from pelt import detect_seams_streaming

    # Detect seams in chunk with context
    seams_chunk = detect_seams_streaming(
        signal_chunk, history=history, penalty=penalty, model=model
    )

    if not seams_chunk:
        return {'seams': [], 'results': pd.DataFrame(), 'summary': {'num_seams': 0}}

    # Refine within chunk
    seams_refined = []
    for tau in seams_chunk:
        tau_refined, _ = refine_seam(signal_chunk, tau, window_size=window_size)
        seams_refined.append(tau_refined)

    return {'seams': seams_refined, 'summary': {'num_seams': len(seams_refined)}}


def analyze_seam_distribution(results_df: pd.DataFrame) -> Dict:
    """
    Analyze distribution of seam types for signal characterization.

    Args:
        results_df: Output from run_pipeline()['results']

    Returns:
        Analysis dictionary with:
            - Type percentages
            - Dominant patterns
            - Recommendations

    Notes:
        Use for interpretability:
        - High S%: Signal is mostly smooth (drift, slow changes)
        - High C%: Many abrupt events (arrhythmias, state switches)
        - High antipode%: Strong periodicities or symmetries
    """
    if results_df.empty:
        return {'error': 'No seams to analyze'}

    # Filter out filtered seams
    valid_df = results_df[results_df['filtered'] == False]

    if valid_df.empty:
        return {'error': 'All seams filtered'}

    total = len(valid_df)

    type_counts = valid_df['seam_type'].value_counts()
    type_percentages = (type_counts / total * 100).to_dict()

    # Antipodal analysis
    antipodal_df = valid_df[valid_df['antipodal_corr'].abs() > 0.6]
    antipodal_percentage = len(antipodal_df) / total * 100

    # Characterize signal
    dominant_type = type_counts.idxmax() if not type_counts.empty else 'Unknown'

    if dominant_type == 'S':
        characterization = 'Signal dominated by smooth, gradual changes (drift or slow trends).'
    elif dominant_type == 'C':
        characterization = 'Signal dominated by abrupt transitions (discrete events or state changes).'
    elif dominant_type == 'T':
        characterization = 'Signal has mixed smooth and sharp transitions (typical complex signal).'
    else:
        characterization = 'Signal characteristics unclear.'

    # Recommendations
    recommendations = []

    if type_percentages.get('C', 0) > 50:
        recommendations.append('High cusp rate suggests event-driven behavior; consider event detection models.')

    if antipodal_percentage > 30:
        recommendations.append('Strong antipodal symmetries detected; exploit for compression.')

    if valid_df['confidence'].mean() < 2.0:
        recommendations.append('Low average confidence; consider tighter PELT penalty or larger min_size.')

    return {
        'type_percentages': type_percentages,
        'dominant_type': dominant_type,
        'antipodal_percentage': antipodal_percentage,
        'characterization': characterization,
        'recommendations': recommendations,
        'mean_curvature': valid_df['curvature'].abs().mean(),
        'mean_confidence': valid_df['confidence'].mean(),
        'mean_sharpness': valid_df['sharpness'].mean()
    }
