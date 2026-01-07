"""
Antipodes: Antipodal pairing and symmetry extraction across seams.

Maps involutions (value reflections, time lags) between pre/post segments
to compress structure via symmetry atoms.
"""

import numpy as np
from typing import Dict, Tuple, Optional
from .utils import safe_correlation


def fit_antipode(pre_segment: np.ndarray,
                post_segment: np.ndarray,
                mode: str = 'value',
                max_lag: Optional[int] = None) -> Dict:
    """
    Fit antipodal pairing between segments across a seam.

    Args:
        pre_segment: Signal before seam
        post_segment: Signal after seam
        mode: 'value' for reflection, 'time' for lag/phase, 'both' for combined
        max_lag: Maximum lag to search (default: min(10, len//4))

    Returns:
        Dictionary with:
            - corr: Best correlation coefficient
            - type: 'direct', 'reflect', 'lag'
            - param: Reflection axis (float) or lag (int)
            - mdl_savings: Estimated compression gain (bits saved)

    Notes:
        Value antipodes (reflection): post ≈ -pre + 2*axis
            where axis = (mean(pre) + mean(post)) / 2 is the reflection midpoint.
            Detects polarity flips in ECG, phase inversions in periodic signals.

        Time antipodes (lag): post(t) ≈ pre(t - δ)
            where δ is the optimal lag maximizing correlation.
            Detects phase shifts, cyclic patterns, delayed repetitions.

        Combined mode: Evaluates both transformations and returns the best fit.

        MDL savings uses Shannon limit proxy:
            savings ≈ L * 0.5 * log2(1 + r²/(1-r²)) - involution_cost
            where L is segment length, r is correlation coefficient.
    """
    pre = np.asarray(pre_segment)
    post = np.asarray(post_segment)

    # Truncate to common length for correlation
    min_len = min(len(pre), len(post))
    if min_len < 3:
        return {'corr': 0.0, 'type': 'none', 'param': None, 'mdl_savings': 0.0}

    pre = pre[-min_len:]
    post = post[:min_len]

    results = []

    if mode in ['value', 'both']:
        # Value-space antipode: direct vs. reflected
        corr_direct = safe_correlation(pre, post)
        corr_reflect = safe_correlation(pre, -post)

        if abs(corr_reflect) > abs(corr_direct):
            # Fit reflection axis: post ≈ -pre + 2*axis
            axis = (np.mean(pre) + np.mean(post)) / 2
            mdl_savings = _estimate_mdl_savings(abs(corr_reflect), len(pre), symmetry='reflect')

            results.append({
                'corr': corr_reflect,
                'type': 'reflect',
                'param': axis,
                'mdl_savings': mdl_savings
            })
        else:
            mdl_savings = _estimate_mdl_savings(abs(corr_direct), len(pre), symmetry='direct')
            results.append({
                'corr': corr_direct,
                'type': 'direct',
                'param': None,
                'mdl_savings': mdl_savings
            })

    if mode in ['time', 'both']:
        # Time-space antipode: find best lag
        if max_lag is None:
            max_lag = min(10, min_len // 4)

        max_lag = min(max_lag, min_len - 1)

        if max_lag > 0:
            corrs = []
            for lag in range(-max_lag, max_lag + 1):
                if lag == 0:
                    corr = safe_correlation(pre, post)
                elif lag > 0:
                    # post is ahead: post[0:] ≈ pre[lag:]
                    corr = safe_correlation(pre[lag:], post[:len(pre) - lag])
                else:  # lag < 0
                    # pre is ahead: pre[0:] ≈ post[-lag:]
                    corr = safe_correlation(pre[:len(pre) + lag], post[-lag:])

                corrs.append((abs(corr), lag, corr))

            best_abs_corr, best_lag, best_corr = max(corrs, key=lambda x: x[0])

            mdl_savings = _estimate_mdl_savings(best_abs_corr, len(pre), symmetry='lag')

            results.append({
                'corr': best_corr,
                'type': 'lag',
                'param': best_lag,
                'mdl_savings': mdl_savings
            })

    if not results:
        return {'corr': 0.0, 'type': 'none', 'param': None, 'mdl_savings': 0.0}

    # Return best by absolute correlation
    return max(results, key=lambda x: abs(x['corr']))


def fit_householder_reflection(pre_segment: np.ndarray,
                                post_segment: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Fit Householder reflection: post ≈ H(u) @ pre where H = I - 2uu^T.

    Args:
        pre_segment: Pre-seam signal
        post_segment: Post-seam signal (must have same length)

    Returns:
        (u, residual):
            - u: Unit vector defining reflection hyperplane
            - residual: Reconstruction error

    Notes:
        For 1D signals, this reduces to scalar reflection: post ≈ -pre + c.
        For multi-dimensional signals (e.g., IMU xyz), fits reflection plane.
    """
    pre = np.asarray(pre_segment)
    post = np.asarray(post_segment)

    if pre.ndim == 1:
        # 1D: Simple reflection through axis
        # post ≈ -pre + 2*axis
        axis = (np.mean(pre) + np.mean(post)) / 2
        reflected = -pre + 2 * axis
        residual = np.mean((post - reflected)**2)

        # Return "direction" as sign
        u = np.array([1.0])  # Reflection axis direction
        return u, residual

    else:
        # Multi-D: Fit Householder reflection
        # Solve for u: post ≈ (I - 2uu^T) @ pre
        # This is the vector from pre to post, normalized

        if len(pre) != len(post):
            raise ValueError("Segments must have same length for Householder fit")

        # Center both
        pre_mean = np.mean(pre, axis=0)
        post_mean = np.mean(post, axis=0)

        pre_centered = pre - pre_mean
        post_centered = post - post_mean

        # Best fit u: direction of (pre + post)
        u = pre_mean + post_mean
        u = u / (np.linalg.norm(u) + 1e-10)

        # Apply reflection
        H = np.eye(len(u)) - 2 * np.outer(u, u)
        reflected = pre_centered @ H.T + post_mean

        residual = np.mean((post - reflected)**2)

        return u, residual


def detect_antipodal_pairs(signal: np.ndarray,
                           seams: list,
                           min_corr: float = 0.6) -> list:
    """
    Detect all antipodal pairs across seams in signal.

    Args:
        signal: Full signal
        seams: List of seam positions
        min_corr: Minimum correlation to qualify as antipodal

    Returns:
        List of dicts with seam indices and antipodal info

    Notes:
        Identifies "flip atoms" (involutions) that can be compressed:
        - Strong reflections (polarity flips in ECG)
        - Phase lags (periodic signals)
        - Symmetries that reduce effective dimensionality
    """
    if not seams:
        return []

    # Segments between seams
    seam_points = [0] + sorted(seams) + [len(signal)]
    segments = [signal[seam_points[i]:seam_points[i + 1]]
                for i in range(len(seam_points) - 1)]

    pairs = []

    for i in range(len(segments) - 1):
        pre = segments[i]
        post = segments[i + 1]

        # Try both value and time antipodes
        antipode = fit_antipode(pre, post, mode='both')

        if abs(antipode['corr']) >= min_corr:
            pairs.append({
                'seam_idx': i,
                'seam_position': seams[i] if i < len(seams) else len(signal),
                'antipode': antipode
            })

    return pairs


def compute_symmetry_compression(signal: np.ndarray,
                                 seams: list,
                                 antipodal_pairs: list) -> Dict:
    """
    Estimate MDL savings from exploiting antipodal symmetries.

    Args:
        signal: Full signal
        seams: Seam positions
        antipodal_pairs: Output from detect_antipodal_pairs

    Returns:
        Dictionary with compression statistics:
            - naive_cost: Cost of encoding all segments independently
            - symmetry_cost: Cost with antipodal atoms
            - savings_ratio: (naive - symmetry) / naive

    Notes:
        Naive: encode each segment independently
        Symmetry: encode one segment + involution (reflection/lag)
        For strong antipodes, this can reduce storage by ~50%
    """
    # Naive cost: sum of segment variances (proxy for encoding cost)
    seam_points = [0] + sorted(seams) + [len(signal)]
    segments = [signal[seam_points[i]:seam_points[i + 1]]
                for i in range(len(seam_points) - 1)]

    naive_cost = sum(np.var(seg) * len(seg) for seg in segments if len(seg) > 0)

    # Symmetry cost: for antipodal pairs, encode only one + involution
    symmetry_cost = 0.0
    used_segments = set()

    for pair in antipodal_pairs:
        i = pair['seam_idx']
        if i in used_segments or i + 1 in used_segments:
            continue

        corr = abs(pair['antipode']['corr'])

        # Cost = encode first segment + encode involution + reconstruction error
        seg_i_cost = np.var(segments[i]) * len(segments[i])
        seg_ip1_cost = np.var(segments[i + 1]) * len(segments[i + 1])

        # With antipode: encode first + involution (cheap) + error
        involution_cost = 10  # Cost to encode reflection axis or lag (bits)
        reconstruction_error = (1 - corr**2) * seg_ip1_cost  # Unexplained variance

        pair_cost = seg_i_cost + involution_cost + reconstruction_error

        symmetry_cost += pair_cost
        used_segments.add(i)
        used_segments.add(i + 1)

    # Add non-antipodal segments at full cost
    for i, seg in enumerate(segments):
        if i not in used_segments and len(seg) > 0:
            symmetry_cost += np.var(seg) * len(seg)

    savings_ratio = (naive_cost - symmetry_cost) / (naive_cost + 1e-10)

    return {
        'naive_cost': naive_cost,
        'symmetry_cost': symmetry_cost,
        'savings_ratio': savings_ratio,
        'num_antipodes': len(antipodal_pairs)
    }


def _estimate_mdl_savings(corr: float, segment_length: int, symmetry: str) -> float:
    """
    Estimate MDL savings from single antipodal pair.

    Args:
        corr: Absolute correlation coefficient
        segment_length: Length of segments
        symmetry: 'direct', 'reflect', or 'lag'

    Returns:
        Bits saved by encoding as antipodal pair vs. independent

    Notes:
        Naive: encode both segments independently (2 * L * bits_per_sample)
        Antipodal: encode first + involution + residual
        Savings ≈ L * log2(1 + SNR) where SNR ∝ corr^2 / (1 - corr^2)
    """
    if corr < 0.1:
        return 0.0

    # Explained variance ratio
    r_squared = corr**2

    # SNR = explained / unexplained
    snr = r_squared / (1 - r_squared + 1e-10)

    # Bits saved per sample (Shannon limit)
    bits_per_sample = 0.5 * np.log2(1 + snr)

    # Total savings
    total_savings = segment_length * bits_per_sample

    # Cost of encoding involution
    involution_cost = {
        'direct': 1,      # No encoding needed (identity)
        'reflect': 16,    # Encode reflection axis (float)
        'lag': 8          # Encode lag (int)
    }.get(symmetry, 10)

    net_savings = max(0, total_savings - involution_cost)

    return net_savings
