"""
Lighthouse: Local refinement around PELT candidates via entropy/novelty.

The "lighthouse" sweeps a narrow beam around each PELT candidate to find
the true high-information boundary within a small window.
"""

import numpy as np
from typing import Tuple, Optional
from scipy.stats import entropy as scipy_entropy
from .utils import compute_delta_cost, safe_correlation


def refine_seam(signal: np.ndarray,
                seam_candidate: int,
                window_size: int = 10,
                method: str = 'delta_cost') -> Tuple[int, float]:
    """
    Lighthouse refinement: Find true seam within window of PELT candidate.

    Args:
        signal: Full input signal
        seam_candidate: PELT-detected τ
        window_size: Delta for search window [τ - δ, τ + δ]
        method: Refinement metric ('delta_cost', 'entropy', 'kl_divergence')

    Returns:
        (t_star, confidence):
            - t_star: Refined seam position
            - confidence: Peak sharpness / signal-to-noise ratio

    Notes:
        ΔC(t) is defined as the cost reduction from splitting at position t:
            ΔC(t) = C([start:t]) + C([t:end]) - C([start:end])
        where C is the segment cost (e.g., sum of squared deviations for 'l2').
        Negative ΔC indicates a good split (cost reduction).

        Methods:
        - delta_cost: Find argmax |ΔC(t)| in window (steepest cost drop)
        - entropy: Local Shannon entropy peak with adaptive binning
        - kl_divergence: Max KL divergence between pre/post distributions

        Confidence is ratio of peak to mean, indicating "obviousness" of seam.
    """
    T = len(signal)

    # Define search window
    start = max(1, seam_candidate - window_size)  # Need at least 1 point on each side
    end = min(T - 1, seam_candidate + window_size)

    if start >= end - 1:
        return seam_candidate, 0.0

    # Compute novelty profile H(t) in window
    novelty = []
    positions = range(start, end)

    for t in positions:
        if method == 'delta_cost':
            # Cost reduction from split at t
            delta_C = compute_delta_cost(signal, t, window=(start - 5, end + 5))
            novelty.append(abs(delta_C))

        elif method == 'entropy':
            # Local Shannon entropy in sliding window
            left_window = signal[max(0, t - 5):t]
            right_window = signal[t:min(T, t + 5)]

            if len(left_window) > 0 and len(right_window) > 0:
                # Adaptive binning: sqrt(n) rule (Sturges/Scott compromise)
                max_window = max(len(left_window), len(right_window))
                bins = max(3, int(np.sqrt(max_window)))  # At least 3 bins

                hist_left, _ = np.histogram(left_window, bins=bins, density=True)
                hist_right, _ = np.histogram(right_window, bins=bins, density=True)

                ent = scipy_entropy(hist_left + 1e-10) + scipy_entropy(hist_right + 1e-10)
                novelty.append(ent)
            else:
                novelty.append(0.0)

        elif method == 'kl_divergence':
            # KL divergence between pre/post distributions
            left_window = signal[max(0, t - 10):t]
            right_window = signal[t:min(T, t + 10)]

            if len(left_window) > 2 and len(right_window) > 2:
                kl = _kl_divergence_gaussian(left_window, right_window)
                novelty.append(kl)
            else:
                novelty.append(0.0)

        else:
            raise ValueError(f"Unknown method: {method}")

    novelty = np.array(novelty)

    if len(novelty) == 0 or np.all(novelty == 0):
        return seam_candidate, 0.0

    # Find peak novelty
    peak_idx = np.argmax(novelty)
    t_star = positions[peak_idx]

    # Confidence: peak vs. mean (higher = sharper, more obvious seam)
    mean_novelty = np.mean(novelty)
    peak_novelty = novelty[peak_idx]

    confidence = (peak_novelty / (mean_novelty + 1e-10)) if mean_novelty > 0 else 0.0

    return t_star, confidence


def refine_seam_multi_scale(signal: np.ndarray,
                            seam_candidate: int,
                            scales: list = [5, 10, 20]) -> Tuple[int, float]:
    """
    Multi-scale lighthouse: Refine at multiple window sizes and vote.

    Args:
        signal: Input signal
        seam_candidate: PELT candidate
        scales: List of window sizes to try

    Returns:
        (t_star, confidence): Consensus refined seam and confidence

    Notes:
        - Coarse scales (large windows) catch global structure
        - Fine scales (small windows) catch sharp transitions
        - Voting reduces sensitivity to noise
    """
    candidates = []

    for window_size in scales:
        t_star, conf = refine_seam(signal, seam_candidate, window_size, method='delta_cost')
        candidates.append((t_star, conf))

    # Weight by confidence
    weighted_positions = [t * conf for t, conf in candidates]
    total_conf = sum(conf for _, conf in candidates)

    if total_conf > 0:
        t_star = int(round(sum(weighted_positions) / total_conf))
        avg_conf = total_conf / len(scales)
    else:
        t_star = seam_candidate
        avg_conf = 0.0

    return t_star, avg_conf


def compute_seam_sharpness(signal: np.ndarray,
                           seam: int,
                           window_size: int = 10) -> float:
    """
    Measure sharpness of transition at seam (for filtering weak candidates).

    Args:
        signal: Input signal
        seam: Seam position
        window_size: Window for comparison

    Returns:
        Sharpness score (higher = sharper transition)

    Notes:
        Sharpness = |mean(post) - mean(pre)| / std(window)
        - Sharp step: high sharpness
        - Gradual drift: low sharpness
        Use this to filter out "boring" seams that PELT flagged spuriously.
    """
    start = max(0, seam - window_size)
    end = min(len(signal), seam + window_size)

    if seam - start < 2 or end - seam < 2:
        return 0.0

    pre = signal[start:seam]
    post = signal[seam:end]
    window = signal[start:end]

    mean_diff = abs(np.mean(post) - np.mean(pre))
    std_window = np.std(window)

    if std_window < 1e-10:
        return 0.0

    return mean_diff / std_window


def _kl_divergence_gaussian(x: np.ndarray, y: np.ndarray) -> float:
    """
    KL divergence between two segments under Gaussian assumption.

    Args:
        x, y: Signal segments

    Returns:
        KL(P_x || P_y) where P_x = N(μ_x, σ_x^2)

    Notes:
        Closed form: KL = log(σ_y/σ_x) + (σ_x^2 + (μ_x - μ_y)^2)/(2σ_y^2) - 1/2
    """
    mu_x, mu_y = np.mean(x), np.mean(y)
    sig_x, sig_y = np.std(x) + 1e-10, np.std(y) + 1e-10

    kl = np.log(sig_y / sig_x) + (sig_x**2 + (mu_x - mu_y)**2) / (2 * sig_y**2) - 0.5

    return max(kl, 0.0)  # KL is non-negative


def filter_weak_seams(signal: np.ndarray,
                      seams: list,
                      min_confidence: float = 1.5,
                      method: str = 'sharpness') -> list:
    """
    Filter out weak or spurious seam candidates.

    Args:
        signal: Input signal
        seams: List of seam positions
        min_confidence: Minimum confidence threshold
        method: 'sharpness', 'delta_cost', or 'both'

    Returns:
        Filtered list of high-confidence seams

    Notes:
        Reduces false positives from PELT over-segmentation.
        Sharpness filters gradual drifts; ΔC filters low-MDL splits.
    """
    filtered = []

    for seam in seams:
        if method == 'sharpness' or method == 'both':
            sharpness = compute_seam_sharpness(signal, seam)
            if sharpness < min_confidence:
                continue

        if method == 'delta_cost' or method == 'both':
            delta_C = abs(compute_delta_cost(signal, seam))
            # Normalize by segment length
            seg_len = min(seam, len(signal) - seam)
            normalized_delta = delta_C / (seg_len + 1e-10)

            if normalized_delta < min_confidence:
                continue

        filtered.append(seam)

    return filtered
