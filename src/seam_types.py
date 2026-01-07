"""
Seam Type Classification via Cost Curvature.

Classifies seams into S (smooth), T (tangent), C (cusp) via second derivative
of cost function ΔC(τ). Stays entirely within MDL framework without extra machinery.
"""

import numpy as np
from typing import Tuple, Dict
from utils import compute_delta_cost, compute_cost


def classify_seam_type(signal: np.ndarray,
                       tau: int,
                       window_size: int = 10,
                       model: str = 'l2',
                       stencil_size: int = 5) -> Tuple[str, float]:
    """
    Classify seam type via cost curvature: S/T/C taxonomy.

    Args:
        signal: Input signal
        tau: Seam position
        window_size: Window for cost computation [tau-w, tau+w]
        model: Cost model ('l2', 'l1', 'ar1')
        stencil_size: Step size for finite difference (default 5 samples)

    Returns:
        (seam_type, curvature):
            - seam_type: 'S' (smooth), 'T' (tangent), 'C' (cusp)
            - curvature: κ = d²(ΔC)/dτ² (second derivative)

    Notes:
        Intuition:
        - S (smooth): ΔC changes gradually, κ ≈ 0 (e.g., slow drift)
        - T (tangent): ΔC has kink, 0 < |κ| < ∞ (e.g., sigmoid transition)
        - C (cusp): ΔC has discontinuity, |κ| → ∞ (e.g., hard step)

        Method:
        1. Compute ΔC at three points: τ-h, τ, τ+h (h = stencil_size)
        2. Second derivative: κ ≈ [ΔC(τ-h) - 2ΔC(τ) + ΔC(τ+h)] / h²
        3. Classify via thresholds on |κ|
    """
    T = len(signal)
    h = stencil_size

    # Compute ΔC at three points using finite difference stencil
    taus = [tau - h, tau, tau + h]
    delta_Cs = []

    for t in taus:
        # Ensure t is valid
        if t < window_size or t > T - window_size:
            # Out of bounds - can't compute curvature reliably
            return 'Unknown', 0.0

        # Compute ΔC(t) in window around t
        window_start = max(0, t - window_size)
        window_end = min(T, t + window_size)

        delta_C = compute_delta_cost(signal, t, model=model, window=(window_start, window_end))
        delta_Cs.append(delta_C)

    if len(delta_Cs) < 3:
        return 'Unknown', 0.0

    # Finite difference approximation of second derivative
    # d²(ΔC)/dτ² ≈ [ΔC(τ-h) - 2ΔC(τ) + ΔC(τ+h)] / h²
    curvature = (delta_Cs[0] - 2 * delta_Cs[1] + delta_Cs[2]) / (h ** 2)

    # Classify via curvature magnitude
    # Thresholds tuned for synthetic validation:
    # - S: gradual drifts (Gaussian bumps, slow trends)
    # - T: sigmoid transitions (continuous derivative, kinked second derivative)
    # - C: hard steps (discontinuous derivative)
    abs_curv = abs(curvature)

    if abs_curv < 0.02:
        # Low curvature: smooth transition (gradual drift)
        return 'S', curvature

    elif abs_curv < 2.0:
        # Moderate curvature: tangent transition (sigmoid-like)
        return 'T', curvature

    else:
        # High curvature: cusp transition (hard step)
        return 'C', curvature


def compute_curvature_profile(signal: np.ndarray,
                              seams: list,
                              window_size: int = 10,
                              stencil_size: int = 5,
                              model: str = 'l2') -> Dict:
    """
    Compute full curvature profile for all seams.

    Args:
        signal: Input signal
        seams: List of seam positions
        window_size: Window for ΔC computation
        stencil_size: Finite difference step
        model: Cost model

    Returns:
        Dictionary with:
            - seam_types: List of (tau, type, curvature) tuples
            - distribution: Count of S/T/C seams
            - sharpness_score: Overall sharpness (mean |κ|)

    Notes:
        Use this for aggregate analysis:
        - High S count: Signal has mostly gradual changes (e.g., drift)
        - High C count: Many abrupt transitions (e.g., arrhythmias)
        - High T count: Mix of smooth and sharp (typical)
    """
    seam_types = []
    type_counts = {'S': 0, 'T': 0, 'C': 0, 'Unknown': 0}
    curvatures = []

    for tau in seams:
        seam_type, curvature = classify_seam_type(
            signal, tau, window_size=window_size, stencil_size=stencil_size, model=model
        )

        seam_types.append((tau, seam_type, curvature))
        type_counts[seam_type] += 1

        if seam_type != 'Unknown':
            curvatures.append(abs(curvature))

    sharpness_score = np.mean(curvatures) if curvatures else 0.0

    return {
        'seam_types': seam_types,
        'distribution': type_counts,
        'sharpness_score': sharpness_score
    }


def adaptive_threshold_classification(signal: np.ndarray,
                                     tau: int,
                                     window_size: int = 10,
                                     model: str = 'l2') -> Tuple[str, float, Dict]:
    """
    Data-adaptive seam classification using local signal statistics.

    Args:
        signal: Input signal
        tau: Seam position
        window_size: Window for analysis
        model: Cost model

    Returns:
        (seam_type, curvature, metrics):
            - seam_type: S/T/C classification
            - curvature: Raw curvature value
            - metrics: Additional diagnostic info

    Notes:
        Instead of fixed thresholds (0.01, 0.5), adapt to local noise level:
        - Normalize κ by signal variance in window
        - Adjust thresholds based on SNR
        More robust for varying signal amplitudes (e.g., ECG lead differences).
    """
    seam_type, curvature = classify_seam_type(
        signal, tau, window_size=window_size, model=model
    )

    # Compute local statistics for normalization
    window_start = max(0, tau - window_size * 2)
    window_end = min(len(signal), tau + window_size * 2)
    window = signal[window_start:window_end]

    signal_std = np.std(window)
    signal_range = np.ptp(window)  # Peak-to-peak

    # Normalize curvature by signal amplitude
    if signal_std > 1e-10:
        normalized_curvature = abs(curvature) / signal_std
    else:
        normalized_curvature = abs(curvature)

    # Adaptive thresholds (scaled by local variability)
    base_threshold_low = 0.01
    base_threshold_high = 0.5

    # If signal is noisy (high std relative to range), be more conservative
    noise_factor = signal_std / (signal_range + 1e-10)
    threshold_low = base_threshold_low * (1 + noise_factor)
    threshold_high = base_threshold_high * (1 + noise_factor)

    # Reclassify with adaptive thresholds
    if normalized_curvature < threshold_low:
        adaptive_type = 'S'
    elif normalized_curvature < threshold_high:
        adaptive_type = 'T'
    else:
        adaptive_type = 'C'

    metrics = {
        'raw_curvature': curvature,
        'normalized_curvature': normalized_curvature,
        'signal_std': signal_std,
        'signal_range': signal_range,
        'noise_factor': noise_factor,
        'threshold_low': threshold_low,
        'threshold_high': threshold_high,
        'fixed_type': seam_type,
        'adaptive_type': adaptive_type
    }

    return adaptive_type, curvature, metrics


def plot_cost_profile(signal: np.ndarray,
                     tau: int,
                     window_size: int = 20,
                     model: str = 'l2') -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute ΔC(t) profile around seam for visualization.

    Args:
        signal: Input signal
        tau: Seam position
        window_size: Window for scan
        model: Cost model

    Returns:
        (positions, delta_costs): Arrays for plotting ΔC vs. position

    Notes:
        Useful for visualizing seam sharpness:
        - S seams: ΔC changes gradually (parabolic)
        - T seams: ΔC has kink (V-shaped)
        - C seams: ΔC has sharp corner (discontinuous derivative)
    """
    start = max(1, tau - window_size)
    end = min(len(signal) - 1, tau + window_size)

    positions = list(range(start, end))
    delta_costs = []

    for t in positions:
        delta_C = compute_delta_cost(signal, t, model=model)
        delta_costs.append(delta_C)

    return np.array(positions), np.array(delta_costs)


def recommend_expensive_ops(seam_types: list,
                           curvatures: list,
                           battery_budget: str = 'medium') -> list:
    """
    Recommend which seams warrant expensive operations (e.g., deep model inference).

    Args:
        seam_types: List of seam type strings ('S', 'T', 'C')
        curvatures: List of curvature values (aligned with seam_types)
        battery_budget: 'low', 'medium', 'high'

    Returns:
        List of indices (into seam_types) to process with expensive ops

    Notes:
        Battery-aware triggering:
        - Low budget: Only C seams (most critical, e.g., arrhythmia onset)
        - Medium budget: C + high-curvature T seams
        - High budget: All seams (or use for validation)

        This is the "Joules-rent" payoff: spend compute where info is highest.
    """
    indices = []

    for i, (stype, curv) in enumerate(zip(seam_types, curvatures)):
        if battery_budget == 'low':
            # Only cusps (hard transitions)
            if stype == 'C':
                indices.append(i)

        elif battery_budget == 'medium':
            # Cusps + sharp tangents
            if stype == 'C' or (stype == 'T' and abs(curv) > 0.3):
                indices.append(i)

        elif battery_budget == 'high':
            # All seams except very smooth
            if stype != 'S':
                indices.append(i)

        else:
            raise ValueError(f"Unknown battery_budget: {battery_budget}")

    return indices
