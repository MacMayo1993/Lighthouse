"""
Utility functions for cost computation and MDL penalties.
"""

import numpy as np
from typing import Union, Tuple


def compute_cost(segment: np.ndarray, model: str = 'l2') -> float:
    """
    Compute segment cost function C(·) under specified model.

    Args:
        segment: Input signal segment (1D array)
        model: Cost model ('l2' for Gaussian, 'l1' for Laplacian, 'ar1' for autoregressive)

    Returns:
        Segment cost (lower is better fit)

    Notes:
        - l2: Sum of squared deviations from mean (Gaussian MLE)
        - l1: Sum of absolute deviations from median (robust)
        - ar1: Residual variance under AR(1) model
    """
    if len(segment) == 0:
        return 0.0

    segment = np.asarray(segment)

    if model == 'l2':
        # Gaussian: sum((x - mean)^2)
        return np.sum((segment - np.mean(segment))**2)

    elif model == 'l1':
        # Laplacian: sum(|x - median|)
        return np.sum(np.abs(segment - np.median(segment)))

    elif model == 'ar1':
        # AR(1): fit phi and compute residual variance
        if len(segment) < 3:
            return compute_cost(segment, 'l2')  # Fallback

        x = segment[:-1]
        y = segment[1:]

        # Least squares: y = phi * x + epsilon
        phi = np.dot(x, y) / (np.dot(x, x) + 1e-10)
        residuals = y - phi * x
        return np.sum(residuals**2)

    else:
        raise ValueError(f"Unknown cost model: {model}")


def compute_delta_cost(signal: np.ndarray, tau: int,
                       model: str = 'l2',
                       window: Union[Tuple[int, int], None] = None) -> float:
    """
    Compute cost reduction ΔC(τ) from splitting at τ within a window.

    Args:
        signal: Full signal
        tau: Proposed split point
        model: Cost model
        window: (start, end) indices for window; if None, use full signal

    Returns:
        ΔC(τ) = C_split - C_unsplit (negative values indicate good splits)

    Notes:
        This is the key quantity for MDL analysis:
        - ΔC < 0: Split reduces description length (good seam)
        - ΔC ≈ 0: Marginal improvement (weak seam)
        - ΔC > 0: Split increases cost (spurious)
    """
    if window is None:
        start, end = 0, len(signal)
    else:
        start, end = window

    # Ensure tau is within bounds
    if tau <= start or tau >= end:
        return np.inf

    # Cost without split
    C_unsplit = compute_cost(signal[start:end], model)

    # Cost with split at tau
    C_left = compute_cost(signal[start:tau], model)
    C_right = compute_cost(signal[tau:end], model)
    C_split = C_left + C_right

    return C_split - C_unsplit


def mdl_penalty(k: int, T: int, d: int = 1) -> float:
    """
    Minimum Description Length penalty for k changepoints.

    Args:
        k: Number of changepoints
        T: Total signal length
        d: Signal dimensionality

    Returns:
        Penalty term (bits) for encoding k changepoints

    Notes:
        Standard MDL: pen(k) = k * (d + 1) * log(T)
        - k segments requires k changepoint positions
        - Each segment has d parameters + 1 variance
        - log(T) factor encodes position precision
    """
    if k == 0:
        return 0.0
    return k * (d + 1) * np.log(T)


def compute_segment_stats(segment: np.ndarray) -> dict:
    """
    Compute descriptive statistics for a segment.

    Args:
        segment: Signal segment

    Returns:
        Dictionary with mean, std, median, min, max, length
    """
    segment = np.asarray(segment)

    return {
        'mean': np.mean(segment),
        'std': np.std(segment),
        'median': np.median(segment),
        'min': np.min(segment),
        'max': np.max(segment),
        'length': len(segment)
    }


def safe_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute Pearson correlation with NaN handling.

    Args:
        x, y: Signal segments (must have same length)

    Returns:
        Correlation coefficient in [-1, 1], or 0.0 if undefined
    """
    if len(x) != len(y) or len(x) == 0:
        return 0.0

    x = np.asarray(x)
    y = np.asarray(y)

    # Remove NaNs
    mask = ~(np.isnan(x) | np.isnan(y))
    x = x[mask]
    y = y[mask]

    if len(x) < 2:
        return 0.0

    # Pearson correlation
    x_centered = x - np.mean(x)
    y_centered = y - np.mean(y)

    numerator = np.dot(x_centered, y_centered)
    denominator = np.sqrt(np.dot(x_centered, x_centered) * np.dot(y_centered, y_centered))

    if denominator < 1e-10:
        return 0.0

    return numerator / denominator


def normalize_signal(signal: np.ndarray) -> np.ndarray:
    """
    Z-score normalize signal (zero mean, unit variance).

    Args:
        signal: Input signal

    Returns:
        Normalized signal
    """
    signal = np.asarray(signal)
    mean = np.mean(signal)
    std = np.std(signal)

    if std < 1e-10:
        return signal - mean

    return (signal - mean) / std
