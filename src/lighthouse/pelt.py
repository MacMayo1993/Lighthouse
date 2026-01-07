"""
PELT (Pruned Exact Linear Time) changepoint detection.

Provides cheap, sparse scaffolding of candidate seams for downstream refinement.
"""

import numpy as np
from typing import List, Optional
from .utils import compute_cost, mdl_penalty


def detect_seams(signal: np.ndarray,
                 penalty: float = 10.0,
                 model: str = 'l2',
                 min_size: int = 2,
                 jump: int = 1) -> List[int]:
    """
    Run PELT for seam candidate detection.

    Args:
        signal: Input time-series (1D array or summary statistic)
        penalty: Beta penalty for number of changepoints (higher = fewer seams)
        model: Cost model ('l2', 'l1', 'ar1')
        min_size: Minimum segment size (prevents micro-segments)
        jump: Stride for candidate evaluation (jump > 1 for speed)

    Returns:
        List of seam positions (indices), excluding end

    Notes:
        - PELT is O(T) on average, O(T^2) worst case
        - Penalty controls sparsity via MDL-style tradeoff
        - Jump parameter trades accuracy for speed (jump=5 typical for ECG)
        - Returns sorted changepoint indices; segments are [0:τ₁), [τ₁:τ₂), ...
    """
    T = len(signal)

    if T < 2 * min_size:
        return []

    # Dynamic programming state
    # F[t] = minimum cost to segment signal[0:t]
    F = np.full(T + 1, np.inf)
    F[0] = -penalty  # No penalty for starting configuration
    R = np.zeros(T + 1, dtype=int)  # Backtracking: last changepoint before t

    # Pruned set: candidate last changepoints
    pruned = {0}

    # Build list of positions to evaluate
    positions = list(range(min_size, T + 1, jump))
    # Always include T
    if T not in positions:
        positions.append(T)
    positions = sorted(set(positions))

    for t in positions:
        candidates = []

        for s in sorted(pruned):
            # s is candidate for last changepoint before t
            # Segment: [s:t)
            if t - s < min_size:
                continue

            segment_cost = compute_cost(signal[s:t], model)
            cost = F[s] + segment_cost + penalty

            candidates.append((cost, s))

        if not candidates:
            continue

        # Take best candidate
        best_cost, best_s = min(candidates)
        F[t] = best_cost
        R[t] = best_s

        # Prune: remove s if F[s] + C[s:t] + penalty > F[t] for all future t
        # Keep only recent promising points
        pruned.add(t)
        # More conservative pruning: keep candidates within reasonable cost
        pruned = {s for s in pruned if F[s] <= F[t] + penalty * 2}

    # Backtrack to recover changepoints
    changepoints = []
    t = T

    while t > 0:
        prev = R[t]
        if prev > 0:
            changepoints.append(prev)
        t = prev

    changepoints.reverse()

    return changepoints


def detect_seams_with_mdl(signal: np.ndarray,
                          model: str = 'l2',
                          min_size: int = 2,
                          k_max: int = 20) -> List[int]:
    """
    Auto-tune penalty via MDL principle: minimize total description length.

    Args:
        signal: Input signal
        model: Cost model
        min_size: Minimum segment size
        k_max: Maximum number of changepoints to consider

    Returns:
        Optimal changepoint set under MDL

    Notes:
        Total description length: L = C + pen(k)
        - C: sum of segment costs (data encoding)
        - pen(k): cost to encode k changepoints (model encoding)
        This auto-selects sparsity without manual penalty tuning.
    """
    T = len(signal)
    best_score = np.inf
    best_changepoints = []

    # Try penalties corresponding to different k values
    for k in range(0, k_max + 1):
        penalty = mdl_penalty(k, T) / max(k, 1)  # Amortize over changepoints

        changepoints = detect_seams(signal, penalty=penalty, model=model, min_size=min_size)

        # Compute MDL score
        k_actual = len(changepoints)
        segments = np.split(signal, changepoints) if changepoints else [signal]
        cost = sum(compute_cost(seg, model) for seg in segments)
        mdl_score = cost + mdl_penalty(k_actual, T)

        if mdl_score < best_score:
            best_score = mdl_score
            best_changepoints = changepoints

    return best_changepoints


def detect_seams_streaming(signal_chunk: np.ndarray,
                           history: Optional[np.ndarray] = None,
                           penalty: float = 10.0,
                           model: str = 'l2',
                           min_size: int = 2) -> List[int]:
    """
    Online PELT variant for streaming data.

    Args:
        signal_chunk: New signal chunk
        history: Previous signal (for context)
        penalty: PELT penalty
        model: Cost model
        min_size: Minimum segment size

    Returns:
        Changepoints in current chunk (relative indices)

    Notes:
        - Maintains small context window from history
        - Returns only changepoints in new chunk
        - Suitable for battery-aware streaming (e.g., wearables)
    """
    context_size = min_size * 3

    if history is not None and len(history) > 0:
        context = history[-context_size:]
        full_signal = np.concatenate([context, signal_chunk])
        offset = len(context)
    else:
        full_signal = signal_chunk
        offset = 0

    changepoints = detect_seams(full_signal, penalty=penalty, model=model, min_size=min_size)

    # Filter to changepoints in new chunk
    chunk_changepoints = [cp - offset for cp in changepoints if cp >= offset]

    return chunk_changepoints
