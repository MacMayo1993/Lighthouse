#!/usr/bin/env python3
"""
Synthetic Data Generator for S/T/C Seam Type Validation

Generates signals with known seam types (Smooth, Tangent, Cusp) for validating
the cost-curvature classification framework.

New features:
- AR(1) noise for realistic dynamics
- Multi-seam signals with varying types
- Configurable SNR levels
- Auto-save for CI integration

Usage:
    python synthetic_data_generator.py --type C --output cusps.csv
    python synthetic_data_generator.py --mixed --plot
    python synthetic_data_generator.py --ar1-noise --phi 0.8
"""

import numpy as np
import pandas as pd
import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pelt_lighthouse import analyze_signal


def add_ar1_noise(signal: np.ndarray, phi: float = 0.5, sigma: float = 0.1) -> np.ndarray:
    """
    Add AR(1) autocorrelated noise to signal.

    Args:
        signal: Clean signal
        phi: AR(1) coefficient (0 < phi < 1 for stationarity)
        sigma: Innovation noise std

    Returns:
        Signal with AR(1) noise

    Notes:
        AR(1) noise is more realistic than white noise for many applications:
        - ECG: baseline wander (phi ~ 0.8-0.9)
        - IMU: sensor drift (phi ~ 0.7-0.9)
        - Network: bursty traffic (phi ~ 0.5-0.7)
    """
    T = len(signal)
    noise = np.zeros(T)
    noise[0] = np.random.normal(0, sigma)

    for t in range(1, T):
        noise[t] = phi * noise[t-1] + np.random.normal(0, sigma)

    return signal + noise


def generate_cusp(length: int = 500, num_seams: int = 3, noise_level: float = 0.1) -> tuple:
    """
    Generate signal with cusp (C) seams: hard steps.

    Args:
        length: Total signal length
        num_seams: Number of cusps
        noise_level: Gaussian noise std

    Returns:
        (signal, true_seams): Signal and ground truth seam positions

    Notes:
        Cusps have discontinuous first derivative: x' jumps.
        Example: Arrhythmia onset, state switches.
    """
    signal = np.zeros(length)

    # Ensure seams are well-separated (at least 50 samples apart)
    min_spacing = max(50, length // (num_seams + 1))
    seam_positions = []
    for i in range(num_seams):
        pos = int((i + 1) * length / (num_seams + 1))
        seam_positions.append(pos)

    segments = [0] + seam_positions + [length]

    # Each segment has a different mean (abrupt jumps with large magnitude)
    levels = np.random.uniform(-5, 5, len(segments))
    # Ensure consecutive levels are well-separated
    for i in range(1, len(levels)):
        if abs(levels[i] - levels[i-1]) < 2:
            levels[i] = levels[i-1] + np.random.choice([-3, 3])

    for i in range(len(segments) - 1):
        start = segments[i]
        end = segments[i + 1]
        signal[start:end] = levels[i]

    # Add noise
    signal += np.random.normal(0, noise_level, length)

    return signal, seam_positions


def generate_tangent(length: int = 500, num_seams: int = 3, noise_level: float = 0.1,
                    transition_width: int = 20) -> tuple:
    """
    Generate signal with tangent (T) seams: smooth transitions with kinks.

    Args:
        length: Total signal length
        num_seams: Number of tangent transitions
        noise_level: Gaussian noise std
        transition_width: Width of sigmoid transition

    Returns:
        (signal, true_seams): Signal and ground truth seam positions

    Notes:
        Tangents have continuous first derivative but kinked second derivative.
        Example: Sigmoid transitions between states.
    """
    signal = np.zeros(length)

    # Ensure seams are well-separated
    seam_positions = []
    for i in range(num_seams):
        pos = int((i + 1) * length / (num_seams + 1))
        seam_positions.append(pos)

    segments = [0] + seam_positions + [length]
    levels = np.random.uniform(-5, 5, len(segments))
    # Ensure level differences
    for i in range(1, len(levels)):
        if abs(levels[i] - levels[i-1]) < 2:
            levels[i] = levels[i-1] + np.random.choice([-3, 3])

    # Build piecewise signal with sigmoid transitions
    for i in range(len(segments) - 1):
        start = segments[i]
        end = segments[i + 1]

        if i == 0:
            # First segment: constant
            signal[start:end] = levels[i]
        else:
            # Transition from previous level to current
            seam = segments[i]
            # Make transition wider for smoother sigmoid
            transition_start = max(start - transition_width, 0)
            transition_end = min(seam + transition_width, length)

            # Wider sigmoid range for gentler transition
            t = np.linspace(-4, 4, transition_end - transition_start)
            sigmoid = 1 / (1 + np.exp(-t))

            # Smooth transition from levels[i-1] to levels[i]
            transition = levels[i - 1] + (levels[i] - levels[i - 1]) * sigmoid

            signal[transition_start:transition_end] = transition

            # Constant after transition
            if transition_end < end:
                signal[transition_end:end] = levels[i]

    # Add noise
    signal += np.random.normal(0, noise_level, length)

    return signal, seam_positions


def generate_smooth(length: int = 500, num_seams: int = 3, noise_level: float = 0.1) -> tuple:
    """
    Generate signal with smooth (S) seams: gradual changes.

    Args:
        length: Total signal length
        num_seams: Number of smooth transitions
        noise_level: Gaussian noise std

    Returns:
        (signal, true_seams): Signal and ground truth seam positions

    Notes:
        Smooth seams have continuous derivatives (C^∞).
        Example: Slow drift, thermal changes.
    """
    # Use sum of sinusoids with different frequencies
    t = np.linspace(0, 10, length)

    # Base signal: slow trend
    signal = np.sin(0.5 * t) + 0.5 * np.sin(0.2 * t)

    # Add gradual shifts at seam positions (well-separated)
    seam_positions = []
    for i in range(num_seams):
        pos = int((i + 1) * length / (num_seams + 1))
        seam_positions.append(pos)

    for seam in seam_positions:
        # Gaussian bump centered at seam
        bump = 2.0 * np.exp(-((t - t[seam])**2) / 2.0)
        signal += bump

    # Add noise
    signal += np.random.normal(0, noise_level, length)

    return signal, seam_positions


def generate_antipodal(length: int = 500, num_pairs: int = 2,
                      antipode_type: str = 'reflect', noise_level: float = 0.1) -> tuple:
    """
    Generate signal with antipodal pairs (for testing antipode detection).

    Args:
        length: Total signal length
        num_pairs: Number of antipodal pairs
        antipode_type: 'reflect' (polarity flip) or 'lag' (phase shift)
        noise_level: Gaussian noise std

    Returns:
        (signal, seam_positions, pair_info): Signal, seams, and pairing metadata

    Notes:
        Reflects typical patterns in ECG (polarity flips) and periodic signals (lags).
    """
    signal = np.zeros(length)
    seam_positions = []
    pair_info = []

    segment_length = length // (2 * num_pairs + 1)

    for i in range(num_pairs):
        # Generate base pattern
        t = np.linspace(0, 2 * np.pi, segment_length)
        base_pattern = np.sin(t) + 0.3 * np.sin(3 * t)  # Rich waveform

        # Place first segment
        start1 = i * 2 * segment_length
        end1 = start1 + segment_length
        signal[start1:end1] = base_pattern

        # Place antipodal segment
        start2 = end1
        end2 = start2 + segment_length

        if antipode_type == 'reflect':
            # Polarity flip: post ≈ -pre
            signal[start2:end2] = -base_pattern
            pair_info.append({'type': 'reflect', 'seam': end1, 'segments': (start1, end2)})

        elif antipode_type == 'lag':
            # Phase shift
            lag = segment_length // 4
            lagged_pattern = np.roll(base_pattern, lag)
            signal[start2:end2] = lagged_pattern
            pair_info.append({'type': 'lag', 'lag': lag, 'seam': end1, 'segments': (start1, end2)})

        seam_positions.append(end1)

    # Add noise
    signal += np.random.normal(0, noise_level, length)

    return signal, seam_positions, pair_info


def generate_mixed(length: int = 1000, noise_level: float = 0.1) -> tuple:
    """
    Generate signal with mix of S/T/C seams for comprehensive testing.

    Args:
        length: Total signal length
        noise_level: Gaussian noise std

    Returns:
        (signal, seam_positions, seam_types): Signal, seams, and ground truth types

    Notes:
        This is the most realistic test case: real signals have diverse seam types.
    """
    num_seams = 6
    seam_types_list = ['S', 'S', 'T', 'T', 'C', 'C']
    np.random.shuffle(seam_types_list)

    signal = np.zeros(length)
    seam_positions = sorted(np.random.choice(range(100, length - 100), num_seams, replace=False))

    segments = [0] + list(seam_positions) + [length]

    for i in range(len(segments) - 1):
        start = segments[i]
        end = segments[i + 1]
        seg_length = end - start

        stype = seam_types_list[i] if i < len(seam_types_list) else 'S'

        if stype == 'C':
            # Cusp: constant level
            signal[start:end] = np.random.uniform(-3, 3)

        elif stype == 'T':
            # Tangent: sigmoid transition
            mid = (start + end) // 2
            t = np.linspace(-6, 6, seg_length)
            signal[start:end] = 3 * (1 / (1 + np.exp(-t)) - 0.5)

        elif stype == 'S':
            # Smooth: sinusoidal
            t = np.linspace(0, 2 * np.pi, seg_length)
            signal[start:end] = 2 * np.sin(t)

    # Add noise
    signal += np.random.normal(0, noise_level, length)

    seam_type_map = {seam: stype for seam, stype in zip(seam_positions, seam_types_list)}

    return signal, seam_positions, seam_type_map


def validate_classification(signal: np.ndarray, true_seams: list, true_types: dict) -> dict:
    """
    Run pipeline and validate seam type classification.

    Args:
        signal: Synthetic signal
        true_seams: Ground truth seam positions
        true_types: Dict mapping seam positions to types

    Returns:
        Validation statistics (accuracy, confusion matrix)
    """
    # Run pipeline with lower penalty for synthetic signals
    output = analyze_signal(
        signal,
        penalty=5.0,
        window_size=10,
        auto_mdl=False,
        compute_antipodes=False,
        verbose=False
    )

    results_df = output['results_df']

    if results_df.empty:
        return {'error': 'No seams detected'}

    # Match detected seams to ground truth
    matched = []
    tolerance = 20  # samples

    for _, row in results_df.iterrows():
        detected_pos = row['tau_lighthouse']
        detected_type = row['seam_type']

        # Find closest true seam
        if true_seams:
            closest_true = min(true_seams, key=lambda x: abs(x - detected_pos))
            distance = abs(closest_true - detected_pos)

            if distance <= tolerance and closest_true in true_types:
                true_type = true_types[closest_true]
                matched.append({
                    'detected_pos': detected_pos,
                    'true_pos': closest_true,
                    'detected_type': detected_type,
                    'true_type': true_type,
                    'correct': detected_type == true_type
                })

    if not matched:
        return {'error': 'No matches within tolerance'}

    # Compute accuracy
    correct = sum(1 for m in matched if m['correct'])
    accuracy = correct / len(matched)

    # Confusion matrix
    confusion = {}
    for m in matched:
        key = (m['true_type'], m['detected_type'])
        confusion[key] = confusion.get(key, 0) + 1

    return {
        'accuracy': accuracy,
        'num_matched': len(matched),
        'num_correct': correct,
        'confusion': confusion,
        'matches': matched
    }


def generate_multi_seam(length: int = 1000,
                       seam_types: list = ['S', 'T', 'C'],
                       num_seams_per_type: int = 2,
                       noise_level: float = 0.1,
                       ar1_phi: float = 0.0) -> tuple:
    """
    Generate signal with multiple seams of different types.

    Args:
        length: Total signal length
        seam_types: List of seam types to include
        num_seams_per_type: Number of each type
        noise_level: Noise std
        ar1_phi: AR(1) coefficient (0 = white noise)

    Returns:
        (signal, seam_positions, seam_type_map): Signal, seams, and ground truth

    Notes:
        Creates realistic test cases with mixed seam types for comprehensive validation.
    """
    total_seams = len(seam_types) * num_seams_per_type
    signal = np.zeros(length)

    # Well-separated seam positions
    seam_positions = []
    for i in range(total_seams):
        pos = int((i + 1) * length / (total_seams + 1))
        seam_positions.append(pos)

    # Assign types cyclically
    type_assignments = []
    for stype in seam_types:
        type_assignments.extend([stype] * num_seams_per_type)
    np.random.shuffle(type_assignments)

    segments = [0] + seam_positions + [length]

    for i in range(len(segments) - 1):
        start = segments[i]
        end = segments[i + 1]
        seg_length = end - start

        if i >= len(type_assignments):
            stype = 'S'
        else:
            stype = type_assignments[i]

        if stype == 'C':
            # Cusp: constant level with abrupt jump
            level = np.random.uniform(-5, 5)
            signal[start:end] = level

        elif stype == 'T':
            # Tangent: sigmoid transition
            t = np.linspace(-4, 4, seg_length)
            sigmoid = 1 / (1 + np.exp(-t))
            level_start = np.random.uniform(-5, 5)
            level_end = level_start + np.random.choice([-3, 3])
            signal[start:end] = level_start + (level_end - level_start) * sigmoid

        elif stype == 'S':
            # Smooth: sinusoidal or polynomial
            t = np.linspace(0, 2*np.pi, seg_length)
            signal[start:end] = 2 * np.sin(t) + np.random.uniform(-2, 2)

    # Add noise (AR(1) or white)
    if ar1_phi > 0:
        signal = add_ar1_noise(signal, phi=ar1_phi, sigma=noise_level)
    else:
        signal += np.random.normal(0, noise_level, length)

    seam_type_map = {seam: stype for seam, stype in zip(seam_positions, type_assignments)}

    return signal, seam_positions, seam_type_map


def generate_snr_test(base_signal: np.ndarray,
                     snr_db_range: list = [0, 10, 20, 30]) -> list:
    """
    Generate signals at different SNR levels for robustness testing.

    Args:
        base_signal: Clean signal
        snr_db_range: List of SNR levels in dB

    Returns:
        List of (snr_db, noisy_signal) tuples

    Notes:
        SNR = 10 * log10(P_signal / P_noise)
        Useful for testing algorithm robustness to measurement noise.
    """
    signal_power = np.mean(base_signal**2)

    results = []
    for snr_db in snr_db_range:
        # Compute noise power from SNR
        snr_linear = 10**(snr_db / 10)
        noise_power = signal_power / snr_linear
        noise_std = np.sqrt(noise_power)

        # Add white noise
        noise = np.random.normal(0, noise_std, len(base_signal))
        noisy_signal = base_signal + noise

        results.append((snr_db, noisy_signal))

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Generate synthetic signals with known seam types',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--type', type=str, default='mixed',
                       choices=['C', 'T', 'S', 'mixed', 'antipodal'],
                       help='Seam type to generate (default: mixed)')
    parser.add_argument('--length', type=int, default=1000,
                       help='Signal length (default: 1000)')
    parser.add_argument('--noise', type=float, default=0.1,
                       help='Noise level (default: 0.1)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output CSV file')
    parser.add_argument('--plot', action='store_true',
                       help='Generate plot')
    parser.add_argument('--validate', action='store_true',
                       help='Run validation (test classification accuracy)')

    parser.add_argument('--ar1-noise', action='store_true',
                       help='Use AR(1) noise instead of white noise')
    parser.add_argument('--phi', type=float, default=0.5,
                       help='AR(1) coefficient (default: 0.5)')
    parser.add_argument('--snr-test', action='store_true',
                       help='Generate SNR robustness test')
    parser.add_argument('--multi-seam', action='store_true',
                       help='Generate multi-seam signal with mixed types')

    args = parser.parse_args()

    # Generate signal
    print(f"Generating synthetic signal: type={args.type}, length={args.length}, noise={args.noise}")

    seam_type_map = {}

    if args.multi_seam:
        signal, seams, seam_type_map = generate_multi_seam(
            args.length,
            seam_types=['S', 'T', 'C'],
            num_seams_per_type=2,
            noise_level=args.noise,
            ar1_phi=args.phi if args.ar1_noise else 0.0
        )
        print(f"Generated multi-seam signal with {len(seams)} seams")

    elif args.type == 'C':
        signal, seams = generate_cusp(args.length, num_seams=5, noise_level=args.noise)
        seam_type_map = {s: 'C' for s in seams}

    elif args.type == 'T':
        signal, seams = generate_tangent(args.length, num_seams=5, noise_level=args.noise)
        seam_type_map = {s: 'T' for s in seams}

    elif args.type == 'S':
        signal, seams = generate_smooth(args.length, num_seams=5, noise_level=args.noise)
        seam_type_map = {s: 'S' for s in seams}

    elif args.type == 'mixed':
        signal, seams, seam_type_map = generate_mixed(args.length, noise_level=args.noise)

    elif args.type == 'antipodal':
        signal, seams, pair_info = generate_antipodal(args.length, num_pairs=3, noise_level=args.noise)
        print(f"Generated {len(pair_info)} antipodal pairs")
        seam_type_map = {s: 'C' for s in seams}  # Antipodes create cusps

    print(f"Generated signal with {len(seams)} seams at positions: {seams}")

    # Save to CSV
    if args.output:
        df = pd.DataFrame({'signal': signal})
        df.to_csv(args.output, index=False)
        print(f"Saved to {args.output}")

    # Plot
    if args.plot:
        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(signal, 'k-', linewidth=0.8, alpha=0.7)

            # Mark true seams
            for seam in seams:
                ax.axvline(seam, color='green', linestyle='--', alpha=0.6, linewidth=1.5)

            ax.set_xlabel('Sample')
            ax.set_ylabel('Amplitude')
            ax.set_title(f'Synthetic Signal: Type {args.type.upper()}')
            ax.grid(alpha=0.3)

            plt.tight_layout()
            plt.show()

        except ImportError:
            print("WARNING: matplotlib not available for plotting")

    # Validate classification
    if args.validate and seam_type_map:
        print(f"\n{'='*80}")
        print("Running validation...")
        print(f"{'='*80}\n")

        validation = validate_classification(signal, seams, seam_type_map)

        if 'error' in validation:
            print(f"ERROR: {validation['error']}")
        else:
            print(f"Accuracy: {validation['accuracy']:.2%}")
            print(f"Matched: {validation['num_matched']} / {len(seams)}")
            print(f"Correct: {validation['num_correct']} / {validation['num_matched']}")

            print("\nConfusion Matrix (true → detected):")
            for (true_type, detected_type), count in validation['confusion'].items():
                print(f"  {true_type} → {detected_type}: {count}")


if __name__ == '__main__':
    main()
