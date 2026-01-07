#!/usr/bin/env python3
"""
ECG Pilot Study: PELT-Lighthouse-Antipodes on MIT-BIH Database

Demonstrates seam detection on ECG signals with arrhythmias.
MIT-BIH provides gold-standard annotations for validation.

Requirements:
    pip install wfdb matplotlib

Usage:
    python ecg_pilot.py --record 100
    python ecg_pilot.py --record 119 --plot
"""

import numpy as np
import pandas as pd
import argparse
import sys
from pathlib import Path
import matplotlib.pyplot as plt

# Add parent directory's src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from lighthouse.pipeline import run_pipeline


def load_mitbih_record(record: str, lead: int = 0, duration: int = None):
    """
    Load MIT-BIH Arrhythmia Database record.

    Args:
        record: Record ID (e.g., '100', '119')
        lead: ECG lead (0 = MLII, 1 = V5)
        duration: Maximum duration in seconds (None = full record)

    Returns:
        (signal, annotation, fs): ECG signal, beat annotations, sampling frequency

    Notes:
        MIT-BIH records are sampled at 360 Hz.
        Annotations mark R-peaks and arrhythmia types.
    """
    try:
        import wfdb
    except ImportError:
        print("ERROR: wfdb package required. Install with: pip install wfdb")
        sys.exit(1)

    # Download from PhysioNet if not cached
    record_path = f'mitdb/{record}'

    try:
        # Load signal
        record_data = wfdb.rdrecord(record_path, pn_dir='mitdb')
        signal = record_data.p_signal[:, lead]
        fs = record_data.fs

        # Load annotations
        annotation = wfdb.rdann(record_path, 'atr', pn_dir='mitdb')

        # Truncate if duration specified
        if duration is not None:
            max_samples = int(duration * fs)
            signal = signal[:max_samples]

            # Filter annotations
            annotation.sample = annotation.sample[annotation.sample < max_samples]
            annotation.symbol = [s for s, idx in zip(annotation.symbol, annotation.sample)
                                if idx < max_samples]

        print(f"Loaded record {record} (lead {lead}): {len(signal)} samples @ {fs} Hz")
        print(f"Duration: {len(signal)/fs:.1f} seconds")
        print(f"Annotations: {len(annotation.sample)} beats")

        return signal, annotation, fs

    except Exception as e:
        print(f"ERROR loading MIT-BIH record {record}: {e}")
        print("First-time download may take a moment...")
        raise


def compare_with_annotations(detected_seams: list,
                            annotations,
                            fs: float,
                            tolerance_ms: float = 100) -> dict:
    """
    Compare detected seams with MIT-BIH beat annotations.

    Args:
        detected_seams: List of seam sample indices
        annotations: wfdb annotation object
        fs: Sampling frequency
        tolerance_ms: Tolerance window in milliseconds

    Returns:
        Dictionary with precision, recall, F1 statistics

    Notes:
        This is a simplified comparison - true validation requires:
        - Mapping seam types to arrhythmia classes
        - Handling multiple annotations per beat
        - Clinical interpretation of false positives/negatives
    """
    tolerance_samples = int(tolerance_ms * fs / 1000)

    true_beats = set(annotations.sample)
    detected_set = set(detected_seams)

    # True positives: detected seams within tolerance of annotated beats
    tp = 0
    matched_annotations = set()

    for seam in detected_seams:
        for beat in true_beats:
            if abs(seam - beat) <= tolerance_samples:
                tp += 1
                matched_annotations.add(beat)
                break

    # False positives: detected seams not near any annotation
    fp = len(detected_seams) - tp

    # False negatives: annotations not matched by any detection
    fn = len(true_beats) - len(matched_annotations)

    # Metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'num_detected': len(detected_seams),
        'num_annotated': len(true_beats)
    }


def plot_results(signal, seams, annotations, fs, record_id, output_path=None):
    """
    Visualize detected seams vs. ground truth annotations.

    Args:
        signal: ECG signal
        seams: Detected seam positions
        annotations: MIT-BIH annotations
        fs: Sampling frequency
        record_id: Record ID for title
        output_path: Save path (if None, display interactively)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("WARNING: matplotlib not available for plotting")
        return

    # Plot first 10 seconds for clarity
    duration = 10
    max_samples = int(duration * fs)

    time = np.arange(min(len(signal), max_samples)) / fs
    signal_segment = signal[:max_samples]

    fig, ax = plt.subplots(figsize=(14, 6))

    # Plot signal
    ax.plot(time, signal_segment, 'k-', linewidth=0.8, label='ECG', alpha=0.7)

    # Plot detected seams
    seams_in_window = [s for s in seams if s < max_samples]
    for seam in seams_in_window:
        ax.axvline(seam / fs, color='red', linestyle='--', alpha=0.6, linewidth=1.5)

    # Plot annotations
    ann_in_window = [idx for idx in annotations.sample if idx < max_samples]
    for ann in ann_in_window:
        ax.axvline(ann / fs, color='green', linestyle=':', alpha=0.5, linewidth=1.5)

    # Labels
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Amplitude (mV)', fontsize=12)
    ax.set_title(f'MIT-BIH Record {record_id}: PELT-Lighthouse Seam Detection\n'
                 f'Red dashed = Detected seams, Green dotted = Annotated beats',
                 fontsize=13)
    ax.grid(alpha=0.3)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='k', linewidth=0.8, label='ECG signal'),
        Line2D([0], [0], color='red', linestyle='--', linewidth=1.5, label='Detected seams'),
        Line2D([0], [0], color='green', linestyle=':', linewidth=1.5, label='Annotated beats')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Saved plot to {output_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='ECG pilot study with MIT-BIH database',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--record', type=str, default='100',
                       help='MIT-BIH record ID (e.g., 100, 119)')
    parser.add_argument('--lead', type=int, default=0,
                       help='ECG lead (0=MLII, 1=V5)')
    parser.add_argument('--duration', type=int, default=None,
                       help='Duration in seconds (default: full record)')
    parser.add_argument('--penalty', type=float, default=50.0,
                       help='PELT penalty (default: 50.0)')
    parser.add_argument('--window', type=int, default=15,
                       help='Lighthouse window (default: 15)')
    parser.add_argument('--jump', type=int, default=5,
                       help='PELT stride (default: 5)')
    parser.add_argument('--plot', action='store_true',
                       help='Generate visualization')
    parser.add_argument('--output-dir', type=str, default='./results_ecg',
                       help='Output directory')

    args = parser.parse_args()

    # Load ECG data
    signal, annotations, fs = load_mitbih_record(args.record, lead=args.lead, duration=args.duration)

    # Run pipeline
    print(f"\n{'='*80}")
    print("Running PELT-Lighthouse-Antipodes pipeline...")
    print(f"{'='*80}\n")

    output = run_pipeline(
        signal,
        penalty=args.penalty,
        window_size=args.window,
        model='l2',
        min_size=10,
        jump=args.jump,
        auto_mdl=False,
        filter_weak=True,
        compute_antipodes=True,
        battery_budget='medium'
    )

    seams = output['seams']
    results_df = output['results']
    summary = output['summary']

    print(f"\nDetected {len(seams)} seams")

    # Compare with annotations
    print(f"\n{'='*80}")
    print("Validation against MIT-BIH annotations")
    print(f"{'='*80}\n")

    comparison = compare_with_annotations(seams, annotations, fs, tolerance_ms=100)

    print(f"True positives:  {comparison['tp']}")
    print(f"False positives: {comparison['fp']}")
    print(f"False negatives: {comparison['fn']}")
    print(f"\nPrecision: {comparison['precision']:.3f}")
    print(f"Recall:    {comparison['recall']:.3f}")
    print(f"F1 score:  {comparison['f1']:.3f}")

    # Save results
    output_path = Path(args.output_dir)
    output_path.mkdir(exist_ok=True)

    results_df.to_csv(output_path / f'seams_record_{args.record}.csv', index=False)
    print(f"\nSaved results to {output_path / f'seams_record_{args.record}.csv'}")

    # Save comparison
    comparison_df = pd.DataFrame([comparison])
    comparison_df.to_csv(output_path / f'validation_record_{args.record}.csv', index=False)

    # Plot if requested
    if args.plot:
        plot_results(signal, seams, annotations, fs, args.record,
                    output_path=output_path / f'plot_record_{args.record}.png')

    print(f"\n{'='*80}")
    print("ECG pilot complete!")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
