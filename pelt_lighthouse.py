#!/usr/bin/env python3
"""
PELT-Lighthouse Refined Analysis Script

This script demonstrates the full pipeline with detailed cost-curvature analysis,
seam type classification (S/T/C), and antipodal pairing (value & time).

Produces tabular output suitable for immediate validation experiments.

Usage:
    python pelt_lighthouse.py <input_signal.csv> [options]

    Or in Python:
    from pelt_lighthouse import analyze_signal
    results = analyze_signal(signal, penalty=10, window_size=10)
"""

import numpy as np
import pandas as pd
import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.pipeline import run_pipeline, analyze_seam_distribution
from src.seam_types import plot_cost_profile


def analyze_signal(signal: np.ndarray,
                  penalty: float = 10.0,
                  window_size: int = 10,
                  model: str = 'l2',
                  min_size: int = 2,
                  jump: int = 5,
                  auto_mdl: bool = False,
                  compute_antipodes: bool = True,
                  battery_budget: str = 'medium',
                  verbose: bool = True) -> dict:
    """
    Run full PELT-Lighthouse-Antipodes analysis with detailed output.

    Args:
        signal: Input 1D signal
        penalty: PELT penalty (higher = fewer seams)
        window_size: Lighthouse refinement window
        model: Cost model ('l2' for Gaussian, 'l1' for robust, 'ar1' for autoregressive)
        min_size: Minimum segment size
        jump: PELT stride (5 typical for ECG)
        auto_mdl: Auto-tune penalty via MDL (recommended for exploratory analysis)
        compute_antipodes: Run antipodal pairing (expensive but informative)
        battery_budget: 'low', 'medium', 'high' for triggering expensive ops
        verbose: Print detailed output

    Returns:
        Dictionary with:
            - results_df: Per-seam analysis (τ_PELT, τ_lighthouse, offset, type, curvature, etc.)
            - summary: Aggregate statistics
            - distribution_analysis: Seam type distribution and recommendations
            - compression: Symmetry compression statistics (if compute_antipodes=True)
    """
    if verbose:
        print(f"\n{'='*80}")
        print("PELT-Lighthouse-Antipodes Pipeline")
        print(f"{'='*80}\n")
        print(f"Signal length: {len(signal)}")
        print(f"Model: {model}")
        print(f"Penalty: {penalty} (auto_mdl={auto_mdl})")
        print(f"Window size: {window_size}")
        print(f"Min segment size: {min_size}")
        print(f"Jump: {jump}")
        print(f"\n{'='*80}\n")

    # Run full pipeline
    pipeline_output = run_pipeline(
        signal,
        penalty=penalty,
        window_size=window_size,
        model=model,
        min_size=min_size,
        jump=jump,
        auto_mdl=auto_mdl,
        filter_weak=True,
        min_confidence=1.5,
        compute_antipodes=compute_antipodes,
        battery_budget=battery_budget
    )

    results_df = pipeline_output['results']
    summary = pipeline_output['summary']
    compression = pipeline_output['compression']

    if verbose:
        print("SUMMARY")
        print(f"{'-'*80}")
        print(f"PELT detected:        {summary['num_seams_pelt']} seams")
        print(f"Lighthouse refined:   {summary['num_seams_refined']} seams")
        print(f"Filtered (weak):      {summary['num_seams_filtered']} seams")
        print(f"\nSeam Type Distribution:")
        for stype, count in summary['seam_distribution'].items():
            pct = count / max(summary['num_seams_refined'], 1) * 100
            print(f"  {stype}: {count} ({pct:.1f}%)")

        print(f"\nMean curvature (|κ|): {summary['mean_curvature']:.4f}")
        print(f"Mean confidence:      {summary['mean_confidence']:.2f}")

        if compression is not None:
            print(f"\nCOMPRESSION ANALYSIS")
            print(f"{'-'*80}")
            print(f"Naive cost:       {compression['naive_cost']:.2f}")
            print(f"Symmetry cost:    {compression['symmetry_cost']:.2f}")
            print(f"Savings ratio:    {compression['savings_ratio']:.2%}")
            print(f"Antipodal pairs:  {compression['num_antipodes']}")

        print(f"\n{'='*80}\n")

        # Detailed per-seam table
        if not results_df.empty:
            print("PER-SEAM ANALYSIS")
            print(f"{'-'*80}")

            # Format for display
            display_df = results_df.copy()

            # Select columns for table
            columns = [
                'tau_pelt', 'tau_lighthouse', 'offset', 'seam_type',
                'curvature', 'sharpness', 'confidence',
                'antipodal_corr', 'antipode_type', 'mdl_savings', 'filtered'
            ]

            display_df = display_df[columns]

            # Round numerical columns
            display_df['curvature'] = display_df['curvature'].round(4)
            display_df['sharpness'] = display_df['sharpness'].round(2)
            display_df['confidence'] = display_df['confidence'].round(2)
            display_df['antipodal_corr'] = display_df['antipodal_corr'].round(3)
            display_df['mdl_savings'] = display_df['mdl_savings'].round(1)

            # Pretty print
            pd.set_option('display.max_rows', None)
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', 150)

            print(display_df.to_string(index=False))

            print(f"\n{'='*80}\n")

    # Distribution analysis
    distribution_analysis = analyze_seam_distribution(results_df)

    if verbose and 'error' not in distribution_analysis:
        print("SIGNAL CHARACTERIZATION")
        print(f"{'-'*80}")
        print(f"Dominant type: {distribution_analysis['dominant_type']}")
        print(f"\n{distribution_analysis['characterization']}\n")

        if distribution_analysis['recommendations']:
            print("Recommendations:")
            for rec in distribution_analysis['recommendations']:
                print(f"  • {rec}")

        print(f"\n{'='*80}\n")

    return {
        'results_df': results_df,
        'summary': summary,
        'distribution_analysis': distribution_analysis,
        'compression': compression,
        'seams': pipeline_output['seams']
    }


def load_signal(filepath: str, column: str = None) -> np.ndarray:
    """
    Load signal from CSV file.

    Args:
        filepath: Path to CSV file
        column: Column name (if None, uses first numeric column)

    Returns:
        1D numpy array
    """
    df = pd.read_csv(filepath)

    if column is not None:
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in {filepath}")
        signal = df[column].values
    else:
        # Use first numeric column
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            raise ValueError(f"No numeric columns found in {filepath}")
        signal = df[numeric_cols[0]].values

    return signal


def save_results(output: dict, output_dir: str = './results'):
    """
    Save analysis results to files.

    Args:
        output: Dictionary from analyze_signal()
        output_dir: Output directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Save per-seam results
    results_df = output['results_df']
    if not results_df.empty:
        results_df.to_csv(output_path / 'seams.csv', index=False)
        print(f"Saved seam analysis to {output_path / 'seams.csv'}")

    # Save summary
    summary_df = pd.DataFrame([output['summary']])
    summary_df.to_csv(output_path / 'summary.csv', index=False)
    print(f"Saved summary to {output_path / 'summary.csv'}")

    # Save compression stats if available
    if output['compression'] is not None:
        compression_df = pd.DataFrame([output['compression']])
        compression_df.to_csv(output_path / 'compression.csv', index=False)
        print(f"Saved compression analysis to {output_path / 'compression.csv'}")


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description='PELT-Lighthouse seam detection with cost-curvature analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('input', type=str, help='Input CSV file with signal')
    parser.add_argument('--column', type=str, default=None,
                       help='Column name for signal (default: first numeric column)')
    parser.add_argument('--penalty', type=float, default=10.0,
                       help='PELT penalty (default: 10.0)')
    parser.add_argument('--window', type=int, default=10,
                       help='Lighthouse window size (default: 10)')
    parser.add_argument('--model', type=str, default='l2', choices=['l2', 'l1', 'ar1'],
                       help='Cost model (default: l2)')
    parser.add_argument('--min-size', type=int, default=2,
                       help='Minimum segment size (default: 2)')
    parser.add_argument('--jump', type=int, default=5,
                       help='PELT stride (default: 5)')
    parser.add_argument('--auto-mdl', action='store_true',
                       help='Auto-tune penalty via MDL')
    parser.add_argument('--no-antipodes', action='store_true',
                       help='Skip antipodal pairing (faster)')
    parser.add_argument('--battery', type=str, default='medium', choices=['low', 'medium', 'high'],
                       help='Battery budget for expensive ops (default: medium)')
    parser.add_argument('--output-dir', type=str, default='./results',
                       help='Output directory for results (default: ./results)')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress verbose output')

    args = parser.parse_args()

    # Load signal
    try:
        signal = load_signal(args.input, column=args.column)
        print(f"Loaded signal from {args.input}: {len(signal)} samples")
    except Exception as e:
        print(f"Error loading signal: {e}")
        sys.exit(1)

    # Run analysis
    output = analyze_signal(
        signal,
        penalty=args.penalty,
        window_size=args.window,
        model=args.model,
        min_size=args.min_size,
        jump=args.jump,
        auto_mdl=args.auto_mdl,
        compute_antipodes=not args.no_antipodes,
        battery_budget=args.battery,
        verbose=not args.quiet
    )

    # Save results
    save_results(output, args.output_dir)

    print("\nAnalysis complete!")


if __name__ == '__main__':
    main()
