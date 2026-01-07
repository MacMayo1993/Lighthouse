# Lighthouse: PELT-Lighthouse-Antipodes Pipeline

**Computationally efficient seam detection for time-series signals via cost-curvature analysis and antipodal symmetry extraction.**

---

## Overview

The **Lighthouse** pipeline provides a principled, battery-aware framework for detecting and classifying regime changes ("seams") in time-series data. It combines:

1. **PELT (Pruned Exact Linear Time)**: Sparse, cheap scaffolding of candidate seams
2. **Lighthouse**: Local entropy-based refinement around candidates
3. **Antipodes**: Symmetry extraction via value-space reflections and time-space lags
4. **Cost-Curvature Seam Typing**: Classification into S (smooth), T (tangent), C (cusp) via MDL-derived second derivatives

The architecture is designed for **resource-constrained devices** (e.g., wearables, IoT sensors) where expensive operations must be triggered sparingly based on information-theoretic criteria.

---

## Key Innovations

### 1. Cost-Curvature Seam Taxonomy (S/T/C)

Seams are classified by computing the **curvature** of the cost function ΔC(τ):

```
κ(τ) = d²(ΔC)/dτ² ≈ [ΔC(τ-h) - 2ΔC(τ) + ΔC(τ+h)] / h²
```

- **S (Smooth)**: |κ| ≈ 0 → Gradual drift (e.g., temperature change)
- **T (Tangent)**: 0 < |κ| < ∞ → Sigmoid-like transition (e.g., state change onset)
- **C (Cusp)**: |κ| → ∞ → Hard step (e.g., arrhythmia, packet drop)

This stays entirely within the **MDL framework** without extra machinery.

### 2. Antipodal Pairing

Maps **involutions** (symmetries) between segments across seams:

- **Value antipodes**: `post ≈ -pre + 2*axis` (polarity flips in ECG)
- **Time antipodes**: `post(t) ≈ pre(t - δ)` (phase lags in periodic signals)

Strong antipodes enable **compression** by encoding one segment + involution instead of two independent segments.

### 3. Battery-Aware Triggering

Expensive operations (e.g., deep model inference) are triggered only on high-information seams:

- **Low budget**: Only C seams (critical events)
- **Medium budget**: C + high-curvature T seams
- **High budget**: All non-smooth seams

This **"Joules-rent" optimization** maximizes information per unit energy.

---

## Installation

```bash
git clone https://github.com/yourusername/Lighthouse.git
cd Lighthouse
pip install -r requirements.txt
```

Or install as a package:

```bash
pip install -e .
```

---

## Quick Start

### Basic Usage

```python
import numpy as np
from src.pipeline import run_pipeline

# Load your signal (1D array)
signal = np.loadtxt('data/your_signal.csv')

# Run full pipeline
results = run_pipeline(
    signal,
    penalty=10.0,           # PELT penalty (higher = fewer seams)
    window_size=10,         # Lighthouse refinement window
    model='l2',             # Cost model (Gaussian)
    compute_antipodes=True, # Enable symmetry extraction
    battery_budget='medium' # Trigger expensive ops moderately
)

# Access results
seams = results['seams']                  # Refined seam positions
results_df = results['results']           # Per-seam DataFrame
summary = results['summary']              # Aggregate statistics
compression = results['compression']      # Symmetry compression stats
```

### Command-Line Interface

```bash
# Run on ECG data
python pelt_lighthouse.py data/ecg_sample.csv --penalty 50 --window 15

# Auto-tune penalty via MDL
python pelt_lighthouse.py data/ecg_sample.csv --auto-mdl

# Save detailed output
python pelt_lighthouse.py data/signal.csv --output-dir ./results
```

---

## Examples

### ECG Pilot (MIT-BIH Database)

```bash
# Run on MIT-BIH record 100 (requires wfdb)
python examples/ecg_pilot.py --record 100 --plot

# Noisy record with arrhythmias
python examples/ecg_pilot.py --record 119 --penalty 50 --plot
```

**Output:**
- Detected seam positions with S/T/C classification
- Validation against MIT-BIH annotations (precision/recall/F1)
- Antipodal pairs (polarity flips across seams)
- Visualization of seams vs. ground truth

### Synthetic Data Validation

```bash
# Generate signal with cusp seams
python examples/synthetic_data_generator.py --type C --plot --validate

# Generate mixed S/T/C signal
python examples/synthetic_data_generator.py --type mixed --plot --validate

# Test antipodal detection
python examples/synthetic_data_generator.py --type antipodal --plot
```

**Use cases:**
- Validate S/T/C classification accuracy
- Test antipodal pairing on known symmetries
- Tune hyperparameters on controlled data

---

## Pipeline Architecture

```
Input Signal
    ↓
┌─────────────────────┐
│  PELT Detection     │  O(T) on average
│  Sparse candidates  │  → τ₁, τ₂, ..., τₖ
└─────────────────────┘
    ↓
┌─────────────────────┐
│  Lighthouse Refine  │  O(k·w) where w = window size
│  Local entropy peak │  → τ₁*, τ₂*, ..., τₖ*
└─────────────────────┘
    ↓
┌─────────────────────┐
│  Filter Weak Seams  │  Remove spurious splits
│  Sharpness threshold│  → Filtered seam set
└─────────────────────┘
    ↓
┌─────────────────────┐
│  Seam Type (S/T/C)  │  Cost curvature κ(τ)
│  Cost curvature     │  → Classification
└─────────────────────┘
    ↓
┌─────────────────────┐
│  Antipodal Pairing  │  O(k·m) where m = segment length
│  Value & time modes │  → Symmetry atoms
└─────────────────────┘
    ↓
┌─────────────────────┐
│  Battery Trigger    │  Recommend expensive ops
│  High-info seams    │  → Priority queue
└─────────────────────┘
    ↓
Output: Seams + Types + Symmetries
```

---

## Output Format

### Per-Seam DataFrame

| Column           | Description                                      |
|------------------|--------------------------------------------------|
| `tau_pelt`       | PELT-detected candidate position                 |
| `tau_lighthouse` | Lighthouse-refined position                      |
| `offset`         | Refinement offset (τ_lighthouse - τ_pelt)        |
| `seam_type`      | S/T/C classification                             |
| `curvature`      | Raw curvature κ(τ)                               |
| `sharpness`      | Transition sharpness score                       |
| `confidence`     | Lighthouse confidence (peak/mean novelty)        |
| `antipodal_corr` | Best antipodal correlation                       |
| `antipode_type`  | 'direct', 'reflect', 'lag', or 'none'            |
| `mdl_savings`    | Estimated bits saved by antipodal encoding       |
| `filtered`       | Boolean: was this seam filtered as weak?         |

### Summary Statistics

- Number of seams detected (PELT, refined, filtered)
- Seam type distribution (S/T/C percentages)
- Mean curvature and confidence
- Compression statistics (naive cost vs. symmetry cost, savings ratio)

---

## Algorithm Details

### PELT (src/pelt.py)

- **Input**: Signal, penalty β, cost model (l2/l1/ar1)
- **Output**: Changepoint set τ₁, ..., τₖ minimizing `C + β·k`
- **Complexity**: O(T) average, O(T²) worst case
- **Features**:
  - Dynamic programming with pruning
  - MDL auto-tuning via `detect_seams_with_mdl()`
  - Streaming variant for online processing

### Lighthouse (src/lighthouse.py)

- **Input**: PELT candidate τ, window size w
- **Output**: Refined position τ*, confidence score
- **Methods**:
  - `delta_cost`: Find argmax |ΔC(t)| in window (steepest cost drop)
  - `entropy`: Shannon entropy peak
  - `kl_divergence`: Max KL divergence between pre/post
- **Filtering**: Remove weak seams via sharpness threshold

### Antipodes (src/antipodes.py)

- **Input**: Pre-seam segment, post-seam segment
- **Output**: Best antipodal pairing (correlation, type, parameter)
- **Modes**:
  - Value: `corr(pre, post)` vs. `corr(pre, -post)` → reflection
  - Time: `max_lag corr(pre, roll(post, lag))` → phase shift
- **Compression**: Estimate MDL savings from symmetry atoms

### Seam Types (src/seam_types.py)

- **Input**: Signal, seam position τ
- **Output**: (S/T/C, curvature κ)
- **Method**: Finite difference second derivative of ΔC(τ)
- **Thresholds**:
  - S: |κ| < 0.01
  - T: 0.01 ≤ |κ| < 0.5
  - C: |κ| ≥ 0.5
- **Adaptive**: Normalize by local signal variance for robustness

---

## Applications

### Digital Health (ECG, PPG)

- **Arrhythmia detection**: C seams flag abrupt rhythm changes
- **Beat segmentation**: T seams mark QRS/R-peak transitions
- **Polarity flips**: Value antipodes detect lead inversions
- **Battery savings**: Trigger deep models only on C seams (~90% compute reduction)

### Network Traffic

- **Packet bursts**: C seams mark congestion onset
- **Protocol switches**: T seams for gradual rate changes
- **Periodic patterns**: Time antipodes detect cyclic traffic

### IMU/Activity Recognition

- **Gait transitions**: C seams for walk→run, sit→stand
- **Phase lags**: Time antipodes between x/y/z axes
- **Fall detection**: High-curvature C seams with low antipodal correlation

### Industrial IoT

- **Equipment failures**: C seams for abrupt sensor changes
- **Thermal drift**: S seams for slow baseline shifts
- **Predictive maintenance**: T seams as early warning (smooth → tangent → cusp)

---

## Tuning Guide

### Penalty (β)

- **Low penalty (β < 5)**: More seams, higher sensitivity, more false positives
- **Medium penalty (β = 10-50)**: Balanced (default for most signals)
- **High penalty (β > 100)**: Fewer seams, high precision, may miss subtle changes
- **Auto-tune**: Use `auto_mdl=True` to select β via MDL principle

### Window Size

- **Small window (w = 5-10)**: Sharp seams (ECG R-peaks, network bursts)
- **Medium window (w = 10-20)**: General-purpose (default)
- **Large window (w > 20)**: Smooth, gradual transitions (temperature drift)

### Jump (PELT Stride)

- **jump=1**: Exact PELT (slowest, most accurate)
- **jump=5**: Fast PELT (typical for ECG @ 360 Hz)
- **jump=10+**: Very fast, may miss narrow seams

### Cost Model

- **l2** (Gaussian): General-purpose, assumes normally distributed noise
- **l1** (Laplacian): Robust to outliers
- **ar1** (Autoregressive): For signals with temporal correlation

---

## Performance

### Computational Complexity

| Stage           | Complexity    | Notes                                    |
|-----------------|---------------|------------------------------------------|
| PELT            | O(T)          | Average case with pruning                |
| Lighthouse      | O(k·w)        | k seams × w window size                  |
| Filtering       | O(k)          | Per-seam sharpness check                 |
| Seam Typing     | O(k)          | Three-point stencil per seam             |
| Antipodes       | O(k·m)        | k seams × m segment length (expensive)   |

**Total**: O(T + k·m) where k << T and m ~ T/k → **O(T)** overall (linear in signal length).

### Battery Savings (Simulated)

For ECG @ 360 Hz, 10-minute record (216,000 samples):

- **Naive**: Process all samples → 100% compute
- **PELT**: ~50 seams → **99.98% reduction** in candidate set
- **Lighthouse + Antipodes** on C seams only (battery=low): ~5 expensive ops → **~99.998% reduction**

**Real-world impact**: Hours to days of battery life extension for wearables.

---

## Testing

```bash
# Run test suite
pip install pytest pytest-cov
pytest tests/ --cov=src --cov-report=html

# Quick smoke test
python -c "from src.pipeline import run_pipeline; import numpy as np; \
           run_pipeline(np.random.randn(1000))"
```

---

## Citation

If you use this code in research, please cite:

```bibtex
@software{lighthouse2026,
  title={Lighthouse: PELT-Lighthouse-Antipodes Pipeline for Seam Detection},
  author={Your Name},
  year={2026},
  url={https://github.com/yourusername/Lighthouse}
}
```

---

## Roadmap

- [ ] Multi-dimensional signals (IMU xyz, multi-lead ECG)
- [ ] GPU acceleration for large datasets
- [ ] Streaming API for real-time processing
- [ ] Pre-trained models for domain-specific seam types
- [ ] Integration with edge ML frameworks (TensorFlow Lite, ONNX)
- [ ] Companion paper with formal proofs and extended validation

---

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Acknowledgments

- **PELT algorithm**: Killick et al. (2012), "Optimal Detection of Changepoints"
- **MIT-BIH Database**: Moody & Mark (2001), PhysioNet
- **MDL principle**: Rissanen (1978), "Modeling by Shortest Data Description"

---

## Contact

For questions, issues, or collaboration:

- **GitHub Issues**: [https://github.com/yourusername/Lighthouse/issues](https://github.com/yourusername/Lighthouse/issues)
- **Email**: your.email@example.com

**Let's build efficient, principled systems that respect both information theory and battery constraints!**
