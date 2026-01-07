# PELT-Lighthouse-Antipodes Pipeline Overview

## Architecture

The pipeline consists of five stages that progressively refine and classify seams:

### 1. PELT Detection (Sparse Scaffolding)

**Goal**: Generate initial candidate seams efficiently.

- **Algorithm**: Pruned Exact Linear Time changepoint detection
- **Complexity**: O(T) on average
- **Output**: Coarse seam positions τ₁, τ₂, ..., τₖ

**Key Idea**: Dynamic programming minimizes total cost:
```
min { Σ C(segment_i) + β·k }
```
where β is the penalty per seam (MDL-style tradeoff).

### 2. Lighthouse Refinement (Local Optimization)

**Goal**: Refine each PELT candidate to precise boundary.

- **Method**: Sweep narrow window [τ - δ, τ + δ] to find local entropy/cost peak
- **Complexity**: O(k·w) where w = window size
- **Output**: Refined positions τ₁*, τ₂*, ..., τₖ*

**Key Idea**: PELT is accurate to ~jump samples; lighthouse zooms in to sample-level precision.

### 3. Weak Seam Filtering

**Goal**: Remove spurious splits from over-segmentation.

- **Method**: Compute sharpness score = |Δmean| / σ
- **Threshold**: Keep only seams with confidence > min_confidence
- **Complexity**: O(k)

**Key Idea**: Gradual drifts are not informative seams; filter them out.

### 4. Seam Type Classification (S/T/C)

**Goal**: Classify seams by transition sharpness via cost curvature.

- **Method**: Compute κ(τ) = d²(ΔC)/dτ² using 3-point stencil
- **Complexity**: O(k)
- **Output**: S (smooth), T (tangent), C (cusp) per seam

**Key Idea**: Curvature of the cost function encodes transition geometry within MDL framework.

### 5. Antipodal Pairing (Symmetry Extraction)

**Goal**: Identify involutions for compression.

- **Value antipodes**: `corr(pre, -post)` for polarity flips
- **Time antipodes**: `argmax_δ corr(pre, roll(post, δ))` for phase lags
- **Complexity**: O(k·m) where m = segment length (expensive!)
- **Output**: Symmetry atoms with MDL savings estimates

**Key Idea**: Strong antipodes enable 2× compression by encoding one segment + involution.

---

## Cost-Curvature Intuition

The cost function ΔC(τ) measures improvement from splitting at τ:

```
ΔC(τ) = [C_left(τ) + C_right(τ)] - C_full
```

Curvature κ(τ) = d²(ΔC)/dτ² distinguishes:

- **S**: ΔC(τ) is parabolic → κ ≈ 0 (gradual change)
- **T**: ΔC(τ) has kink → κ moderate (sigmoid-like)
- **C**: ΔC(τ) has corner → κ large (discontinuous derivative)

Analogy: Driving on a road:
- S = gentle curve (no steering adjustment)
- T = sharp turn (steer smoothly)
- C = right angle (slam the brakes)

---

## Battery-Aware Triggering

Not all seams warrant expensive operations (e.g., deep models, long-range queries).

**Strategy**: Prioritize by information content:

1. **High-info seams**: C seams (critical events like arrhythmias)
2. **Medium-info seams**: High-curvature T seams
3. **Low-info seams**: S seams (drifts, ignore)

**Budget levels**:
- **Low**: Process only C seams (~5% of total)
- **Medium**: C + sharp T (~20% of total)
- **High**: All non-smooth (~50% of total)

**Impact**: 5-20× reduction in compute for equivalent information extraction.

---

## MDL Throughout

Every component respects the Minimum Description Length principle:

- **PELT**: Minimizes data cost + model cost (β·k)
- **Lighthouse**: Maximizes local information (entropy peaks)
- **Seam types**: Derived from cost geometry (no ad-hoc features)
- **Antipodes**: Estimate compression via R² → MDL savings

No external classifiers, feature engineering, or black boxes—pure information theory.

---

## Computational Profile

For typical ECG (360 Hz, 10 minutes = 216,000 samples):

| Stage          | Time (ms) | Memory    | Notes                    |
|----------------|-----------|-----------|--------------------------|
| PELT           | ~50       | O(T)      | Bottleneck if jump=1     |
| Lighthouse     | ~5        | O(k·w)    | k ≈ 50, w ≈ 10           |
| Filtering      | <1        | O(k)      | Negligible               |
| Seam typing    | ~2        | O(k)      | 3-point stencil per seam |
| Antipodes      | ~100      | O(k·m)    | Optional, expensive      |
| **Total**      | ~150 ms   | O(T + k·m)| Real-time capable        |

**Speedup tricks**:
- Use `jump=5` in PELT (5× faster, minimal accuracy loss)
- Skip antipodes on low-budget devices
- Cache cost computations for repeated queries

---

## Future Extensions

### Multi-dimensional Signals

For IMU (x, y, z) or multi-lead ECG:
- Run PELT on principal component or norm(x, y, z)
- Compute antipodes in full d-dimensional space
- Householder reflections replace scalar flips

### Streaming Mode

For online processing:
- Maintain sliding window of context
- Run PELT on chunks
- Update seam database incrementally

### Hierarchical Seams

For multi-scale structure:
- Run PELT at multiple penalties
- Build tree: coarse seams → fine seams
- Prune tree via MDL

---

## References

1. Killick, R., Fearnhead, P., & Eckley, I. A. (2012). Optimal detection of changepoints with a linear computational cost. *Journal of the American Statistical Association*.

2. Rissanen, J. (1978). Modeling by shortest data description. *Automatica*.

3. Truong, C., Oudre, L., & Vayatis, N. (2020). Selective review of offline change point detection methods. *Signal Processing*.

---

**Next steps**: Run `examples/ecg_pilot.py` or `examples/synthetic_data_generator.py` to see the pipeline in action!
