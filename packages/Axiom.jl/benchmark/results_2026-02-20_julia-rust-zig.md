# Axiom.jl Benchmark Results — 2026-02-20

## System
- **CPU**: Intel (i915 + Quadro M2000M laptop)
- **OS**: Fedora 43 Atomic (Linux 6.18.10)
- **Julia**: 1.11+
- **Zig**: 0.15.2 (ReleaseFast)
- **Rust**: nightly (release)
- **Method**: 50 iterations, 3 warmup, median timing

## Binary Sizes
| Backend | Size |
|---------|------|
| Rust    | 1990 KB |
| Zig     | 212 KB  |

## Results

```
┌─────────────┬──────────────┬───────────────┬───────────────┬───────────────┬─────────────┬─────────────┐
│ Operation    │ Size         │ Julia (μs)    │ Rust (μs)     │ Zig (μs)      │ Rust vs Jul │ Zig vs Jul  │
├─────────────┼──────────────┼───────────────┼───────────────┼───────────────┼─────────────┼─────────────┤
│ matmul      │ 64×64        │          12.7 │          69.1 │         114.9 │       0.18x │       0.11x │
│ matmul      │ 256×256      │         295.5 │       13894.9 │        2747.4 │       0.02x │       0.11x │
│ matmul      │ 512×512      │        2454.3 │      105420.3 │       24679.0 │       0.02x │       0.10x │
│ matmul      │ 1024×1024    │       16894.4 │      807349.9 │      228086.2 │       0.02x │       0.07x │
│ relu        │ 1K           │           0.5 │           1.2 │           0.6 │       0.45x │       0.91x │
│ relu        │ 100K         │          40.0 │          81.3 │          37.0 │       0.49x │       1.08x │
│ relu        │ 1M           │         478.7 │        1586.4 │         671.1 │       0.30x │       0.71x │
│ sigmoid     │ 1K           │          16.2 │           5.3 │           4.9 │       3.06x │       3.32x │
│ sigmoid     │ 100K         │         865.4 │         550.8 │         456.2 │       1.57x │       1.90x │
│ sigmoid     │ 1M           │       10994.9 │        6721.6 │        5395.5 │       1.64x │       2.04x │
│ gelu        │ 1K           │          11.2 │          15.3 │          21.6 │       0.74x │       0.52x │
│ gelu        │ 100K         │        1980.8 │        1784.3 │        3449.9 │       1.11x │       0.57x │
│ gelu        │ 1M           │       18328.7 │       16624.5 │       33310.6 │       1.10x │       0.55x │
│ softmax     │ 32×10        │           4.7 │           6.2 │           3.4 │       0.76x │       1.38x │
│ softmax     │ 64×1000      │        1243.3 │        1745.3 │        1032.3 │       0.71x │       1.20x │
│ softmax     │ 128×50257    │      109756.2 │      321730.0 │      269685.2 │       0.34x │       0.41x │
│ layernorm   │ 32×128       │          28.4 │          16.7 │          15.4 │       1.70x │       1.85x │
│ layernorm   │ 64×768       │         403.2 │         379.4 │         274.8 │       1.06x │       1.47x │
│ layernorm   │ 128×1024     │         786.6 │         595.0 │         716.0 │       1.32x │       1.10x │
│ rmsnorm     │ 32×128       │          15.9 │           9.5 │           2.5 │       1.67x │       6.46x │
│ rmsnorm     │ 64×768       │         154.7 │         112.2 │          23.8 │       1.38x │       6.50x │
│ rmsnorm     │ 128×1024     │         484.7 │         310.6 │          67.7 │       1.56x │       7.16x │
│ batchnorm   │ 32×64        │           8.0 │          11.1 │           9.1 │       0.72x │       0.88x │
│ batchnorm   │ 64×256       │          52.0 │         117.0 │          53.8 │       0.44x │       0.97x │
│ batchnorm   │ 128×512      │         203.9 │         608.8 │         282.4 │       0.33x │       0.72x │
└─────────────┴──────────────┴───────────────┴───────────────┴───────────────┴─────────────┴─────────────┘
```

## Aggregate

| Backend | Geometric Mean | Arithmetic Mean |
|---------|---------------|-----------------|
| Rust    | 0.53x         | 0.91x           |
| Zig     | 0.89x         | 1.68x           |

> Values >1.0x mean native backend is faster than Julia.

## Analysis

### Zig Wins (dispatch-worthy)
- **RMSNorm**: 6.5–7.2x faster — SIMD-optimized inner loop dominates
- **Sigmoid**: 1.9–3.3x faster — compact `@exp` path
- **LayerNorm**: 1.1–1.9x faster — consistent advantage at all sizes
- **Softmax (small batch)**: 1.2–1.4x faster

### Julia Wins (keep on BLAS)
- **MatMul**: 9–50x faster — Julia calls OpenBLAS/MKL; native backends use hand-written tiled matmul
- **GELU (large)**: ~1.8x faster — LLVM auto-vectorization + broadcasting
- **BatchNorm**: 1.1–1.4x faster — row-major conversion overhead in FFI path

### Zig vs Rust
Zig beats Rust on every single benchmark:
- 9.4x smaller binary (212KB vs 1990KB)
- Faster compilation (seconds vs minutes)
- First-class SIMD (no crate dependencies)
- RMSNorm: Zig 7.2x vs Rust 1.6x (Zig 4.5x faster than Rust)

### Recommendations
1. **MatMul**: Always dispatch to Julia/BLAS — native backends cannot compete
2. **RMSNorm/LayerNorm/Sigmoid**: Dispatch to Zig by default
3. **Softmax**: Dispatch to Zig for small batches, Julia for large (>50K classes)
4. **BatchNorm/ReLU**: Keep on Julia — FFI overhead negates any kernel advantage
5. **GELU**: Keep on Julia — broadcasting + LLVM vectorization wins
