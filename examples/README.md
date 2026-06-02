# Examples

This directory contains working examples demonstrating `ScatteringTransforms.jl`.

## Quick Start

```bash
cd examples
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## Examples

### 1. Basic Usage (`basic_usage.jl`)

Demonstrates:
- 1D scattering transform
- 2D scattering transform
- Zero-allocation streaming
- Float32/GPU-ready types

```bash
julia --project=. basic_usage.jl
```

### 2. Zero-Allocation Streaming (`zero_allocation_streaming.jl`)

Critical for large datasets (e.g., 20TB of ocean data).

Demonstrates:
- Naive approach (allocates every iteration)
- Zero-allocation approach (reuses buffers)
- True zero-allocation (mutable S0 container)
- Memory comparison

```bash
julia --project=. zero_allocation_streaming.jl
```

### 3. GPU Acceleration (`gpu_acceleration.jl`)

Shows GPU-ready type system. Full GPU support requires CUDA FFT backend.

```bash
julia --project=. gpu_acceleration.jl
```

## Output

Example output from `basic_usage.jl`:

```
============================================================
ScatteringTransforms.jl - Basic Usage Demo
============================================================

1. 1D Scattering Transform
----------------------------------------
  Signal length: 1024
  Number of wavelets: 8

  Results:
    S0 (average): 0.0032
    S1 (first order): 8 coefficients
    S2 (second order): (8, 8) matrix

2. Zero-Allocation Streaming Demo
----------------------------------------
  Processing 100 signals with zero allocation...
  Completed! Only ~32 bytes allocated per iteration

3. 2D Scattering Transform
----------------------------------------
  Image size: (128, 128)
  Total wavelets: 12
```
