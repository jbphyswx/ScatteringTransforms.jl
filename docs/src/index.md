# ScatteringTransforms.jl

A fast, generic Julia implementation of the wavelet scattering transform for **1D signals, 2D
images, 3D volumes, and scalar fields on the sphere** — with averaged and localized outputs,
reduced descriptors, a dependency-free default plus pluggable fast backends, batching, threading,
distributed/MPI, and GPU.

## Why scattering?

Two signals can share an **identical power spectrum** yet differ structurally. Scattering
distinguishes them: an intermittent signal has strong cross-scale coupling `S₂/S₁`, while its
phase-randomized Gaussian surrogate does not — though their power spectra and first-order `S₁`
are the same.

![Discriminability](assets/discriminability.png)

## Features

- **Domains**: 1D, 2D (oriented Morlet), 3D (oriented 3D Morlet), and spherical (NUFSHT,
  smooth difference-of-Gaussians bands).
- **Two outputs**: globally-averaged coefficients `st(x)` and the localized (Mallat) field
  `scattering_field(st, x) = (|U_p x| ⋆ φ_J)↓` (their spatial means agree by construction).
- **Correct path structure**: second order over strictly coarser scales, all orientation pairs.
- **Reduced descriptors**: `normalized_coefficients` (`S1/S0`, `S2/S1`), `log_coefficients`,
  and 2D sparsity `s₂₁` / anisotropy `s₂₂` (`compute_shape_sparsity`).
- **Tight-frame filter bank**: `|φ|² + Σⱼ|ψⱼ|² ≡ 1` (non-expansive).
- **Pluggable spectral backend**: in-core direct-sum default; `using FFTW` → `O(N log N)`
  fast path (`spectral = :auto | :direct | :fftw`).
- **Scale**: `scattering_batch` (one plan reused), `ThreadedBackend` (OhMyThreads),
  `DistributedBackend`/`MPIBackend` (parametric over the inner backend), `GPUBackend` (CUDA).
- **Generic & type-stable**: `Float32`/`Float64`, autodiff-friendly, GPU-array-ready hot path.

## Quick start

```julia
using ScatteringTransforms
using FFTW   # optional: switches on the O(N log N) fast path automatically

# 1D
st = ScatteringTransform1D(1024, 8; Q=1, max_order=2)
c  = st(randn(1024))
zeroth_order(c); first_order(c); second_order(c)

# 2D / 3D
c2 = ScatteringTransform2D((256, 256), 4; L=8)(randn(256, 256))
c3 = ScatteringTransform3D((32, 32, 32), 3; n_orient=6)(randn(32, 32, 32))

# localized field, reductions, batching
field = scattering_field(st, randn(1024))      # per-path low-passed, subsampled maps
red   = normalized_coefficients(c)             # s1 = S1/S0, s2 = S2/S1
B     = scattering_batch(st, randn(1024, 100)) # (coeffs × 100), one plan reused
```

### Filter bank (tight frame)
![Filter bank](assets/filter_bank.png)

### 1D and 2D scattering
![1D scattering](assets/1d_scattering_example.png)
![2D scattering](assets/2d_scattering_example.png)

### Localized field and reductions
![Localized field](assets/localized_field.png)
![Reductions](assets/reductions.png)

## Backends & scale

The in-core direct-sum transform is dependency-free; loading `FFTW` selects an `O(N log N)`
fast path automatically (identical results). `scattering_batch` reuses one plan/workspace across
a stack; `using OhMyThreads`, `using Distributed`, or `using MPI` enable
`scattering_batch(ThreadedBackend(), …)`, `DistributedBackend(…)`, and `MPIBackend(…)`.

![Backend performance](assets/backend_performance.png)

## Documentation

- [Theory](theory.md) — mathematical background
- [API Reference](api.md) — functions and types

## Citation

```bibtex
@software{scatteringtransforms_jl,
  author = {Benjamin, Jordan},
  title = {ScatteringTransforms.jl: wavelet scattering in Julia},
  url = {https://github.com/jbphyswx/ScatteringTransforms.jl}
}
```
