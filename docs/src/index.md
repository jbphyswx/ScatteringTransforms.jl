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
- **Reconstruction**: exact linear wavelet-frame inverse (`wavelet_transform`/`iwavelet`),
  phase retrieval (`reconstruct_phase`), and gradient-descent `synthesize` from coefficients
  (DifferentiationInterface extension, any `ADTypes` backend).
- **Monogenic (Riesz) scattering**: `MonogenicScattering` (1D/2D/3D) with the rotation-covariant
  monogenic amplitude + continuous orientation/phase (`monogenic_components`), and
  `spherical_monogenic_scattering` on S².
- **Tight-frame filter bank**: `|φ|² + Σⱼ|ψⱼ|² ≡ 1` (non-expansive).
- **Pluggable spectral backend**: in-core direct-sum default; `using FFTW` → `O(N log N)`
  fast path (`spectral = AutoSpectral() | DirectSumBackend() | FFTBackend()`).
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

### Reconstruction & synthesis
The complex (pre-modulus) wavelet layer is exactly invertible (`iwavelet`, machine precision);
from the scattering *coefficients*, `synthesize` descends `‖S(x̂)−S(x)‖²` (via
DifferentiationInterface) to draw a new sample with matching multiscale statistics — in 1D and 2D.

```julia
using DifferentiationInterface, Mooncake
# exact linear inverse (machine precision)
x̂ = iwavelet(st, wavelet_transform(st, signal))
# coefficient synthesis from noise (a matching sample, not the original field)
res = synthesize(st, signal; backend = AutoMooncake(), iters = 400)
```

![Reconstruction & synthesis (1D)](assets/reconstruction_synthesis.png)
![Reconstruction & synthesis (2D)](assets/reconstruction_2d.png)

### Monogenic (Riesz) scattering
`MonogenicScattering` uses the rotation-covariant monogenic amplitude in place of the oriented
modulus, and `monogenic_components` recovers the local amplitude envelope and a **continuous**
orientation (not quantized into bins).

![Monogenic analysis](assets/monogenic.png)

### Spherical scattering
On S² (scattered points, via NUFSHT) both the analytic and monogenic transforms are available;
the monogenic Riesz energy is computed with spin-0 transforms via a Bochner identity.

![Spherical scattering](assets/spherical_scattering.png)

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
