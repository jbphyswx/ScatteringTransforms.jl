# ScatteringTransforms.jl

[![Build Status](https://github.com/jbphyswx/ScatteringTransforms.jl/workflows/CI/badge.svg)](https://github.com/jbphyswx/ScatteringTransforms.jl/actions)

Fast, generic wavelet scattering transforms in Julia.

## Features

- **1D / 2D / 3D**: signals, images, and volumes with oriented Morlet wavelets.
- **Two outputs**: globally-averaged coefficients (`st(x)`) and the localized (Mallat) field
  `scattering_field(st, x) = (|U_p x| ⋆ φ_J) ↓ s` (their spatial means agree by construction).
- **Correct path structure**: second order over strictly coarser scales, all orientation pairs
  (enumerated once into a `ScatteringTree`).
- **Reduced descriptors**: normalized (`S1/S0`, `S2/S1`), log, and 2D sparsity `s₂₁` /
  anisotropy `s₂₂` (`compute_shape_sparsity`).
- **Pluggable spectral backend**: dependency-free direct-sum default; `using FFTW` switches on
  an `O(N log N)` fast path automatically (`spectral = :auto | :direct | :fftw`).
- **Batching & threading**: `scattering_batch` reuses one plan/workspace; `using OhMyThreads`
  enables `scattering_batch(ThreadedCPU(), …)`.
- **Generic & type-stable**: `Float32`/`Float64`, autodiff-friendly, GPU-array-ready hot path,
  in-place `!` methods over pre-allocated buffers.

## Quick Start

```julia
using ScatteringTransforms

# 1D Scattering
N = 1024
signal = randn(N)
st = ScatteringTransform1D(N, 8; Q=1, max_order=2)
coeffs = st(signal)

@show coeffs.S0  # 0th order (average)
@show coeffs.S1  # 1st order (scale amplitudes)
@show coeffs.S2  # 2nd order (scale interactions)

# 2D Scattering
image = randn(256, 256)
st2d = ScatteringTransform2D((256, 256), 4; L=8, max_order=2)
coeffs_2d = st2d(image)

# 3D volumetric scattering
st3d = ScatteringTransform3D((32, 32, 32), 3; n_orient=6, max_order=2)
coeffs_3d = st3d(randn(32, 32, 32))

# Localized (Mallat) field, reduced descriptors, batching
field  = scattering_field(st, signal)              # per-path low-passed, subsampled maps
norm   = normalized_coefficients(coeffs)           # s1 = S1/S0, s2 = S2/S1
batch  = scattering_batch(st, randn(N, 100))       # (coeffs × 100), one plan reused

using FFTW         # → automatic O(N log N) fast path
using OhMyThreads  # → scattering_batch(ThreadedBackend(), st, X)
```

## Why scattering? (the headline result)

Two signals can have **identical power spectra** yet very different structure. The scattering
transform tells them apart: an intermittent signal (sparse bursts) has strong **cross-scale
coupling** `S₂/S₁`, while its phase-randomized Gaussian surrogate does not — even though their
power spectra (and first-order `S₁`) are the same.

![Discriminability](docs/src/assets/discriminability.png)

## Visualizations

### Filter bank — a tight frame
1D Morlet filter bank in the frequency domain. The Littlewood–Paley sum `|φ|² + Σⱼ|ψⱼ|²` is
flat at **1** (a tight frame ⇒ the transform is non-expansive — no frequency is amplified).

![Morlet Filter Bank](docs/src/assets/filter_bank.png)

### 1D scattering
A 1/f (turbulent) signal with its `S₀`, first-order `S₁(j)`, and second-order `S₂(j₁,j₂)`
(only admissible strictly-coarser scale pairs are populated).

![1D Scattering Example](docs/src/assets/1d_scattering_example.png)

### 2D scattering
A turbulent 2D field with `S₁` over scale × orientation and `S₂` over wavelet pairs. The `S₂`
panel is a `(J·L)²` matrix whose only populated blocks are **strictly coarser scale pairs**
(all orientation pairs); the same-scale diagonal blocks are empty by construction.

![2D Scattering Example](docs/src/assets/2d_scattering_example.png)

### Localized (Mallat) field
`scattering_field` returns `S_p x = (|U_p x| ⋆ φ_J)↓` per path — energy localized in **scale
and space** (here: two bursts at different scales light up different rows/positions).

![Localized field](docs/src/assets/localized_field.png)

### Reduced descriptors — anisotropy
The shape reduction `s₂₂` (second angular harmonic of `S₂`) cleanly separates an oriented
texture from an isotropic one.

![Reductions](docs/src/assets/reductions.png)

### Spectral backends
The in-core direct-sum default is dependency-free but `O(N²)`; loading `FFTW` switches on an
`O(N log N)` fast path automatically (≈100–1000× faster), with identical results.

![Backend performance](docs/src/assets/backend_performance.png)

## Documentation

- [Documentation](https://jbphyswx.github.io/ScatteringTransforms.jl/dev/)
- [Theory](docs/src/theory.md) - Mathematical background
- [API Reference](docs/src/api.md) - Function documentation

## Examples

See the [examples/](examples/) directory:

- `basic_usage.jl` - Getting started with 1D and 2D scattering
- `zero_allocation_streaming.jl` - High-performance streaming for large datasets
- `gpu_acceleration.jl` - GPU-ready type system demonstration

Run examples:
```bash
cd examples
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. basic_usage.jl
```

## Zero-Allocation Streaming

For large datasets (e.g., ocean data), use the in-place API:

```julia
st = ScatteringTransform1D(N, 8; Q=1, max_order=2)
coeffs = ScatteringCoefficients1D(length(st.filter_bank.wavelets), Float64)

for slice in dataset
    coeffs = scattering_transform!(coeffs, st, slice)
    process(coeffs)
end
```

## References

- Mallat (2012): Group invariant scattering. *Comm. Pure Appl. Math.*
- Bruna & Mallat (2013): Invariant Scattering Convolution Networks. *IEEE PAMI*
- Cheng & Ménard (2021): [How to quantify fields or textures?](https://arxiv.org/pdf/2112.01288)
- Related packages: [scattering_transform](https://github.com/SihaoCheng/scattering_transform), [ScatteringTransform.jl](https://github.com/dsweber2/ScatteringTransform.jl)

## Citation

```bibtex
@software{scatteringtransforms_jl,
  author = {Benjamin, Jordan},
  title = {ScatteringTransforms.jl: Fast wavelet scattering in Julia},
  url = {https://github.com/jbphyswx/ScatteringTransforms.jl}
}
```
