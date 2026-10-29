# ScatteringTransforms.jl

[![Build Status](https://github.com/jbphyswx/ScatteringTransforms.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/jbphyswx/ScatteringTransforms.jl/actions/workflows/CI.yml)
[![Dev Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://jbphyswx.github.io/ScatteringTransforms.jl/dev/)
[![Coverage](https://codecov.io/gh/jbphyswx/ScatteringTransforms.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/jbphyswx/ScatteringTransforms.jl)

Fast, generic wavelet scattering transforms in Julia.

**[Documentation](https://jbphyswx.github.io/ScatteringTransforms.jl/dev/)** —
[API reference](https://jbphyswx.github.io/ScatteringTransforms.jl/dev/api/) ·
[theory](https://jbphyswx.github.io/ScatteringTransforms.jl/dev/theory/)

## Features

- **1D / 2D / 3D**: signals, images, and volumes with oriented Morlet wavelets.
- **Two outputs**: globally-averaged coefficients (`st(x)`) and the localized (Mallat) field
  `scattering_field(st, x) = (|U_p x| ⋆ φ_J) ↓ s` (their spatial means agree by construction).
- **Correct path structure**: second order over strictly coarser scales, all orientation pairs
  (enumerated once into a `ScatteringTree`).
- **Reduced descriptors**: normalized (`S1/S0`, `S2/S1`), log, and 2D sparsity `s₂₁` /
  anisotropy `s₂₂` (`compute_shape_sparsity`).
- **Reconstruction**: exact linear wavelet-frame inverse (`wavelet_transform`/`iwavelet`), phase
  retrieval (`reconstruct_phase`), and gradient-descent `synthesize` from coefficients
  (DifferentiationInterface extension; any `ADTypes` backend — with Enzyme, pass
  `AutoEnzyme(; mode = Enzyme.set_runtime_activity(Enzyme.Reverse))`).
- **Monogenic (Riesz) scattering**: `MonogenicScatteringTransform` (1D/2D/3D) with the rotation-covariant
  monogenic amplitude + continuous orientation/phase (`monogenic_components`); on S²,
  `spherical_monogenic_scattering` (amplitude from the exact surface-gradient energy) plus
  `spherical_monogenic_components` for pointwise orientation/phase (spin-1 Riesz vector, dependency-free
  via the surface gradient, or NUFSHT's spin-weighted synthesis when loaded).
- **Grid-support matrix**: Cartesian × spherical, on uniform/structured and nonuniform/scattered
  sampling — gridded `ScatteringTransform{D}` (FFT), scattered-planar `scattered_planar_scattering`
  (exact direct NUDFT), structured-sphere `structured_spherical_scattering` (exact direct SHT), and
  scattered-sphere `spherical_scattering` (exact direct SHT). **Every cell has an in-core,
  dependency-free default**; `using NonuniformFFTs` or `using FINUFFT` / `using NUFSHT` /
  `using FastSphericalHarmonics` add faster paths selected via the `transform` keyword.
- **Pluggable spectral backend**: dependency-free direct-sum default; `using FFTW` switches on
  an `O(N log N)` fast path automatically
  (`transform = AutoSpectralBackend() | DirectSumSpectralBackend() | FFTSpectralBackend()`).
- **Batching & threading**: `scattering_batch` retains plans and workspaces for gridded 1D/2D/3D,
  monogenic, scattered-planar, and spherical transforms; `using OhMyThreads` enables
  `scattering_batch(…; execution = ThreadedBackend())`, equal to serial bit for bit on the gridded
  surfaces and to the solve tolerance on the scattered ones.
- **Periodized cascade**: every convolution is produced directly on the decimated grid its band
  needs (`oversampling` keyword), exactly — `oversampling ≥ J` reproduces the undecimated cascade
  bit for bit, and lower values trade accuracy for speed.
- **Vendor-neutral GPU**: `using KernelAbstractions, AbstractFFTs` + a device backend (e.g.
  `using CUDA`) gives a device-resident transform via `GPUBackend(CUDA.CUDABackend())` and a
  batched-throughput `scattering_batch(st, X; execution = gpu)`.
  Covers the gridded 1D/2D/3D surfaces (with their localized fields and the wavelet-frame inverse),
  `MonogenicScatteringTransform`, and scattered planar and spherical transforms. On scattered points everything
  downstream of the points follows their array type — bank, buffers, cascade — so the same cascade
  runs on the device, and only the NUFFT is bound per vendor: `NonuniformFFTs` is
  KernelAbstractions-native and runs on any of them, `cuFINUFFT` is bound for CUDA. The
  array paths are tested with JLArrays and scalar indexing disabled. Hardware GPU performance is
  outside these tests. The FastSphericalHarmonics structured-grid route runs on the host, since
  `FastSphericalHarmonics` has no device path.
- **Generic & type-stable**: `Float32`/`Float64`, autodiff-friendly, GPU-array-ready hot path,
  in-place `!` methods over pre-allocated buffers.

## Quick Start

```julia
using ScatteringTransforms: ScatteringTransforms as ST
using ComputationalBackends: ComputationalBackends as CB
using FFTW: FFTW                # → automatic O(N log N) fast path
using OhMyThreads: OhMyThreads  # → ST.scattering_batch(st, X; execution = CB.ThreadedBackend())

# 1D Scattering
N = 1024
signal = randn(N)
st = ST.ScatteringCore.ScatteringTransform(N, 8; Q=1, max_order=2)
coeffs = st(signal)

ST.Coefficients.zeroth_order(coeffs)   # 0th order (average)
ST.Coefficients.first_order(coeffs)    # 1st order (scale amplitudes)
ST.Coefficients.second_order(coeffs)   # 2nd order (scale interactions)

# 2D Scattering
image = randn(256, 256)
st2d = ST.ScatteringCore.ScatteringTransform((256, 256), 4; L=8, max_order=2)
coeffs_2d = st2d(image)

# 3D volumetric scattering
st3d = ST.ScatteringCore.ScatteringTransform((32, 32, 32), 3; n_orient=6, max_order=2)
coeffs_3d = st3d(randn(32, 32, 32))

# Localized (Mallat) field, reduced descriptors, batching
field  = ST.ScatteringFields.scattering_field(st, signal)   # per-path low-passed, subsampled maps
norm   = ST.Reductions.normalized_coefficients(coeffs)      # s1 = S1/S0, s2 = S2/S1
X      = randn(N, 100)
batch  = ST.scattering_batch(st, X)                         # (coeffs × 100), one plan reused
ST.scattering_batch(st, X; execution = CB.ThreadedBackend()) == batch
```

## Why scattering? (the headline result)

Two signals can have **identical power spectra** yet very different structure. The scattering
transform tells them apart: an intermittent signal (sparse bursts) has strong **cross-scale
coupling** `S₂/S₁`, while its phase-randomized Gaussian surrogate does not — even though their
power spectra (and first-order `S₁`) are the same.

![Discriminability](docs/src/assets/discriminability.png)

## Visualizations

### Filter bank and its Littlewood–Paley sum
1D Morlet filter bank in the frequency domain. For a real field the Littlewood–Paley sum
`A(ω) = |φ̂|² + ½Σⱼ(|ψ̂ⱼ(ω)|² + |ψ̂ⱼ(−ω)|²)` lies in `(0, 1]`, so the wavelet layer amplifies no
frequency and is invertible (Andén & Mallat 2014).

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

### Reconstruction & synthesis
The complex (pre-modulus) wavelet layer is exactly invertible (`iwavelet`, machine precision);
from the scattering *coefficients*, `synthesize` (DifferentiationInterface + an AD backend)
descends `‖S(x̂)−S(x)‖²` to draw a new sample with matching multiscale statistics — 1D and 2D.

![Reconstruction & synthesis (1D)](docs/src/assets/reconstruction_synthesis.png)
![Reconstruction & synthesis (2D)](docs/src/assets/reconstruction_2d.png)

### Monogenic (Riesz) scattering
The rotation-covariant monogenic amplitude replaces the oriented modulus; `monogenic_components`
recovers the local amplitude envelope and a **continuous** orientation (a radial pinwheel here),
not quantized orientation bins.

![Monogenic analysis](docs/src/assets/monogenic.png)

### Nonuniform / scattered planar grids
Off-lattice / gappy planar data (scattered `(x, y)` points) is analysed onto a uniform Fourier mode
grid by a nonuniform DFT (`scattered_planar_scattering`), and the gridded transform's cascade, with its
Morlet wavelet bank, runs on that grid.
The caller gives the domain's size, `period`, which sets every mode's wavelength, or a FlowGeometries
grid that carries it. On a uniform grid it reproduces the gridded FFT transform to transform accuracy.
`analysis = ST.Plans.LeastSquares()` recovers the
band-limited coefficients for irregular sampling by LSMR least squares — minimum-norm where the
samples do not determine the modes, with `damp` for explicit Tikhonov regularisation. The default is
an in-core exact direct NUDFT (no dependencies); `using NonuniformFFTs` or `using FINUFFT` enables the
nonuniform FFT, and with both loaded `transform = FlowTransformBindings.NonuniformFFTsBackend()` or
`FlowTransformBindings.FINUFFTBackend()` names one.

The default analysis is `ST.Plans.WeightedAdjoint()`. `tol` sets NUFFT accuracy and `nthreads`
sets its library thread count. A FlowGeometries periodic axis supplies its period; a bounded axis
uses its extent plus the coarsest wavelet width. An explicit `period` overrides these values.

![Scattered planar scattering](docs/src/assets/scattered_planar.png)

### Spherical scattering
On S², both **scattered points** (`spherical_scattering`; in-core direct SHT by default, NUFSHT fast
path with `using NUFSHT`) and a **structured** equiangular grid (`structured_spherical_scattering`,
via the fast SHT in FastSphericalHarmonics) give matching multi-scale coefficients. Scattered samples
are analysed once, and the cascade runs on the Gauss–Legendre grid of the band limit, so each
coefficient is that grid's mean of a modulus of the field the analysis recovers:

Spherical results use `ST.Coefficients.ScatteringCoefficients1D`, with scale 1 finest and
`S2[j1, j2]` populated for `j2 > j1`. The scattered constructors also accept a spherical
FlowGeometries grid. Their default `LeastSquares()` input fit returns its best iterate at an
iteration or condition limit; diagnostics stay in the fit workspace.

![Spherical scattering](docs/src/assets/spherical_scattering.png)
![Structured spherical scattering](docs/src/assets/structured_spherical.png)

The monogenic *amplitude* takes the spin-1 Riesz energy `|∇_S g|²` from three scalar syntheses of
`g`'s coefficients, exact at the band limit; pointwise orientation/phase
(`spherical_monogenic_components`) synthesizes the spin-1 Riesz tangent vector on S².

![Spherical monogenic components](docs/src/assets/spherical_monogenic.png)

### Spectral backends
The in-core direct-sum default is dependency-free but `O(N²)`; loading `FFTW` switches on an
`O(N log N)` fast path automatically, with the same results to rounding.

![Backend performance](docs/src/assets/backend_performance.png)

## Documentation

- [Documentation](https://jbphyswx.github.io/ScatteringTransforms.jl/dev/)
- [Theory](docs/src/theory.md) - Mathematical background
- [API Reference](docs/src/api.md) - Function documentation

## Examples

See the [examples/](examples/) directory:

- `basic_usage.jl` - 1D/2D/3D scattering, localized field, reductions, batching/threading
- `zero_allocation_streaming.jl` - High-performance streaming for large datasets
- `backends.jl` - spectral (direct-sum vs FFTW) and compute (serial/threaded/GPU) backends
- `synthesis_and_inverse.jl` - exact inverse, phase retrieval, and coefficient synthesis
- `monogenic.jl` - monogenic (Riesz) scattering: amplitude envelope + continuous orientation

Run examples:
```bash
cd examples
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. basic_usage.jl
```

## Zero-Allocation Streaming

For large datasets, use the in-place API:

```julia
st = ST.ScatteringCore.ScatteringTransform(N, 8; Q=1, max_order=2)
coeffs = ST.Coefficients.ScatteringCoefficients1D(length(st.filter_bank.wavelets), Float64)

for slice in dataset
    coeffs = ST.Scattering1D.scattering_transform!(coeffs, st, slice)
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
