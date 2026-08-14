# API Reference

Symbols are accessed via fully-qualified submodule paths (the package does not re-export names into
its top-level namespace); `using ScatteringTransforms: ScatteringTransforms as ST`, then e.g.
`ST.Scattering2D.ScatteringTransform2D(...)`.

## Grid-support matrix

Planar (Cartesian) and spherical scattering, on uniform/structured and nonuniform/scattered sampling:

| domain | uniform / structured | nonuniform / scattered |
|---|---|---|
| Cartesian | `ScatteringTransform{1,2,3}D` (FFT/direct sum; GPU via `GPUBackend`) | `scattered_planar_scattering` (exact direct NUDFT; FINUFFT fast path) |
| Sphere (S²) | `structured_spherical_scattering` (exact direct SHT; FastSphericalHarmonics fast path) | `spherical_scattering` (exact direct SHT; NUFSHT fast path) |

**Every cell has an in-core, dependency-free default** (direct summation), with an optional fast path
selected by the `spectral` keyword, which takes a
[SpectralBackends.jl](https://github.com/jbphyswx/SpectralBackends.jl) tag. `DirectSumSpectralBackend`
is the in-core default everywhere; the fast paths are `FFTSpectralBackend` (FFTW) on a grid,
`NUFFTSpectralBackend` (FINUFFT or NonuniformFFTs) for scattered points, `NUFSHTSpectralBackend`
(NUFSHT) for the scattered sphere, and `FSHTSpectralBackend` (FastSphericalHarmonics) for the
structured sphere. `AutoSpectralBackend` (the default) picks the fast path if its extension is
loaded, else the direct sum — so nothing requires an external library. Naming a backend explicitly is
honoured exactly: if its extension is absent it raises rather than silently downgrading.

Monogenic (Riesz) variants exist on both sphere paths; see below (pointwise
`spherical_monogenic_components` additionally needs the NUFSHT spin-1 synthesis).

## Transforms

```@docs
ScatteringTransforms.Scattering1D.ScatteringTransform1D
ScatteringTransforms.Scattering2D.ScatteringTransform2D
ScatteringTransforms.Scattering3D.ScatteringTransform3D
ScatteringTransforms.Scattering1D.scattering_transform!
ScatteringTransforms.Scattering2D.scattering_transform2d!
ScatteringTransforms.Scattering3D.scattering_transform3d!
ScatteringTransforms.ScatteringCore.scattering
ScatteringTransforms.scattered_planar_scattering
ScatteringTransforms.spherical_scattering
ScatteringTransforms.structured_spherical_scattering
ScatteringTransforms.structured_sphere_points
```

## Reconstruction & synthesis

```@docs
ScatteringTransforms.Inverse.wavelet_transform
ScatteringTransforms.Inverse.iwavelet
ScatteringTransforms.Inverse.reconstruct_phase
ScatteringTransforms.synthesize
ScatteringTransforms.scattering_loss
```

## Monogenic (Riesz) scattering

```@docs
ScatteringTransforms.Monogenic.MonogenicScattering
ScatteringTransforms.Monogenic.MonogenicFilterBank
ScatteringTransforms.Monogenic.build_monogenic_bank
ScatteringTransforms.Monogenic.riesz_multipliers
ScatteringTransforms.Monogenic.monogenic_amplitude
ScatteringTransforms.Monogenic.monogenic_components
ScatteringTransforms.spherical_monogenic_scattering
ScatteringTransforms.spherical_monogenic_components
ScatteringTransforms.structured_spherical_monogenic_scattering
```

## Localized (Mallat) field

```@docs
ScatteringTransforms.ScatteringFields.scattering_field
ScatteringTransforms.ScatteringFields.scattering_field!
ScatteringTransforms.ScatteringFields.ScatteringField1D
ScatteringTransforms.ScatteringFields.ScatteringField2D
ScatteringTransforms.ScatteringFields.path_field
```

## Coefficients & reductions

```@docs
ScatteringTransforms.Coefficients.ScatteringCoefficients1D
ScatteringTransforms.Coefficients.ScatteringCoefficients2D
ScatteringTransforms.Coefficients.zeroth_order
ScatteringTransforms.Coefficients.first_order
ScatteringTransforms.Coefficients.second_order
ScatteringTransforms.Coefficients.flatten1d
ScatteringTransforms.Coefficients.flatten2d
ScatteringTransforms.Coefficients.flatten_length
ScatteringTransforms.Scattering2D.compute_shape_sparsity
ScatteringTransforms.Reductions.normalized_coefficients
ScatteringTransforms.Reductions.log_coefficients
```

## Batching & backends

Where a transform runs is chosen with a backend from
[ComputationalBackends.jl](https://github.com/jbphyswx/ComputationalBackends.jl) — `SerialBackend`,
`ThreadedBackend`, `GPUBackend`, `DistributedBackend`, `MPIBackend`, `AutoBackend` — passed as the
second argument to `scattering_batch`. Which spectral algorithm it uses is chosen with a
[SpectralBackends.jl](https://github.com/jbphyswx/SpectralBackends.jl) tag passed as `spectral=` at
construction. Each is honoured exactly: naming a backend whose extension is not loaded raises rather
than falling back.

```@docs
ScatteringTransforms.scattering_batch
ScatteringTransforms.scattering_batch!
ScatteringTransforms.batch_coeffs
ScatteringTransforms.flat_rows
ScatteringTransforms.batch_workspace
ScatteringTransforms.Batched.BatchWorkspace
ScatteringTransforms.Batched.batch_cascade!
```

## Multi-resolution second order

The second order runs on a decimated grid, which is where most of a transform's work is. Opt-in and
approximate; `oversampling` converges it to the exact transform.

```@docs
ScatteringTransforms.SubsampledScattering.MultiResolutionScattering
ScatteringTransforms.SubsampledScattering.SubsampledScattering1D
ScatteringTransforms.SubsampledScattering.SubsampledScattering2D
ScatteringTransforms.SubsampledScattering.SubsampledScattering3D
ScatteringTransforms.SubsampledScattering.subsampled_scattering!
```

## Filter banks, filters & path graph

```@docs
ScatteringTransforms.FilterBanks.FilterBank1D
ScatteringTransforms.FilterBanks.FilterBank2D
ScatteringTransforms.FilterBanks.FilterBank3D
ScatteringTransforms.FilterBanks.WaveletMeta
ScatteringTransforms.FilterBanks.build_filter_bank1d
ScatteringTransforms.FilterBanks.build_filter_bank2d
ScatteringTransforms.FilterBanks.build_filter_bank3d
ScatteringTransforms.Filters.Morlet1D
ScatteringTransforms.Filters.Morlet2D
ScatteringTransforms.Filters.Morlet3D
ScatteringTransforms.Filters.frequency_response
ScatteringTransforms.Filters.fibonacci_directions
ScatteringTransforms.PathGraph.ScatteringTree
ScatteringTransforms.PathGraph.build_tree
```

## Spectral plans & core operations

```@docs
ScatteringTransforms.Plans.AbstractScatteringPlan
ScatteringTransforms.Plans.DirectSumPlan
ScatteringTransforms.Plans.forward_transform!
ScatteringTransforms.Plans.inverse_transform!
ScatteringTransforms.Plans.forward_transform
ScatteringTransforms.Plans.inverse_transform
ScatteringTransforms.Plans.make_plan
ScatteringTransforms.ScatteringCore.wavelet_convolve!
ScatteringTransforms.ScatteringCore.apply_modulus!
ScatteringTransforms.ScatteringCore.spatial_average
```
