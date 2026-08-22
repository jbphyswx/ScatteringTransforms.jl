# API Reference

Symbols are accessed via fully-qualified submodule paths (the package does not re-export names into
its top-level namespace); `using ScatteringTransforms: ScatteringTransforms as ST`, then e.g.
`ST.Scattering2D.ScatteringTransform2D(...)`.

```@docs
ScatteringTransforms.ScatteringTransforms
```

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
ScatteringTransforms.Inverse.ReconstructionWorkspace
ScatteringTransforms.Inverse.wavelet_transform!
ScatteringTransforms.Inverse.iwavelet!
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
ScatteringTransforms.ScatteringFields.subsample_factor
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
ScatteringTransforms.Coefficients.flatten1d!
ScatteringTransforms.Coefficients.flatten2d!
ScatteringTransforms.Coefficients.flatten_length
ScatteringTransforms.Coefficients.flat_length
ScatteringTransforms.Coefficients.update_S0
ScatteringTransforms.Scattering2D.compute_shape_sparsity
ScatteringTransforms.Reductions.normalized_coefficients
ScatteringTransforms.Reductions.log_coefficients
```

`flat_length`'s docstring also covers the row accessors `flat_row_s0`, `flat_row_s1` and
`flat_row_s2`, which are the layout `flatten1d!`, `flatten2d!` and the batched paths all walk.

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
ScatteringTransforms.PathGraph.order2_groups
ScatteringTransforms.PathGraph.order_range
ScatteringTransforms.PathGraph.path_indices
ScatteringTransforms.PathGraph.npaths
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
ScatteringTransforms.Plans.make_scattered_plan
ScatteringTransforms.Plans.spectral_backend
ScatteringTransforms.Plans.task_local_plan
ScatteringTransforms.Plans.batch_width
ScatteringTransforms.Plans.plan_points
ScatteringTransforms.Plans.plan_analysis
ScatteringTransforms.Plans.nufft_guru_make
ScatteringTransforms.Plans.with_fft_nthreads
ScatteringTransforms.Plans.PLANNER_LOCK
ScatteringTransforms.ScatteringCore.wavelet_convolve
ScatteringTransforms.ScatteringCore.wavelet_convolve!
ScatteringTransforms.ScatteringCore.apply_modulus
ScatteringTransforms.ScatteringCore.apply_modulus!
ScatteringTransforms.ScatteringCore.modulus_mean
ScatteringTransforms.ScatteringCore.modulus_mean!
ScatteringTransforms.ScatteringCore.spatial_average
ScatteringTransforms.ScatteringCore.task_workspace
```

### Plans supplied by extensions

These are declared in the core and given a method by the corresponding extension. Calling one
without its package loaded raises with the `using` line to run.

```@docs
ScatteringTransforms.Plans.fftw_plan
ScatteringTransforms.Plans.abstractffts_plan
ScatteringTransforms.Plans.finufft_scattered_plan
ScatteringTransforms.Plans.nonuniformffts_scattered_plan
ScatteringTransforms.Plans.FINUFFTBackend
ScatteringTransforms.Plans.NonuniformFFTsBackend
```

## Execution backends

```@docs
ScatteringTransforms.Execution.resolve_backend
ScatteringTransforms.Execution.check_available
ScatteringTransforms.Execution.have_threads
ScatteringTransforms.Execution.have_gpu
ScatteringTransforms.Execution.have_distributed
ScatteringTransforms.Execution.have_mpi
```

## Cascade internals

The cascade each gridded transform runs, and the per-surface in-place entry points.

```@docs
ScatteringTransforms.Scattering1D.cascade!
ScatteringTransforms.Scattering2D.cascade!
ScatteringTransforms.Scattering3D.cascade!
ScatteringTransforms.ScatteredPlanar.ScatteredPlanarScattering
ScatteringTransforms.ScatteredPlanar.scattered_planar_scattering!
ScatteringTransforms.ScatteredPlanar.scattered_planar_scattering_batch!
ScatteringTransforms.SubsampledScattering.Level
```

## Spherical scattering

A spherical plan implements three primitives — analysis, multiply-and-synthesise, and the spherical
mean — and everything above them is shared. The in-core direct plan is always available; NUFSHT and
FastSphericalHarmonics supply the fast ones.

```@docs
ScatteringTransforms.SphericalCore.AbstractSphericalPlan
ScatteringTransforms.SphericalCore.sphere_coeffs
ScatteringTransforms.SphericalCore.sphere_coeffs!
ScatteringTransforms.SphericalCore.sphere_coeffs_buffer
ScatteringTransforms.SphericalCore.sphere_apply!
ScatteringTransforms.SphericalCore.sphere_mean
ScatteringTransforms.SphericalCore.SphericalScattering
ScatteringTransforms.SphericalCore.SphericalMonogenicScattering
ScatteringTransforms.SphericalCore.SphericalWorkspace
ScatteringTransforms.SphericalCore.SphericalMonogenicWorkspace
ScatteringTransforms.SphericalCore.spherical_scattering!
ScatteringTransforms.SphericalCore.spherical_scattering_batch!
ScatteringTransforms.SphericalCore.spherical_monogenic_scattering!
ScatteringTransforms.SphericalCore.monogenic_amplitude!
ScatteringTransforms.SphericalCore.task_local
ScatteringTransforms.SphericalCore.band_multiplier
ScatteringTransforms.SphericalCore.dog_sigma2
ScatteringTransforms.SphericalCore.structured_grid
ScatteringTransforms.SphericalCore.plan_points
ScatteringTransforms.SphericalCore.plan_weights
ScatteringTransforms.SphericalCore.plan_solver
ScatteringTransforms.SphericalCore.batch_plan
ScatteringTransforms.SphericalCore.supports_batch
ScatteringTransforms.SphericalCore.plan_nufft
ScatteringTransforms.SphericalCore.plan_spin
ScatteringTransforms.SphericalCore.AnalysisNotConverged
ScatteringTransforms.SphericalCore.default_rtol
ScatteringTransforms.SphericalCore.with_serial_ft
ScatteringTransforms.SphericalCore.make_spherical_plan
ScatteringTransforms.SphericalCore.make_structured_plan
ScatteringTransforms.SphericalCore.nusht_spherical_plan
ScatteringTransforms.SphericalCore.fsh_structured_plan
```

## Plotting

Methods are supplied by the CairoMakie extension.

```@docs
ScatteringTransforms.plot_coefficients
ScatteringTransforms.plot_filter_bank
```
