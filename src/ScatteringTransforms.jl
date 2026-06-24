"""
    ScatteringTransforms.jl — Native Julia implementation of wavelet scattering transforms

Supports 1D, 2D planar (gridded), and (via extension) 2D spherical scattering.

## Quick Start

```julia
using ScatteringTransforms

# 1D scattering (N = signal length, J = number of octaves)
signal = randn(1024)
st = ScatteringTransform1D(1024, 8; Q=1, max_order=2)
coeffs = st(signal)

# 2D planar scattering (N = image size, J = scales, L = orientations)
image = randn(256, 256)
st2d = ScatteringTransform2D((256, 256), 4; L=8, max_order=2)
coeffs2d = st2d(image)
```

## Implementation Notes

- FFT-based convolutions for O(N log N) performance
- Frequency-domain Morlet filter banks
- Depth-first tree traversal (memory efficient)
- Modular extensions for spherical (NUFSHT) and GPU support

## References

- Mallat (2012): Group invariant scattering. Comm. Pure Appl. Math.
- Bruna & Mallat (2013): Invariant Scattering Convolution Networks. IEEE PAMI.
- Cheng & Ménard (2021): How to quantify fields or textures? A guide to the scattering transform.
"""
module ScatteringTransforms

# Include core components (creates submodules)
# Order matters: Coefficients must be before Scattering1D/Scattering2D
include("Backends.jl")
include("Domains.jl")
include("Plans.jl")
include("Filters.jl")
include("FilterBanks.jl")
include("ScatteringCore.jl")
include("Coefficients.jl")
include("PathGraph.jl")
include("ScatteringFields.jl")
include("Scattering1D.jl")
include("Scattering2D.jl")
include("Scattering3D.jl")
include("SubsampledScattering.jl")
include("Reductions.jl")
include("Monogenic.jl")
include("Inverse.jl")

# Import submodules using X: X pattern
using .Backends: Backends
using .Domains: Domains
using .Plans: Plans
using .Filters: Filters
using .FilterBanks: FilterBanks
using .ScatteringCore: ScatteringCore
using .PathGraph: PathGraph
using .ScatteringFields: ScatteringFields
using .Scattering1D: Scattering1D
using .Scattering2D: Scattering2D
using .Scattering3D: Scattering3D
using .SubsampledScattering: SubsampledScattering
using .Coefficients: Coefficients
using .Reductions: Reductions
using .Monogenic: Monogenic
using .Inverse: Inverse

# Re-export key types from submodules (using X: X pattern)
const ScatteringTransform1D = Scattering1D.ScatteringTransform1D
const ScatteringTransform2D = Scattering2D.ScatteringTransform2D
const ScatteringTransform3D = Scattering3D.ScatteringTransform3D
const SubsampledScattering1D = SubsampledScattering.SubsampledScattering1D
const FilterBank1D = FilterBanks.FilterBank1D
const FilterBank2D = FilterBanks.FilterBank2D
const FilterBank3D = FilterBanks.FilterBank3D
const Morlet3D = Filters.Morlet3D
const build_filter_bank3d = FilterBanks.build_filter_bank3d
const scattering_transform3d! = Scattering3D.scattering_transform3d!
const WaveletMeta = FilterBanks.WaveletMeta
const ScatteringTree = PathGraph.ScatteringTree
# Backend taxonomy (ecosystem names; DistributedBackend/MPIBackend parametric over inner backend)
const SerialBackend = Backends.SerialBackend
const ThreadedBackend = Backends.ThreadedBackend
const GPUBackend = Backends.GPUBackend
const AutoBackend = Backends.AutoBackend
const DistributedBackend = Backends.DistributedBackend
const MPIBackend = Backends.MPIBackend
# Domain tags
const Line1D = Domains.Line1D
const Plane2D = Domains.Plane2D
const Volume3D = Domains.Volume3D
const Sphere = Domains.Sphere
# Spectral plans
const DirectSumPlan = Plans.DirectSumPlan
const AbstractScatteringPlan = Plans.AbstractScatteringPlan
const forward_transform! = Plans.forward_transform!
const inverse_transform! = Plans.inverse_transform!
const forward_transform = Plans.forward_transform
const inverse_transform = Plans.inverse_transform
const scattering = ScatteringCore.scattering
# Spectral backend selectors (type-dispatched, not symbols)
const AbstractSpectralBackend = Plans.AbstractSpectralBackend
const DirectSumBackend = Plans.DirectSumBackend
const FFTBackend = Plans.FFTBackend
const AutoSpectral = Plans.AutoSpectral
const ScatteringField1D = ScatteringFields.ScatteringField1D
const ScatteringField2D = ScatteringFields.ScatteringField2D
const path_field = ScatteringFields.path_field
const scattering_field = ScatteringFields.scattering_field    # 1D + 2D methods added in submodules
const scattering_field! = ScatteringFields.scattering_field!
const Morlet1D = Filters.Morlet1D
const Morlet2D = Filters.Morlet2D
const ScatteringCoefficients1D = Coefficients.ScatteringCoefficients1D
const ScatteringCoefficients2D = Coefficients.ScatteringCoefficients2D
const frequency_response = Filters.frequency_response
const build_filter_bank1d = FilterBanks.build_filter_bank1d
const build_filter_bank2d = FilterBanks.build_filter_bank2d
const zeroth_order = Coefficients.zeroth_order
const first_order = Coefficients.first_order
const second_order = Coefficients.second_order
const flatten1d = Coefficients.flatten1d
const flatten2d = Coefficients.flatten2d
const flatten1d! = Coefficients.flatten1d!
const flatten2d! = Coefficients.flatten2d!
const flatten_length = Coefficients.flatten_length
const scattering_transform!    = Scattering1D.scattering_transform!
const scattering_transform2d!  = Scattering2D.scattering_transform2d!
const compute_S1_2d! = Scattering2D.compute_S1_2d!
const compute_S2_2d! = Scattering2D.compute_S2_2d!
const compute_shape_sparsity = Scattering2D.compute_shape_sparsity
const normalized_coefficients = Reductions.normalized_coefficients
const log_coefficients = Reductions.log_coefficients
# Reconstruction (Inverse.jl): exact linear wavelet-frame inverse + phase retrieval
const wavelet_transform = Inverse.wavelet_transform
const iwavelet = Inverse.iwavelet
const reconstruct_phase = Inverse.reconstruct_phase
# Monogenic (Riesz) scattering — isotropic bank + monogenic amplitude (1D/2D/3D)
const MonogenicScattering = Monogenic.MonogenicScattering
const MonogenicFilterBank = Monogenic.MonogenicFilterBank
const build_monogenic_bank = Monogenic.build_monogenic_bank
const riesz_multipliers = Monogenic.riesz_multipliers
const monogenic_amplitude = Monogenic.monogenic_amplitude
const monogenic_components = Monogenic.monogenic_components

# ============================================================================
# Batched transforms — process a stack of signals/images reusing one plan + workspace
# ============================================================================

"""
    scattering_batch(st::ScatteringTransform1D, X) -> Matrix

Apply a 1D scattering transform to a batch of signals `X` of size `(N, B)` (signals as columns),
returning a `(flatten_length, B)` matrix whose column `b` is `flatten1d(st(X[:, b]))`. The plan
and all workspace buffers are reused across the batch (only the small scalar-S0 wrapper is
re-allocated per column).
"""
function scattering_batch(st::ScatteringTransform1D, X::AbstractMatrix)
    N, B = size(X)
    nw = length(st.filter_bank.wavelets)
    T = eltype(st.filter_bank.averaging) |> real
    coeffs = Coefficients.ScatteringCoefficients1D(nw, T; compute_S2 = st.max_order >= 2)
    out = Matrix{T}(undef, Coefficients.flatten_length(coeffs), B)
    @inbounds for b in 1:B
        c = Scattering1D.scattering_transform!(coeffs, st, view(X, :, b))
        Coefficients.flatten1d!(view(out, :, b), c)
    end
    return out
end

"""
    scattering_batch(st::ScatteringTransform2D, X) -> Matrix

Apply a 2D scattering transform to a batch of images `X` of size `(Ny, Nx, B)`, returning a
`(flatten_length, B)` matrix whose column `b` is `flatten2d(st(X[:, :, b]))`. Plan and workspace
are reused across the batch.
"""
function scattering_batch(st::ScatteringTransform2D, X::AbstractArray{<:Any,3})
    Ny, Nx, B = size(X)
    T = eltype(st.filter_bank.averaging) |> real
    coeffs = Coefficients.ScatteringCoefficients2D(st.filter_bank.J, st.filter_bank.L, T;
                                                   compute_S2 = st.max_order >= 2)
    out = Matrix{T}(undef, Coefficients.flatten_length(coeffs), B)
    @inbounds for b in 1:B
        c = Scattering2D.scattering_transform2d!(coeffs, st, view(X, :, :, b))
        Coefficients.flatten2d!(view(out, :, b), c)
    end
    return out
end

# Serializable build spec for reconstructing a transform on a remote worker (FFTW/CUFFT plans
# are not serializable, so distributed workers rebuild rather than receive the transform).
_spectral_backend(st) = st.plan isa Plans.DirectSumPlan ? DirectSumBackend() : FFTBackend()
transform_spec(st::ScatteringTransform1D) =
    (kind = :st1d, N = length(st.buffer_mod), J = st.filter_bank.J, Q = st.filter_bank.Q,
     max_order = st.max_order, T = real(eltype(st.filter_bank.averaging)), spectral = _spectral_backend(st))
transform_spec(st::ScatteringTransform2D) =
    (kind = :st2d, N = size(st.buffer_mod), J = st.filter_bank.J, L = st.filter_bank.L,
     max_order = st.max_order, T = real(eltype(st.filter_bank.averaging)), spectral = _spectral_backend(st))

function rebuild_transform(spec)
    if spec.kind === :st1d
        return ScatteringTransform1D(spec.N, spec.J; Q = spec.Q, max_order = spec.max_order,
                                     T = spec.T, spectral = spec.spectral)
    elseif spec.kind === :st2d
        return ScatteringTransform2D(spec.N, spec.J; L = spec.L, max_order = spec.max_order,
                                     T = spec.T, spectral = spec.spectral)
    else
        throw(ArgumentError("unknown transform spec kind $(spec.kind)"))
    end
end

# Backend-dispatched batched transforms. Serial runs in-process; ThreadedBackend / Distributed /
# MPI methods are added by the corresponding extensions (per-task/per-worker workspace).
scattering_batch(::Backends.SerialBackend, st, X) = scattering_batch(st, X)
function scattering_batch(b::Backends.AbstractExecutionBackend, st, X)
    throw(ArgumentError(
        "scattering_batch on backend $(typeof(b)) is not loaded — run `using OhMyThreads` " *
        "(ThreadedBackend), `using Distributed` (DistributedBackend), or `using MPI` (MPIBackend)."))
end

# Spherical scattering on S² (scattered points); method added by the NUFSHT extension.
"""
    spherical_scattering(pts_theta, pts_phi, lmax, J; max_order=2)

Build a spherical scattering transform for a scalar field at scattered points `(θ, φ)` on S²,
using smooth difference-of-Gaussians band-pass wavelets. Requires `using NUFSHT`.
"""
spherical_scattering(args...; kwargs...) = throw(ArgumentError(
    "spherical scattering requires the NUFSHT extension — run `using NUFSHT`."))

# Spherical *monogenic* scattering on S² (scattered points); method added by the NUFSHT extension.
"""
    spherical_monogenic_scattering(pts_theta, pts_phi, lmax, J; max_order=2)

Build a **monogenic** spherical scattering transform for a scalar field at scattered points
`(θ, φ)` on S². The nonlinearity is the spherical monogenic amplitude
`A_j = √(U⁰_j² + |U^R_j|²)`, where `U⁰_j` is the difference-of-Gaussians band-pass and `U^R_j` is
the spin-1 Riesz field `R = ð∘(−Δ_S)^{-1/2}`. The Riesz energy `|U^R_j|² = |∇_S g_j|²` (with
`g_j = (−Δ_S)^{-1/2} U⁰_j`) is evaluated using only spin-0 spherical-harmonic transforms via the
identity `|∇_S g|² = ½ Δ_S(g²) − g Δ_S g`, so no spin-weighted synthesis is required. Requires
`using NUFSHT`.
"""
spherical_monogenic_scattering(args...; kwargs...) = throw(ArgumentError(
    "spherical monogenic scattering requires the NUFSHT extension — run `using NUFSHT`."))

"""
    spherical_monogenic_components(st, field, j)

Pointwise spherical monogenic decomposition (band-pass field, spin-1 Riesz *vector*, amplitude,
phase, orientation) at scale `j` — the S² analogue of the planar [`monogenic_components`](@ref).

!!! warning "Not implemented yet"
    This requires **spin-1 (spin-weighted) synthesis at scattered points**, which NUFSHT does not
    yet expose (the spin-0 surface-gradient shortcut recovers the amplitude exactly but gives
    unreliable orientation). The amplitude-only [`spherical_monogenic_scattering`](@ref) is
    complete and does not depend on this. Tracked: ScatteringTransforms.jl#1 (and upstream
    NUFSHT.jl#1, for which FastSphericalHarmonics already provides the spin-weighted primitives).
"""
function spherical_monogenic_components(args...; kwargs...)
    error("`spherical_monogenic_components` (pointwise orientation/phase on S²) is not implemented " *
          "yet — it needs spin-1 scattered synthesis in NUFSHT. See ScatteringTransforms.jl#1 and " *
          "NUFSHT.jl#1. The amplitude-only `spherical_monogenic_scattering` is available now.")
end

"""
    scattering_loss(c, target) -> Real

Default [`synthesize`](@ref) objective: the normalized squared error between the coefficient
container `c` and the `target`, summed over the first- and (when present) second-order
coefficients, `‖S₁(c)−S₁(t)‖² + ‖S₂(c)−S₂(t)‖²` divided by the target energy. Differentiable in
`c`, so it composes with `scattering(st, ·)` under autodiff.
"""
function scattering_loss(c, target)
    s1c = first_order(c)
    s1t = first_order(target)
    num = sum(abs2, s1c .- s1t)
    den = sum(abs2, s1t)
    s2t = second_order(target)
    if !isempty(s2t)
        num = num + sum(abs2, second_order(c) .- s2t)
        den = den + sum(abs2, s2t)
    end
    return num / (den + eps(float(real(eltype(s1t)))))
end

# Gradient-descent synthesis from scattering coefficients (method added by the
# DifferentiationInterface extension; the differentiable forward is `scattering(st, x)`).
"""
    synthesize(st, target; backend, init=nothing, iters=500, lr=0.05, loss=normalized_l2) -> (; field, losses)

Reconstruct a field whose scattering coefficients match `target` by gradient descent
(Bruna & Mallat microcanonical synthesis): starting from `init` (random by default), minimize
`loss(scattering(st, x̂), target)` with Adam. `target` may be a precomputed coefficient container
or a field (its coefficients are taken first). The gradient is obtained through
DifferentiationInterface, so `backend` is any `ADTypes` backend (e.g. `AutoMooncake()`); the
synthesized result is a *sample* with matching statistics, not the exact original (the modulus
discards local phase). Requires `using DifferentiationInterface` and an AD backend package.
"""
function synthesize(args...; kwargs...)
    throw(ArgumentError(
        "`synthesize` requires the DifferentiationInterface extension and an AD backend — run " *
        "`using DifferentiationInterface` and e.g. `using Mooncake`, then pass `backend = AutoMooncake()`."))
end

# Plotting stubs (implemented in ScatteringTransformsCairoMakieExt)
function plot_filter_bank end
function plot_coefficients end

export ScatteringTransform1D, ScatteringTransform2D, ScatteringTransform3D, SubsampledScattering1D
export FilterBank1D, FilterBank2D, FilterBank3D
export build_filter_bank3d, scattering_transform3d!
export WaveletMeta, ScatteringTree
export SerialBackend, ThreadedBackend, GPUBackend, AutoBackend, DistributedBackend, MPIBackend
export Line1D, Plane2D, Volume3D, Sphere
export DirectSumPlan, AbstractScatteringPlan, forward_transform!, inverse_transform!
export forward_transform, inverse_transform, scattering
export AbstractSpectralBackend, DirectSumBackend, FFTBackend, AutoSpectral
export Morlet1D, Morlet2D, Morlet3D
export ScatteringCoefficients1D, ScatteringCoefficients2D
export zeroth_order, first_order, second_order
export flatten1d, flatten2d
export frequency_response
export build_filter_bank1d, build_filter_bank2d
export scattering_transform!, scattering_transform2d!
export scattering_batch
export flatten1d!, flatten2d!, flatten_length
export scattering_field, scattering_field!
export ScatteringField1D, ScatteringField2D, path_field
export compute_S1_2d!, compute_S2_2d!
export compute_shape_sparsity, normalized_coefficients, log_coefficients
export wavelet_transform, iwavelet, reconstruct_phase
export MonogenicScattering, MonogenicFilterBank, build_monogenic_bank
export riesz_multipliers, monogenic_amplitude, monogenic_components
export spherical_scattering, spherical_monogenic_scattering, spherical_monogenic_components
export synthesize, scattering_loss
export plot_filter_bank, plot_coefficients

# Precompile the hot paths (using the dependency-free direct-sum backend, so no weakdep is
# required at precompile time) to cut time-to-first-transform.
using PrecompileTools: @setup_workload, @compile_workload
@setup_workload begin
    @compile_workload begin
        st1 = ScatteringTransform1D(32, 3; Q = 1, max_order = 2, spectral = DirectSumBackend())
        c1 = st1(zeros(Float64, 32))
        flatten1d(c1)
        scattering_field(st1, zeros(Float64, 32); subsample = 1)
        scattering_batch(st1, zeros(Float64, 32, 2))
        normalized_coefficients(c1)

        st2 = ScatteringTransform2D((16, 16), 2; L = 4, max_order = 2, spectral = DirectSumBackend())
        c2 = st2(zeros(Float64, 16, 16))
        compute_shape_sparsity(first_order(c2), second_order(c2), st2.filter_bank.meta)

        st3 = ScatteringTransform3D((8, 8, 8), 2; n_orient = 6, max_order = 2, spectral = DirectSumBackend())
        st3(zeros(Float64, 8, 8, 8))
    end
end

end # module
