"""
    ScatteringTransforms.jl — Native Julia implementation of wavelet scattering transforms

Supports 1D, 2D planar (gridded), and (via extension) 2D spherical scattering.

## Quick Start

Symbols are accessed via fully-qualified submodule paths (the package does not re-export names into
the top-level namespace — see `docs/src/api.md`); alias the package for brevity.

```julia
using ScatteringTransforms: ScatteringTransforms as ST

# 1D scattering (N = signal length, J = number of octaves)
signal = randn(1024)
st = ST.Scattering1D.ScatteringTransform1D(1024, 8; Q=1, max_order=2)
coeffs = st(signal)

# 2D planar scattering (N = image size, J = scales, L = orientations)
image = randn(256, 256)
st2d = ST.Scattering2D.ScatteringTransform2D((256, 256), 4; L=8, max_order=2)
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
include("SphericalCore.jl")
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
using .SphericalCore: SphericalCore
using .Inverse: Inverse


# ============================================================================
# Batched transforms — process a stack of signals/images reusing one plan + workspace
# ============================================================================

"""
    scattering_batch(st::Scattering1D.ScatteringTransform1D, X) -> Matrix

Apply a 1D scattering transform to a batch of signals `X` of size `(N, B)` (signals as columns),
returning a `(flatten_length, B)` matrix whose column `b` is `flatten1d(st(X[:, b]))`. The plan
and all workspace buffers are reused across the batch (only the small scalar-S0 wrapper is
re-allocated per column).
"""
function scattering_batch(st::Scattering1D.ScatteringTransform1D, X::AbstractMatrix)
    T = eltype(st.filter_bank.averaging) |> real
    nw = length(st.filter_bank.wavelets)
    flen = Coefficients.flatten_length(
        Coefficients.ScatteringCoefficients1D(nw, T; compute_S2 = st.max_order >= 2))
    return scattering_batch!(Matrix{T}(undef, flen, size(X, 2)), st, X)
end

"""
    scattering_batch!(out, st, X) -> out

In-place counterpart of [`scattering_batch`](@ref): write the flattened coefficients of each column
(1D) / slice (2D) of `X` into the preallocated `(flatten_length, B)` matrix `out`, reusing one plan +
workspace across the batch. Backend-dispatched `!` methods (`scattering_batch!(out, backend, st, X)`)
are added by the corresponding extensions.
"""
function scattering_batch!(out::AbstractMatrix, st::Scattering1D.ScatteringTransform1D, X::AbstractMatrix)
    nw = length(st.filter_bank.wavelets)
    T = eltype(st.filter_bank.averaging) |> real
    # Mutable-S0 container so `scattering_transform!` updates in place — no per-column wrapper alloc.
    S2 = st.max_order >= 2 ? zeros(T, nw, nw) : Matrix{T}(undef, 0, 0)
    coeffs = Coefficients.ScatteringCoefficients1D(Vector{T}(undef, nw), S2; S0 = [zero(T)])
    @inbounds for b in 1:size(X, 2)
        Scattering1D.scattering_transform!(coeffs, st, view(X, :, b))
        Coefficients.flatten1d!(view(out, :, b), coeffs)
    end
    return out
end

"""
    scattering_batch(st::Scattering2D.ScatteringTransform2D, X) -> Matrix

Apply a 2D scattering transform to a batch of images `X` of size `(Ny, Nx, B)`, returning a
`(flatten_length, B)` matrix whose column `b` is `flatten2d(st(X[:, :, b]))`. Plan and workspace
are reused across the batch.
"""
function scattering_batch(st::Scattering2D.ScatteringTransform2D, X::AbstractArray{<:Any,3})
    T = eltype(st.filter_bank.averaging) |> real
    flen = Coefficients.flatten_length(
        Coefficients.ScatteringCoefficients2D(st.filter_bank.J, st.filter_bank.L, T;
                                              compute_S2 = st.max_order >= 2))
    return scattering_batch!(Matrix{T}(undef, flen, size(X, 3)), st, X)
end

function scattering_batch!(out::AbstractMatrix, st::Scattering2D.ScatteringTransform2D, X::AbstractArray{<:Any,3})
    T = eltype(st.filter_bank.averaging) |> real
    J, L = st.filter_bank.J, st.filter_bank.L
    n = J * L
    # Mutable-S0 container so `scattering_transform2d!` updates in place — no per-slice wrapper alloc.
    S2 = st.max_order >= 2 ? zeros(T, n, n) : Matrix{T}(undef, 0, 0)
    coeffs = Coefficients.ScatteringCoefficients2D(Vector{T}(undef, n), S2;
                                                   S0 = [zero(T)], n_scales = J, n_orientations = L)
    @inbounds for b in 1:size(X, 3)
        Scattering2D.scattering_transform2d!(coeffs, st, view(X, :, :, b))
        Coefficients.flatten2d!(view(out, :, b), coeffs)
    end
    return out
end

# Serializable build spec for reconstructing a transform on a remote worker (FFTW/CUFFT plans
# are not serializable, so distributed workers rebuild rather than receive the transform).
_spectral_backend(st) = st.plan isa Plans.DirectSumPlan ? Plans.DirectSumBackend() : Plans.FFTBackend()
transform_spec(st::Scattering1D.ScatteringTransform1D) =
    (kind = :st1d, N = length(st.buffer_mod), J = st.filter_bank.J, Q = st.filter_bank.Q,
     max_order = st.max_order, T = real(eltype(st.filter_bank.averaging)), spectral = _spectral_backend(st))
transform_spec(st::Scattering2D.ScatteringTransform2D) =
    (kind = :st2d, N = size(st.buffer_mod), J = st.filter_bank.J, L = st.filter_bank.L,
     max_order = st.max_order, T = real(eltype(st.filter_bank.averaging)), spectral = _spectral_backend(st))

function rebuild_transform(spec)
    if spec.kind === :st1d
        return Scattering1D.ScatteringTransform1D(spec.N, spec.J; Q = spec.Q, max_order = spec.max_order,
                                                  T = spec.T, spectral = spec.spectral)
    elseif spec.kind === :st2d
        return Scattering2D.ScatteringTransform2D(spec.N, spec.J; L = spec.L, max_order = spec.max_order,
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
        "(ThreadedBackend), `using Distributed` (DistributedBackend), `using MPI` (MPIBackend), or " *
        "`using KernelAbstractions, AbstractFFTs` plus a device backend such as `using CUDA` " *
        "(GPUBackend)."))
end

# Nonuniform / scattered planar scattering via NUFFT; method added by the FINUFFT extension.
"""
    scattered_planar_scattering(x, y, ms, J; L=8, max_order=2, T=Float64, period=nothing,
                                solve=false, weights=nothing, eps=..., maxiter=100, rtol=1e-8)

Build a 2D planar scattering transform for a scalar field sampled at scattered points `(x, y)`, using
the same oriented Morlet wavelet bank as the gridded [`ScatteringTransform2D`] but computing the
wavelet convolutions on a uniform Fourier **mode grid** of size `ms = (m1, m2)` via a NUFFT: analysis
maps the scattered points to the mode grid, the wavelet multiply happens there, and synthesis
evaluates the filtered field back at the points. Apply it to a length-`M` vector of samples.

`period` is the physical domain size per axis (the Fourier period); it defaults so that a uniform
`0:m-1` grid reproduces the gridded FFT transform exactly. `solve=false` uses the fast NUFFT adjoint
(type-1) — exact for adequately-sampled band-limited fields, approximate on gappy/irregular data;
`solve=true` uses a conjugate-gradient least-squares inversion for the true band-limited coefficients
(slower, needed for irregular sampling). `weights` (length `M`, summing to 1) sets the quadrature for
the spatial mean; the default is the uniform sample mean. Requires `using FINUFFT`.
"""
scattered_planar_scattering(args...; kwargs...) = throw(ArgumentError(
    "scattered / nonuniform planar scattering requires the FINUFFT extension — run `using FINUFFT`."))

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

# Structured (uniform) spherical scattering via a fast SHT on a Clenshaw–Curtis grid; methods added
# by the FastSphericalHarmonics extension. The structured analogue of the scattered
# `spherical_scattering` (NUFSHT).
"""
    structured_spherical_scattering(lmax, J; max_order=2)

Build a spherical scattering transform for a scalar field sampled on the structured Clenshaw–Curtis
grid of a fast spherical-harmonic transform (`Nθ = lmax+1`, `Nφ = 2lmax+1`), using the same smooth
difference-of-Gaussians band-pass wavelets as [`spherical_scattering`](@ref). Apply it to a
`(Nθ, Nφ)` grid of samples; obtain the grid points with [`structured_sphere_points`](@ref).
Requires `using FastSphericalHarmonics`.
"""
structured_spherical_scattering(args...; kwargs...) = throw(ArgumentError(
    "structured spherical scattering requires the FastSphericalHarmonics extension — run " *
    "`using FastSphericalHarmonics`."))

"""
    structured_spherical_monogenic_scattering(lmax, J; max_order=2)

Structured-grid counterpart of [`spherical_monogenic_scattering`](@ref) (fast SHT on a
Clenshaw–Curtis grid). Requires `using FastSphericalHarmonics`.
"""
structured_spherical_monogenic_scattering(args...; kwargs...) = throw(ArgumentError(
    "structured spherical monogenic scattering requires the FastSphericalHarmonics extension — run " *
    "`using FastSphericalHarmonics`."))

"""
    structured_sphere_points(lmax) -> (Θ, Φ)

Colatitudes `Θ` (length `lmax+1`) and longitudes `Φ` (length `2lmax+1`) of the Clenshaw–Curtis grid
used by [`structured_spherical_scattering`](@ref); sample a field as `[f(θ, φ) for θ in Θ, φ in Φ]`.
Requires `using FastSphericalHarmonics`.
"""
structured_sphere_points(args...; kwargs...) = throw(ArgumentError(
    "structured_sphere_points requires the FastSphericalHarmonics extension — run " *
    "`using FastSphericalHarmonics`."))

"""
    spherical_monogenic_components(st, field, j) -> (; bandpass, riesz, amplitude, phase, orientation)

Pointwise spherical monogenic decomposition of `field` band-passed at scale `j` — the S² analogue of
the planar [`monogenic_components`](@ref ScatteringTransforms.Monogenic.monogenic_components). Returns
the band-pass field `bandpass = U⁰_j`, the spin-1
Riesz tangent vector `riesz = (u_θ, u_φ)` (`U^R_j = ð∘(−Δ_S)^{-1/2} U⁰_j`), the monogenic `amplitude`
`√(U⁰² + ‖U^R‖²)`, the `phase = atan(‖U^R‖, U⁰)`, and the local `orientation = atan(u_φ, u_θ)` of the
Riesz vector. Uses spin-weighted scattered synthesis from NUFSHT (`st` is a
[`spherical_monogenic_scattering`](@ref) transform). Requires `using NUFSHT`.
"""
spherical_monogenic_components(args...; kwargs...) = throw(ArgumentError(
    "spherical monogenic components requires the NUFSHT extension — run `using NUFSHT`."))

"""
    scattering_loss(c, target) -> Real

Default [`synthesize`](@ref) objective: the normalized squared error between the coefficient
container `c` and the `target`, summed over the first- and (when present) second-order
coefficients, `‖S₁(c)−S₁(t)‖² + ‖S₂(c)−S₂(t)‖²` divided by the target energy. Differentiable in
`c`, so it composes with `scattering(st, ·)` under autodiff.
"""
function scattering_loss(c, target)
    s1c = Coefficients.first_order(c)
    s1t = Coefficients.first_order(target)
    num = sum(abs2, s1c .- s1t)
    den = sum(abs2, s1t)
    s2t = Coefficients.second_order(target)
    if !isempty(s2t)
        num = num + sum(abs2, Coefficients.second_order(c) .- s2t)
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

# Plotting (implemented in ScatteringTransformsCairoMakieExt). Fallbacks give a helpful, consistent
# error when CairoMakie isn't loaded (matching `synthesize`/`spherical_scattering`).
"""
    plot_filter_bank(fb) — plot a filter bank. Requires `using CairoMakie`.
"""
plot_filter_bank(args...; kwargs...) = throw(ArgumentError(
    "plotting requires the CairoMakie extension — run `using CairoMakie`."))

"""
    plot_coefficients(c; …) — plot scattering coefficients. Requires `using CairoMakie`.
"""
plot_coefficients(args...; kwargs...) = throw(ArgumentError(
    "plotting requires the CairoMakie extension — run `using CairoMakie`."))


# Precompile the hot paths (using the dependency-free direct-sum backend, so no weakdep is
# required at precompile time) to cut time-to-first-transform.
using PrecompileTools: @setup_workload, @compile_workload
@setup_workload begin
    @compile_workload begin
        st1 = Scattering1D.ScatteringTransform1D(32, 3; Q = 1, max_order = 2, spectral = Plans.DirectSumBackend())
        c1 = st1(zeros(Float64, 32))
        Coefficients.flatten1d(c1)
        ScatteringFields.scattering_field(st1, zeros(Float64, 32); subsample = 1)
        scattering_batch(st1, zeros(Float64, 32, 2))
        Reductions.normalized_coefficients(c1)

        st2 = Scattering2D.ScatteringTransform2D((16, 16), 2; L = 4, max_order = 2, spectral = Plans.DirectSumBackend())
        c2 = st2(zeros(Float64, 16, 16))
        Scattering2D.compute_shape_sparsity(Coefficients.first_order(c2), Coefficients.second_order(c2), st2.filter_bank.meta)

        st3 = Scattering3D.ScatteringTransform3D((8, 8, 8), 2; n_orient = 6, max_order = 2, spectral = Plans.DirectSumBackend())
        st3(zeros(Float64, 8, 8, 8))
    end
end

end # module
