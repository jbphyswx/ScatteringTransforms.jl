"""
    ScatteringTransforms.jl — Native Julia implementation of wavelet scattering transforms

Surfaces: 1D signals, 2D and 3D gridded fields, scattered planar points (NUFFT), and the sphere —
both on a structured grid and at scattered points. Each has a monogenic variant; the gridded ones
also have a localized (Mallat) field output and a multi-resolution second order. The spherical and
scattered paths run dependency-free by default, with fast paths supplied by extensions.

## Quick Start

Names live in submodules and are reached by their full path; alias the package for brevity.

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
- Breadth-first CSR path list, walked grouped by first-order wavelet
- Modular extensions for spherical (NUFSHT) and GPU support

## References

- Mallat (2012): Group invariant scattering. Comm. Pure Appl. Math.
- Bruna & Mallat (2013): Invariant Scattering Convolution Networks. IEEE PAMI.
- Cheng & Ménard (2021): How to quantify fields or textures? A guide to the scattering transform.
"""
module ScatteringTransforms

# Include core components (creates submodules). 
include("Execution.jl")
include("Plans.jl")
include("Filters.jl")
include("FilterBanks.jl")
include("ScatteringCore.jl")
include("Coefficients.jl")
include("PathGraph.jl")
include("Batched.jl")
include("ScatteringFields.jl")
include("Scattering1D.jl")
include("Scattering2D.jl")
include("Scattering3D.jl")
include("ScatteredPlanar.jl")
include("SubsampledScattering.jl")
include("Reductions.jl")
include("Monogenic.jl")
include("SphericalCore.jl")
include("Inverse.jl")

# Shared ecosystem dispatch tags: where compute runs (ComputationalBackends) and which spectral
# algorithm it uses (SpectralBackends).
using ComputationalBackends: ComputationalBackends as CB
using SpectralBackends: SpectralBackends as SB

# Import submodules using X: X pattern
using .Execution: Execution
using .Plans: Plans
using .Filters: Filters
using .FilterBanks: FilterBanks
using .ScatteringCore: ScatteringCore
using .PathGraph: PathGraph
using .Batched: Batched
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
# Batched transforms — one plan over the whole (spatial…, B) stack
# ============================================================================

"""
    batch_workspace(st, B; spectral = ..., fft_nthreads = 1) -> Batched.BatchWorkspace

Build the batched workspace for transform `st` at batch size `B`: a spectral plan over the whole
`(spatial…, B)` stack plus the scratch [`Batched.batch_cascade!`](@ref) runs in.

The plan is rebuilt because a batched transform is a different plan from a single-slice one — that
is exactly what makes it one execution instead of `B`. Build it once and pass it back in through
`scattering_batch!` when transforming many stacks of the same size.

`fft_nthreads` is the FFT library's own thread count, baked into the plan. It defaults to `1`: the
cascade is memory-bandwidth bound and the transform is only ~a quarter of a cascade step, so
threading inside the FFT is capped near `1.3×` by Amdahl no matter how many cores it gets, while
threading over the batch or wavelet axis parallelises the whole step. Measured on 8 threads it
returns `1.08–1.17×` on large stacks and `0.33×` on small ones, where the library's per-execution
task spawn (which also allocates) exceeds the transform itself. Raise it only for a single stack
large enough to be worth it, transformed with no parallelism above.
"""
function batch_workspace(st, B::Int; spectral = Plans.spectral_backend(st.plan),
                         fft_nthreads::Int = 1, kwargs...)
    T = real(eltype(st.filter_bank.averaging))
    spatial = size(st.buffer_mod)
    stack = (spatial..., B)
    plan = Plans.make_plan(spectral, T, spatial; nbatch = B, batched = true,
                           fft_nthreads = fft_nthreads, kwargs...)
    cz() = zeros(Complex{T}, stack)
    red = zeros(T, (ntuple(_ -> 1, length(spatial))..., B))
    # Shape the filters and the reduction target once, so the cascade's inner loop never calls
    # `reshape` — each call would allocate a fresh array object, once per wavelet and per path.
    wavelets = [reshape(psi, (spatial..., 1)) for psi in st.filter_bank.wavelets]
    return Batched.BatchWorkspace(
        plan, zeros(T, stack), cz(), cz(), cz(), cz(), zeros(T, stack),
        red, reshape(red, B), wavelets, length(wavelets), st.groups, 1 / prod(spatial))
end

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
    batch_coeffs(st, T) -> coefficient container

A coefficient container for `st` whose `S0` is a 1-element array, so a loop over a batch writes every
field in place instead of rebuilding the struct once per slice.
"""
batch_coeffs(st::Scattering1D.ScatteringTransform1D, ::Type{T}) where {T} =
    (nw = length(st.filter_bank.wavelets);
     Coefficients.ScatteringCoefficients1D(
        Vector{T}(undef, nw),
        st.max_order >= 2 ? zeros(T, nw, nw) : Matrix{T}(undef, 0, 0); S0 = [zero(T)]))

_batch_coeffs2d(st, ::Type{T}, ns, no) where {T} = Coefficients.ScatteringCoefficients2D(
    Vector{T}(undef, ns * no),
    st.max_order >= 2 ? zeros(T, ns * no, ns * no) : Matrix{T}(undef, 0, 0);
    S0 = [zero(T)], n_scales = ns, n_orientations = no)

batch_coeffs(st::Scattering2D.ScatteringTransform2D, ::Type{T}) where {T} =
    _batch_coeffs2d(st, T, st.filter_bank.J, st.filter_bank.L)
batch_coeffs(st::Scattering3D.ScatteringTransform3D, ::Type{T}) where {T} =
    _batch_coeffs2d(st, T, st.filter_bank.J, st.filter_bank.n_orient)

"""
    scattering_batch!(out, st, X; workspace = nothing) -> out

In-place counterpart of [`scattering_batch`](@ref): write the flattened coefficients of each slice of
`X` into the preallocated `(flatten_length, B)` matrix `out`. Backend-dispatched `!` methods
(`scattering_batch!(out, backend, st, X)`) are added by the corresponding extensions.

The default transforms one slice at a time against the transform's own single-slice plan. Passing a
[`batch_workspace`](@ref) instead runs the whole stack through one batched plan
([`Batched.batch_cascade!`](@ref)).

Per-slice is the default because it is faster on a CPU at every batch size measured — `1.2–1.5×`
serial and `4.5×` threaded. The cascade is memory-bandwidth bound and issues `O(nw + paths)`
operations against the *same* data, so what governs is whether that data stays resident: a slice
does, a `B`-slice stack does not. The batched plan's amortised per-call overhead does not recover
the difference. It wins on a device, where the batch is what fills the machine — which is why the
GPU extension builds one.
"""
function scattering_batch! end

for (Mod, TT, XT, tfun, flat, ND) in (
        (:Scattering1D, :ScatteringTransform1D, :AbstractMatrix, :scattering_transform!, :flatten1d!, 2),
        (:Scattering2D, :ScatteringTransform2D, :(AbstractArray{<:Any,3}), :scattering_transform2d!, :flatten2d!, 3),
        (:Scattering3D, :ScatteringTransform3D, :(AbstractArray{<:Any,4}), :scattering_transform3d!, :flatten2d!, 4))
    @eval function scattering_batch!(out::AbstractMatrix, st::$Mod.$TT, X::$XT; workspace = nothing)
        workspace === nothing || return Batched.batch_cascade!(out, workspace, X)
        T = real(eltype(st.filter_bank.averaging))
        coeffs = batch_coeffs(st, T)
        @inbounds for b in 1:size(X, $ND)
            c = $Mod.$tfun(coeffs, st, selectdim(X, $ND, b))
            Coefficients.$flat(view(out, :, b), c)
        end
        return out
    end
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

"""
    scattering_batch(st::Scattering3D.ScatteringTransform3D, X) -> Matrix

Apply a 3D scattering transform to a batch of volumes `X` of size `(Nz, Ny, Nx, B)`, returning a
`(flatten_length, B)` matrix. Plan and workspace are reused across the batch.
"""
function scattering_batch(st::Scattering3D.ScatteringTransform3D, X::AbstractArray{<:Any,4})
    T = eltype(st.filter_bank.averaging) |> real
    flen = Coefficients.flatten_length(
        Coefficients.ScatteringCoefficients2D(st.filter_bank.J, st.filter_bank.n_orient, T;
                                              compute_S2 = st.max_order >= 2))
    return scattering_batch!(Matrix{T}(undef, flen, size(X, 4)), st, X)
end

# ----------------------------------------------------------------------------
# Batches on the remaining surfaces
#
# These share a point set / grid across the batch: the plan, filter bank and workspace are built
# once and every slice reuses them, which is the whole cost on a nonuniform or spherical transform
# (the plan dominates construction). A batch of *different* geometries is a batch of transforms, not
# a batch of slices, and stays a loop over transforms.
# ----------------------------------------------------------------------------

batch_coeffs(st::SubsampledScattering.MultiResolutionScattering{<:Any,1}, ::Type{T}) where {T} =
    (nw = length(st.filter_bank.wavelets);
     Coefficients.ScatteringCoefficients1D(
        Vector{T}(undef, nw),
        st.max_order >= 2 ? zeros(T, nw, nw) : Matrix{T}(undef, 0, 0); S0 = [zero(T)]))

batch_coeffs(st::SubsampledScattering.MultiResolutionScattering, ::Type{T}) where {T} =
    _batch_coeffs2d(st, T, st.filter_bank.J,
                    SubsampledScattering._orient_count(st.filter_bank))

batch_coeffs(st::ScatteredPlanar.ScatteredPlanarScattering, ::Type{T}) where {T} =
    _batch_coeffs2d(st, T, st.filter_bank.J, st.filter_bank.L)

"""
    flat_rows(st) -> Int

Rows a flattened coefficient column of `st` occupies — the height of `scattering_batch`'s output.
"""
flat_rows(st::SubsampledScattering.MultiResolutionScattering) =
    Coefficients.flat_length(length(st.filter_bank.wavelets))
flat_rows(st::ScatteredPlanar.ScatteredPlanarScattering) =
    Coefficients.flat_length(st.filter_bank.J * st.filter_bank.L)
flat_rows(st::SphericalCore.SphericalScattering) = Coefficients.flat_length(st.J)
flat_rows(st::SphericalCore.SphericalMonogenicScattering) = Coefficients.flat_length(st.J)

# 1D reports scales along one axis, so it flattens through `flatten1d!`; 2D and 3D report
# scales × orientations and flatten through `flatten2d!`, exactly as their exact counterparts do.
_flatten_into!(col, c::Coefficients.ScatteringCoefficients1D) = Coefficients.flatten1d!(col, c)
_flatten_into!(col, c::Coefficients.ScatteringCoefficients2D) = Coefficients.flatten2d!(col, c)

"""
    scattering_batch(st::SubsampledScattering.MultiResolutionScattering, X) -> Matrix
    scattering_batch(st::ScatteredPlanar.ScatteredPlanarScattering, X) -> Matrix

Transform every slice of `X` against one transform, returning a `(flatten_length, B)` matrix. `X` is
`(dims…, B)` for a multi-resolution transform and `(M, B)` for the scattered planar one, where `M` is
the plan's point count.
"""
function scattering_batch(st::SubsampledScattering.MultiResolutionScattering{T}, X::AbstractArray) where {T}
    return scattering_batch!(Matrix{T}(undef, flat_rows(st), size(X)[end]), st, X)
end

function scattering_batch!(out::AbstractMatrix, st::SubsampledScattering.MultiResolutionScattering,
                           X::AbstractArray)
    coeffs = batch_coeffs(st, eltype(out))
    D = ndims(X)
    @inbounds for b in 1:size(X, D)
        c = SubsampledScattering.subsampled_scattering!(coeffs, st, selectdim(X, D, b))
        _flatten_into!(view(out, :, b), c)
    end
    return out
end

function scattering_batch(st::ScatteredPlanar.ScatteredPlanarScattering, X::AbstractMatrix)
    T = real(eltype(st.filter_bank.averaging))
    return scattering_batch!(Matrix{T}(undef, flat_rows(st), size(X, 2)), st, X)
end

function scattering_batch!(out::AbstractMatrix, st::ScatteredPlanar.ScatteredPlanarScattering,
                           X::AbstractMatrix)
    coeffs = batch_coeffs(st, eltype(out))
    @inbounds for b in 1:size(X, 2)
        c = ScatteredPlanar.scattered_planar_scattering!(coeffs, st, view(X, :, b))
        Coefficients.flatten2d!(view(out, :, b), c)
    end
    return out
end

"""
    scattering_batch(st::SphericalCore.SphericalScattering, X) -> Matrix
    scattering_batch(st::SphericalCore.SphericalMonogenicScattering, X) -> Matrix

Transform a stack of `B` fields sampled by the same spherical plan — `X` of size
`(field_size…, B)`, so `(M, B)` for a scattered point set and `(nθ, nφ, B)` on a structured grid.

Rows follow the flat layout of [`Coefficients.flat_row_s0`](@ref) and friends. The spherical cascade
pairs each scale with strictly *coarser* ones (`j2 < j1`), so a pair lands in the row that layout
assigns to the unordered pair, `flat_row_s2(j2, j1, J)`.
"""
scattering_batch(st::SphericalCore.SphericalScattering{T}, X::AbstractArray) where {T} =
    scattering_batch!(Matrix{T}(undef, flat_rows(st), size(X)[end]), st, X)

scattering_batch(st::SphericalCore.SphericalMonogenicScattering{T}, X::AbstractArray) where {T} =
    scattering_batch!(Matrix{T}(undef, flat_rows(st), size(X)[end]), st, X)

# Writes (S0, S1, S2) into a flat column. `S2` is lower-triangular here (`j2 < j1`); the flat layout
# stores one row per unordered pair, so the transpose indexing is the same slot, not a relabelling.
function _flatten_spherical!(col::AbstractVector, S0, S1::AbstractVector, S2::AbstractMatrix, J::Int)
    col[Coefficients.flat_row_s0()] = S0
    @inbounds for j in 1:J
        col[Coefficients.flat_row_s1(j, J)] = S1[j]
    end
    @inbounds for j1 in 1:J, j2 in 1:(j1 - 1)
        col[Coefficients.flat_row_s2(j2, j1, J)] = isempty(S2) ? zero(eltype(col)) : S2[j1, j2]
    end
    return col
end

for (TT, WS, fun) in (
        (:SphericalScattering, :SphericalWorkspace, :spherical_scattering!),
        (:SphericalMonogenicScattering, :SphericalMonogenicWorkspace, :spherical_monogenic_scattering!))
    @eval function scattering_batch!(out::AbstractMatrix, st::SphericalCore.$TT{T},
                                     X::AbstractArray) where {T}
        D = ndims(X)
        field1 = selectdim(X, D, 1)
        ws = SphericalCore.$WS(st, field1)
        S1 = zeros(T, st.J)
        S2 = st.max_order >= 2 ? zeros(T, st.J, st.J) : Matrix{T}(undef, 0, 0)
        @inbounds for b in 1:size(X, D)
            r = SphericalCore.$fun(S1, S2, st, ws, selectdim(X, D, b))
            _flatten_spherical!(view(out, :, b), r.S0, r.S1, r.S2, st.J)
        end
        return out
    end
end

# Serializable build spec for reconstructing a transform on a remote worker (FFTW/device plans are
# not serializable, so distributed workers rebuild rather than receive the transform). The spectral
# tag is carried on the transform rather than guessed from the plan type, so a device-resident
# transform is not rebuilt as a host FFTW one.
transform_spec(st::Scattering1D.ScatteringTransform1D) =
    (kind = :st1d, N = length(st.buffer_mod), J = st.filter_bank.J, Q = st.filter_bank.Q,
     max_order = st.max_order, T = real(eltype(st.filter_bank.averaging)),
     spectral = Plans.spectral_backend(st.plan))
transform_spec(st::Scattering2D.ScatteringTransform2D) =
    (kind = :st2d, N = size(st.buffer_mod), J = st.filter_bank.J, L = st.filter_bank.L,
     max_order = st.max_order, T = real(eltype(st.filter_bank.averaging)),
     spectral = Plans.spectral_backend(st.plan))
transform_spec(st::Scattering3D.ScatteringTransform3D) =
    (kind = :st3d, N = size(st.buffer_mod), J = st.filter_bank.J,
     n_orient = st.filter_bank.n_orient, max_order = st.max_order,
     T = real(eltype(st.filter_bank.averaging)), spectral = Plans.spectral_backend(st.plan))

function rebuild_transform(spec)
    if spec.kind === :st1d
        return Scattering1D.ScatteringTransform1D(spec.T, spec.N, spec.J; Q = spec.Q,
                                                  max_order = spec.max_order, spectral = spec.spectral)
    elseif spec.kind === :st2d
        return Scattering2D.ScatteringTransform2D(spec.T, spec.N, spec.J; L = spec.L,
                                                  max_order = spec.max_order, spectral = spec.spectral)
    elseif spec.kind === :st3d
        return Scattering3D.ScatteringTransform3D(spec.T, spec.N, spec.J; n_orient = spec.n_orient,
                                                  max_order = spec.max_order, spectral = spec.spectral)
    else
        throw(ArgumentError("unknown transform spec kind $(spec.kind)"))
    end
end

# Backend-dispatched batched transforms. Serial runs in-process; ThreadedBackend / Distributed /
# MPI / GPU methods are added by the corresponding extensions (per-task/per-worker workspace).
# `AutoBackend` resolves on real capability; every other request is honoured exactly or refused.
scattering_batch(::CB.AbstractSerialBackend, st, X) = scattering_batch(st, X)
scattering_batch(b::CB.AbstractAutoBackend, st, X) =
    scattering_batch(Execution.resolve_backend(b), st, X)
scattering_batch(b::CB.AbstractExecutionBackend, st, X) =
    (Execution.check_available(b); throw(ArgumentError(
        "scattering_batch has no method for backend $(typeof(b)) on a $(typeof(st)).")))

# Nonuniform / scattered planar scattering. Dependency-free by default (exact direct-summation NUDFT
# in `Plans`); the FINUFFT extension supplies a faster spectral plan for the same cascade.
"""
    scattered_planar_scattering(x, y, ms, J; L=8, max_order=2, T=Float64,
                                spectral=SpectralBackends.AutoSpectralBackend(), period=nothing,
                                solve=false, weights=nothing, eps=nothing, maxiter=100, rtol=1e-8)

Build a 2D planar scattering transform for a scalar field sampled at scattered points `(x, y)`, using
the same oriented Morlet wavelet bank as the gridded [`ScatteringTransform2D`] but computing the
wavelet convolutions on a uniform Fourier **mode grid** of size `ms = (m1, m2)` via a nonuniform DFT:
analysis maps the scattered points to the mode grid, the wavelet multiply happens there, and synthesis
evaluates the filtered field back at the points. Apply it to a length-`M` vector of samples.

`spectral` selects the transform: `SpectralBackends.DirectSumSpectralBackend` is the in-core,
dependency-free exact NUDFT (always available, `O(M·prod(ms))`);
[`Plans.FINUFFTBackend`](@ref ScatteringTransforms.Plans.FINUFFTBackend) and
[`Plans.NonuniformFFTsBackend`](@ref ScatteringTransforms.Plans.NonuniformFFTsBackend) select a
specific fast library (`using FINUFFT` / `using NonuniformFFTs`);
`SpectralBackends.NUFFTSpectralBackend` takes whichever of those is loaded; and
`SpectralBackends.AutoSpectralBackend` (the default) picks a fast library if one is loaded, else the
direct sum. `period` is the physical domain size per
axis (the Fourier period); it defaults so a uniform `0:m-1` grid reproduces the gridded FFT transform
exactly. `solve=false` uses the fast adjoint (type-1) — exact for adequately-sampled band-limited
fields, approximate on gappy/irregular data; `solve=true` uses a conjugate-gradient least-squares
inversion for the true band-limited coefficients (slower, needed for irregular sampling). `weights`
(length `M`, summing to 1) sets the quadrature for the spatial mean; the default is the uniform sample
mean. `eps` is the FINUFFT tolerance (ignored by the exact direct sum).
"""
scattered_planar_scattering(::Type{T}, x::AbstractVector, y::AbstractVector, ms::NTuple{2, Int},
                            J::Int; kwargs...) where {T} =
    ScatteredPlanar.build(T, x, y, ms, J; kwargs...)
scattered_planar_scattering(x::AbstractVector, y::AbstractVector, ms::NTuple{2, Int}, J::Int; kwargs...) =
    ScatteredPlanar.build(Float64, x, y, ms, J; kwargs...)

# Spherical scattering on S² (scattered points). Dependency-free by default (in-core direct real-SH
# least-squares transform in `SphericalCore`); the NUFSHT extension supplies a faster spectral plan.
"""
    spherical_scattering(pts_theta, pts_phi, lmax, J; max_order=2,
                         spectral=SpectralBackends.AutoSpectralBackend(), rtol=1e-8, maxiter=500)

Build a spherical scattering transform for a scalar field at scattered points `(θ, φ)` on S², using
smooth difference-of-Gaussians band-pass wavelets. `spectral` selects the spherical-harmonic transform:
`SpectralBackends.DirectSumSpectralBackend` is the in-core, dependency-free exact least-squares
transform (always available, `O(M·(lmax+1)²)`); `SpectralBackends.NUFSHTSpectralBackend` uses the
NUFSHT fast path (needs `using NUFSHT`); `SpectralBackends.AutoSpectralBackend` (the default) picks
NUFSHT if its extension is loaded, else the direct transform. Accurate analysis
needs the sampling to resolve the band limit, i.e. roughly `M ≳ (lmax+1)²` well-distributed points.
"""
function spherical_scattering(pts_theta::AbstractVector{TT}, pts_phi::AbstractVector, lmax::Int, J::Int;
                              max_order::Int = 2,
                              spectral::SB.AbstractSpectralBackend = SB.AutoSpectralBackend(),
                              rtol::Real = 1.0e-8, maxiter::Int = 500) where {TT<:Real}
    T = float(TT)
    plan = SphericalCore.make_spherical_plan(spectral, pts_theta, pts_phi, lmax, T;
                                             rtol = rtol, maxiter = maxiter)
    return SphericalCore.SphericalScattering(lmax, J, max_order, plan,
                                             SphericalCore.dog_sigma2(lmax, J, T))
end

# Spherical *monogenic* scattering on S² (scattered points); method added by the NUFSHT extension.
"""
    spherical_monogenic_scattering(pts_theta, pts_phi, lmax, J; max_order=2)

Build a **monogenic** spherical scattering transform for a scalar field at scattered points
`(θ, φ)` on S². The nonlinearity is the spherical monogenic amplitude
`A_j = √(U⁰_j² + |U^R_j|²)`, where `U⁰_j` is the difference-of-Gaussians band-pass and `U^R_j` is
the spin-1 Riesz field `R = ð∘(−Δ_S)^{-1/2}`. The Riesz energy `|U^R_j|² = |∇_S g_j|²` (with
`g_j = (−Δ_S)^{-1/2} U⁰_j`) is evaluated using only spin-0 spherical-harmonic transforms via the
identity `|∇_S g|² = ½ Δ_S(g²) − g Δ_S g`, so no spin-weighted synthesis is required. `spectral`
selects the spherical-harmonic transform as in [`spherical_scattering`](@ref) (dependency-free
direct SH transform by default, NUFSHT fast path when loaded).
"""
function spherical_monogenic_scattering(pts_theta::AbstractVector{TT}, pts_phi::AbstractVector,
                                        lmax::Int, J::Int; max_order::Int = 2,
                                        spectral::SB.AbstractSpectralBackend = SB.AutoSpectralBackend(),
                                        rtol::Real = 1.0e-8, maxiter::Int = 500) where {TT<:Real}
    T = float(TT)
    plan = SphericalCore.make_spherical_plan(spectral, pts_theta, pts_phi, lmax, T;
                                             rtol = rtol, maxiter = maxiter)
    return SphericalCore.SphericalMonogenicScattering(lmax, J, max_order, plan,
                                                      SphericalCore.dog_sigma2(lmax, J, T))
end

# Structured (uniform-grid) spherical scattering on the equiangular grid. Dependency-free by default
# (in-core direct SHT on the grid); the FastSphericalHarmonics extension supplies the fast exact SHT.
"""
    structured_spherical_scattering(lmax, J; max_order=2,
                                    spectral=SpectralBackends.AutoSpectralBackend(), T=Float64,
                                    rtol=1e-8, maxiter=500)

Build a spherical scattering transform for a scalar field sampled on the structured equiangular grid
(`Nθ = lmax+1` colatitudes, `Nφ = 2lmax+1` longitudes), using the same smooth difference-of-Gaussians
band-pass wavelets as [`spherical_scattering`](@ref). Apply it to a `(Nθ, Nφ)` grid of samples; obtain
the grid points with [`structured_sphere_points`](@ref). `spectral` selects the transform:
`SpectralBackends.DirectSumSpectralBackend` (in-core, dependency-free) by default, or
`SpectralBackends.FSHTSpectralBackend` (the fast exact SHT, needs `using FastSphericalHarmonics`);
`SpectralBackends.AutoSpectralBackend` picks the fast path if its extension is loaded, else the
direct SHT.
"""
function structured_spherical_scattering(::Type{T}, lmax::Int, J::Int; max_order::Int = 2,
                                         spectral::SB.AbstractSpectralBackend = SB.AutoSpectralBackend(),
                                         rtol::Real = 1.0e-8, maxiter::Int = 500) where {T}
    plan = SphericalCore.make_structured_plan(spectral, lmax, T; rtol = rtol, maxiter = maxiter)
    return SphericalCore.SphericalScattering(lmax, J, max_order, plan,
                                             SphericalCore.dog_sigma2(lmax, J, T))
end
structured_spherical_scattering(lmax::Int, J::Int; kwargs...) =
    structured_spherical_scattering(Float64, lmax, J; kwargs...)

"""
    structured_spherical_monogenic_scattering(lmax, J; max_order=2,
                                              spectral=SpectralBackends.AutoSpectralBackend(),
                                              T=Float64, rtol=1e-8, maxiter=500)

Structured-grid counterpart of [`spherical_monogenic_scattering`](@ref). `spectral` selects the SH
transform as in [`structured_spherical_scattering`](@ref) (dependency-free direct SHT by default, fast
exact SHT with `using FastSphericalHarmonics`).
"""
function structured_spherical_monogenic_scattering(::Type{T}, lmax::Int, J::Int; max_order::Int = 2,
                                         spectral::SB.AbstractSpectralBackend = SB.AutoSpectralBackend(),
                                         rtol::Real = 1.0e-8, maxiter::Int = 500) where {T}
    plan = SphericalCore.make_structured_plan(spectral, lmax, T; rtol = rtol, maxiter = maxiter)
    return SphericalCore.SphericalMonogenicScattering(lmax, J, max_order, plan,
                                                      SphericalCore.dog_sigma2(lmax, J, T))
end
structured_spherical_monogenic_scattering(lmax::Int, J::Int; kwargs...) =
    structured_spherical_monogenic_scattering(Float64, lmax, J; kwargs...)

"""
    structured_sphere_points(lmax) -> (Θ, Φ)

Colatitudes `Θ` (length `lmax+1`) and longitudes `Φ` (length `2lmax+1`) of the equiangular structured
grid used by [`structured_spherical_scattering`](@ref); sample a field as
`[f(θ, φ) for θ in Θ, φ in Φ]`. In-core (no dependency); matches `FastSphericalHarmonics.sph_points`.
"""
structured_sphere_points(lmax::Int) = SphericalCore.structured_grid(lmax, Float64)

"""
    spherical_monogenic_components(st, field, j) -> (; bandpass, riesz, amplitude, phase, orientation)

Pointwise spherical monogenic decomposition of `field` band-passed at scale `j` — the S² analogue of
the planar [`monogenic_components`](@ref ScatteringTransforms.Monogenic.monogenic_components). Returns
the band-pass field `bandpass = U⁰_j`, the spin-1
Riesz tangent vector `riesz = (u_θ, u_φ)` (`U^R_j = ð∘(−Δ_S)^{-1/2} U⁰_j`), the monogenic `amplitude`
`√(U⁰² + ‖U^R‖²)`, the `phase = atan(‖U^R‖, U⁰)`, and the local `orientation = atan(u_φ, u_θ)` of the
Riesz vector. `st` is a [`spherical_monogenic_scattering`](@ref) transform. With the in-core direct SH
plan (dependency-free) the Riesz field is synthesized as the surface gradient of `g = (−Δ_S)^{-1/2}U⁰`;
with a NUFSHT-backed plan it uses spin-weighted synthesis — the two agree to solver accuracy.
"""
spherical_monogenic_components(args...; kwargs...) = throw(ArgumentError(
    "spherical_monogenic_components expects a `spherical_monogenic_scattering` transform (with an " *
    "in-core direct SH plan or a NUFSHT-backed plan) plus a field and scale index."))

# Dependency-free pointwise monogenic decomposition on the in-core direct SH plan: the spin-1 Riesz
# field is synthesized as the surface gradient of g (see `SphericalCore.direct_monogenic_components`).
# The NUFSHT extension provides the equivalent via spin-weighted synthesis for its plan.
spherical_monogenic_components(st::SphericalCore.SphericalMonogenicScattering{<:Any, <:SphericalCore.DirectSHTSphericalPlan},
                               field::AbstractVector, j::Int) =
    SphericalCore.direct_monogenic_components(st, field, j)

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
    synthesize(st, target; backend, init=nothing, iters=500, lr=0.05, loss=scattering_loss) -> (; field, losses)

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
        st1 = Scattering1D.ScatteringTransform1D(32, 3; Q = 1, max_order = 2, spectral = SB.DirectSumSpectralBackend())
        c1 = st1(zeros(Float64, 32))
        Coefficients.flatten1d(c1)
        ScatteringFields.scattering_field(st1, zeros(Float64, 32); subsample = 1)
        scattering_batch(st1, zeros(Float64, 32, 2))
        Reductions.normalized_coefficients(c1)

        st2 = Scattering2D.ScatteringTransform2D((16, 16), 2; L = 4, max_order = 2, spectral = SB.DirectSumSpectralBackend())
        c2 = st2(zeros(Float64, 16, 16))
        Scattering2D.compute_shape_sparsity(Coefficients.first_order(c2), Coefficients.second_order(c2), st2.filter_bank.meta)

        st3 = Scattering3D.ScatteringTransform3D((8, 8, 8), 2; n_orient = 6, max_order = 2, spectral = SB.DirectSumSpectralBackend())
        st3(zeros(Float64, 8, 8, 8))
    end
end

end # module
