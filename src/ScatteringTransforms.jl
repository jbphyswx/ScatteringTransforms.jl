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

## References

- Mallat (2012): Group invariant scattering. Comm. Pure Appl. Math.
- Bruna & Mallat (2013): Invariant Scattering Convolution Networks. IEEE PAMI.
- Cheng & Ménard (2021): How to quantify fields or textures? A guide to the scattering transform.
"""
module ScatteringTransforms

# Loaded first, for its `__init__`: it selects the OpenMP runtime's thread-local mode before
# FastTransforms loads that runtime; see `FlowTransformBindings.with_fasttransforms_threads`.
using FlowTransformBindings: FlowTransformBindings

include("Execution.jl")
include("Plans.jl")
include("Filters.jl")
include("FilterBanks.jl")
include("ScatteringCore.jl")
include("Coefficients.jl")
include("PathGraph.jl")
include("Cascade.jl")
include("Batched.jl")
include("ScatteringFields.jl")
include("Scattering1D.jl")
include("Scattering2D.jl")
include("Scattering3D.jl")
include("ScatteredPlanar.jl")
include("Reductions.jl")
include("Monogenic.jl")
include("SphericalCore.jl")
include("Inverse.jl")

using ComputationalBackends: ComputationalBackends as CB
using SpectralBackends: SpectralBackends as SB

using .Execution: Execution
using .Plans: Plans
using .Filters: Filters
using .FilterBanks: FilterBanks
using .ScatteringCore: ScatteringCore
using .PathGraph: PathGraph
using .Cascade: Cascade
using .Batched: Batched
using .ScatteringFields: ScatteringFields
using .Scattering1D: Scattering1D
using .Scattering2D: Scattering2D
using .Scattering3D: Scattering3D
using .Coefficients: Coefficients
using .Reductions: Reductions
using .Monogenic: Monogenic
using .SphericalCore: SphericalCore
using .Inverse: Inverse


# ============================================================================
# Batched transforms
# ============================================================================

"""
    batch_workspace(st, B; transform = nothing, fft_nthreads = 1, kwargs...) -> Batched.BatchWorkspace

Build the batched workspace for transform `st` at batch size `B`: a spectral plan over the whole
`(spatial…, B)` stack plus the scratch [`Batched.batch_cascade!`](@ref) runs in, on the array type of
`st`'s own buffers. Build it once and pass it as `workspace` to [`scattering_batch!`](@ref) for every
stack of `B` slices.

The stack's plan is of the kind `st.plan` is ([`Plans.plan_like`](@ref)), so a device-resident `st`
gets a device workspace; a `SpectralBackends` tag as `transform` builds a host plan of that kind.
`fft_nthreads` is the FFT library's own thread count, fixed in the plan; leave it at `1` when the
batch or wavelet axis is already threaded. `kwargs` go to the plan builder.
"""
function batch_workspace(st, B::Int; transform = nothing, fft_nthreads::Int = 1, kwargs...)
    T = real(eltype(st.filter_bank.averaging))
    like = st.buffer_input
    zeros_like(::Type{S}, dims) where {S} = fill!(similar(like, S, dims), zero(S))
    makeplan = transform === nothing ?
        (rd -> Plans.plan_like(st.plan, zeros_like(Complex{T}, (rd..., B)), length(rd);
                               fft_nthreads = fft_nthreads, kwargs...)) :
        (rd -> Plans.make_plan(transform, T, rd; nbatch = B, batched = true,
                               fft_nthreads = fft_nthreads, kwargs...))
    return Batched.build_workspace(st, B, T; zeros_like = zeros_like, makeplan = makeplan,
                                   shapefilter = f -> reshape(f, (size(f)..., 1)))
end

"""
    scattering_batch(st::Scattering1D.ScatteringTransform1D, X; execution = SerialBackend()) -> Matrix

Apply a 1D scattering transform to a batch of signals `X` of size `(N, B)` (signals as columns),
returning a `(flatten_length, B)` matrix whose column `b` is `flatten1d(st(X[:, b]))`. The plan,
the workspace and one coefficient container are reused across the batch. `execution` is where the
batch runs, as in [`scattering_batch!`](@ref).
"""
scattering_batch(st::Scattering1D.ScatteringTransform1D, X::AbstractMatrix;
                 execution::CB.AbstractExecutionBackend = CB.SerialBackend()) =
    _scattering_batch(execution, st, X)

function _scattering_batch(b::CB.AbstractSerialBackend, st::Scattering1D.ScatteringTransform1D,
                           X::AbstractMatrix)
    T = eltype(st.filter_bank.averaging) |> real
    nw = FilterBanks.nwavelets(st.filter_bank)
    flen = Coefficients.flatten_length(
        Coefficients.ScatteringCoefficients1D(nw, T; compute_S2 = st.max_order >= 2))
    return _scattering_batch!(Matrix{_flat_eltype(T, X)}(undef, flen, size(X, 2)), b, st, X)
end

# Element type a flattened batch column needs: `S1`/`S2` are moduli and stay real, but `S0` is the
# field mean and follows the input, so a complex field widens the column.
_flat_eltype(::Type{T}, X::AbstractArray) where {T} = promote_type(T, eltype(X))

"""
    batch_coeffs(st, T, S0T = T) -> coefficient container

A coefficient container for `st` whose `S0` is a 1-element array, so a loop over a batch writes every
field in place.

`T` is the element type of `S1`/`S2`, which are moduli; `S0T` is that of `S0`, the field mean,
which is complex for a complex field.
"""
batch_coeffs(st::Scattering1D.ScatteringTransform1D, ::Type{T}, ::Type{S0T} = T) where {T, S0T} =
    (nw = FilterBanks.nwavelets(st.filter_bank);
     Coefficients.ScatteringCoefficients1D(
        Vector{T}(undef, nw),
        st.max_order >= 2 ? zeros(T, nw, nw) : Matrix{T}(undef, 0, 0); S0 = [zero(S0T)]))

_batch_coeffs2d(st, ::Type{T}, ::Type{S0T}, ns, no) where {T, S0T} =
    Coefficients.ScatteringCoefficients2D(
        Vector{T}(undef, ns * no),
        st.max_order >= 2 ? zeros(T, ns * no, ns * no) : Matrix{T}(undef, 0, 0);
        S0 = [zero(S0T)], n_scales = ns, n_orientations = no)

batch_coeffs(st::Scattering2D.ScatteringTransform2D, ::Type{T}, ::Type{S0T} = T) where {T, S0T} =
    _batch_coeffs2d(st, T, S0T, st.filter_bank.J, st.filter_bank.L)
batch_coeffs(st::Scattering3D.ScatteringTransform3D, ::Type{T}, ::Type{S0T} = T) where {T, S0T} =
    _batch_coeffs2d(st, T, S0T, st.filter_bank.J, st.filter_bank.n_orient)

"""
    scattering_batch!(out, st, X; execution = SerialBackend(), workspace = nothing) -> out

In-place counterpart of [`scattering_batch`](@ref): write the flattened coefficients of each slice of
`X` into the preallocated `(flatten_length, B)` matrix `out`.

`execution` is where the batch runs: `SerialBackend` in this task, `ThreadedBackend` and
`GPUBackend` through their extensions (OhMyThreads; KernelAbstractions, for the gridded transforms),
and `AutoBackend` as [`Execution.resolve_backend`](@ref) resolves it. `DistributedBackend` and
`MPIBackend` gather a new matrix, so they run through [`scattering_batch`](@ref) alone. A backend
whose extension is not loaded throws.

On `SerialBackend` a gridded transform runs one slice at a time against its own single-slice plan,
which on a CPU keeps each slice's working set in cache. Passing a [`batch_workspace`](@ref) runs the
whole stack through one batched plan ([`Batched.batch_cascade!`](@ref)), as the GPU extension does.
"""
scattering_batch!(out::AbstractMatrix, st, X::AbstractArray;
                  execution::CB.AbstractExecutionBackend = CB.SerialBackend(), kwargs...) =
    _scattering_batch!(out, execution, st, X; kwargs...)

for (Mod, TT, XT, tfun, flat, ND) in (
        (:Scattering1D, :ScatteringTransform1D, :AbstractMatrix, :scattering_transform!, :flatten1d!, 2),
        (:Scattering2D, :ScatteringTransform2D, :(AbstractArray{<:Any,3}), :scattering_transform2d!, :flatten2d!, 3),
        (:Scattering3D, :ScatteringTransform3D, :(AbstractArray{<:Any,4}), :scattering_transform3d!, :flatten2d!, 4))
    @eval function _scattering_batch!(out::AbstractMatrix, ::CB.AbstractSerialBackend, st::$Mod.$TT,
                                      X::$XT; workspace = nothing)
        workspace === nothing || return Batched.batch_cascade!(out, workspace, X)
        # `out` decides: real columns get a real `S0` slot, complex columns a complex one.
        coeffs = batch_coeffs(st, real(eltype(out)), eltype(out))
        @inbounds for b in 1:size(X, $ND)
            c = $Mod.$tfun(coeffs, st, selectdim(X, $ND, b))
            Coefficients.$flat(view(out, :, b), c)
        end
        return out
    end
end

"""
    scattering_batch(st::Scattering2D.ScatteringTransform2D, X; execution = SerialBackend()) -> Matrix

Apply a 2D scattering transform to a batch of images `X` of size `(Ny, Nx, B)`, returning a
`(flatten_length, B)` matrix whose column `b` is `flatten2d(st(X[:, :, b]))`. Plan and workspace
are reused across the batch. `execution` is as in [`scattering_batch!`](@ref).
"""
scattering_batch(st::Scattering2D.ScatteringTransform2D, X::AbstractArray{<:Any,3};
                 execution::CB.AbstractExecutionBackend = CB.SerialBackend()) =
    _scattering_batch(execution, st, X)

function _scattering_batch(b::CB.AbstractSerialBackend, st::Scattering2D.ScatteringTransform2D,
                           X::AbstractArray{<:Any,3})
    T = eltype(st.filter_bank.averaging) |> real
    flen = Coefficients.flatten_length(
        Coefficients.ScatteringCoefficients2D(st.filter_bank.J, st.filter_bank.L, T;
                                              compute_S2 = st.max_order >= 2))
    return _scattering_batch!(Matrix{_flat_eltype(T, X)}(undef, flen, size(X, 3)), b, st, X)
end

"""
    scattering_batch(st::Scattering3D.ScatteringTransform3D, X; execution = SerialBackend()) -> Matrix

Apply a 3D scattering transform to a batch of volumes `X` of size `(Nz, Ny, Nx, B)`, returning a
`(flatten_length, B)` matrix. Plan and workspace are reused across the batch. `execution` is as in
[`scattering_batch!`](@ref).
"""
scattering_batch(st::Scattering3D.ScatteringTransform3D, X::AbstractArray{<:Any,4};
                 execution::CB.AbstractExecutionBackend = CB.SerialBackend()) =
    _scattering_batch(execution, st, X)

function _scattering_batch(b::CB.AbstractSerialBackend, st::Scattering3D.ScatteringTransform3D,
                           X::AbstractArray{<:Any,4})
    T = eltype(st.filter_bank.averaging) |> real
    flen = Coefficients.flatten_length(
        Coefficients.ScatteringCoefficients2D(st.filter_bank.J, st.filter_bank.n_orient, T;
                                              compute_S2 = st.max_order >= 2))
    return _scattering_batch!(Matrix{_flat_eltype(T, X)}(undef, flen, size(X, 4)), b, st, X)
end

# ----------------------------------------------------------------------------
# Batches on the remaining surfaces, where every slice shares the transform's point set or grid
# ----------------------------------------------------------------------------

batch_coeffs(st::ScatteredPlanar.ScatteredPlanarScattering, ::Type{T},
             ::Type{S0T} = T) where {T, S0T} =
    _batch_coeffs2d(st, T, S0T, st.filter_bank.J, st.filter_bank.L)

"""
    flat_rows(st) -> Int

Rows a flattened coefficient column of `st` occupies — the height of `scattering_batch`'s output.
"""
flat_rows(st::ScatteredPlanar.ScatteredPlanarScattering) =
    Coefficients.flat_length(st.filter_bank.J * st.filter_bank.L)
flat_rows(st::SphericalCore.SphericalScattering) = Coefficients.flat_length(st.J)
flat_rows(st::SphericalCore.SphericalMonogenicScattering) = Coefficients.flat_length(st.J)

_flatten_into!(col, c::Coefficients.ScatteringCoefficients1D) = Coefficients.flatten1d!(col, c)
_flatten_into!(col, c::Coefficients.ScatteringCoefficients2D) = Coefficients.flatten2d!(col, c)

"""
    scattering_batch(st::ScatteredPlanar.ScatteredPlanarScattering, X; execution = SerialBackend())
        -> Matrix

Transform every slice of `X` against one transform, returning a `(flatten_length, B)` matrix. `X` is
`(M, B)`, where `M` is the plan's point count. A transform built with `ntrans = W > 1` runs the
columns `W` at a time, through one execution per cascade step for each group, the last group padded
with zero columns. `execution` is as in [`scattering_batch!`](@ref).
"""
scattering_batch(st::ScatteredPlanar.ScatteredPlanarScattering, X::AbstractMatrix;
                 execution::CB.AbstractExecutionBackend = CB.SerialBackend()) =
    _scattering_batch(execution, st, X)

function _scattering_batch(b::CB.AbstractSerialBackend, st::ScatteredPlanar.ScatteredPlanarScattering,
                           X::AbstractMatrix)
    T = real(eltype(st.filter_bank.averaging))
    return _scattering_batch!(Matrix{_flat_eltype(T, X)}(undef, flat_rows(st), size(X, 2)), b, st, X)
end

function _scattering_batch!(out::AbstractMatrix, ::CB.AbstractSerialBackend,
                            st::ScatteredPlanar.ScatteredPlanarScattering, X::AbstractMatrix)
    # `out` decides: real columns get a real `S0` slot, complex columns a complex one.
    coeffs = batch_coeffs(st, real(eltype(out)), eltype(out))
    B = size(X, 2)
    W = Plans.batch_width(st.plan)
    if W == 1
        @inbounds for b in 1:B
            c = ScatteredPlanar.scattered_planar_scattering!(coeffs, st, view(X, :, b))
            Coefficients.flatten2d!(view(out, :, b), c)
        end
        return out
    end
    stage = _PlanarStage(st, eltype(out), eltype(X))
    for g in 1:cld(B, W)
        _planar_group!(out, st, X, ((g - 1) * W + 1):min(g * W, B), stage, coeffs)
    end
    return out
end

# What a transform `W = batch_width(st.plan)` wide runs a group of columns through: the group padded
# with zero columns to `W`, the group's `S0`, `S1` and `S2` on the device of the transform's points,
# and their host copies (the same arrays on the host).
struct _PlanarStage{XP, A0, A1, A2, H0, H1, H2}
    pad::XP
    S0::A0
    S1::A1
    S2::A2
    h0::H0
    h1::H1
    h2::H2
end

_host_copy(A::Array) = A
_host_copy(A::AbstractArray) = Array{eltype(A)}(undef, size(A))
_to_host!(h, A) = h === A ? h : copyto!(h, A)

function _PlanarStage(st::ScatteredPlanar.ScatteredPlanarScattering, ::Type{TO},
                      ::Type{TX}) where {TO, TX}
    T = real(TO)
    W = Plans.batch_width(st.plan)
    M = length(st.weights)
    nw = FilterBanks.nwavelets(st.filter_bank)
    zeros_like(::Type{S}, dims) where {S} = fill!(similar(st.weights, S, dims), zero(S))
    pad = zeros_like(TX <: Complex ? Complex{T} : T, (M, W))
    S0 = zeros_like(TO, (W,))
    S1 = zeros_like(T, (nw, W))
    S2 = st.max_order >= 2 ? zeros_like(T, (nw, nw, W)) : zeros_like(T, (0, 0, 0))
    return _PlanarStage(pad, S0, S1, S2, _host_copy(S0), _host_copy(S1), _host_copy(S2))
end

# Columns `cols` of `X`, at most `W` of them, written into the same columns of `out`.
function _planar_group!(out::AbstractMatrix, st::ScatteredPlanar.ScatteredPlanarScattering,
                        X::AbstractMatrix, cols::UnitRange{Int}, stage::_PlanarStage, coeffs)
    n = length(cols)
    n < size(stage.pad, 2) && fill!(stage.pad, zero(eltype(stage.pad)))
    copyto!(view(stage.pad, :, 1:n), view(X, :, cols))
    r = ScatteredPlanar.scattered_planar_scattering_batch!(stage.S0, stage.S1, stage.S2, st, stage.pad)
    h0 = _to_host!(stage.h0, r.S0)
    h1 = _to_host!(stage.h1, r.S1)
    h2 = _to_host!(stage.h2, r.S2)
    @inbounds for i in 1:n
        copyto!(coeffs.S1, view(h1, :, i))
        isempty(coeffs.S2) || copyto!(coeffs.S2, view(h2, :, :, i))
        Coefficients.flatten2d!(view(out, :, cols[i]), Coefficients.update_S0(coeffs, h0[i]))
    end
    return out
end

"""
    scattering_batch(st::SphericalCore.SphericalScattering, X; execution = SerialBackend()) -> Matrix
    scattering_batch(st::SphericalCore.SphericalMonogenicScattering, X; execution = SerialBackend())
        -> Matrix

Transform a stack of `B` fields sampled by the same spherical plan — `X` of size
`(field_size…, B)`, so `(M, B)` for a scattered point set and `(nθ, nφ, B)` on a structured grid.
`execution` is as in [`scattering_batch!`](@ref).

Rows follow the flat layout of [`Coefficients.flat_length`](@ref) and its row accessors. The spherical cascade
pairs each scale with strictly *coarser* ones (`j2 < j1`), so a pair lands in the row that layout
assigns to the unordered pair, `flat_row_s2(j2, j1, J)`.
"""
scattering_batch(st::SphericalCore.SphericalScattering, X::AbstractArray;
                 execution::CB.AbstractExecutionBackend = CB.SerialBackend()) =
    _scattering_batch(execution, st, X)

scattering_batch(st::SphericalCore.SphericalMonogenicScattering, X::AbstractArray;
                 execution::CB.AbstractExecutionBackend = CB.SerialBackend()) =
    _scattering_batch(execution, st, X)

_scattering_batch(b::CB.AbstractSerialBackend, st::SphericalCore.SphericalScattering{T},
                  X::AbstractArray) where {T} =
    _scattering_batch!(Matrix{T}(undef, flat_rows(st), size(X)[end]), b, st, X)

_scattering_batch(b::CB.AbstractSerialBackend, st::SphericalCore.SphericalMonogenicScattering{T},
                  X::AbstractArray) where {T} =
    _scattering_batch!(Matrix{T}(undef, flat_rows(st), size(X)[end]), b, st, X)

# `S2` is lower-triangular (`j2 < j1`), and the flat layout stores one row per unordered pair
# `(j2, j1)`.
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
    @eval function _spherical_batch_perfield!(out::AbstractMatrix, st::SphericalCore.$TT{T},
                                              X::AbstractArray) where {T}
        D = ndims(X)
        # An empty block (more MPI ranks than columns) has no field to size the workspace from.
        size(X, D) == 0 && return out
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

# The monogenic cascade has no batched form: its amplitude needs the Riesz gradient, which is built
# from per-field spin-0 transforms.
_scattering_batch!(out::AbstractMatrix, ::CB.AbstractSerialBackend,
                   st::SphericalCore.SphericalMonogenicScattering, X::AbstractArray) =
    _spherical_batch_perfield!(out, st, X)

# A widened plan takes a scattered `(M, B)` stack with `B > 1`; a structured grid's fields are
# matrices.
_spherical_batches(plan, X::AbstractArray) =
    size(X)[end] > 1 && ndims(X) == 2 && SphericalCore.supports_batch(plan)

# Scattered-point backends that support `batch_plan` run one transform per cascade step for the
# whole stack. Structured and in-core plans take the per-field loop.
function _scattering_batch!(out::AbstractMatrix, ::CB.AbstractSerialBackend,
                            st::SphericalCore.SphericalScattering{T}, X::AbstractArray) where {T}
    B = size(X)[end]
    _spherical_batches(st.plan, X) || return _spherical_batch_perfield!(out, st, X)

    # `batch_plan` hands back `st.plan` at its own width and otherwise a plan this call owns.
    bp = SphericalCore.batch_plan(st.plan, B)
    owned = Plans.batch_width(st.plan) != B
    try
        stb = SphericalCore.SphericalScattering(st.lmax, st.J, st.max_order, bp, st.sigma2)
        ws = SphericalCore.SphericalWorkspace(stb, X)
        S1 = zeros(T, st.J, B)
        S2 = st.max_order >= 2 ? zeros(T, st.J, st.J, B) : Array{T, 3}(undef, 0, 0, 0)
        r = SphericalCore.spherical_scattering_batch!(S1, S2, stb, ws, X)
        empty2 = Matrix{T}(undef, 0, 0)
        @inbounds for b in 1:B
            S2b = isempty(r.S2) ? empty2 : view(r.S2, :, :, b)
            _flatten_spherical!(view(out, :, b), r.S0[b], view(r.S1, :, b), S2b, st.J)
        end
    finally
        owned && Plans.close_plan!(bp)
    end
    return out
end

"""
    close_transform!(st) -> nothing

`Plans.close_plan!` every spectral plan `st` owns. `st` must not be in use by any task, and is not
usable after.
"""
close_transform!(st) = (Plans.close_plan!(st.plan); nothing)
function close_transform!(st::SphericalCore.SphericalScattering)
    Plans.close_plan!(st.plan)
    st.bands === nothing || foreach(Plans.close_plan!, st.bands)
    return nothing
end

# A serializable build spec, from which `rebuild_transform` reconstructs `st` on another process;
# spectral plans are not serializable. `transform` is the plan's own tag, so a device-resident plan,
# which has none, throws here.
transform_spec(st::Scattering1D.ScatteringTransform1D) =
    (kind = :st1d, N = st.dims[1], J = st.filter_bank.J, Q = st.filter_bank.Q,
     max_order = st.max_order, T = real(eltype(st.filter_bank.averaging)),
     oversampling = st.pw.oversampling, cache = !FilterBanks.iscomputed(st.filter_bank),
     transform = Plans.spectral_backend(st.plan))
transform_spec(st::Scattering2D.ScatteringTransform2D) =
    (kind = :st2d, N = st.dims, J = st.filter_bank.J, L = st.filter_bank.L,
     max_order = st.max_order, T = real(eltype(st.filter_bank.averaging)),
     oversampling = st.pw.oversampling, cache = !FilterBanks.iscomputed(st.filter_bank),
     transform = Plans.spectral_backend(st.plan))
transform_spec(st::Scattering3D.ScatteringTransform3D) =
    (kind = :st3d, N = st.dims, J = st.filter_bank.J,
     n_orient = st.filter_bank.n_orient, max_order = st.max_order,
     T = real(eltype(st.filter_bank.averaging)), oversampling = st.pw.oversampling,
     cache = !FilterBanks.iscomputed(st.filter_bank),
     transform = Plans.spectral_backend(st.plan))

# The nonuniform and spherical specs carry the sample locations and omit the batch width: a rebuilt
# transform is single-field, so it accepts a column block of any width.
function transform_spec(st::ScatteredPlanar.ScatteredPlanarScattering)
    x, y = Plans.plan_points(st.plan)
    # On irregular sampling `solve = true` and the type-1 adjoint are different transforms, so the
    # analysis settings travel too.
    a = Plans.plan_analysis(st.plan)
    return (kind = :planar, x = x, y = y, ms = st.plan.ms, J = st.filter_bank.J,
            L = st.filter_bank.L, max_order = st.max_order, weights = st.weights,
            solve = a.solve, maxiter = a.maxiter, rtol = a.rtol, damp = a.damp, eps = a.eps,
            nufft_nthreads = a.nufft_nthreads,
            T = real(eltype(st.filter_bank.averaging)), transform = Plans.spectral_backend(st.plan))
end

# The spec carries every setting the plan was built with, including its quadrature weights and
# solver tolerance.
for (TT, scat, struc) in ((:SphericalScattering, :sphere, :sphere_struct),
                          (:SphericalMonogenicScattering, :sphere_mono, :sphere_mono_struct))
    @eval function transform_spec(st::SphericalCore.$TT{T}) where {T}
        pts = SphericalCore.plan_points(st.plan)
        transform = Plans.spectral_backend(st.plan)
        # No points means the plan is defined by its band limit alone (a structured backend).
        pts === nothing &&
            return (kind = $(QuoteNode(struc)), lmax = st.lmax, J = st.J,
                    max_order = st.max_order, T = T, transform = transform)
        # Every scattered plan analyses by iterative solve, so it has solver settings; only the
        # structured backends return `nothing`, and those left through the branch above.
        solver = SphericalCore.plan_solver(st.plan)
        # `plan_points`/`plan_weights` return the plan's own arrays; the spec copies them so a plan
        # rebuilt from it shares no state with `st`.
        w = SphericalCore.plan_weights(st.plan)
        return (kind = $(QuoteNode(scat)), theta = copy(pts[1]), phi = copy(pts[2]),
                weights = w === nothing ? nothing : copy(w),
                rtol = solver.rtol, maxiter = solver.maxiter,
                nufft = SphericalCore.plan_nufft(st.plan),
                lmax = st.lmax, J = st.J, max_order = st.max_order, T = T, transform = transform)
    end
end

function rebuild_transform(spec)
    if spec.kind === :st1d
        return Scattering1D.ScatteringTransform1D(spec.T, spec.N, spec.J; Q = spec.Q,
            max_order = spec.max_order, oversampling = spec.oversampling, cache = spec.cache,
            transform = spec.transform)
    elseif spec.kind === :st2d
        return Scattering2D.ScatteringTransform2D(spec.T, spec.N, spec.J; L = spec.L,
            max_order = spec.max_order, oversampling = spec.oversampling, cache = spec.cache,
            transform = spec.transform)
    elseif spec.kind === :st3d
        return Scattering3D.ScatteringTransform3D(spec.T, spec.N, spec.J; n_orient = spec.n_orient,
            max_order = spec.max_order, oversampling = spec.oversampling, cache = spec.cache,
            transform = spec.transform)
    elseif spec.kind === :planar
        # The retained points already lie on the plan's domain of period `2π`.
        return ScatteredPlanar.build(spec.T, spec.x, spec.y, spec.ms, spec.J; L = spec.L,
            max_order = spec.max_order, transform = spec.transform, weights = spec.weights,
            period = (2π, 2π), solve = spec.solve, maxiter = spec.maxiter, rtol = spec.rtol,
            damp = spec.damp, eps = spec.eps, nufft_nthreads = spec.nufft_nthreads)
    elseif spec.kind === :sphere
        # The element type follows from the retained points, so it is not passed separately.
        return spherical_scattering(spec.theta, spec.phi, spec.lmax, spec.J;
            max_order = spec.max_order, transform = spec.transform, weights = spec.weights,
            rtol = spec.rtol, maxiter = spec.maxiter, nufft = spec.nufft)
    elseif spec.kind === :sphere_mono
        return spherical_monogenic_scattering(spec.theta, spec.phi, spec.lmax, spec.J;
            max_order = spec.max_order, transform = spec.transform, weights = spec.weights,
            rtol = spec.rtol, maxiter = spec.maxiter, nufft = spec.nufft)
    elseif spec.kind === :sphere_struct
        return structured_spherical_scattering(spec.T, spec.lmax, spec.J;
            max_order = spec.max_order, transform = spec.transform)
    elseif spec.kind === :sphere_mono_struct
        return structured_spherical_monogenic_scattering(spec.T, spec.lmax, spec.J;
            max_order = spec.max_order, transform = spec.transform)
    else
        throw(ArgumentError("unknown transform spec kind $(spec.kind)"))
    end
end

# The batch entries dispatch on their resolved `execution` here. Serial runs in-process;
# ThreadedBackend / Distributed / MPI / GPU methods of `_scattering_batch` and `_scattering_batch!`
# are added by the corresponding extensions (per-task/per-worker workspace). `AutoBackend` resolves
# on real capability; every other request is honoured exactly or refused.
_scattering_batch(b::CB.AbstractAutoBackend, st, X) =
    _scattering_batch(Execution.resolve_backend(b), st, X)
_scattering_batch(b::CB.AbstractExecutionBackend, st, X) =
    (Execution.check_available(b); throw(ArgumentError(
        "scattering_batch has no method for backend $(typeof(b)) on a $(typeof(st)).")))
_scattering_batch!(out, b::CB.AbstractAutoBackend, st, X; kwargs...) =
    _scattering_batch!(out, Execution.resolve_backend(b), st, X; kwargs...)
_scattering_batch!(out, b::CB.AbstractExecutionBackend, st, X; kwargs...) =
    (Execution.check_available(b); throw(ArgumentError(
        "scattering_batch! has no method for backend $(typeof(b)) on a $(typeof(st)).")))

"""
    scattered_planar_scattering([T=Float64,] x, y, ms, J; period, origin=(0, 0), L=8, max_order=2,
                                transform=SpectralBackends.AutoSpectralBackend(), solve=false,
                                weights=nothing, eps=nothing, maxiter=100,
                                rtol=Plans.default_solver_rtol(T, transform, eps), damp=0,
                                ntrans=1, nufft_nthreads=0, cache=true)
    scattered_planar_scattering([T,] grid, ms, J; kwargs...)

Build a 2D planar scattering transform for a scalar field sampled at scattered points `(x, y)`, using
the same oriented Morlet wavelet bank as the gridded `Scattering2D.ScatteringTransform2D` but
computing the wavelet convolutions on a uniform Fourier **mode grid** of size `ms = (m1, m2)` via a
nonuniform DFT: analysis maps the scattered points to the mode grid, the wavelet multiply happens
there, and synthesis evaluates the filtered field back at the points. Apply it to a length-`M`
vector of samples.

`period` is the length of the domain the points sample along each axis and `origin` its lower
corner (a number for both axes, or a 2-tuple): mode `k` has wavelength `period/k`, so the wavelets'
physical scales are those of the domain, and point sets drawn from one domain share them. Points
`0:m-1` with `period = m` are the nodes of the length-`m` DFT. With FlowGeometries loaded, a 2-D
Cartesian `FlowGeometries.Grids` grid stands in for the points: its active cells are the points, its
`Grids.domain_length` the period, the lowest cell centre the origin and its cell measure the default
`weights`.

`transform` names the nonuniform transform: `SpectralBackends.DirectSumSpectralBackend` is the in-core,
dependency-free exact NUDFT (always available, `O(M·prod(ms))`);
`FlowTransformBindings.NonuniformFFTsBackend()` and `FlowTransformBindings.FINUFFTBackend()` select a
nonuniform-FFT library (`using NonuniformFFTs` / `using FINUFFT`);
and `SpectralBackends.AutoSpectralBackend` (the default) takes NonuniformFFTs when it is loaded, FINUFFT
when only it is, and the direct sum otherwise. `solve=false` uses the fast adjoint (type-1) — exact
for adequately-sampled band-limited fields, approximate on gappy/irregular data; `solve=true`
recovers the true band-limited coefficients by least squares (slower, needed for irregular
sampling). `weights` (length `M`, summing to 1) sets the quadrature for the spatial mean; the default
is the uniform sample mean. `eps` is the NUFFT tolerance (ignored by the exact direct sum).
`ntrans = B` builds a transform that runs `B` fields per call through one execution per cascade step
(see [`scattering_batch`](@ref)). `cache = false` evaluates the wavelets on demand.

The solve is LSMR (`FlowTransformBindings.lsmr!`), which costs one type-2 and one type-1 per
iteration and keeps both `‖b - Af‖` and `‖A†(b - Af)‖` monotone, so stopping at `maxiter` returns the
best iterate reached. It stops on the tests of Fong & Saunders, `‖b - Af‖ ≤ rtol ‖b‖` or
`‖A†(b - Af)‖ ≤ rtol ‖A‖ ‖b - Af‖`, and `rtol` defaults to the accuracy of the transform under it:
`sqrt(eps(T))` for the exact direct sum and `~10·eps` for a fast library, whose type-1 and type-2
are adjoints only to its own tolerance
([`Plans.default_solver_rtol`](@ref ScatteringTransforms.Plans.default_solver_rtol)). `damp` is a
Tikhonov `λ` minimising `‖Af - b‖² + λ²‖f‖²`, `0` by default; with `prod(ms) > M` the problem is
underdetermined, construction warns, and LSMR returns the minimum-norm solution unless a `λ` is given.

The Krylov process terminates within `prod(ms)` steps, and `maxiter` caps it earlier. A truncated
solve returns the best iterate reached but is reproducible only to the accuracy of the transform
under it: a multi-threaded NUFFT accumulates its spreading in a run-dependent order, which an
unconverged solve amplifies by about `cond(A)²`. Raise `maxiter` until the solve converges for an
answer independent of the thread count, or set `nufft_nthreads = 1` for bitwise-identical repeats.

`nufft_nthreads` is the fast library's own thread count, honoured by every copy of the plan,
including the per-task copies a threaded backend builds. `0` (the default) takes `Threads.nthreads()`
on a serial backend and gives each per-task copy one thread. `1` keeps the library single-threaded
everywhere, and its plan executes without allocating; a multi-threaded FFTW plan runs its parallel
loop on Julia tasks.
"""
scattered_planar_scattering(::Type{T}, x::AbstractVector, y::AbstractVector, ms::NTuple{2, Int},
                            J::Int; kwargs...) where {T} =
    ScatteredPlanar.build(T, x, y, ms, J; kwargs...)
scattered_planar_scattering(x::AbstractVector, y::AbstractVector, ms::NTuple{2, Int}, J::Int; kwargs...) =
    ScatteredPlanar.build(Float64, x, y, ms, J; kwargs...)

"""
    spherical_scattering(pts_theta, pts_phi, lmax, J; max_order=2,
                         transform=SpectralBackends.AutoSpectralBackend(),
                         rtol=SphericalCore.default_rtol(transform), maxiter=500, weights=nothing,
                         ntrans=1, nufft=SpectralBackends.AutoSpectralBackend())

Build a spherical scattering transform for a scalar field at scattered points `(θ, φ)` on S², using
smooth difference-of-Gaussians band-pass wavelets. `transform` names the spherical-harmonic transform:
`SpectralBackends.DirectSumSpectralBackend` is the in-core, dependency-free exact least-squares
transform (always available, `O(M·(lmax+1)²)`); `SpectralBackends.NUFSHTSpectralBackend` uses the
NUFSHT fast path (needs `using NUFSHT`); `SpectralBackends.AutoSpectralBackend` (the default) picks
NUFSHT if its extension is loaded, else the direct transform. Accurate analysis
needs the sampling to resolve the band limit, i.e. roughly `M ≳ (lmax+1)²` well-distributed points.

`rtol` and `maxiter` configure the analysis solve, a least-squares fit by LSMR that returns its
best iterate at `maxiter`. `weights` are quadrature weights for the spherical mean, the unweighted
sample mean by default. `ntrans = B` builds a plan that transforms `B` co-located fields per call.
`nufft` names the NUFFT library NUFSHT runs on; the in-core plan runs none and accepts only an `Auto`
`nufft`.
"""
function spherical_scattering(pts_theta::AbstractVector{TT}, pts_phi::AbstractVector, lmax::Int, J::Int;
                              max_order::Int = 2,
                              transform::SB.AbstractSpectralBackend = SB.AutoSpectralBackend(),
                              rtol::Real = SphericalCore.default_rtol(transform),
                              maxiter::Int = 500, weights = nothing, ntrans::Int = 1,
                              nufft::SB.AbstractSpectralBackend = SB.AutoSpectralBackend()) where {TT<:Real}
    T = float(TT)
    plan = SphericalCore.make_spherical_plan(transform, pts_theta, pts_phi, lmax, T;
                                             rtol = rtol, maxiter = maxiter, weights = weights,
                                             ntrans = ntrans, nufft = nufft)
    return SphericalCore.SphericalScattering(lmax, J, max_order, plan,
                                             SphericalCore.dog_sigma2(lmax, J, T))
end

"""
    spherical_monogenic_scattering(pts_theta, pts_phi, lmax, J; max_order=2,
                                   transform=SpectralBackends.AutoSpectralBackend(),
                                   rtol=SphericalCore.default_rtol(transform), maxiter=500,
                                   weights=nothing, nufft=SpectralBackends.AutoSpectralBackend())

Build a **monogenic** spherical scattering transform for a scalar field at scattered points
`(θ, φ)` on S². The nonlinearity is the spherical monogenic amplitude
`A_j = √(U⁰_j² + |U^R_j|²)`, where `U⁰_j` is the difference-of-Gaussians band-pass and `U^R_j` is
the spin-1 Riesz field `R = ð∘(−Δ_S)^{-1/2}`. The Riesz energy `|U^R_j|² = |∇_S g_j|²` (with
`g_j = (−Δ_S)^{-1/2} U⁰_j`) is synthesised exactly from `g_j`'s coefficients
(`SphericalCore.sphere_riesz_energy!`). The keywords are as in [`spherical_scattering`](@ref).
"""
function spherical_monogenic_scattering(pts_theta::AbstractVector{TT}, pts_phi::AbstractVector,
                                        lmax::Int, J::Int; max_order::Int = 2,
                                        transform::SB.AbstractSpectralBackend = SB.AutoSpectralBackend(),
                                        rtol::Real = SphericalCore.default_rtol(transform),
                                        maxiter::Int = 500, weights = nothing,
                                        nufft::SB.AbstractSpectralBackend = SB.AutoSpectralBackend()) where {TT<:Real}
    T = float(TT)
    # `spin = true` builds the spin-weighted plan pair `spherical_monogenic_components` synthesises
    # through on a NUFSHT plan.
    plan = SphericalCore.make_spherical_plan(transform, pts_theta, pts_phi, lmax, T;
                                             rtol = rtol, maxiter = maxiter, weights = weights,
                                             nufft = nufft, spin = true)
    return SphericalCore.SphericalMonogenicScattering(lmax, J, max_order, plan,
                                                      SphericalCore.dog_sigma2(lmax, J, T))
end

"""
    structured_spherical_scattering([T=Float64,] lmax, J; max_order=2,
                                    transform=SpectralBackends.AutoSpectralBackend(),
                                    band_headroom=Inf, rtol=1e-8, maxiter=500)

Build a spherical scattering transform for a scalar field sampled on the structured equiangular grid
(`Nθ = lmax+1` colatitudes, `Nφ = 2lmax+1` longitudes), using the same smooth difference-of-Gaussians
band-pass wavelets as [`spherical_scattering`](@ref). Apply it to a `(Nθ, Nφ)` grid of samples; obtain
the grid points with [`structured_sphere_points`](@ref). `transform` names the spherical-harmonic
transform:
`SpectralBackends.DirectSumSpectralBackend` (in-core, dependency-free) by default, or
`SpectralBackends.FSHTSpectralBackend` (the fast exact SHT, needs `using FastSphericalHarmonics`);
`SpectralBackends.AutoSpectralBackend` picks the fast path if its extension is loaded, else the
direct SHT.

On a backend that can narrow its band limit (FastSphericalHarmonics), a finite `band_headroom`
synthesises band `j` at band limit `band_headroom·ℓ_j`, discarding the tail
[`SphericalCore.band_lmax`](@ref) describes; the default `Inf` reproduces the full-band-limit
cascade bit for bit. `rtol` and `maxiter` configure the in-core plan's analysis solve.
"""
function structured_spherical_scattering(::Type{T}, lmax::Int, J::Int; max_order::Int = 2,
                                         transform::SB.AbstractSpectralBackend = SB.AutoSpectralBackend(),
                                         band_headroom::Real = Inf,
                                         rtol::Real = 1.0e-8, maxiter::Int = 500) where {T}
    plan = SphericalCore.make_structured_plan(transform, lmax, T; rtol = rtol, maxiter = maxiter)
    bands = SphericalCore.band_plans(plan, lmax, J, band_headroom)
    return SphericalCore.SphericalScattering(lmax, J, max_order, plan,
                                             SphericalCore.dog_sigma2(lmax, J, T), bands)
end
structured_spherical_scattering(lmax::Int, J::Int; kwargs...) =
    structured_spherical_scattering(Float64, lmax, J; kwargs...)

"""
    structured_spherical_monogenic_scattering([T=Float64,] lmax, J; max_order=2,
                                              transform=SpectralBackends.AutoSpectralBackend(),
                                              rtol=1e-8, maxiter=500)

Structured-grid counterpart of [`spherical_monogenic_scattering`](@ref). `transform` names the SH
transform as in [`structured_spherical_scattering`](@ref) (dependency-free direct SHT by default, fast
exact SHT with `using FastSphericalHarmonics`).
"""
function structured_spherical_monogenic_scattering(::Type{T}, lmax::Int, J::Int; max_order::Int = 2,
                                         transform::SB.AbstractSpectralBackend = SB.AutoSpectralBackend(),
                                         rtol::Real = 1.0e-8, maxiter::Int = 500) where {T}
    plan = SphericalCore.make_structured_plan(transform, lmax, T; rtol = rtol, maxiter = maxiter)
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

"""
    synthesize(st, target; backend, init=nothing, iters=500, lr=0.05, loss=scattering_loss) -> (; field, losses)

Reconstruct a field whose scattering coefficients match `target` by gradient descent
(Bruna & Mallat microcanonical synthesis): starting from `init` (random by default), minimize
`loss(scattering(st, x̂), target)` with Adam. `target` may be a precomputed coefficient container
or a field (its coefficients are taken first). The gradient is obtained through
DifferentiationInterface, so `backend` is any `ADTypes` backend. The modulus discards local phase,
so the result is a *sample* with matching statistics and the original is not recovered. Requires
`using DifferentiationInterface` and an AD backend package.

With Enzyme, pass `AutoEnzyme(; mode = Enzyme.set_runtime_activity(Enzyme.Reverse))`: the cascade's
closures capture constant filters alongside the active input, which Enzyme's static activity
analysis cannot separate, and a plain `AutoEnzyme()` raises `EnzymeRuntimeActivityError`.
"""
function synthesize(args...; kwargs...)
    throw(ArgumentError(
        "`synthesize` requires the DifferentiationInterface extension and an AD backend — run " *
        "`using DifferentiationInterface` and e.g. `using Mooncake`, then pass `backend = AutoMooncake()`."))
end

"""
    plot_filter_bank(fb) — plot a 1D or 2D filter bank. Requires `using CairoMakie`.
"""
plot_filter_bank(args...; kwargs...) = throw(ArgumentError(
    "plotting requires the CairoMakie extension — run `using CairoMakie`."))

"""
    plot_coefficients(c; signal=nothing) — plot 1D scattering coefficients.
    plot_coefficients(c; image=nothing) — plot 2D scattering coefficients.

Passing the input adds a panel showing it. Requires `using CairoMakie`.
"""
plot_coefficients(args...; kwargs...) = throw(ArgumentError(
    "plotting requires the CairoMakie extension — run `using CairoMakie`."))


# The precompile workload uses the direct-sum backend, which needs no weak dependency.
using PrecompileTools: @setup_workload, @compile_workload
@setup_workload begin
    @compile_workload begin
        st1 = Scattering1D.ScatteringTransform1D(32, 3; Q = 1, max_order = 2, transform = SB.DirectSumSpectralBackend())
        c1 = st1(zeros(Float64, 32))
        Coefficients.flatten1d(c1)
        ScatteringFields.scattering_field(st1, zeros(Float64, 32); subsample = 1)
        scattering_batch(st1, zeros(Float64, 32, 2))
        Reductions.normalized_coefficients(c1)

        st2 = Scattering2D.ScatteringTransform2D((16, 16), 2; L = 4, max_order = 2, transform = SB.DirectSumSpectralBackend())
        c2 = st2(zeros(Float64, 16, 16))
        Scattering2D.compute_shape_sparsity(Coefficients.first_order(c2), Coefficients.second_order(c2), st2.filter_bank.meta)

        st3 = Scattering3D.ScatteringTransform3D((8, 8, 8), 2; n_orient = 6, max_order = 2, transform = SB.DirectSumSpectralBackend())
        st3(zeros(Float64, 8, 8, 8))
    end
end

end # module
