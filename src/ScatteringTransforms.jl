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
st = ST.ScatteringCore.ScatteringTransform(1024, 8; Q=1, max_order=2)
coeffs = st(signal)

# 2D planar scattering (N = image size, J = scales, L = orientations)
image = randn(256, 256)
st2d = ST.ScatteringCore.ScatteringTransform((256, 256), 4; L=8, max_order=2)
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

using ComputationalBackends: ComputationalBackends, ComputationalBackends as CB
using SpectralBackends: SpectralBackends, SpectralBackends as SB

public ComputationalBackends, CB
public SpectralBackends, SB

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

function ScatteringCore.task_workspace(st::ScatteringCore.ScatteringTransform)
    fb = FilterBanks.task_bank(st.filter_bank)
    pw = Cascade.task_copy(st.pw, fb)
    return ScatteringCore.ScatteringTransform(fb, st.tree, st.groups, st.max_order, first(pw.plans),
        similar(st.buffer_input), similar(st.buffer_signal_fft), st.dims, pw)
end

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
        (rd -> Plans.plan_like(st.plan, zeros_like(Complex{T}, (rd..., B)), Plans.spatial_rank(rd);
                               fft_nthreads = fft_nthreads, kwargs...)) :
        (rd -> Plans.make_plan(transform, T, rd; batch = B, fft_nthreads = fft_nthreads, kwargs...))
    fb = FilterBanks.task_bank(st.filter_bank)
    return Batched.build_workspace(fb, st.groups, st.dims, st.pw.oversampling, B, T;
                                   zeros_like = zeros_like, makeplan = makeplan,
                                   shapefilter = f -> reshape(f, (size(f)..., 1)))
end

"""
    scattering_batch(st::ScatteringCore.ScatteringTransform{1}, X; execution = SerialBackend()) -> Matrix

Apply a 1D scattering transform to a batch of signals `X` of size `(N, B)` (signals as columns),
returning a `(flatten_length, B)` matrix whose column `b` is `flatten1d(st(X[:, b]))`. The plan,
the workspace and one coefficient container are reused across the batch. `execution` is where the
batch runs, as in [`scattering_batch!`](@ref).
"""
scattering_batch(st::ScatteringCore.ScatteringTransform{1}, X::AbstractMatrix;
                 execution::CB.AbstractExecutionBackend = CB.SerialBackend()) =
    _scattering_batch(execution, st, X)

function _scattering_batch(b::CB.AbstractSerialBackend, st::ScatteringCore.ScatteringTransform{1},
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
batch_coeffs(st::Union{ScatteringCore.ScatteringTransform{1},Monogenic.MonogenicScatteringTransform},
             ::Type{T}, ::Type{S0T} = T) where {T, S0T} =
    (nw = FilterBanks.nwavelets(st.filter_bank);
     Coefficients.ScatteringCoefficients1D(
        Vector{T}(undef, nw),
        st.max_order >= 2 ? zeros(T, nw, nw) : Matrix{T}(undef, 0, 0); S0 = [zero(S0T)]))

_batch_coeffs2d(st, ::Type{T}, ::Type{S0T}, ns, no) where {T, S0T} =
    Coefficients.ScatteringCoefficients2D(
        Vector{T}(undef, ns * no),
        st.max_order >= 2 ? zeros(T, ns * no, ns * no) : Matrix{T}(undef, 0, 0);
        S0 = [zero(S0T)], n_scales = ns, n_orientations = no)

batch_coeffs(st::ScatteringCore.ScatteringTransform{2}, ::Type{T}, ::Type{S0T} = T) where {T, S0T} =
    _batch_coeffs2d(st, T, S0T, st.filter_bank.J, st.filter_bank.L)
batch_coeffs(st::ScatteringCore.ScatteringTransform{3}, ::Type{T}, ::Type{S0T} = T) where {T, S0T} =
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

@inline _batch_column(X::AbstractArray{T,N}, b) where {T,N} =
    view(X, ntuple(_ -> Colon(), Val(N - 1))..., b)

for (Mod, D, XT, tfun, flat, ND) in (
        (:Scattering1D, 1, :AbstractMatrix, :scattering_transform!, :flatten1d!, 2),
        (:Scattering2D, 2, :(AbstractArray{<:Any,3}), :scattering_transform2d!, :flatten2d!, 3),
        (:Scattering3D, 3, :(AbstractArray{<:Any,4}), :scattering_transform3d!, :flatten2d!, 4))
    @eval function _scattering_batch!(out::AbstractMatrix, ::CB.AbstractSerialBackend,
                                      st::ScatteringCore.ScatteringTransform{$D},
                                      X::$XT; workspace = nothing)
        workspace === nothing || return Batched.batch_cascade!(out, workspace, X)
        # `out` decides: real columns get a real `S0` slot, complex columns a complex one.
        coeffs = batch_coeffs(st, real(eltype(out)), eltype(out))
        @inbounds for b in 1:size(X, $ND)
            c = $Mod.$tfun(coeffs, st, _batch_column(X, b))
            Coefficients.$flat(view(out, :, b), c)
        end
        return out
    end
end

"""
    scattering_batch(st::ScatteringCore.ScatteringTransform{2}, X; execution = SerialBackend()) -> Matrix

Apply a 2D scattering transform to a batch of images `X` of size `(Ny, Nx, B)`, returning a
`(flatten_length, B)` matrix whose column `b` is `flatten2d(st(X[:, :, b]))`. Plan and workspace
are reused across the batch. `execution` is as in [`scattering_batch!`](@ref).
"""
scattering_batch(st::ScatteringCore.ScatteringTransform{2}, X::AbstractArray{<:Any,3};
                 execution::CB.AbstractExecutionBackend = CB.SerialBackend()) =
    _scattering_batch(execution, st, X)

function _scattering_batch(b::CB.AbstractSerialBackend, st::ScatteringCore.ScatteringTransform{2},
                           X::AbstractArray{<:Any,3})
    T = eltype(st.filter_bank.averaging) |> real
    flen = Coefficients.flatten_length(
        Coefficients.ScatteringCoefficients2D(st.filter_bank.J, st.filter_bank.L, T;
                                              compute_S2 = st.max_order >= 2))
    return _scattering_batch!(Matrix{_flat_eltype(T, X)}(undef, flen, size(X, 3)), b, st, X)
end

"""
    scattering_batch(st::ScatteringCore.ScatteringTransform{3}, X; execution = SerialBackend()) -> Matrix

Apply a 3D scattering transform to a batch of volumes `X` of size `(Nz, Ny, Nx, B)`, returning a
`(flatten_length, B)` matrix. Plan and workspace are reused across the batch. `execution` is as in
[`scattering_batch!`](@ref).
"""
scattering_batch(st::ScatteringCore.ScatteringTransform{3}, X::AbstractArray{<:Any,4};
                 execution::CB.AbstractExecutionBackend = CB.SerialBackend()) =
    _scattering_batch(execution, st, X)

function _scattering_batch(b::CB.AbstractSerialBackend, st::ScatteringCore.ScatteringTransform{3},
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

batch_coeffs(st::ScatteredPlanar.ScatteredPlanarScatteringTransform, ::Type{T},
             ::Type{S0T} = T) where {T, S0T} =
    _batch_coeffs2d(st, T, S0T, st.filter_bank.J, st.filter_bank.L)

"""
    flat_rows(st) -> Int

Rows a flattened coefficient column of `st` occupies — the height of `scattering_batch`'s output.
"""
flat_rows(st::ScatteredPlanar.ScatteredPlanarScatteringTransform) =
    Coefficients.flat_length(st.filter_bank.J * st.filter_bank.L)
flat_rows(st::SphericalCore.SphericalScatteringTransform) = Coefficients.flat_length(st.J)
flat_rows(st::SphericalCore.SphericalMonogenicScatteringTransform) = Coefficients.flat_length(st.J)
flat_rows(st::Monogenic.MonogenicScatteringTransform) = Coefficients.flat_length(FilterBanks.nwavelets(st.filter_bank))
flat_rows(st::ScatteringCore.ScatteringTransform) =
    Coefficients.flat_length(FilterBanks.nwavelets(st.filter_bank))

_coefficient_eltype(st) = real(eltype(st.filter_bank.averaging))
_coefficient_eltype(st::Union{SphericalCore.SphericalScatteringTransform,SphericalCore.SphericalMonogenicScatteringTransform}) =
    eltype(st.sigma2)

batch_coeffs(st::Union{SphericalCore.SphericalScatteringTransform,SphericalCore.SphericalMonogenicScatteringTransform},
             ::Type{T}, ::Type{S0T} = T) where {T,S0T} =
    Coefficients.ScatteringCoefficients1D(Vector{T}(undef, st.J),
        st.max_order >= 2 ? zeros(T, st.J, st.J) : Matrix{T}(undef, 0, 0); S0 = [zero(S0T)])

_single_scattering!(c, st::ScatteringCore.ScatteringTransform{1}, x) = Scattering1D.scattering_transform!(c, st, x)
_single_scattering!(c, st::ScatteringCore.ScatteringTransform{2}, x) = Scattering2D.scattering_transform2d!(c, st, x)
_single_scattering!(c, st::ScatteringCore.ScatteringTransform{3}, x) = Scattering3D.scattering_transform3d!(c, st, x)
_single_scattering!(c, st::Monogenic.MonogenicScatteringTransform, x) = Monogenic.monogenic_scattering!(c, st, x)
_single_scattering!(c, st::SphericalCore.SphericalScatteringTransform, x) =
    SphericalCore.spherical_scattering!(c.S1, c.S2, st, st.ws, x)
_single_scattering!(c, st::SphericalCore.SphericalMonogenicScatteringTransform, x) =
    SphericalCore.spherical_monogenic_scattering!(c.S1, c.S2, st, st.ws, x)

scattering_batch(st::Monogenic.MonogenicScatteringTransform, X::AbstractArray;
                 execution::CB.AbstractExecutionBackend = CB.SerialBackend()) =
    _scattering_batch(execution, st, X)

function _scattering_batch(b::CB.AbstractSerialBackend, st::Monogenic.MonogenicScatteringTransform,
                           X::AbstractArray)
    T = real(eltype(st.buffer_signal_fft))
    return _scattering_batch!(Matrix{_flat_eltype(T, X)}(undef, flat_rows(st), size(X)[end]), b, st, X)
end

function _scattering_batch!(out::AbstractMatrix, ::CB.AbstractSerialBackend,
                            st::Monogenic.MonogenicScatteringTransform{D}, X::AbstractArray) where {D}
    coeffs = batch_coeffs(st, real(eltype(out)), eltype(out))
    flat = Vector{eltype(out)}(undef, size(out, 1))
    for b in axes(out, 2)
        c = Monogenic.monogenic_scattering!(coeffs, st, _batch_column(X, b))
        Coefficients.flatten1d!(flat, c)
        copyto!(view(out, :, b), flat)
    end
    return out
end

_flatten_into!(col, c::Coefficients.ScatteringCoefficients1D) = Coefficients.flatten1d!(col, c)
_flatten_into!(col, c::Coefficients.ScatteringCoefficients2D) = Coefficients.flatten2d!(col, c)

"""
    scattering_batch(st::ScatteredPlanar.ScatteredPlanarScatteringTransform, X; execution = SerialBackend())
        -> Matrix

Transform every slice of `X` against one transform, returning a `(flatten_length, B)` matrix. `X` is
`(M, B)`, where `M` is the plan's point count. A transform built with `ntrans = W > 1` runs the
columns `W` at a time, through one execution per cascade step for each group, the last group padded
with zero columns. `execution` is as in [`scattering_batch!`](@ref).
"""
scattering_batch(st::ScatteredPlanar.ScatteredPlanarScatteringTransform, X::AbstractMatrix;
                 execution::CB.AbstractExecutionBackend = CB.SerialBackend()) =
    _scattering_batch(execution, st, X)

function _scattering_batch(b::CB.AbstractSerialBackend, st::ScatteredPlanar.ScatteredPlanarScatteringTransform,
                           X::AbstractMatrix)
    T = real(eltype(st.filter_bank.averaging))
    return _scattering_batch!(Matrix{_flat_eltype(T, X)}(undef, flat_rows(st), size(X, 2)), b, st, X)
end

function _scattering_batch!(out::AbstractMatrix, ::CB.AbstractSerialBackend,
                            st::ScatteredPlanar.ScatteredPlanarScatteringTransform, X::AbstractMatrix)
    B = size(X, 2)
    W = Plans.batch_width(st.plan)
    for g in 1:cld(B, W)
        _planar_group!(out, st, X, ((g - 1) * W + 1):min(g * W, B))
    end
    return out
end

_planar_flat(st, out::StridedMatrix) = ScatteredPlanar._host_coefficients!(st)
_planar_flat(st, out::AbstractMatrix) = st.flat

_copy_planar_flat!(out::AbstractMatrix{<:Real}, flat) = (out .= real.(flat); out)
_copy_planar_flat!(out::AbstractMatrix{<:Complex}, flat) = (copyto!(out, flat); out)

# Columns `cols` of `X`, at most `W` of them, written into the same columns of `out`.
function _planar_group!(out::AbstractMatrix, st::ScatteredPlanar.ScatteredPlanarScatteringTransform,
                        X::AbstractMatrix, cols::UnitRange{Int})
    n = length(cols)
    pad = ScatteredPlanar._samples(st, X)
    n < size(pad, 2) && fill!(pad, zero(eltype(pad)))
    copyto!(view(pad, :, 1:n), view(X, :, cols))
    spectrum = ScatteredPlanar._analyse!(st, pad)
    Batched.batch_spectral_cascade!(st.flat, st.batch, spectrum)
    _copy_planar_flat!(view(out, :, cols), view(_planar_flat(st, out), :, 1:n))
    return out
end

"""
    scattering_batch(st::SphericalCore.SphericalScatteringTransform, X; execution = SerialBackend()) -> Matrix
    scattering_batch(st::SphericalCore.SphericalMonogenicScatteringTransform, X; execution = SerialBackend())
        -> Matrix

Transform a stack of `B` fields sampled by the same spherical plan — `X` of size
`(field_size…, B)`, so `(M, B)` for a scattered point set and `(nθ, nφ, B)` on a structured grid.
`execution` is as in [`scattering_batch!`](@ref).

Rows follow [`Coefficients.flat_length`](@ref). Scales are finest-first and second-order
pairs satisfy `j2 > j1`, with row `flat_row_s2(j1, j2, J)`.
"""
scattering_batch(st::SphericalCore.SphericalScatteringTransform, X::AbstractArray;
                 execution::CB.AbstractExecutionBackend = CB.SerialBackend()) =
    _scattering_batch(execution, st, X)

scattering_batch(st::SphericalCore.SphericalMonogenicScatteringTransform, X::AbstractArray;
                 execution::CB.AbstractExecutionBackend = CB.SerialBackend()) =
    _scattering_batch(execution, st, X)

_scattering_batch(b::CB.AbstractSerialBackend, st::SphericalCore.SphericalScatteringTransform{T},
                  X::AbstractArray) where {T} =
    _scattering_batch!(Matrix{T}(undef, flat_rows(st), size(X)[end]), b, st, X)

_scattering_batch(b::CB.AbstractSerialBackend, st::SphericalCore.SphericalMonogenicScatteringTransform{T},
                  X::AbstractArray) where {T} =
    _scattering_batch!(Matrix{T}(undef, flat_rows(st), size(X)[end]), b, st, X)

for (TT, WS, fun) in (
        (:SphericalScatteringTransform, :SphericalWorkspace, :spherical_scattering!),
        (:SphericalMonogenicScatteringTransform, :SphericalMonogenicWorkspace, :spherical_monogenic_scattering!))
    @eval function _spherical_batch_perfield!(out::AbstractMatrix, st::SphericalCore.$TT{T},
                                              X::AbstractArray) where {T}
        D = ndims(X)
        # An empty block (more MPI ranks than columns) has no field to size the workspace from.
        size(X, D) == 0 && return out
        ws = st.ws
        S1 = zeros(T, st.J)
        S2 = st.max_order >= 2 ? zeros(T, st.J, st.J) : Matrix{T}(undef, 0, 0)
        flat = Vector{eltype(out)}(undef, size(out, 1))
        @inbounds for b in 1:size(X, D)
            r = SphericalCore.$fun(S1, S2, st, ws, selectdim(X, D, b))
            Coefficients.flatten1d!(flat, r)
            copyto!(view(out, :, b), flat)
        end
        return out
    end
end

# The monogenic cascade has no batched form: its amplitude needs the Riesz gradient, which is built
# from per-field spin-0 transforms.
_scattering_batch!(out::AbstractMatrix, ::CB.AbstractSerialBackend,
                   st::SphericalCore.SphericalMonogenicScatteringTransform, X::AbstractArray) =
    _spherical_batch_perfield!(out, st, X)

# A widened plan takes a scattered `(M, B)` stack with `B > 1`; a structured grid's fields are
# matrices.
_spherical_batches(plan, X::AbstractArray) =
    SphericalCore.supports_batch(plan)

# Scattered-point backends that support `batch_plan` run one transform per cascade step for the
# whole stack. Structured and in-core plans take the per-field loop.
function _scattering_batch!(out::AbstractMatrix, ::CB.AbstractSerialBackend,
                            st::SphericalCore.SphericalScatteringTransform{T}, X::AbstractArray) where {T}
    B = size(X)[end]
    B == 0 && return out
    _spherical_batches(st.plan, X) || return _spherical_batch_perfield!(out, st, X)

    ScatteringCore.with_batch_workspaces(() -> SphericalBatchWorkspace(st, B), st,
                                         B, CB.SerialBackend(), 1) do entries
        _apply_cached_batch!(out, first(entries), X)
    end
    return out
end

struct SphericalBatchWorkspace{S,I,A1,A2,F}
    transform::S
    input::I
    S1::A1
    S2::A2
    flat::F
end

Base.show(io::IO, ws::SphericalBatchWorkspace) =
    print(io, "SphericalBatchWorkspace(J=", ws.transform.J, ", B=", size(ws.input, 2), ")")

function SphericalBatchWorkspace(st, B::Int)
    stb = SphericalCore.task_local_batch(st, B)
    prototype = SphericalCore.sphere_field_buffer(stb.plan)
    T = real(eltype(prototype))
    S1 = similar(prototype, T, st.J, B)
    S2 = similar(prototype, T, st.max_order >= 2 ? (st.J, st.J, B) : (0, 0, B))
    flat = similar(prototype, T, flat_rows(st), B)
    return SphericalBatchWorkspace(stb, prototype, S1, S2, flat)
end

_apply_cached_batch!(out, entry::ScatteringCore.CachedBatch, X) =
    _apply_batch!(out, entry.workspace, X)
_apply_batch!(out, ws::Batched.BatchWorkspace, X) = Batched.batch_cascade!(out, ws, X)

function _apply_batch!(out, ws::SphericalBatchWorkspace, X)
    st = ws.transform
    n = size(out, 2)
    n < size(ws.input, 2) && fill!(ws.input, zero(eltype(ws.input)))
    copyto!(view(ws.input, :, 1:n), X)
    r = SphericalCore.spherical_scattering_batch!(ws.S1, ws.S2, st, st.ws, ws.input)
    fill!(ws.flat, zero(eltype(ws.flat)))
    view(ws.flat, 1, :) .= r.S0
    view(ws.flat, 2:(st.J + 1), :) .= r.S1
    if !isempty(r.S2)
        for j1 in 1:st.J, j2 in (j1 + 1):st.J
            view(ws.flat, Coefficients.flat_row_s2(j1, j2, st.J), :) .= view(r.S2, j1, j2, :)
        end
    end
    copyto!(out, view(ws.flat, :, 1:n))
    return out
end

"""
    close_transform!(st) -> nothing

`Plans.close_plan!` every spectral plan `st` owns. `st` must not be in use by any task, and is not
usable after.
"""
function close_transform!(st)
    _close_workspaces!(st)
    Plans.close_plan!(st.plan)
    pw = _cascade_workspace(st)
    foreach(Plans.close_plan!, pw.plans)
    return nothing
end
_cascade_workspace(st) = st.pw
_cascade_workspace(st::Monogenic.MonogenicScatteringTransform) = st.mw.pw

function _close_workspaces!(st)
    pool = st.workspaces
    lock(pool.lock) do
        foreach(close_transform!, pool.tasks)
        foreach(w -> ScatteringCore.close_batch_workspace!(w.workspace), pool.batches)
        foreach(close_worker!, pool.workers)
        empty!(pool.tasks)
        empty!(pool.batches)
        empty!(pool.workers)
    end
    return nothing
end

function close_worker! end
ScatteringCore.close_batch_workspace!(ws::Batched.BatchWorkspace) = ScatteringCore.close_batch_workspace!(ws.spectral)
ScatteringCore.close_batch_workspace!(ws::Batched.SpectralBatchWorkspace) = (foreach(Plans.close_plan!, ws.pw.plans); nothing)
ScatteringCore.close_batch_workspace!(ws::SphericalBatchWorkspace) = close_transform!(ws.transform)

function close_transform!(st::SphericalCore.SphericalScatteringTransform)
    _close_workspaces!(st)
    Plans.close_plan!(st.plan)
    st.bands === nothing || foreach(Plans.close_plan!, st.bands)
    st.grid === nothing || Plans.close_plan!(st.grid)
    return nothing
end
function close_transform!(st::SphericalCore.SphericalMonogenicScatteringTransform)
    _close_workspaces!(st)
    Plans.close_plan!(st.plan)
    st.grid === nothing || Plans.close_plan!(st.grid)
    return nothing
end

# What `transform_spec(st)` returns: `rebuild_transform` calls `build(args...; kwargs...)` to
# construct `st` again on another process, where its spectral plans, which are not serializable,
# cannot be sent. `build` is a type or a module-level function, both serialized by name, so the
# workers load the package. `transform` is the plan's own tag, so a device-resident plan, which has
# none, throws in `transform_spec`.
struct TransformSpec{F, A <: Tuple, K <: NamedTuple}
    build::F
    args::A
    kwargs::K
end

rebuild_transform(spec::TransformSpec) = spec.build(spec.args...; spec.kwargs...)

# The gridded constructors take `(T, N, J)` and the bank's orientation keyword `bank`.
_gridded_spec(build, st, N, bank::NamedTuple) =
    TransformSpec(build, (real(eltype(st.filter_bank.averaging)), N, st.filter_bank.J),
                  (; bank..., max_order = st.max_order, oversampling = st.pw.oversampling,
                   filters = FilterBanks.filter_storage(st.filter_bank),
                   transform = Plans.spectral_backend(st.plan)))

transform_spec(st::ScatteringCore.ScatteringTransform{1}) =
    _gridded_spec(ScatteringCore.ScatteringTransform{1}, st, st.dims[1], (Q = st.filter_bank.Q,))
transform_spec(st::ScatteringCore.ScatteringTransform{2}) =
    _gridded_spec(ScatteringCore.ScatteringTransform{2}, st, st.dims, (L = st.filter_bank.L,))
transform_spec(st::ScatteringCore.ScatteringTransform{3}) =
    _gridded_spec(ScatteringCore.ScatteringTransform{3}, st, st.dims,
                  (n_orient = st.filter_bank.n_orient,))

transform_spec(st::Monogenic.MonogenicScatteringTransform) =
    TransformSpec(Monogenic.MonogenicScatteringTransform,
                  (real(eltype(st.buffer_signal_fft)), st.dims, st.filter_bank.J),
                  (Q = st.filter_bank.Q, max_order = st.max_order,
                   oversampling = st.mw.pw.oversampling,
                   filters = FilterBanks.filter_storage(st.filter_bank),
                   transform = Plans.spectral_backend(st.plan)))

# The nonuniform and spherical specs carry the sample locations and omit the batch width: a rebuilt
# transform is single-field, so it accepts a column block of any width.
function transform_spec(st::ScatteredPlanar.ScatteredPlanarScatteringTransform)
    x, y = Plans.plan_points(st.plan)
    a = Plans.plan_analysis(st.plan)
    # The plan's points lie on its domain of period `2π`.
    return TransformSpec(ScatteredPlanar.build,
                         (real(eltype(st.filter_bank.averaging)), x, y, st.plan.ms, st.filter_bank.J),
                         (L = st.filter_bank.L, max_order = st.max_order,
                          transform = Plans.spectral_backend(st.plan), weights = st.weights,
                          period = (2π, 2π), analysis = a.analysis, maxiter = a.maxiter, rtol = a.rtol,
                          damp = a.damp, tol = a.tol, nthreads = a.nthreads,
                          oversampling = st.pw.oversampling,
                          filters = FilterBanks.filter_storage(st.filter_bank)))
end

function transform_spec(st::Union{SphericalCore.SphericalScatteringTransform,
                                  SphericalCore.SphericalMonogenicScatteringTransform})
    pts = SphericalCore.plan_points(st.plan)
    # No points means the plan is defined by its band limit alone (a structured backend).
    pts === nothing && return _structured_spec(st)
    solver = SphericalCore.plan_solver(st.plan)
    w = SphericalCore.plan_weights(st.plan)
    # Copies, so a transform rebuilt in this process shares no array with `st`. The element type
    # follows from the points.
    return TransformSpec(_scattered_build(st), (copy(pts[1]), copy(pts[2]), st.lmax, st.J),
                         (max_order = st.max_order, transform = Plans.spectral_backend(st.plan),
                          weights = w === nothing ? nothing : copy(w), rtol = solver.rtol,
                          maxiter = solver.maxiter, analysis = solver.analysis,
                          nufft = SphericalCore.plan_nufft(st.plan),
                          grid_lmax = SphericalCore.grid_band(st.grid)))
end

_scattered_build(::SphericalCore.SphericalScatteringTransform) = spherical_scattering
_scattered_build(::SphericalCore.SphericalMonogenicScatteringTransform) = spherical_monogenic_scattering

_structured_spec(st::SphericalCore.SphericalScatteringTransform{T}) where {T} =
    TransformSpec(_structured_scattering,
                  (T, st.lmax, st.J, st.bands === nothing ? nothing :
                                                           map(SphericalCore.sphere_lmax, st.bands)),
                  (max_order = st.max_order, transform = Plans.spectral_backend(st.plan)))
_structured_spec(st::SphericalCore.SphericalMonogenicScatteringTransform{T}) where {T} =
    TransformSpec(structured_spherical_monogenic_scattering, (T, st.lmax, st.J),
                  (max_order = st.max_order, transform = Plans.spectral_backend(st.plan)))

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
                                transform=SpectralBackends.AutoSpectralBackend(),
                                analysis=Plans.WeightedAdjoint(), weights=nothing, tol=nothing,
                                maxiter=100, rtol=Plans.default_solver_rtol(T, transform, tol), damp=0,
                                ntrans=1, nthreads=0, filters=StoredFilters(), oversampling=J)
    scattered_planar_scattering([T,] grid, ms, J; kwargs...)

Build a 2D planar scattering transform for a scalar field sampled at scattered points `(x, y)`. A
nonuniform DFT analyses the samples onto a uniform Fourier **mode grid** of size `ms = (m1, m2)`, and
the cascade of the gridded `ScatteringCore.ScatteringTransform{2}`, with its oriented Morlet bank and
`oversampling`, runs on that grid: each coefficient is the domain mean of a modulus of the field the
analysis recovers. Apply it to a length-`M` vector of samples.

`period` is the length of the domain the points sample along each axis and `origin` its lower
corner (a number for both axes, or a 2-tuple): mode `k` has wavelength `period/k`, so the wavelets'
physical scales are those of the domain, and point sets drawn from one domain share them. Points
`0:m-1` with `period = m` are the nodes of the length-`m` DFT. With FlowGeometries loaded, a 2-D
Cartesian `FlowGeometries.Grids` grid stands in for the points: its active cells are the points, its
periodic axes supply their periods, bounded axes their extents plus the coarsest wavelet width,
the lowest cell centre supplies the origin, and cell measures supply default `weights`.

`transform` names the nonuniform transform: `SpectralBackends.DirectSumSpectralBackend` is the in-core,
dependency-free exact NUDFT (always available, `O(M·prod(ms))`);
`FlowTransformBindings.NonuniformFFTsBackend()` and `FlowTransformBindings.FINUFFTBackend()` select a
nonuniform-FFT library (`using NonuniformFFTs` / `using FINUFFT`);
and `SpectralBackends.AutoSpectralBackend` (the default) takes NonuniformFFTs when it is loaded, FINUFFT
when only it is, and the direct sum otherwise. `analysis=Plans.WeightedAdjoint()` takes the adjoint weighted by
`weights`, the quadrature of each Fourier coefficient's domain integral: exact on the nodes of the
DFT, an estimate elsewhere. `analysis=Plans.LeastSquares()` fits the band-limited coefficients by least squares (slower,
for irregular sampling). `weights` (length `M`, normalized to sum to 1) are uniform by default; `S0` is
the mean of the analysed field. `tol` is the NUFFT tolerance (ignored by the exact direct sum).
`ntrans = B` builds a transform that runs `B` fields per call through one execution per cascade step
(see [`scattering_batch`](@ref)). `filters = FilterBanks.ComputedFilters()` evaluates the wavelets on
demand.

The solve is LSMR (`FlowTransformBindings.lsmr!`), which costs one type-2 and one type-1 per
iteration and keeps both `‖b - Af‖` and `‖A†(b - Af)‖` monotone, so stopping at `maxiter` returns the
best iterate reached. `rtol` targets `‖A†(b - Af)‖ ≤ rtol ‖A†b‖` and defaults to the accuracy of
the transform under it: `sqrt(eps(T))` for the exact direct sum and at least `10·tol` for a fast library, whose type-1 and type-2
are adjoints only to its own tolerance
([`Plans.default_solver_rtol`](@ref ScatteringTransforms.Plans.default_solver_rtol)). `damp` is a
Tikhonov `λ` minimising `‖Af - b‖² + λ²‖f‖²`, `0` by default; with `prod(ms) > M` the problem is
underdetermined, construction warns, and LSMR returns the minimum-norm solution unless a `λ` is given.

The Krylov process terminates within `prod(ms)` steps, and `maxiter` caps it earlier. A truncated
solve returns the best iterate reached but is reproducible only to the accuracy of the transform
under it: a multi-threaded NUFFT accumulates its spreading in a run-dependent order, which an
unconverged solve amplifies by about `cond(A)²`. Raise `maxiter` until the solve converges for an
answer independent of the thread count, or set `nthreads = 1` for bitwise-identical repeats.

`nthreads` is the fast library's own thread count, honoured by every copy of the plan,
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
                         transform=SpectralBackends.AutoSpectralBackend(), analysis=Plans.LeastSquares(),
                         rtol=SphericalCore.default_rtol(transform), maxiter=500, weights=nothing,
                         ntrans=1, nufft=SpectralBackends.AutoSpectralBackend(), grid_lmax=2lmax)

Build a spherical scattering transform for a scalar field at scattered points `(θ, φ)` on S², using
smooth difference-of-Gaussians band-pass wavelets. `transform` names the spherical-harmonic transform:
`SpectralBackends.DirectSumSpectralBackend` is the in-core, dependency-free exact least-squares
transform (always available, `O(M·(lmax+1)²)`); `SpectralBackends.NUFSHTSpectralBackend` uses the
NUFSHT fast path (needs `using NUFSHT`); `SpectralBackends.AutoSpectralBackend` (the default) picks
NUFSHT if its extension is loaded, else the direct transform. Accurate analysis
needs the sampling to resolve the band limit, i.e. roughly `M ≳ (lmax+1)²` well-distributed points.

The samples are analysed once, onto the harmonics of degree `≤ lmax`. Every band, modulus, mean and
second analysis after that runs on the Gauss–Legendre grid of band limit `grid_lmax`
([`SphericalCore.gauss_legendre_grid`](@ref)): each coefficient is the spherical mean of a modulus of the
field the analysis recovers, and `S0` is that field's mean. A modulus is not band limited, and the grid's
analysis of it onto degree `≤ lmax` is exact for its content up to degree `2grid_lmax + 1 − lmax`.

`analysis = Plans.LeastSquares()` fits by LSMR, configured by `rtol` and `maxiter`, and returns
its best iterate at a stopping limit. `Plans.WeightedAdjoint()` takes the adjoint weighted by
`weights`: the expansion of a degree-`lmax` field where the weights integrate the
product of two degree-`lmax` harmonics exactly, and an approximation elsewhere. `weights` (length `M`,
normalized to sum to 1) are the quadrature of that adjoint and the weights of the fit, uniform by
default. `ntrans = B` builds a plan that transforms `B` co-located fields per call. `nufft` names the
NUFFT library NUFSHT runs on; the in-core plan runs none and accepts only an `Auto` `nufft`.
"""
function spherical_scattering(pts_theta::AbstractVector{TT}, pts_phi::AbstractVector, lmax::Int, J::Int;
                              max_order::Int = 2,
                              transform::SB.AbstractSpectralBackend = SB.AutoSpectralBackend(),
                              rtol::Real = SphericalCore.default_rtol(transform),
                              maxiter::Int = 500, weights = nothing, ntrans::Int = 1,
                              nufft::SB.AbstractSpectralBackend = SB.AutoSpectralBackend(),
                              analysis::Plans.AbstractAnalysis = Plans.LeastSquares(),
                              grid_lmax::Int = 2lmax) where {TT<:Real}
    T = float(TT)
    plan = SphericalCore.make_spherical_plan(SphericalCore.SphericalScatteringTransform, transform, pts_theta, pts_phi, lmax, T;
                                             rtol = rtol, maxiter = maxiter, weights = weights,
                                             ntrans = ntrans, nufft = nufft, analysis = analysis)
    return SphericalCore.SphericalScatteringTransform(lmax, J, max_order, plan,
                                             SphericalCore.dog_sigma2(lmax, J, T), nothing,
                                             SphericalCore.grid_plan(plan, _grid_lmax(grid_lmax, lmax)))
end

_grid_lmax(L::Int, lmax::Int) =
    L >= lmax ? L : throw(ArgumentError("grid_lmax = $L is below the band limit lmax = $lmax"))

"""
    spherical_monogenic_scattering(pts_theta, pts_phi, lmax, J; max_order=2,
                                   transform=SpectralBackends.AutoSpectralBackend(), analysis=Plans.LeastSquares(),
                                   rtol=SphericalCore.default_rtol(transform), maxiter=500,
                                   weights=nothing, nufft=SpectralBackends.AutoSpectralBackend(),
                                   grid_lmax=2lmax)

Build a **monogenic** spherical scattering transform for a scalar field at scattered points
`(θ, φ)` on S². The nonlinearity is the spherical monogenic amplitude
`A_j = √(U⁰_j² + |U^R_j|²)`, where `U⁰_j` is the difference-of-Gaussians band-pass and `U^R_j` is
the spin-1 Riesz field `R = ð∘(−Δ_S)^{-1/2}`. The Riesz energy `|U^R_j|² = |∇_S g_j|²` (with
`g_j = (−Δ_S)^{-1/2} U⁰_j`) is synthesised exactly from `g_j`'s coefficients
(`SphericalCore.sphere_riesz_energy!`). The cascade runs on the Gauss–Legendre grid and the keywords
are as in [`spherical_scattering`](@ref).
"""
function spherical_monogenic_scattering(pts_theta::AbstractVector{TT}, pts_phi::AbstractVector,
                                        lmax::Int, J::Int; max_order::Int = 2,
                                        transform::SB.AbstractSpectralBackend = SB.AutoSpectralBackend(),
                                        rtol::Real = SphericalCore.default_rtol(transform),
                                        maxiter::Int = 500, weights = nothing,
                                        nufft::SB.AbstractSpectralBackend = SB.AutoSpectralBackend(),
                                        analysis::Plans.AbstractAnalysis = Plans.LeastSquares(),
                                        grid_lmax::Int = 2lmax) where {TT<:Real}
    T = float(TT)
    plan = SphericalCore.make_spherical_plan(SphericalCore.SphericalMonogenicScatteringTransform, transform, pts_theta, pts_phi, lmax, T;
                                             rtol = rtol, maxiter = maxiter, weights = weights,
                                             nufft = nufft, analysis = analysis)
    return SphericalCore.SphericalMonogenicScatteringTransform(lmax, J, max_order, plan,
                                                      SphericalCore.dog_sigma2(lmax, J, T),
                                                      SphericalCore.grid_plan(plan, _grid_lmax(grid_lmax, lmax)))
end

"""
    structured_spherical_scattering([T=Float64,] lmax, J; max_order=2,
                                    transform=SpectralBackends.AutoSpectralBackend(),
                                    band_headroom=nothing, rtol=1e-8, maxiter=500)

Build a spherical scattering transform for a scalar field sampled on the structured equiangular grid
(`Nθ = lmax+1` colatitudes, `Nφ = 2lmax+1` longitudes), using the same smooth difference-of-Gaussians
band-pass wavelets as [`spherical_scattering`](@ref). Apply it to a `(Nθ, Nφ)` grid of samples; obtain
the grid points with [`structured_sphere_points`](@ref). `transform` names the spherical-harmonic
transform:
`SpectralBackends.DirectSumSpectralBackend` (in-core, dependency-free), or
`SpectralBackends.FSHTSpectralBackend` (the fast exact SHT, needs `using FastSphericalHarmonics`);
`SpectralBackends.AutoSpectralBackend` picks the fast path if its extension is loaded, else the
direct SHT.

On a backend that can narrow its band limit (FastSphericalHarmonics), a finite `band_headroom`
synthesises band `j` at band limit `band_headroom·ℓ_j`, discarding the tail
[`SphericalCore.band_lmax`](@ref) describes; the default `nothing` runs the full-band-limit
cascade bit for bit. `rtol` and `maxiter` configure the in-core plan's analysis solve.
"""
structured_spherical_scattering(::Type{T}, lmax::Int, J::Int; max_order::Int = 2,
                                transform::SB.AbstractSpectralBackend = SB.AutoSpectralBackend(),
                                band_headroom::Union{Nothing,Real} = nothing,
                                rtol::Real = 1.0e-8, maxiter::Int = 500) where {T} =
    _structured_scattering(T, lmax, J, _band_limits(lmax, J, band_headroom);
                           max_order = max_order, transform = transform, rtol = rtol,
                           maxiter = maxiter)

# The structured transform with scale `j` at band limit `ls[j]`; on a backend that cannot narrow,
# every scale runs at `lmax`.
_band_limits(lmax, J, ::Nothing) = nothing
_band_limits(lmax, J, headroom::Real) =
    [SphericalCore.band_lmax(lmax, J, j, headroom) for j in 1:J]

function _structured_scattering(::Type{T}, lmax::Int, J::Int, ls;
                                max_order::Int, transform::SB.AbstractSpectralBackend,
                                rtol::Real = 1.0e-8, maxiter::Int = 500) where {T}
    plan = SphericalCore.make_structured_plan(transform, lmax, T; rtol = rtol, maxiter = maxiter)
    return SphericalCore.SphericalScatteringTransform(lmax, J, max_order, plan,
                                             SphericalCore.dog_sigma2(lmax, J, T),
                                             SphericalCore.band_plans(plan, lmax, ls))
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
    return SphericalCore.SphericalMonogenicScatteringTransform(lmax, J, max_order, plan,
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
Riesz vector. `st` is a scattered or structured spherical monogenic transform. With the in-core direct SH
plan (dependency-free) the Riesz field is synthesized as the surface gradient of `g = (−Δ_S)^{-1/2}U⁰`;
with a NUFSHT-backed plan it uses spin-weighted synthesis — the two agree to solver accuracy.
"""
spherical_monogenic_components(args...; kwargs...) = throw(ArgumentError(
    "spherical_monogenic_components expects a spherical monogenic transform, a field and a scale index."))

spherical_monogenic_components(st::SphericalCore.SphericalMonogenicScatteringTransform{<:Any, <:SphericalCore.DirectSHTSphericalPlan},
                               field::AbstractArray, j::Int) =
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
        st1 = ScatteringCore.ScatteringTransform(32, 3; Q = 1, max_order = 2, transform = SB.DirectSumSpectralBackend())
        c1 = st1(zeros(Float64, 32))
        Coefficients.flatten1d(c1)
        ScatteringFields.scattering_field(st1, zeros(Float64, 32); subsample = 1)
        scattering_batch(st1, zeros(Float64, 32, 2))
        Reductions.normalized_coefficients(c1)

        st2 = ScatteringCore.ScatteringTransform((16, 16), 2; L = 4, max_order = 2, transform = SB.DirectSumSpectralBackend())
        c2 = st2(zeros(Float64, 16, 16))
        Scattering2D.compute_shape_sparsity(Coefficients.first_order(c2), Coefficients.second_order(c2), st2.filter_bank.meta)

        st3 = ScatteringCore.ScatteringTransform((8, 8, 8), 2; n_orient = 6, max_order = 2, transform = SB.DirectSumSpectralBackend())
        st3(zeros(Float64, 8, 8, 8))
    end
end

end # module
