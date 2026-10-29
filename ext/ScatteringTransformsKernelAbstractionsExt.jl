module ScatteringTransformsKernelAbstractionsExt

# Device execution on `CB.GPUBackend{B}` for any `KernelAbstractions.Backend` `B`. The cascade is
# array-type generic (broadcasts and plan executions), so this extension supplies only device
# allocation and `AbstractFFTs` plans, which dispatch on the array type. It adds device-resident
# constructors `ScatteringTransform{D}([T,] dims, J, gpu::GPUBackend; …)` and
# `MonogenicScatteringTransform([T,] dims, J, gpu::GPUBackend; …)`, and the batched
# `scattering_batch(st, X; execution = gpu)` / `scattering_batch!(out, st, X; execution = gpu)`.

using KernelAbstractions: KernelAbstractions as KA
using AbstractFFTs: AbstractFFTs
using LinearAlgebra: LinearAlgebra
using ComputationalBackends: ComputationalBackends as CB
using ScatteringTransforms: ScatteringTransforms as ST

ST.Execution.available(::CB.AbstractGPUBackend) = true

# ---------------------------------------------------------------------------
# Vendor-neutral spectral plan (cuFFT / rocFFT / FFTW via AbstractFFTs dispatch)
# ---------------------------------------------------------------------------

"""
    AbstractFFTsScatteringPlan{T,FP,IP,IPI}

Spectral plan backed by any `AbstractFFTs` plan (cuFFT / rocFFT / FFTW), as `FFTWScatteringPlan` is
by FFTW: in-place `!` transforms through `mul!`, allocating non-`!` transforms through `plan * x`.
"""
struct AbstractFFTsScatteringPlan{T, FP, IP, IPI} <: ST.Plans.AbstractScatteringPlan
    fwd::FP
    inv::IP
    inv!::IPI
end

function ST.Plans.abstractffts_plan(dummy::AbstractArray{Complex{T}}; region = 1:ndims(dummy),
                                    fft_nthreads::Int = 1) where {T}
    # On a host array `AbstractFFTs` dispatches to FFTW, whose process-global thread count other
    # packages change, so the plan names its own. Device plans ignore it.
    return ST.Plans.with_fft_nthreads(fft_nthreads) do
        fwd = AbstractFFTs.plan_fft(dummy, region)
        inv = AbstractFFTs.plan_ifft(dummy, region)
        invp = AbstractFFTs.plan_ifft!(dummy, region)
        return AbstractFFTsScatteringPlan{T, typeof(fwd), typeof(inv), typeof(invp)}(fwd, inv, invp)
    end
end

# The default `show` of an `AbstractFFTs` plan can reach FFTW's `fftw_sprint_plan`, which can
# segfault; and a device plan prints its whole device buffer. One line instead.
Base.show(io::IO, ::AbstractFFTsScatteringPlan{T}) where {T} =
    print(io, "AbstractFFTsScatteringPlan{", T, "}()")
Base.show(io::IO, ::MIME"text/plain", p::AbstractFFTsScatteringPlan) = show(io, p)

# A tag cannot rebuild a device plan without its device, so the plan declares no spectral backend and
# `transform_spec` throws for it.
ST.Plans.task_local_plan(p::AbstractFFTsScatteringPlan) = p

ST.Plans.forward_transform!(out::AbstractArray, p::AbstractFFTsScatteringPlan, x::AbstractArray) =
    (LinearAlgebra.mul!(out, p.fwd, x); out)
ST.Plans.inverse_transform!(out::AbstractArray, p::AbstractFFTsScatteringPlan, x::AbstractArray) =
    out === x ? (p.inv! * x) : (LinearAlgebra.mul!(out, p.inv, x); out)
ST.Plans.inplace_inverse(::AbstractFFTsScatteringPlan) = true
# The prototype array carries the device.
ST.Plans.plan_like(::AbstractFFTsScatteringPlan, prototype::AbstractArray) =
    ST.Plans.abstractffts_plan(prototype; region = 1:ndims(prototype))
ST.Plans.plan_like(::AbstractFFTsScatteringPlan, prototype::AbstractArray, ::Val{D};
                   fft_nthreads::Int = 1) where {D} =
    ST.Plans.abstractffts_plan(prototype; region = 1:D, fft_nthreads = fft_nthreads)
ST.Plans.forward_transform(p::AbstractFFTsScatteringPlan, x::AbstractArray) = p.fwd * x
ST.Plans.inverse_transform(p::AbstractFFTsScatteringPlan, x::AbstractArray) = p.inv * x

# ---------------------------------------------------------------------------
# Device allocation helpers (device-agnostic via KernelAbstractions)
# ---------------------------------------------------------------------------

# Move a host/device array onto `backend`, preserving element type and shape.
function _to_device(backend, A::AbstractArray)
    d = KA.allocate(backend, eltype(A), size(A))
    copyto!(d, A)
    return d
end

# Zero-initialised device array of element type `T` and size `dims` on `backend`.
function _dzeros(backend, ::Type{T}, dims) where {T}
    a = KA.allocate(backend, T, dims)
    fill!(a, zero(T))
    return a
end

_device_bank(b, fb::ST.FilterBanks.FilterBank1D) =
    ST.FilterBanks.FilterBank1D(map(f -> _to_device(b, f), fb.wavelets), _to_device(b, fb.averaging),
                                 fb.meta, fb.J, fb.Q)
_device_bank(b, fb::ST.FilterBanks.FilterBank2D) =
    ST.FilterBanks.FilterBank2D(map(f -> _to_device(b, f), fb.wavelets), _to_device(b, fb.averaging),
                                 fb.meta, fb.J, fb.L)
_device_bank(b, fb::ST.FilterBanks.FilterBank3D) =
    ST.FilterBanks.FilterBank3D(map(f -> _to_device(b, f), fb.wavelets), _to_device(b, fb.averaging),
                                 fb.meta, fb.J, fb.n_orient)
_device_bank(b, fb::ST.FilterBanks.ComputedFilterBank1D) =
    ST.FilterBanks.ComputedFilterBank1D(fb.morlets, fb.rescale, _to_device(b, fb.averaging),
                                         KA.allocate(b, eltype(fb.scratch), size(fb.scratch)), fb.meta, fb.J, fb.Q)
_device_bank(b, fb::ST.FilterBanks.ComputedFilterBank2D) =
    ST.FilterBanks.ComputedFilterBank2D(fb.morlets, fb.rescale, _to_device(b, fb.averaging),
                                         KA.allocate(b, eltype(fb.scratch), size(fb.scratch)), fb.meta, fb.J, fb.L)
_device_bank(b, fb::ST.FilterBanks.ComputedFilterBank3D) =
    ST.FilterBanks.ComputedFilterBank3D(fb.morlets, fb.rescale, _to_device(b, fb.averaging),
                                         KA.allocate(b, eltype(fb.scratch), size(fb.scratch)), fb.meta, fb.J, fb.n_orient)

function _device_bank(b, fb::ST.Monogenic.MonogenicFilterBank{D,T}) where {D,T}
    wavelets = map(f -> _to_device(b, f), fb.wavelets)
    riesz = map(f -> _to_device(b, f), fb.riesz)
    averaging = _to_device(b, fb.averaging)
    return ST.Monogenic.MonogenicFilterBank{D,T,typeof(averaging),typeof(wavelets),typeof(riesz),typeof(fb.meta)}(
        wavelets, riesz, averaging, fb.meta, fb.J, fb.Q)
end

function _device_bank(b, fb::ST.Monogenic.ComputedMonogenicFilterBank{D,T}) where {D,T}
    riesz = map(f -> _to_device(b, f), fb.riesz)
    averaging = _to_device(b, fb.averaging)
    return ST.Monogenic.ComputedMonogenicFilterBank{D,T,typeof(averaging),typeof(riesz),typeof(fb.meta),typeof(fb.profiles)}(
        fb.profiles, fb.rescale, riesz, averaging, KA.allocate(b, T, size(fb.scratch)), fb.meta, fb.J, fb.Q)
end

# ---------------------------------------------------------------------------
# Device-resident transform constructors
# ---------------------------------------------------------------------------

# The periodized cascade's workspace, allocated and planned on the device, with the transform's own
# plan as level 1. Each path's mean is summed on the device, so a single-field transform synchronises
# with the host once per call.
_device_pw(b, fb, groups, dims, ::Type{T}, α::Int, D::Int, plan) where {T} =
    ST.Cascade.build(fb, groups, dims, T, α;
                     alloc = rd -> _dzeros(b, Complex{T}, rd),
                     makeplan = rd -> ST.Plans.abstractffts_plan(_dzeros(b, Complex{T}, rd);
                                                                 region = 1:D),
                     level1 = plan, device_zeros = (S, d) -> _dzeros(b, S, d))

function ST.ScatteringCore.ScatteringTransform{1}(::Type{T}, N::Int, J::Int, gpu::CB.GPUBackend;
                                               Q::Int = 1, max_order::Int = 2, oversampling::Int = J,
                                               filters::ST.FilterBanks.AbstractFilterStorage = ST.FilterBanks.StoredFilters()) where {T}
    b = gpu.backend
    filter_bank = _device_bank(b, ST.FilterBanks.build_filter_bank1d(T, N, J, filters; Q))
    tree = ST.PathGraph.build_tree([m.j_eff for m in filter_bank.meta], max_order)
    groups = _groups(tree, ST.FilterBanks.nwavelets(filter_bank), max_order)
    plan = ST.Plans.abstractffts_plan(_dzeros(b, Complex{T}, (N,)))
    pw = _device_pw(b, filter_bank, groups, (N,), T, oversampling, 1, plan)
    return ST.ScatteringCore.ScatteringTransform{1}(filter_bank, tree, groups, max_order, plan,
        _dzeros(b, Complex{T}, (N,)), _dzeros(b, Complex{T}, (N,)), (N,), pw)
end
# Scattered planar: everything downstream of the points follows their array type, so this moves the
# points to the device. `transform` picks the NUFFT; NonuniformFFTs runs on KernelAbstractions
# devices.
function ST.ScatteredPlanar.build(::Type{T}, x::AbstractVector, y::AbstractVector,
                                  ms::NTuple{2,Int}, J::Int, gpu::CB.GPUBackend;
                                  kwargs...) where {T}
    b = gpu.backend
    return ST.ScatteredPlanar.build(T, _to_device(b, T.(x)), _to_device(b, T.(y)), ms, J; kwargs...)
end
ST.ScatteredPlanar.build(x::AbstractVector, y::AbstractVector, ms::NTuple{2,Int}, J::Int,
                         gpu::CB.GPUBackend; kwargs...) =
    ST.ScatteredPlanar.build(Float32, x, y, ms, J, gpu; kwargs...)

ST.ScatteringCore.ScatteringTransform{1}(N::Int, J::Int, gpu::CB.GPUBackend; kwargs...) =
    ST.ScatteringCore.ScatteringTransform{1}(Float32, N, J, gpu; kwargs...)

function ST.ScatteringCore.ScatteringTransform{2}(::Type{T}, N::NTuple{2,Int}, J::Int, gpu::CB.GPUBackend;
                                               L::Int = 8, max_order::Int = 2, oversampling::Int = J,
                                               filters::ST.FilterBanks.AbstractFilterStorage = ST.FilterBanks.StoredFilters()) where {T}
    b = gpu.backend
    filter_bank = _device_bank(b, ST.FilterBanks.build_filter_bank2d(T, N, J, filters; L))
    tree = ST.PathGraph.build_tree([m.j_eff for m in filter_bank.meta], max_order)
    groups = _groups(tree, ST.FilterBanks.nwavelets(filter_bank), max_order)
    plan = ST.Plans.abstractffts_plan(_dzeros(b, Complex{T}, N))
    pw = _device_pw(b, filter_bank, groups, N, T, oversampling, 2, plan)
    return ST.ScatteringCore.ScatteringTransform{2}(filter_bank, tree, groups, max_order, plan,
        _dzeros(b, Complex{T}, N), _dzeros(b, Complex{T}, N), N, pw)
end
function ST.ScatteringCore.ScatteringTransform{3}(::Type{T}, N::NTuple{3,Int}, J::Int, gpu::CB.GPUBackend;
                                               n_orient::Int = 6, max_order::Int = 2, oversampling::Int = J,
                                               filters::ST.FilterBanks.AbstractFilterStorage = ST.FilterBanks.StoredFilters()) where {T}
    b = gpu.backend
    filter_bank = _device_bank(b, ST.FilterBanks.build_filter_bank3d(T, N, J, filters; n_orient))
    tree = ST.PathGraph.build_tree([m.j_eff for m in filter_bank.meta], max_order)
    groups = _groups(tree, ST.FilterBanks.nwavelets(filter_bank), max_order)
    plan = ST.Plans.abstractffts_plan(_dzeros(b, Complex{T}, N))
    pw = _device_pw(b, filter_bank, groups, N, T, oversampling, 3, plan)
    return ST.ScatteringCore.ScatteringTransform{3}(filter_bank, tree, groups, max_order, plan,
        _dzeros(b, Complex{T}, N), _dzeros(b, Complex{T}, N), N, pw)
end
ST.ScatteringCore.ScatteringTransform{D}(dims::NTuple{D,Int}, J::Int, gpu::CB.GPUBackend; kwargs...) where {D} =
    ST.ScatteringCore.ScatteringTransform{D}(Float32, dims, J, gpu; kwargs...)
ST.ScatteringCore.ScatteringTransform{1}(::Type{T}, dims::NTuple{1,Int}, J::Int,
                                        gpu::CB.GPUBackend; kwargs...) where {T} =
    ST.ScatteringCore.ScatteringTransform{1}(T, first(dims), J, gpu; kwargs...)
ST.ScatteringCore.ScatteringTransform(::Type{T}, dims::NTuple{D,Int}, J::Int,
                                     gpu::CB.GPUBackend; kwargs...) where {T,D} =
    ST.ScatteringCore.ScatteringTransform{D}(T, dims, J, gpu; kwargs...)
ST.ScatteringCore.ScatteringTransform(::Type{T}, N::Int, J::Int,
                                     gpu::CB.GPUBackend; kwargs...) where {T} =
    ST.ScatteringCore.ScatteringTransform{1}(T, N, J, gpu; kwargs...)
ST.ScatteringCore.ScatteringTransform(dims::NTuple{D,Int}, J::Int,
                                     gpu::CB.GPUBackend; kwargs...) where {D} =
    ST.ScatteringCore.ScatteringTransform(Float32, dims, J, gpu; kwargs...)
ST.ScatteringCore.ScatteringTransform(N::Int, J::Int, gpu::CB.GPUBackend; kwargs...) =
    ST.ScatteringCore.ScatteringTransform(Float32, N, J, gpu; kwargs...)

function ST.Monogenic.MonogenicScatteringTransform(::Type{T}, dims::NTuple{D,Int}, J::Int, gpu::CB.GPUBackend;
                                          Q::Int = 1, max_order::Int = 2,
                                          oversampling::Int = J,
                                          filters::ST.FilterBanks.AbstractFilterStorage = ST.FilterBanks.StoredFilters()) where {T, D}
    b = gpu.backend
    fb = _device_bank(b, ST.Monogenic.build_monogenic_bank(T, dims, J, filters; Q))
    tree, groups = ST.Monogenic.tree_groups(fb, max_order)
    plan = ST.Plans.abstractffts_plan(_dzeros(b, Complex{T}, dims))
    pw = _device_pw(b, fb, groups, dims, T, oversampling, D, plan)
    return ST.Monogenic.assemble(fb, tree, groups, max_order, plan, pw)
end
ST.Monogenic.MonogenicScatteringTransform(::Type{T}, N::Int, J::Int, gpu::CB.GPUBackend; kwargs...) where {T} =
    ST.Monogenic.MonogenicScatteringTransform(T, (N,), J, gpu; kwargs...)
ST.Monogenic.MonogenicScatteringTransform(dims::NTuple{D,Int}, J::Int, gpu::CB.GPUBackend; kwargs...) where {D} =
    ST.Monogenic.MonogenicScatteringTransform(Float32, dims, J, gpu; kwargs...)
ST.Monogenic.MonogenicScatteringTransform(N::Int, J::Int, gpu::CB.GPUBackend; kwargs...) =
    ST.Monogenic.MonogenicScatteringTransform(Float32, (N,), J, gpu; kwargs...)

# The cascade's work list. Path topology is integer bookkeeping, so it stays on the host even for a
# device-resident transform.
_groups(tree, nw::Int, max_order::Int) =
    max_order >= 2 ? ST.PathGraph.order2_groups(tree, nw) : [(j, Int[], Int[]) for j in 1:nw]

# ---------------------------------------------------------------------------
# Batched transforms on the device
# ---------------------------------------------------------------------------

"""
    gpu_batch_workspace(gpu, st, B) -> ST.Batched.BatchWorkspace

Device-resident state for repeated `scattering_batch!(out, st, X; execution = gpu, workspace)`
calls at a fixed batch size `B`: the batched `AbstractFFTs` plan, `(spatial…, B)` scratch and device
copies of the filters, so a host-built `st` runs on the device through `ST.Batched.batch_cascade!`.
"""
function gpu_batch_workspace(gpu::CB.GPUBackend, st, B::Int)
    b = gpu.backend
    T = real(eltype(st.filter_bank.averaging))
    D = length(st.dims)                           # (N,) 1D, (Ny,Nx) 2D, (Nz,Ny,Nx) 3D
    # Filters carry a trailing singleton so they broadcast over the batch axis.
    fb = _device_bank(b, st.filter_bank)
    return ST.Batched.build_workspace(fb, st.groups, st.dims, st.pw.oversampling, B, T;
                                      zeros_like = (S, d) -> _dzeros(b, S, d),
                                      makeplan = rd -> ST.Plans.abstractffts_plan(
                                          _dzeros(b, Complex{T}, (rd..., B)); region = 1:D),
                                      shapefilter = f -> reshape(f, (size(f)..., 1)))
end

for (D, CF, ND, dimarg) in (
        (1, :_flen1d, 2, :(size(X, 2))),
        (2, :_flen2d, 3, :(size(X, 3))),
        (3, :_flen3d, 4, :(size(X, 4))))
    @eval begin
        # A complex stack has complex columns, for its complex `S0`.
        function ST._scattering_batch(gpu::CB.GPUBackend, st::ST.ScatteringCore.ScatteringTransform{$D},
                                      X::AbstractArray{<:Any,$ND})
            T = real(eltype(st.filter_bank.averaging))
            out = _dzeros(gpu.backend, ST._flat_eltype(T, X), ($CF(st, T), $dimarg))
            return ST._scattering_batch!(out, gpu, st, X)
        end

        # The annotation keeps the `workspace` keyword concretely typed, so the call does not box it.
        function ST._scattering_batch!(out::AbstractMatrix, gpu::CB.GPUBackend,
                                       st::ST.ScatteringCore.ScatteringTransform{$D},
                                       X::AbstractArray{<:Any,$ND};
                                       workspace::Union{Nothing,ST.Batched.BatchWorkspace} = nothing)
            workspace === nothing || return ST.Batched.batch_cascade!(out, workspace, X)
            ST.ScatteringCore.with_batch_workspaces(() -> gpu_batch_workspace(gpu, st, $dimarg),
                                                   st, $dimarg, gpu, 1) do entries
                ST._apply_cached_batch!(out, first(entries), X)
            end
            return out
        end
    end
end

_memory_buffer(st::ST.ScatteredPlanar.ScatteredPlanarScatteringTransform) = st.X_modes
_memory_buffer(st::ST.Monogenic.MonogenicScatteringTransform) = st.buffer_signal_fft
_memory_buffer(st::Union{ST.SphericalCore.SphericalScatteringTransform,ST.SphericalCore.SphericalMonogenicScatteringTransform}) =
    st.ws.C

function _require_backend(gpu, st)
    KA.get_backend(_memory_buffer(st)) == gpu.backend || throw(ArgumentError(
        "build the transform on the requested device before applying this batch"))
    return nothing
end

const _DeviceResidentTransform = Union{ST.ScatteredPlanar.ScatteredPlanarScatteringTransform,
    ST.SphericalCore.SphericalScatteringTransform, ST.SphericalCore.SphericalMonogenicScatteringTransform,
    ST.Monogenic.MonogenicScatteringTransform}

function ST._scattering_batch(gpu::CB.GPUBackend, st::_DeviceResidentTransform, X::AbstractArray)
    _require_backend(gpu, st)
    T = ST._flat_eltype(ST._coefficient_eltype(st), X)
    out = KA.allocate(gpu.backend, T, (ST.flat_rows(st), size(X)[end]))
    return ST._scattering_batch!(out, gpu, st, X)
end

function ST._scattering_batch!(out::AbstractMatrix, gpu::CB.GPUBackend, st::_DeviceResidentTransform,
                               X::AbstractArray)
    _require_backend(gpu, st)
    return ST._scattering_batch!(out, CB.SerialBackend(), st, X)
end

_flen1d(st, ::Type{T}) where {T} = ST.Coefficients.flatten_length(
    ST.Coefficients.ScatteringCoefficients1D(ST.FilterBanks.nwavelets(st.filter_bank), T;
                                             compute_S2 = st.max_order >= 2))
_flen2d(st, ::Type{T}) where {T} = ST.Coefficients.flatten_length(
    ST.Coefficients.ScatteringCoefficients2D(st.filter_bank.J, st.filter_bank.L, T;
                                             compute_S2 = st.max_order >= 2))
_flen3d(st, ::Type{T}) where {T} = ST.Coefficients.flatten_length(
    ST.Coefficients.ScatteringCoefficients2D(st.filter_bank.J, st.filter_bank.n_orient, T;
                                             compute_S2 = st.max_order >= 2))

end # module ScatteringTransformsKernelAbstractionsExt
