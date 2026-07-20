module ScatteringTransformsKernelAbstractionsExt

"""
    ScatteringTransformsKernelAbstractionsExt — vendor-neutral GPU scattering

The device-agnostic GPU execution path, dispatched on `Backends.GPUBackend{B}` where `B` is any
`KernelAbstractions.Backend` (CUDA/ROCm/oneAPI/Metal for real hardware, `KA.CPU()` for CPU-parity CI).

The scattering engine (`ScatteringCore`, `Scattering{1,2,3}D`) is already array-type-generic — every hot
op is a fused broadcast or a `mul!` through the plan interface — so this path needs **no hand-written
`@kernel`s**. The only vendor-specific pieces are:

  1. device allocation — `KA.allocate(backend, T, dims)` + `copyto!` / `fill!`;
  2. the FFT plan — `AbstractFFTs.plan_fft(device_array)` / `plan_ifft`, which dispatch on the array
     *type* (cuFFT for `CuArray`, rocFFT for `ROCArray`, FFTW for a plain `Array`).

Two entry points are provided:

  * **device-resident constructors** `ScatteringTransform{1,2,3}D(N, J, gpu::GPUBackend; …)` — build a
    transform whose filter bank + workspace + plan live on the device; the existing generic `st(x_dev)`
    then runs on-device unchanged;
  * **batched-FFT throughput** `scattering_batch(gpu, st, X)` / `scattering_batch!(out, gpu, st, X)` —
    transform a whole `(N…, B)` stack with one batched plan and fused broadcasts.

Core elementwise ops (`apply_modulus!`, complexify) are not overridden by a broad `::AbstractArray`
method: that would also capture CPU arrays once this extension loads, and base broadcast is already
correct on device arrays.
"""

using KernelAbstractions: KernelAbstractions as KA
using AbstractFFTs: AbstractFFTs
using LinearAlgebra: LinearAlgebra
using ScatteringTransforms: ScatteringTransforms

const ST = ScatteringTransforms
const Plans = ST.Plans
const Backends = ST.Backends
const Coefficients = ST.Coefficients
const PathGraph = ST.PathGraph
const FilterBanks = ST.FilterBanks

# ---------------------------------------------------------------------------
# Vendor-neutral spectral plan (cuFFT / rocFFT / FFTW via AbstractFFTs dispatch)
# ---------------------------------------------------------------------------

"""
    AbstractFFTsScatteringPlan{T,FP,IP}

Spectral plan backed by any `AbstractFFTs` plan (cuFFT / rocFFT / FFTW). Mirrors `FFTWScatteringPlan`:
in-place `!` transforms via `mul!`, allocating non-`!` transforms via `plan * x` (autodiff-friendly).
"""
struct AbstractFFTsScatteringPlan{T, FP, IP} <: Plans.AbstractScatteringPlan
    fwd::FP
    inv::IP
end

function Plans.abstractffts_plan(dummy::AbstractArray{Complex{T}}; region = 1:ndims(dummy)) where {T}
    fwd = AbstractFFTs.plan_fft(dummy, region)
    inv = AbstractFFTs.plan_ifft(dummy, region)
    return AbstractFFTsScatteringPlan{T, typeof(fwd), typeof(inv)}(fwd, inv)
end

Plans.forward_transform!(out::AbstractArray, p::AbstractFFTsScatteringPlan, x::AbstractArray) =
    (LinearAlgebra.mul!(out, p.fwd, x); out)
Plans.inverse_transform!(out::AbstractArray, p::AbstractFFTsScatteringPlan, x::AbstractArray) =
    (LinearAlgebra.mul!(out, p.inv, x); out)
Plans.forward_transform(p::AbstractFFTsScatteringPlan, x::AbstractArray) = p.fwd * x
Plans.inverse_transform(p::AbstractFFTsScatteringPlan, x::AbstractArray) = p.inv * x

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

# ---------------------------------------------------------------------------
# Device-resident transform constructors (parity with the CUDA ext, vendor-neutral)
# ---------------------------------------------------------------------------

function ST.Scattering1D.ScatteringTransform1D(N::Int, J::Int, gpu::Backends.GPUBackend;
                                               Q::Int = 1, max_order::Int = 2, T::Type = Float32)
    b = gpu.backend
    cpu_fb = FilterBanks.build_filter_bank1d(N, J; Q = Q, T = T)
    wavelets = [_to_device(b, ψ) for ψ in cpu_fb.wavelets]
    averaging = _to_device(b, cpu_fb.averaging)
    filter_bank = FilterBanks.FilterBank1D(wavelets, averaging, cpu_fb.meta, cpu_fb.J, cpu_fb.Q)
    tree = PathGraph.build_tree([m.j_eff for m in filter_bank.meta], max_order)
    plan = Plans.abstractffts_plan(_dzeros(b, Complex{T}, (N,)))
    num_w = length(filter_bank.wavelets)
    buffer_input      = _dzeros(b, Complex{T}, (N,))
    buffer_signal_fft = _dzeros(b, Complex{T}, (N,))
    buffer_conv       = _dzeros(b, Complex{T}, (N,))
    buffer_mod        = _dzeros(b, T, (N,))
    U1_buffers, U1_fft_buffers = _u1_buffers(b, T, (N,), num_w, max_order)
    return ST.Scattering1D.ScatteringTransform1D(filter_bank, tree, max_order, plan,
        buffer_input, buffer_signal_fft, buffer_conv, buffer_mod, U1_buffers, U1_fft_buffers)
end

function ST.Scattering2D.ScatteringTransform2D(N::NTuple{2,Int}, J::Int, gpu::Backends.GPUBackend;
                                               L::Int = 8, max_order::Int = 2, T::Type = Float32)
    b = gpu.backend
    cpu_fb = FilterBanks.build_filter_bank2d(N, J; L = L, T = T)
    wavelets = [_to_device(b, ψ) for ψ in cpu_fb.wavelets]
    averaging = _to_device(b, cpu_fb.averaging)
    filter_bank = FilterBanks.FilterBank2D(wavelets, averaging, cpu_fb.meta, cpu_fb.J, cpu_fb.L)
    tree = PathGraph.build_tree([m.j_eff for m in filter_bank.meta], max_order)
    plan = Plans.abstractffts_plan(_dzeros(b, Complex{T}, N))
    num_w = length(filter_bank.wavelets)
    buffer_input      = _dzeros(b, Complex{T}, N)
    buffer_signal_fft = _dzeros(b, Complex{T}, N)
    buffer_conv       = _dzeros(b, Complex{T}, N)
    buffer_mod        = _dzeros(b, T, N)
    U1_buffers, U1_fft_buffers = _u1_buffers(b, T, N, num_w, max_order)
    return ST.Scattering2D.ScatteringTransform2D(filter_bank, tree, max_order, plan,
        buffer_input, buffer_signal_fft, buffer_conv, buffer_mod, U1_buffers, U1_fft_buffers)
end

function ST.Scattering3D.ScatteringTransform3D(N::NTuple{3,Int}, J::Int, gpu::Backends.GPUBackend;
                                               n_orient::Int = 6, max_order::Int = 2, T::Type = Float32)
    b = gpu.backend
    cpu_fb = FilterBanks.build_filter_bank3d(N, J; n_orient = n_orient, T = T)
    wavelets = [_to_device(b, ψ) for ψ in cpu_fb.wavelets]
    averaging = _to_device(b, cpu_fb.averaging)
    filter_bank = FilterBanks.FilterBank3D(wavelets, averaging, cpu_fb.meta, cpu_fb.J, cpu_fb.n_orient)
    tree = PathGraph.build_tree([m.j_eff for m in filter_bank.meta], max_order)
    plan = Plans.abstractffts_plan(_dzeros(b, Complex{T}, N))
    num_w = length(filter_bank.wavelets)
    buffer_input      = _dzeros(b, Complex{T}, N)
    buffer_signal_fft = _dzeros(b, Complex{T}, N)
    buffer_conv       = _dzeros(b, Complex{T}, N)
    buffer_mod        = _dzeros(b, T, N)
    U1_buffers, U1_fft_buffers = _u1_buffers(b, T, N, num_w, max_order)
    return ST.Scattering3D.ScatteringTransform3D(filter_bank, tree, max_order, plan,
        buffer_input, buffer_signal_fft, buffer_conv, buffer_mod, U1_buffers, U1_fft_buffers)
end

# Per-wavelet order-2 scratch (empty typed vectors when max_order < 2).
function _u1_buffers(backend, ::Type{T}, dims, num_w, max_order) where {T}
    real_proto    = _dzeros(backend, T, dims)
    complex_proto = _dzeros(backend, Complex{T}, dims)
    if max_order >= 2
        U1_buffers     = [_dzeros(backend, T, dims) for _ in 1:num_w]
        U1_fft_buffers = [_dzeros(backend, Complex{T}, dims) for _ in 1:num_w]
        return U1_buffers, U1_fft_buffers
    else
        return typeof(real_proto)[], typeof(complex_proto)[]
    end
end

# ---------------------------------------------------------------------------
# Batched-FFT throughput path — the performant `scattering_batch(::GPUBackend, …)`
# ---------------------------------------------------------------------------

# Flattened-output row (1-based) of the upper-triangular S2 entry (j1, j2), j2 > j1, matching
# `Coefficients.flatten2d!`/`flatten1d!`: [S0; S1(1:n); pairs (j1,j2) for j1=1:n, j2=j1+1:n].
@inline _tri_row(j1::Int, j2::Int, n::Int) =
    1 + n + ((j1 - 1) * n - ((j1 - 1) * j1) ÷ 2) + (j2 - j1)

"""
    GPUBatchWorkspace

Preallocated device state for repeated `scattering_batch!` calls at a fixed batch size `B`: the batched
FFT plan, all scratch buffers, **device copies of the filter wavelets** (so a host-built `st` also works
on a real GPU), and the **precomputed admissible order-2 pair structure**. Once constructed,
`scattering_batch!` does no data-proportional allocation — every hot-loop op is in place against these
buffers. All container/array fields are type parameters, so the struct stays concretely typed.

Built from `GPUBatchWorkspace(gpu, st, B)`; the transform grid and element type are taken from `st`.
"""
struct GPUBatchWorkspace{P, RA, CA, WV, G}
    plan::P        # batched AbstractFFTs plan over the spatial dims of the (N…, B) stack
    xr::RA         # real input,  (N…, B)
    xf::CA         # signal FFT,  (N…, B)  (persistent across the batch)
    c1::CA         # complex scratch A (multiply result / complexify input)
    c2::CA         # complex scratch B (inverse output / U1 FFT)
    c3::CA         # complex scratch C (order-2 child inverse; keeps c2=U1f intact)
    rmod::RA       # real modulus scratch, (N…, B)
    sbuf::RA       # spatial-reduction target, (1…, B)
    wavelets::WV   # device copies of the filter wavelets (indexed 1:n)
    n::Int         # number of wavelets
    groups::G      # precomputed admissible (j1, children) pairs; empty when max_order < 2
end

function GPUBatchWorkspace(gpu::Backends.GPUBackend, st, B::Int)
    b = gpu.backend
    T = real(eltype(st.filter_bank.averaging))
    spatial = size(st.buffer_mod)                 # (N,) for 1D, (Ny, Nx) for 2D
    D = length(spatial)
    stack = (spatial..., B)
    xr = _dzeros(b, T, stack)
    xf = _dzeros(b, Complex{T}, stack)
    plan = Plans.abstractffts_plan(_dzeros(b, Complex{T}, stack); region = 1:D)
    wavelets = [_to_device(b, ψ) for ψ in st.filter_bank.wavelets]
    groups = _order2_groups(st.tree, length(wavelets), st.max_order)
    return GPUBatchWorkspace(plan, xr, xf,
        _dzeros(b, Complex{T}, stack), _dzeros(b, Complex{T}, stack), _dzeros(b, Complex{T}, stack),
        _dzeros(b, T, stack), _dzeros(b, T, (ntuple(_ -> 1, D)..., B)),
        wavelets, length(wavelets), groups)
end

# ---- 2D batched ----

function ST.scattering_batch(gpu::Backends.GPUBackend, st::ST.Scattering2D.ScatteringTransform2D,
                             X::AbstractArray{<:Any,3})
    T = real(eltype(st.filter_bank.averaging))
    flen = Coefficients.flatten_length(
        Coefficients.ScatteringCoefficients2D(st.filter_bank.J, st.filter_bank.L, T;
                                              compute_S2 = st.max_order >= 2))
    out = _dzeros(gpu.backend, T, (flen, size(X, 3)))
    return ST.scattering_batch!(out, gpu, st, X)
end

function ST.scattering_batch!(out::AbstractMatrix, gpu::Backends.GPUBackend,
                              st::ST.Scattering2D.ScatteringTransform2D, X::AbstractArray{<:Any,3};
                              workspace::GPUBatchWorkspace = GPUBatchWorkspace(gpu, st, size(X, 3)))
    Ny, Nx, _ = size(X)
    T = real(eltype(st.filter_bank.averaging))
    _batch_common!(out, workspace, X, one(T) / (Ny * Nx), (Ny, Nx))
    return out
end

# ---- 1D batched ----

function ST.scattering_batch(gpu::Backends.GPUBackend, st::ST.Scattering1D.ScatteringTransform1D,
                             X::AbstractMatrix)
    T = real(eltype(st.filter_bank.averaging))
    nw = length(st.filter_bank.wavelets)
    flen = Coefficients.flatten_length(
        Coefficients.ScatteringCoefficients1D(nw, T; compute_S2 = st.max_order >= 2))
    out = _dzeros(gpu.backend, T, (flen, size(X, 2)))
    return ST.scattering_batch!(out, gpu, st, X)
end

function ST.scattering_batch!(out::AbstractMatrix, gpu::Backends.GPUBackend,
                              st::ST.Scattering1D.ScatteringTransform1D, X::AbstractMatrix;
                              workspace::GPUBatchWorkspace = GPUBatchWorkspace(gpu, st, size(X, 2)))
    N = size(X, 1)
    T = real(eltype(st.filter_bank.averaging))
    _batch_common!(out, workspace, X, one(T) / N, (N,))
    return out
end

# Shared batched cascade for 1D/2D. `spatial` is the grid tuple ((N,) or (Ny,Nx)); each wavelet filter
# is reshaped to `(spatial..., 1)` so it broadcasts over the batch axis. All ops write into the
# preallocated workspace buffers (no data-proportional allocation).
function _batch_common!(out::AbstractMatrix, w::GPUBatchWorkspace, X, invN, spatial::NTuple{D,Int}) where {D}
    B = size(X)[end]
    n = w.n
    fshape = (spatial..., 1)

    copyto!(w.xr, X)
    @. w.c1 = complex(w.xr)
    Plans.forward_transform!(w.xf, w.plan, w.c1)                 # w.xf = FFT(input)

    # S0 (row 1); clear the S2 block so non-admissible pairs read as zero (matching flatten2d!/1d!)
    Base.sum!(w.sbuf, w.xr)
    @views out[1, :] .= reshape(w.sbuf, B) .* invN
    @views out[(n + 2):end, :] .= zero(eltype(out))

    # S1 (rows 2 .. n+1). Reshapes are hoisted out of `@.` (which would broadcast `reshape` itself).
    @inbounds for j in 1:n
        ψ = reshape(w.wavelets[j], fshape)
        @. w.c1 = w.xf * ψ
        Plans.inverse_transform!(w.c2, w.plan, w.c1)
        @. w.rmod = abs(w.c2)
        Base.sum!(w.sbuf, w.rmod)
        @views out[1 + j, :] .= reshape(w.sbuf, B) .* invN
    end

    # S2 — j1-outer so only one U1/U1f buffer is live at a time
    @inbounds for (j1, children) in w.groups
        ψ1 = reshape(w.wavelets[j1], fshape)
        @. w.c1 = w.xf * ψ1
        Plans.inverse_transform!(w.c2, w.plan, w.c1)
        @. w.rmod = abs(w.c2)                                    # U1_{j1}
        @. w.c1 = complex(w.rmod)
        Plans.forward_transform!(w.c2, w.plan, w.c1)             # w.c2 = FFT(U1_{j1})
        for j2 in children
            ψ2 = reshape(w.wavelets[j2], fshape)
            @. w.c1 = w.c2 * ψ2
            Plans.inverse_transform!(w.c3, w.plan, w.c1)
            @. w.rmod = abs(w.c3)
            Base.sum!(w.sbuf, w.rmod)
            @views out[_tri_row(j1, j2, n), :] .= reshape(w.sbuf, B) .* invN
        end
    end
    return out
end

# Precompute the admissible order-2 structure once: for each first-order wavelet j1 with ≥1 admissible
# child, the children j2 (from the path tree). Empty when max_order < 2.
function _order2_groups(tree, n::Int, max_order::Int)
    groups = Tuple{Int, Vector{Int}}[]
    max_order >= 2 || return groups
    for j1 in 1:n
        children = Int[]
        for p in PathGraph.order_range(tree, 2)
            idx = PathGraph.path_indices(tree, p)
            idx[1] == j1 && push!(children, idx[2])
        end
        isempty(children) || push!(groups, (j1, children))
    end
    return groups
end

end # module ScatteringTransformsKernelAbstractionsExt
