module ScatteringTransformsKernelAbstractionsExt

"""
    ScatteringTransformsKernelAbstractionsExt — vendor-neutral GPU scattering

The device-agnostic GPU execution path, dispatched on `CB.GPUBackend{B}` where `B` is any
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
using ComputationalBackends: ComputationalBackends as CB
using ScatteringTransforms: ScatteringTransforms as ST

# ---------------------------------------------------------------------------
# Vendor-neutral spectral plan (cuFFT / rocFFT / FFTW via AbstractFFTs dispatch)
# ---------------------------------------------------------------------------

"""
    AbstractFFTsScatteringPlan{T,FP,IP}

Spectral plan backed by any `AbstractFFTs` plan (cuFFT / rocFFT / FFTW). Mirrors `FFTWScatteringPlan`:
in-place `!` transforms via `mul!`, allocating non-`!` transforms via `plan * x` (autodiff-friendly).
"""
struct AbstractFFTsScatteringPlan{T, FP, IP} <: ST.Plans.AbstractScatteringPlan
    fwd::FP
    inv::IP
end

function ST.Plans.abstractffts_plan(dummy::AbstractArray{Complex{T}}; region = 1:ndims(dummy),
                                    fft_nthreads::Int = 1) where {T}
    # On a host array `AbstractFFTs` dispatches to FFTW, whose thread count is process-global and is
    # raised by unrelated packages simply being loaded — so without pinning it here, how many threads
    # this plan uses (and therefore how many tasks it spawns, and allocates, per execution) would
    # depend on load order. Device plans ignore this; the hook is a no-op when FFTW is absent.
    #
    # Serialised on the package-wide planner lock, which is also what makes that pin of a process
    # global safe against a concurrent build restoring it underneath this one.
    return Base.@lock ST.Plans.PLANNER_LOCK ST.Plans.with_fft_nthreads(fft_nthreads) do
        fwd = AbstractFFTs.plan_fft(dummy, region)
        inv = AbstractFFTs.plan_ifft(dummy, region)
        return AbstractFFTsScatteringPlan{T, typeof(fwd), typeof(inv)}(fwd, inv)
    end
end

# The default `show` of an `AbstractFFTs` plan can reach FFTW's `fftw_sprint_plan`, which can
# segfault; and a device plan prints its whole device buffer. One line instead.
Base.show(io::IO, ::AbstractFFTsScatteringPlan{T}) where {T} =
    print(io, "AbstractFFTsScatteringPlan{", T, "}()")
Base.show(io::IO, ::MIME"text/plain", p::AbstractFFTsScatteringPlan) = show(io, p)

# A device plan is not reconstructible from a tag alone — it needs its device too — so it declares
# no spectral backend and `transform_spec` refuses rather than silently reporting a host FFTW plan.
ST.Plans.task_local_plan(p::AbstractFFTsScatteringPlan) = p

ST.Plans.forward_transform!(out::AbstractArray, p::AbstractFFTsScatteringPlan, x::AbstractArray) =
    (LinearAlgebra.mul!(out, p.fwd, x); out)
ST.Plans.inverse_transform!(out::AbstractArray, p::AbstractFFTsScatteringPlan, x::AbstractArray) =
    (LinearAlgebra.mul!(out, p.inv, x); out)
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

# ---------------------------------------------------------------------------
# Device-resident transform constructors (parity with the CUDA ext, vendor-neutral)
# ---------------------------------------------------------------------------

function ST.Scattering1D.ScatteringTransform1D(::Type{T}, N::Int, J::Int, gpu::CB.GPUBackend;
                                               Q::Int = 1, max_order::Int = 2) where {T}
    b = gpu.backend
    cpu_fb = ST.FilterBanks.build_filter_bank1d(T, N, J; Q = Q)
    wavelets = [_to_device(b, ψ) for ψ in cpu_fb.wavelets]
    averaging = _to_device(b, cpu_fb.averaging)
    filter_bank = ST.FilterBanks.FilterBank1D(wavelets, averaging, cpu_fb.meta, cpu_fb.J, cpu_fb.Q)
    tree = ST.PathGraph.build_tree([m.j_eff for m in filter_bank.meta], max_order)
    groups = _groups(tree, length(wavelets), max_order)
    plan = ST.Plans.abstractffts_plan(_dzeros(b, Complex{T}, (N,)))
    return ST.Scattering1D.ScatteringTransform1D(filter_bank, tree, groups, max_order, plan,
        _dzeros(b, Complex{T}, (N,)), _dzeros(b, Complex{T}, (N,)), _dzeros(b, Complex{T}, (N,)),
        _dzeros(b, T, (N,)), _dzeros(b, T, (N,)), _dzeros(b, Complex{T}, (N,)))
end
# Scattered-planar parity: everything downstream of the points follows their array type, so the
# device constructor only has to put the points on the device. Which NUFFT runs there is the
# caller's `spectral` choice — NonuniformFFTs is the KernelAbstractions-native one.
function ST.ScatteredPlanar.build(::Type{T}, x::AbstractVector, y::AbstractVector,
                                  ms::NTuple{2,Int}, J::Int, gpu::CB.GPUBackend;
                                  kwargs...) where {T}
    b = gpu.backend
    return ST.ScatteredPlanar.build(T, _to_device(b, T.(x)), _to_device(b, T.(y)), ms, J; kwargs...)
end
ST.ScatteredPlanar.build(x::AbstractVector, y::AbstractVector, ms::NTuple{2,Int}, J::Int,
                         gpu::CB.GPUBackend; kwargs...) =
    ST.ScatteredPlanar.build(Float32, x, y, ms, J, gpu; kwargs...)

ST.Scattering1D.ScatteringTransform1D(N::Int, J::Int, gpu::CB.GPUBackend; kwargs...) =
    ST.Scattering1D.ScatteringTransform1D(Float32, N, J, gpu; kwargs...)

function ST.Scattering2D.ScatteringTransform2D(::Type{T}, N::NTuple{2,Int}, J::Int, gpu::CB.GPUBackend;
                                               L::Int = 8, max_order::Int = 2) where {T}
    b = gpu.backend
    cpu_fb = ST.FilterBanks.build_filter_bank2d(T, N, J; L = L)
    wavelets = [_to_device(b, ψ) for ψ in cpu_fb.wavelets]
    averaging = _to_device(b, cpu_fb.averaging)
    filter_bank = ST.FilterBanks.FilterBank2D(wavelets, averaging, cpu_fb.meta, cpu_fb.J, cpu_fb.L)
    tree = ST.PathGraph.build_tree([m.j_eff for m in filter_bank.meta], max_order)
    groups = _groups(tree, length(wavelets), max_order)
    plan = ST.Plans.abstractffts_plan(_dzeros(b, Complex{T}, N))
    return ST.Scattering2D.ScatteringTransform2D(filter_bank, tree, groups, max_order, plan,
        _dzeros(b, Complex{T}, N), _dzeros(b, Complex{T}, N), _dzeros(b, Complex{T}, N),
        _dzeros(b, T, N), _dzeros(b, T, N), _dzeros(b, Complex{T}, N))
end
ST.Scattering2D.ScatteringTransform2D(N::NTuple{2,Int}, J::Int, gpu::CB.GPUBackend; kwargs...) =
    ST.Scattering2D.ScatteringTransform2D(Float32, N, J, gpu; kwargs...)

function ST.Scattering3D.ScatteringTransform3D(::Type{T}, N::NTuple{3,Int}, J::Int, gpu::CB.GPUBackend;
                                               n_orient::Int = 6, max_order::Int = 2) where {T}
    b = gpu.backend
    cpu_fb = ST.FilterBanks.build_filter_bank3d(T, N, J; n_orient = n_orient)
    wavelets = [_to_device(b, ψ) for ψ in cpu_fb.wavelets]
    averaging = _to_device(b, cpu_fb.averaging)
    filter_bank = ST.FilterBanks.FilterBank3D(wavelets, averaging, cpu_fb.meta, cpu_fb.J, cpu_fb.n_orient)
    tree = ST.PathGraph.build_tree([m.j_eff for m in filter_bank.meta], max_order)
    groups = _groups(tree, length(wavelets), max_order)
    plan = ST.Plans.abstractffts_plan(_dzeros(b, Complex{T}, N))
    return ST.Scattering3D.ScatteringTransform3D(filter_bank, tree, groups, max_order, plan,
        _dzeros(b, Complex{T}, N), _dzeros(b, Complex{T}, N), _dzeros(b, Complex{T}, N),
        _dzeros(b, T, N), _dzeros(b, T, N), _dzeros(b, Complex{T}, N))
end
ST.Scattering3D.ScatteringTransform3D(N::NTuple{3,Int}, J::Int, gpu::CB.GPUBackend; kwargs...) =
    ST.Scattering3D.ScatteringTransform3D(Float32, N, J, gpu; kwargs...)

# The cascade's work list. Path topology is integer bookkeeping, so it stays on the host even for a
# device-resident transform.
_groups(tree, nw::Int, max_order::Int) =
    max_order >= 2 ? ST.PathGraph.order2_groups(tree, nw) : [(j, Int[], Int[]) for j in 1:nw]

# ---------------------------------------------------------------------------
# Batched-FFT throughput path — the performant `scattering_batch(::GPUBackend, …)`
# ---------------------------------------------------------------------------

"""
    gpu_batch_workspace(gpu, st, B) -> ST.Batched.BatchWorkspace

Device-resident state for repeated `scattering_batch!` calls at a fixed batch size `B`: the batched
`AbstractFFTs` plan, `(spatial…, B)` scratch, and **device copies of the filter wavelets**, so a
host-built `st` also runs on a device.

The cascade itself is [`ST.Batched.batch_cascade!`](@ref) — the same code the CPU path runs. It
names no device: every operation is a broadcast, a plan execution, or a `mapreduce`, all of which
dispatch on the array type. This extension therefore supplies allocation and a plan, and nothing
else.
"""
function gpu_batch_workspace(gpu::CB.GPUBackend, st, B::Int)
    b = gpu.backend
    T = real(eltype(st.filter_bank.averaging))
    spatial = size(st.buffer_mod)                 # (N,) 1D, (Ny,Nx) 2D, (Nz,Ny,Nx) 3D
    D = length(spatial)
    stack = (spatial..., B)
    plan = ST.Plans.abstractffts_plan(_dzeros(b, Complex{T}, stack); region = 1:D)
    cz() = _dzeros(b, Complex{T}, stack)
    red = _dzeros(b, T, (ntuple(_ -> 1, D)..., B))
    # Filters carry the trailing singleton so they broadcast over the batch axis; shaping them here
    # keeps `reshape` (which allocates an array object) out of the cascade's inner loop.
    wavelets = [reshape(_to_device(b, ψ), (spatial..., 1)) for ψ in st.filter_bank.wavelets]
    return ST.Batched.BatchWorkspace(plan, _dzeros(b, T, stack), cz(), cz(), cz(), cz(),
        _dzeros(b, T, stack), red, reshape(red, B), wavelets, length(wavelets),
        st.groups, 1 / prod(spatial))
end

for (Mod, TT, CF, ND, dimarg) in (
        (:Scattering1D, :ScatteringTransform1D, :_flen1d, 2, :(size(X, 2))),
        (:Scattering2D, :ScatteringTransform2D, :_flen2d, 3, :(size(X, 3))),
        (:Scattering3D, :ScatteringTransform3D, :_flen3d, 4, :(size(X, 4))))
    @eval begin
        function ST.scattering_batch(gpu::CB.GPUBackend, st::ST.$Mod.$TT,
                                     X::AbstractArray{<:Any,$ND})
            T = real(eltype(st.filter_bank.averaging))
            out = _dzeros(gpu.backend, T, ($CF(st, T), $dimarg))
            return ST.scattering_batch!(out, gpu, st, X)
        end

        # `workspace` is annotated so the keyword is concretely typed: an unannotated keyword is
        # `Any`, and the call then boxes it on every invocation.
        function ST.scattering_batch!(out::AbstractMatrix, gpu::CB.GPUBackend, st::ST.$Mod.$TT,
                                      X::AbstractArray{<:Any,$ND};
                                      workspace::ST.Batched.BatchWorkspace =
                                          gpu_batch_workspace(gpu, st, $dimarg))
            return ST.Batched.batch_cascade!(out, workspace, X)
        end
    end
end

_flen1d(st, ::Type{T}) where {T} = ST.Coefficients.flatten_length(
    ST.Coefficients.ScatteringCoefficients1D(length(st.filter_bank.wavelets), T;
                                             compute_S2 = st.max_order >= 2))
_flen2d(st, ::Type{T}) where {T} = ST.Coefficients.flatten_length(
    ST.Coefficients.ScatteringCoefficients2D(st.filter_bank.J, st.filter_bank.L, T;
                                             compute_S2 = st.max_order >= 2))
_flen3d(st, ::Type{T}) where {T} = ST.Coefficients.flatten_length(
    ST.Coefficients.ScatteringCoefficients2D(st.filter_bank.J, st.filter_bank.n_orient, T;
                                             compute_S2 = st.max_order >= 2))

end # module ScatteringTransformsKernelAbstractionsExt
