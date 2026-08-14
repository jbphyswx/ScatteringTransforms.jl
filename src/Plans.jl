module Plans

"""
    Plans.jl — Spectral transform plans

The scattering transform needs a forward/inverse spectral transform to convolve with the
frequency-domain wavelets. The core ships a dependency-free default — a **direct-summation DFT** —
and fast paths live in extensions (FFTW for uniform sampling, `AbstractFFTs` on device, FINUFFT /
NonuniformFFTs for scattered points).

A plan implements two in-place primitives:

    forward_transform!(out, plan, x)   # x -> X̂   (fft convention)
    inverse_transform!(out, plan, x)   # X̂ -> x   (ifft convention, 1/N scaled)

so the transform engine never references FFTW/CUDA directly. Plans transform the leading `D`
dimensions of their argument and leave any trailing dimension as an untouched batch axis, matching
`AbstractFFTs.plan_fft(A, 1:D)`.

Which transform a plan performs is selected by a `SpectralBackends` tag, not by a `Symbol`.
"""

using LinearAlgebra: LinearAlgebra
using SpectralBackends: SpectralBackends as SB

"""
    AbstractScatteringPlan

Supertype for spectral transform plans. A plan implements [`forward_transform!`](@ref) and
[`inverse_transform!`](@ref); the in-core default is [`DirectSumPlan`](@ref), with fast paths
(FFTW, `AbstractFFTs`/device, FINUFFT, NonuniformFFTs) provided by extensions.
"""
abstract type AbstractScatteringPlan end

"""
    forward_transform!(out, plan, x) -> out

In-place forward (fft-convention) spectral transform. Methods provided by concrete plans.
"""
function forward_transform! end

"""
    inverse_transform!(out, plan, x) -> out

In-place inverse (ifft-convention, `1/N`-scaled) spectral transform.
"""
function inverse_transform! end

"""
    forward_transform(plan, x) -> X̂
    inverse_transform(plan, x) -> x

Non-mutating, allocating, element-type-generic spectral transforms — the autodiff-friendly
counterparts of the in-place `forward_transform!`/`inverse_transform!`. They never touch the plan's
preallocated scratch and never mutate their inputs, so they accept `ForwardDiff.Dual`/`Float32`
inputs and are differentiable by reverse-mode backends. Used by the non-mutating `scattering(st, x)`
path; the in-place `!` versions remain the production hot path.

[`DirectSumPlan`](@ref) implements these as dense per-axis DFT matrix-multiplies (`W*x`) —
differentiable by every AD backend with no special rules. The matrices are built once on first use
and cached on the plan, so the `O(N²)` build is not repeated per call.
"""
function forward_transform end

"""
    inverse_transform(plan, x) -> x

See [`forward_transform`](@ref).
"""
function inverse_transform end

"""
    spectral_backend(plan) -> SpectralBackends.AbstractSpectralBackend

The tag that would rebuild `plan`. Each plan type declares its own, so a transform can be
reconstructed faithfully on a remote worker; plans that cannot be rebuilt from a tag alone (a
device-resident FFT plan needs its device too) throw rather than report a host plan.
"""
spectral_backend(plan::AbstractScatteringPlan) = throw(ArgumentError(
    "no spectral backend tag is registered for $(typeof(plan)), so a transform using it cannot be " *
    "rebuilt from a serialisable spec — construct the transform on each worker explicitly."))

"""
    task_local_plan(plan) -> plan

A plan equivalent to `plan` that is safe to use concurrently with it. Stateless plans (FFTW,
`AbstractFFTs`) return themselves; plans carrying mutable scratch return a copy that shares their
read-only tables and owns fresh scratch. Called once per task by the parallel backends.
"""
task_local_plan(plan::AbstractScatteringPlan) = plan

"""
    with_fft_nthreads(f, n) -> f()

Run `f` with the FFT library's global thread count set to `n`, restoring it afterwards.

FFTW's thread count is process-global and is raised as a side effect of loading unrelated packages,
so a plan built without pinning it inherits whatever was last set — and a plan built for more threads
than it needs spawns (and allocates) a task per thread on *every* execution. Plan builders wrap
construction in this so a plan's threading is a property of the plan, not of load order.

The default is a no-op: only the FFTW extension has a global count to set. It takes `args...` so the
extension's fixed-arity method is strictly more specific and *adds* a method rather than overwriting
this one — overwriting is an error during precompilation.
"""
with_fft_nthreads(f, args...) = f()

"""
    fftw_plan(T, dims; nbatch, planning, fft_nthreads) -> AbstractScatteringPlan

Build the FFTW-backed plan. The real method lives in the FFTW extension; the definition here is a
throwing stub, so an explicit `FFTSpectralBackend()` request dispatches straight to the extension
when it is loaded and to an actionable error when it is not — with no capability lookup on either
path.
"""
fftw_plan(args...; kwargs...) = throw(ArgumentError(
    "FFTSpectralBackend requires the FFTW extension. Run `using FFTW`."))

"""
    abstractffts_plan(dummy_device_array; region=1:ndims(dummy_device_array))

Vendor-neutral device FFT plan constructor. Declaration only — the sole method is provided by the
KernelAbstractions extension, which builds forward/inverse plans via `AbstractFFTs.plan_fft` /
`plan_ifft` on `dummy_device_array`. Because those dispatch on the array *type*, the same builder
yields cuFFT (`CuArray`), rocFFT (`ROCArray`), or FFTW (plain `Array`) plans — so the GPU scattering
path is device-agnostic. `region` selects the transformed dimensions (e.g. `(1, 2)` for a batched
`(Ny, Nx, B)` stack, leaving the batch axis untouched).
"""
function abstractffts_plan end

# Which fast paths exist right now. Only `Auto*` and the "whichever NUFFT library is loaded"
# resolution call these, because only they have a choice to make; an explicitly named backend
# dispatches straight to its extension's plan builder — or to that builder's throwing stub — and
# performs no lookup at all. `hasmethod` cannot substitute here: the stubs make it always true.
_have_fftw() = Base.get_extension(parentmodule(@__MODULE__), :ScatteringTransformsFFTWExt) !== nothing
_have_finufft() = Base.get_extension(parentmodule(@__MODULE__), :ScatteringTransformsFINUFFTExt) !== nothing
_have_nonuniformffts() =
    Base.get_extension(parentmodule(@__MODULE__), :ScatteringTransformsNonuniformFFTsExt) !== nothing

# ---------------------------------------------------------------------------
# Spectral backend tags
#
# Uniform-grid transforms take `SpectralBackends` tags directly. The two NUFFT libraries need
# distinguishing, which a single `NUFFTSpectralBackend` cannot do, so each gets its own tag under
# the shared abstract supertype.
# ---------------------------------------------------------------------------

"FINUFFT fast path for scattered/nonuniform planar points; requires `using FINUFFT`."
struct FINUFFTBackend <: SB.AbstractNUFFTSpectralBackend end

"NonuniformFFTs.jl fast path for scattered/nonuniform planar points; requires `using NonuniformFFTs`."
struct NonuniformFFTsBackend <: SB.AbstractNUFFTSpectralBackend end

"""
    make_plan(spectral, T, dims; nbatch=1, kwargs...) -> AbstractScatteringPlan

Build the spectral plan selected by `spectral` for arrays whose leading dimensions are `dims` and
whose element type is `Complex{T}`, with a trailing batch axis of length `nbatch`.

`SpectralBackends.DirectSumSpectralBackend` is the dependency-free in-core default;
`SpectralBackends.FFTSpectralBackend` requires `using FFTW`;
`SpectralBackends.AutoSpectralBackend` takes the FFTW fast path when its extension is loaded and
otherwise the in-core direct sum.
"""
make_plan(::SB.AbstractDirectSumSpectralBackend, ::Type{T}, dims; nbatch::Int = 1, kwargs...) where {T} =
    DirectSumPlan(T, dims; nbatch = nbatch)

make_plan(::SB.AbstractFFTSpectralBackend, ::Type{T}, dims; nbatch::Int = 1, kwargs...) where {T} =
    fftw_plan(T, dims; nbatch = nbatch, kwargs...)

make_plan(::SB.AbstractAutoSpectralBackend, ::Type{T}, dims; nbatch::Int = 1, kwargs...) where {T} =
    _have_fftw() ? fftw_plan(T, dims; nbatch = nbatch, kwargs...) :
    DirectSumPlan(T, dims; nbatch = nbatch)

"""
    finufft_scattered_plan(x, y, ms, T; period, solve, maxiter, rtol, eps)
    nonuniformffts_scattered_plan(x, y, ms, T; period, solve, maxiter, rtol, eps)

Fast-path scattered-planar plan constructors. The real methods live in the FINUFFT and
NonuniformFFTs extensions; the definitions here are throwing stubs, so naming one of those backends
explicitly costs a plain dispatch rather than a capability lookup.
"""
finufft_scattered_plan(args...; kwargs...) = throw(ArgumentError(
    "FINUFFTBackend requires the FINUFFT extension. Run `using FINUFFT`."))

"See [`finufft_scattered_plan`](@ref)."
nonuniformffts_scattered_plan(args...; kwargs...) = throw(ArgumentError(
    "NonuniformFFTsBackend requires the NonuniformFFTs extension. Run `using NonuniformFFTs`."))

"""
    make_scattered_plan(spectral, x, y, ms, T; period, solve, maxiter, rtol, eps) -> AbstractScatteringPlan

Build the scattered/nonuniform planar plan selected by `spectral` over points `(x, y)` and a uniform
mode grid of size `ms`. `SpectralBackends.DirectSumSpectralBackend` is the dependency-free exact
NUDFT; [`FINUFFTBackend`](@ref) and [`NonuniformFFTsBackend`](@ref) select a specific fast library;
`SpectralBackends.NUFFTSpectralBackend` takes whichever fast library is loaded, and
`SpectralBackends.AutoSpectralBackend` falls back to the exact direct sum when neither is.
"""
make_scattered_plan(::SB.AbstractDirectSumSpectralBackend, x, y, ms, ::Type{T}; period = nothing,
                    solve::Bool = false, maxiter::Int = 100, rtol::Real = 1.0e-8,
                    eps = nothing) where {T} =
    DirectNUFFTPlan(x, y, ms, T; period, solve, maxiter, rtol)

make_scattered_plan(::FINUFFTBackend, x, y, ms, ::Type{T}; kwargs...) where {T} =
    finufft_scattered_plan(x, y, ms, T; kwargs...)

make_scattered_plan(::NonuniformFFTsBackend, x, y, ms, ::Type{T}; kwargs...) where {T} =
    nonuniformffts_scattered_plan(x, y, ms, T; kwargs...)

function make_scattered_plan(::SB.AbstractNUFFTSpectralBackend, x, y, ms, ::Type{T}; kwargs...) where {T}
    _have_finufft() && return finufft_scattered_plan(x, y, ms, T; kwargs...)
    _have_nonuniformffts() && return nonuniformffts_scattered_plan(x, y, ms, T; kwargs...)
    throw(ArgumentError("NUFFTSpectralBackend requires a fast NUFFT library. Run `using FINUFFT` " *
                        "or `using NonuniformFFTs`, or pass DirectSumSpectralBackend() for the " *
                        "exact O(M·prod(ms)) direct summation."))
end

function make_scattered_plan(::SB.AbstractAutoSpectralBackend, x, y, ms, ::Type{T}; kwargs...) where {T}
    _have_finufft() && return finufft_scattered_plan(x, y, ms, T; kwargs...)
    _have_nonuniformffts() && return nonuniformffts_scattered_plan(x, y, ms, T; kwargs...)
    return DirectNUFFTPlan(x, y, ms, T; kwargs...)
end

# ---------------------------------------------------------------------------
# In-core direct-summation DFT plan
#
# This is the dependency-free reference: `O(N²)` per axis by construction, so it is the ground
# truth every fast plan is validated against — not a fast path. Load FFTW (or a device FFT) for
# `O(N log N)`. Within that `O(N²)` contract the kernel is written to be optimal: `O(N)` memory (a
# per-axis table of roots of unity, plus its conjugate so the inverse carries no branch), an
# incrementally-advanced twiddle index rather than a modulo in the inner loop, and a lane-major
# traversal so non-leading axes and the batch axis vectorise instead of striding.
# ---------------------------------------------------------------------------

"""
    DirectSumPlan{T,V,D,S}

Direct-summation DFT plan: evaluates `X_k = Σ_n x_n e^{-2πi kn/N}` (and its inverse) by direct
summation, separably over the leading `D` dimensions, leaving a trailing batch axis of length
`nbatch` untouched. Memory is `O(N)` — a per-axis table of roots of unity and its conjugate, never
an `N×N` matrix. `scratch` is `nothing` for `D == 1` and a full-size complex array otherwise.
Containers stay parametric.
"""
struct DirectSumPlan{T, V <: AbstractVector{Complex{T}}, D, S} <: AbstractScatteringPlan
    twiddle::NTuple{D, V}       # twiddle[d][m+1] = exp(-2πi m / N_d)
    twiddle_conj::NTuple{D, V}  # its conjugate — the inverse direction, branch-free
    dims::NTuple{D, Int}        # transformed (leading) dimensions
    nbatch::Int                 # length of the trailing batch axis
    scratch::S
    invscale::T                 # 1 / prod(dims)
    # Explicit inner constructor: binds `T,V,D,S` from the call rather than inferring them from a
    # possibly-empty tuple, and enforces that each per-axis twiddle table has exactly its axis length.
    function DirectSumPlan{T, V, D, S}(twiddle::NTuple{D, V}, twiddle_conj::NTuple{D, V},
                                       dims::NTuple{D, Int}, nbatch::Int, scratch::S,
                                       invscale::T) where {T, V <: AbstractVector{Complex{T}}, D, S}
        for d in 1:D
            length(twiddle[d]) == dims[d] || throw(ArgumentError(
                "DirectSumPlan: twiddle table for axis $d has length $(length(twiddle[d])), " *
                "expected dims[$d] = $(dims[d])"))
        end
        return new{T, V, D, S}(twiddle, twiddle_conj, dims, nbatch, scratch, invscale)
    end
end

function _twiddles(::Type{T}, N::Int) where {T}
    tw = Vector{Complex{T}}(undef, N)
    @inbounds for m in 0:(N - 1)
        tw[m + 1] = cispi(-2 * T(m) / T(N))
    end
    return tw
end

"""
    DirectSumPlan(T, dims; nbatch = 1) -> DirectSumPlan

Build a direct-summation DFT plan over leading dimensions `dims` (an `Int` or `NTuple{D,Int}`) with
a trailing batch axis of length `nbatch`.
"""
function DirectSumPlan(::Type{T}, dims::NTuple{D, Int}; nbatch::Int = 1) where {T, D}
    nbatch >= 1 || throw(ArgumentError("nbatch must be positive, got $nbatch"))
    tw = ntuple(d -> _twiddles(T, dims[d]), D)
    twc = ntuple(d -> conj.(tw[d]), D)
    scratch = D == 1 ? nothing : zeros(Complex{T}, prod(dims) * nbatch)
    return DirectSumPlan{T, Vector{Complex{T}}, D, typeof(scratch)}(
        tw, twc, dims, nbatch, scratch, inv(T(prod(dims))))
end
DirectSumPlan(::Type{T}, N::Int; kwargs...) where {T} = DirectSumPlan(T, (N,); kwargs...)

Base.show(io::IO, p::DirectSumPlan{T, V, D}) where {T, V, D} =
    print(io, "DirectSumPlan{", T, "}(dims=", p.dims, ", nbatch=", p.nbatch, ")")

spectral_backend(::DirectSumPlan) = SB.DirectSumSpectralBackend()

task_local_plan(p::DirectSumPlan{T, V, D, S}) where {T, V, D, S} =
    DirectSumPlan{T, V, D, S}(p.twiddle, p.twiddle_conj, p.dims, p.nbatch,
                              p.scratch === nothing ? nothing : similar(p.scratch), p.invscale)

# Transform the middle axis of the (nb, n, na) view of `x` into `out`, out-of-place.
#
#     out[l, k, ia] = Σ_j tw[(k·j mod n) + 1] · x[l, j, ia]
#
# The twiddle exponent advances by `k` per step and wraps at most once, so the inner loop needs a
# compare-subtract rather than an integer division. `nb == 1` (a leading-axis transform) accumulates
# in a register; `nb > 1` accumulates into the contiguous lane run, which stays in L1 and vectorises.
function _dft_axis!(out, x, tw::AbstractVector, nb::Int, n::Int, na::Int)
    if nb == 1
        @inbounds for ia in 0:(na - 1)
            off = ia * n
            for k in 0:(n - 1)
                acc = zero(eltype(out))
                idx = 0
                for j in 1:n
                    acc += tw[idx + 1] * x[off + j]
                    idx += k
                    idx >= n && (idx -= n)
                end
                out[off + k + 1] = acc
            end
        end
    else
        @inbounds for ia in 0:(na - 1)
            aoff = ia * nb * n
            for k in 0:(n - 1)
                di = aoff + k * nb
                for l in 1:nb
                    out[di + l] = zero(eltype(out))
                end
                idx = 0
                for j in 0:(n - 1)
                    w = tw[idx + 1]
                    si = aoff + j * nb
                    for l in 1:nb
                        out[di + l] += w * x[si + l]
                    end
                    idx += k
                    idx >= n && (idx -= n)
                end
            end
        end
    end
    return out
end

# Separable multi-axis execution, ping-ponging so the final axis writes into `out`.
function _execute!(out, x, p::DirectSumPlan{T, V, D}, twt::NTuple{D}) where {T, V, D}
    total = length(out)
    total == length(x) || throw(DimensionMismatch(
        "output has $(length(out)) elements, input has $(length(x))"))
    total == prod(p.dims) * p.nbatch || throw(DimensionMismatch(
        "plan is for dims $(p.dims) × nbatch $(p.nbatch), got $total elements"))
    if D == 1
        n = p.dims[1]
        return _dft_axis!(out, x, twt[1], 1, n, total ÷ n)
    end
    sc = p.scratch
    dst = isodd(D) ? out : sc
    src = x
    nb = 1
    for d in 1:D
        n = p.dims[d]
        _dft_axis!(dst, src, twt[d], nb, n, total ÷ (nb * n))
        src = dst
        dst = dst === out ? sc : out
        nb *= n
    end
    return out
end

forward_transform!(out::AbstractArray, p::DirectSumPlan, x::AbstractArray) =
    _execute!(out, x, p, p.twiddle)

function inverse_transform!(out::AbstractArray, p::DirectSumPlan, x::AbstractArray)
    _execute!(out, x, p, p.twiddle_conj)
    s = p.invscale
    @inbounds for i in eachindex(out)
        out[i] *= s
    end
    return out
end

# ---------------------------------------------------------------------------
# Non-mutating, autodiff-friendly direct-sum transforms.
#
# The sum is contracted against the plan's `O(N)` twiddle table rather than through a dense `N×N`
# DFT matrix. Both are `O(N²)` in time, but a matrix formulation cannot serve this path's purpose:
# Enzyme has no derivative rule for complex `gemm`, so `W*x` on `ComplexF64` is undifferentiable.
# Contracting the table instead is plain scalar arithmetic every AD backend can trace, allocates
# only the output (measured at N=4096: 64 KiB and 56 ms, against 256 MiB and 143 ms for building the
# matrix per call), and stays element-type generic so `Dual`/`Float32` flow through.
# ---------------------------------------------------------------------------

# Contract dimension `d` of `A` against the DFT kernel from `tw`. The 1↔d transposition is an
# involution, so the same permutation restores the layout.
function _dft_along(tw::AbstractVector, n::Int, A::AbstractArray, d::Int)
    perm = ntuple(i -> i == 1 ? d : (i == d ? 1 : i), ndims(A))
    Ap = d == 1 ? A : permutedims(A, perm)
    sz = size(Ap)
    M = reshape(Ap, n, :)
    # Written as an explicit loop over a fresh output rather than a comprehension: the generator of
    # a comprehension closes over both the constant twiddle table and the active input, which
    # Enzyme's static activity analysis cannot separate. A loop reads both as plain arguments.
    R = similar(M, promote_type(eltype(tw), eltype(M)))
    @inbounds for c in axes(M, 2)
        for k in 0:(n - 1)
            acc = zero(eltype(R))
            idx = 0
            for j in 1:n
                acc += tw[idx + 1] * M[j, c]
                idx += k
                idx >= n && (idx -= n)
            end
            R[k + 1, c] = acc
        end
    end
    Rr = reshape(R, sz...)
    return d == 1 ? Rr : permutedims(Rr, perm)
end

function forward_transform(p::DirectSumPlan{T, V, D}, x::AbstractArray) where {T, V, D}
    y = x
    for d in 1:D
        y = _dft_along(p.twiddle[d], p.dims[d], y, d)
    end
    return y
end

function inverse_transform(p::DirectSumPlan{T, V, D}, x::AbstractArray) where {T, V, D}
    y = x
    for d in 1:D
        y = _dft_along(p.twiddle_conj[d], p.dims[d], y, d)
    end
    return y .* p.invscale
end

# ---------------------------------------------------------------------------
# Dependency-free scattered / nonuniform planar transform (exact direct-summation NUDFT).
#
# The `O(M·prod(ms))` reference that lets `scattered_planar_scattering` run with no external NUFFT
# library — the nonuniform counterpart of `DirectSumPlan`. Same numeric contract as the fast
# `NUFFTScatteringPlan`s: `forward_transform!` is the Type-1 adjoint (points → uniform mode grid),
# or a conjugate-gradient least-squares inversion when `solve`; `inverse_transform!` is the Type-2
# synthesis (modes → points) scaled by `1/prod(ms)`. The mode grid uses FFT ordering, so on a
# uniform `0:m-1` grid these reduce exactly to `fft`/`ifft` and the lattice matches the wavelet
# bank's `fftfreq` layout.
#
# Exact NUDFT by direct summation, separated per axis (`s = 2π(p−min)/period` scaled coordinates,
# `f_d` the FFT-ordered integer frequencies):
#   Type-1:  X[k₁,k₂] = Σ_n c_n · e^{-i f₁[k₁]·sx_n} · e^{-i f₂[k₂]·sy_n}
#   Type-2:  c_n      = Σ_{k₁,k₂} X[k₁,k₂] · e^{+i f₁[k₁]·sx_n} · e^{+i f₂[k₂]·sy_n}
#
# `Ex`/`Exc` are stored `(ms₁, M)` and `Ey`/`Eyc` `(M, ms₂)`/`(ms₂, M)` — each in the layout its
# consumer reads contiguously, so neither the `Sbuf` fill nor the type-2 accumulation gathers.
# ---------------------------------------------------------------------------

# FFT-ordered integer frequencies for a length-`m` axis: 0,1,…,⌈m/2⌉−1, −⌊m/2⌋,…,−1.
_fftfreqs(m::Int) = Int[i <= (m - 1) ÷ 2 ? i : i - m for i in 0:(m - 1)]

struct DirectNUFFTPlan{T, EM <: AbstractMatrix{Complex{T}},
                       CV <: AbstractVector{Complex{T}}} <: AbstractScatteringPlan
    ms::NTuple{2, Int}
    M::Int
    invN::T                 # 1/prod(ms) — makes synthesis the ifft-convention inverse
    solve::Bool
    maxiter::Int
    rtol::T
    Ex::EM                  # (ms[1], M)  e^{-i f₁[k]·sx_n}
    Ey::EM                  # (M, ms[2])  e^{-i f₂[k]·sy_n}   (point index contiguous)
    Exc::EM                 # (ms[1], M)  conj(Ex)
    Eyc::EM                 # (ms[2], M)  conj(Ey)ᵀ
    cj::CV                  # (M) values buffer (shared by Type-1/Type-2)
    Sbuf::EM                # (M, ms[2]) Type-1 scratch
    T1::EM                  # (ms[1], M) Type-2 scratch
    r::EM                   # (ms) CG residual / rhs
    p::EM                   # (ms) CG search direction
    Ap::EM                  # (ms) CG A†A·p
    tmp_pts::CV             # (M) CG scratch (points)
end

function DirectNUFFTPlan(x::AbstractVector, y::AbstractVector, ms::NTuple{2, Int}, ::Type{T};
                         period = nothing, solve::Bool = false, maxiter::Int = 100,
                         rtol::Real = 1.0e-8, eps = nothing) where {T}
    M = length(x)
    length(y) == M || throw(DimensionMismatch("x and y must have equal length"))
    xmin, ymin = T(minimum(x)), T(minimum(y))
    # Default period so a uniform 0:m-1 grid (span m-1) maps to the exact DFT nodes 2π·(0:m-1)/m.
    px = period === nothing ? (T(maximum(x)) - xmin) * ms[1] / (ms[1] - 1) : T(period[1])
    py = period === nothing ? (T(maximum(y)) - ymin) * ms[2] / (ms[2] - 1) : T(period[2])
    sx = T(2π) .* (T.(x) .- xmin) ./ px
    sy = T(2π) .* (T.(y) .- ymin) ./ py
    f1, f2 = _fftfreqs(ms[1]), _fftfreqs(ms[2])
    Ex = Complex{T}[cis(-f1[k] * sx[n]) for k in 1:ms[1], n in 1:M]
    Ey = Complex{T}[cis(-f2[k] * sy[n]) for n in 1:M, k in 1:ms[2]]
    return DirectNUFFTPlan{T, Matrix{Complex{T}}, Vector{Complex{T}}}(
        ms, M, one(T) / prod(ms), solve, maxiter, T(rtol),
        Ex, Ey, conj.(Ex), Matrix(conj.(transpose(Ey))),
        Vector{Complex{T}}(undef, M),
        Matrix{Complex{T}}(undef, M, ms[2]), Matrix{Complex{T}}(undef, ms[1], M),
        Matrix{Complex{T}}(undef, ms), Matrix{Complex{T}}(undef, ms), Matrix{Complex{T}}(undef, ms),
        Vector{Complex{T}}(undef, M))
end

Base.show(io::IO, p::DirectNUFFTPlan{T}) where {T} =
    print(io, "DirectNUFFTPlan{", T, "}(ms=", p.ms, ", M=", p.M, ", solve=", p.solve, ")")

spectral_backend(::DirectNUFFTPlan) = SB.DirectSumSpectralBackend()

function task_local_plan(p::DirectNUFFTPlan{T, EM, CV}) where {T, EM, CV}
    return DirectNUFFTPlan{T, EM, CV}(
        p.ms, p.M, p.invN, p.solve, p.maxiter, p.rtol, p.Ex, p.Ey, p.Exc, p.Eyc,
        similar(p.cj), similar(p.Sbuf), similar(p.T1), similar(p.r), similar(p.p), similar(p.Ap),
        similar(p.tmp_pts))
end

# Type-1 (points → modes): X = Ex · (c ⊙ Ey), all preallocated.
function _nudft_type1!(X::AbstractMatrix, plan::DirectNUFFTPlan, c::AbstractVector)
    @inbounds for k2 in 1:plan.ms[2], n in 1:plan.M
        plan.Sbuf[n, k2] = c[n] * plan.Ey[n, k2]
    end
    LinearAlgebra.mul!(X, plan.Ex, plan.Sbuf)
    return X
end

# Type-2 (modes → points): c_n = Σ_{k₁} Exc[k₁,n]·(X·Eyc)[k₁,n].
function _nudft_type2!(c::AbstractVector, plan::DirectNUFFTPlan{T}, X::AbstractMatrix) where {T}
    LinearAlgebra.mul!(plan.T1, X, plan.Eyc)
    @inbounds for n in 1:plan.M
        acc = zero(Complex{T})
        for k1 in 1:plan.ms[1]
            acc += plan.Exc[k1, n] * plan.T1[k1, n]
        end
        c[n] = acc
    end
    return c
end

inverse_transform!(out_pts::AbstractVector, plan::DirectNUFFTPlan, Xmodes::AbstractMatrix) =
    (_nudft_type2!(plan.cj, plan, Xmodes); @. out_pts = plan.cj * plan.invN; out_pts)

function forward_transform!(Xmodes::AbstractMatrix, plan::DirectNUFFTPlan, x_pts::AbstractVector)
    if plan.solve
        _cg_solve_nudft!(Xmodes, plan, x_pts)
    else
        copyto!(plan.cj, x_pts)
        _nudft_type1!(Xmodes, plan, plan.cj)
    end
    return Xmodes
end

# CG least-squares inversion of the normal equations (A†A)f = A†(N·x), A = Type-2, A† = Type-1 — so
# synthesis (Type-2/N) of the recovered modes reproduces the sampled values. Mirrors the FINUFFT path.
function _cg_solve_nudft!(f::AbstractMatrix, plan::DirectNUFFTPlan{T}, x_pts::AbstractVector) where {T}
    N = one(T) / plan.invN
    copyto!(plan.cj, x_pts)
    _nudft_type1!(plan.r, plan, plan.cj)                       # r = A†x  (modes)
    plan.r .*= N                                               # r = A†(N·x) = rhs
    fill!(f, zero(Complex{T}))
    copyto!(plan.p, plan.r)
    rsold = real(LinearAlgebra.dot(vec(plan.r), vec(plan.r)))
    rs0 = rsold
    rs0 == 0 && return f
    @inbounds for _ in 1:plan.maxiter
        _nudft_type2!(plan.tmp_pts, plan, plan.p)              # tmp = A·p    (points)
        _nudft_type1!(plan.Ap, plan, plan.tmp_pts)             # Ap  = A†A·p  (modes)
        α = rsold / real(LinearAlgebra.dot(vec(plan.p), vec(plan.Ap)))
        f .+= α .* plan.p
        plan.r .-= α .* plan.Ap
        rsnew = real(LinearAlgebra.dot(vec(plan.r), vec(plan.r)))
        sqrt(rsnew) <= plan.rtol * sqrt(rs0) && break
        plan.p .= plan.r .+ (rsnew / rsold) .* plan.p
        rsold = rsnew
    end
    return f
end

end # module Plans
