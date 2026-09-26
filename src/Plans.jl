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
using FlowTransformBindings: FlowTransformBindings as FTB

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
    inplace_inverse(plan) -> Bool

Whether `inverse_transform!(x, plan, x)` is valid. When it is, a transform points its convolution
output at its multiply scratch and owns one fewer field-sized array; when it is not, the two stay
distinct. Default `false`: the direct-sum plan alternates through its own scratch and aliasing would
corrupt an odd number of axes.
"""
inplace_inverse(::AbstractScatteringPlan) = false

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
    close_plan!(plan) -> nothing

Release the foreign-library resources `plan` owns, now rather than at collection. No-op by default,
and safe to call more than once. Whoever *builds* a plan per task calls this when the task is done, so
the memory of one task's plans is returned before the next batch builds its own; cuFINUFFT plans are
released by this alone.
"""
close_plan!(::Any) = nothing

"""
    per_task_nthreads(requested) -> Int

Thread count for a library plan built inside a task: `requested` if the caller asked for one, else 1.

An explicit request is honoured here and not just at construction, or `nufft_nthreads` would be a
keyword that silently does nothing under a threaded backend. The default is one because
[`task_local_plan`](@ref) builds a plan per task, so the tasks have already claimed the cores.

Deriving the default from `Sys.CPU_THREADS` instead gets this wrong twice: those are logical cores, so
on a hyperthreaded machine running one Julia task per *physical* core — already a full machine — the
arithmetic hands every task a second library thread.
"""
per_task_nthreads(requested::Integer) = requested > 0 ? Int(requested) : 1

"""
    AnalysisNotConverged <: Exception

Thrown when an iterative analysis cannot produce a usable answer.

Not thrown merely for stopping early: LSMR decreases both `‖r‖` and `‖A†r‖` monotonically, so an
iterate truncated at `maxiter` is the best one seen and is returned. This is for the cases where the
answer means nothing — a non-finite residual, or a conditioning estimate past what the precision can
represent — because those propagate into every coefficient downstream, and a silent `NaN` is worse
than a stop.
"""
struct AnalysisNotConverged{T <: Real} <: Exception
    residual::T
    rtol::T
    iters::Int
    maxiter::Int
    detail::String
end

AnalysisNotConverged(residual::Real, rtol::Real, iters::Integer, maxiter::Integer,
                     detail::AbstractString) =
    AnalysisNotConverged{promote_type(typeof(residual), typeof(rtol))}(
        residual, rtol, Int(iters), Int(maxiter), String(detail))

function Base.showerror(io::IO, e::AnalysisNotConverged)
    print(io, "AnalysisNotConverged: reached relative residual ", e.residual, " after ", e.iters,
          " of ", e.maxiter, " iterations, against rtol = ", e.rtol, ". ")
    return print(io, e.detail)
end

# A column is unusable when its condition estimate reached `conlim` or `1/eps`
# (`FTB.LSMR_CONDITION`) or its residual is not finite; every other stop, the iteration limit
# included, leaves a usable iterate.
function _check_solve(ws::FTB.LSMRWorkspace, M::Integer, ms::Tuple, rtol::Real, maxiter::Integer)
    @inbounds for k in eachindex(ws.status)
        (ws.status[k] == FTB.LSMR_CONDITION || !isfinite(ws.normr[k])) &&
            _refuse_solve(ws, k, M, ms, rtol, maxiter)
    end
    return nothing
end

@noinline function _refuse_solve(ws::FTB.LSMRWorkspace, k::Int, M::Integer, ms::Tuple, rtol::Real,
                                 maxiter::Integer)
    n = prod(ms)
    advice = n > M ?
        "The mode grid asks for more coefficients than there are samples ($n modes, $M points), so " *
        "$(n - M) mode directions are constrained by no sample: reduce `ms`, supply more points, or " *
        "raise `damp` to regularise them." :
        "The sampling may not determine the mode grid ($n modes, $M points) — check the point set " *
        "for gaps, reduce `ms`, or raise `damp`."
    throw(AnalysisNotConverged(ws.normr[k], rtol, ws.iterations[k], maxiter,
                              "cond(A) ≈ $(Float32(ws.condA[k])), stopped on $(ws.status[k]). " * advice))
end

"""
    default_solver_rtol(::Type{T}, spectral, eps) -> T

Stopping tolerance for the scattered least-squares solve, chosen from the accuracy of the transform
the solve is built on rather than from a fixed constant.

Dispatched on the spectral backend, exactly as `SphericalCore.default_rtol` is: the in-core direct
summation evaluates an exact conjugate pair, so the arithmetic is the only limit and `sqrt(Base.eps(T))`
is the conventional least-squares target. A fast NUFFT's Type-1 and Type-2 are adjoints of each other
only to about `eps`, so the operator the solver sees is Hermitian only to that accuracy and a target
below it asks for something the transform cannot express — which is how a `Float32` plan came to be
asked for `1e-8`, two orders under its own `1e-6`.

`Base.eps` is spelled out because `eps` is a keyword-argument name in every plan constructor here.
"""
default_solver_rtol(::Type{T}, ::SB.AbstractDirectSumSpectralBackend, eps) where {T} =
    T(sqrt(Base.eps(real(float(T)))))
default_solver_rtol(::Type{T}, ::SB.AbstractSpectralBackend, eps) where {T} =
    T(max(10 * Float64(eps === nothing ? FTB.default_tolerance(real(float(T))) : eps),
          sqrt(Base.eps(real(float(T))))))

"""
    warn_underdetermined(M, ms, solve, damp) -> nothing

Warn, once per session, that a solve has more modes than samples and no damping to resolve them.

With `prod(ms) > M` at least `prod(ms) - M` mode directions are constrained by no sample, so the
least-squares problem has no unique solution and the answer depends on the regularisation. LSMR
returns the minimum-norm iterate and its early termination is itself a regulariser, which is why
this warns rather than throws — but the caller is the only one who can say whether that is the
answer they wanted, or whether they should coarsen `ms`, add samples, or set `damp` for explicit
Tikhonov. Both numbers are named because the ratio is what decides how ill-posed the solve is.

No `λ` is chosen for the caller. Sweeping `λ` against a dense Tikhonov reference on this operator
shows why: on quasi-uniform points `A` is a scaled isometry and every `λ` across six decades returns
the same coefficients to `1e-14`, while on clustered points a `λ` small enough to be safe changes
neither the residual nor `‖x‖` in the first five digits. A default that cannot be observed to do
anything is worse than none, since it would still be a number the caller has to reason about.
"""
function warn_underdetermined(M::Integer, ms::Tuple, solve::Bool, damp::Real)
    n = prod(ms)
    (!solve || n <= M || damp > 0) && return nothing
    @warn "Scattered solve is underdetermined: $n modes from $M samples leaves at least $(n - M) " *
          "mode directions unconstrained. LSMR returns the minimum-norm solution; pass `damp` for " *
          "Tikhonov regularisation, or reduce `ms`." maxlog = 1
    return nothing
end

"""
    batch_width(plan) -> Int

Number of co-located fields this plan transforms per execution — `1` unless it was built for a batch.

A nonuniform plan's batch width is fixed when its guru plan is built and cannot vary per execution,
so a plan is either single-field or batched. Callers use this to decide whether a stack can go
through one execution per cascade step instead of one per field.
"""
batch_width(::Any) = 1

"""
    task_local_plan(plan) -> plan

A plan equivalent to `plan` that is safe to use concurrently with it. Stateless plans (FFTW,
`AbstractFFTs`) return themselves; plans carrying mutable scratch return a copy that shares their
read-only tables and owns fresh scratch. Called once per task by the parallel backends.

Two obligations follow from *where* this is called. A method that **builds** rather than shares takes
its thread count from [`per_task_nthreads`](@ref), since the caller has already claimed the cores; and
whoever built it must [`close_plan!`](@ref) it when the task ends.
"""
task_local_plan(plan::AbstractScatteringPlan) = plan

"""
    with_fft_nthreads(f, n) -> f()

Run `f`, which builds plans, holding FFTW.jl's planner lock with FFTW's planner thread count set to `n`
and restored after: `FFTW.set_num_threads(f, n)`, the FFTW extension's method.

FFTW's planner state is process-global and unsafe to use from two threads at once, and FFTW.jl,
FINUFFT, NonuniformFFTs and FastTransforms all plan on that one libfftw3, all serialised by that one
lock. The count is read into a plan when it is built, and a plan built for more threads than it needs
spawns (and allocates) a task per thread on every execution, so plan builders name the count here.
The extensions that plan through `AbstractFFTs` or FastSphericalHarmonics reach FFTW.jl only through
this.

A no-op without FFTW loaded. It takes `args...` so the extension's fixed-arity method is strictly more
specific and adds a method rather than overwriting this one — overwriting is an error during
precompilation.
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

# Whether the FFTW fast path exists right now. Only `Auto` calls this, because only it has a choice to
# make; an explicitly named backend dispatches straight to its extension's plan builder — or to that
# builder's throwing stub — and performs no lookup at all.
_have_fftw() = Base.get_extension(parentmodule(@__MODULE__), :ScatteringTransformsFFTWExt) !== nothing

# The NUFFT library `AutoSpectralBackend` takes: NonuniformFFTs when it is loaded, FINUFFT when only it
# is; `nothing` when neither is.
function _nufft_library()
    for b in (FTB.NonuniformFFTsBackend(), FTB.FINUFFTBackend())
        FTB.is_available(b) && return b
    end
    return nothing
end

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
    plan_like(plan, prototype) -> plan′

A plan of the same kind as `plan`, sized for arrays like `prototype`.

Sizing from an *array* rather than from dimensions plus a backend tag is what lets one piece of code
plan on host and device alike: a device plan has no spectral-backend tag to look up — `spectral_backend`
deliberately throws on it — but it does have arrays, and `AbstractFFTs` dispatches on their type. Used
where a cascade needs plans at resolutions not known when the transform was built.
"""
plan_like(plan::AbstractScatteringPlan, prototype::AbstractArray) =
    make_plan(spectral_backend(plan), real(eltype(prototype)), size(prototype))

"""
    plan_analysis(plan) -> (; solve, maxiter, rtol, damp, eps, nufft_nthreads)

How a scattered-planar plan turns samples into modes, in the form its constructor takes.

Carried rather than re-derived, because none of it follows from the points: a plan rebuilt on another
process without `solve` analyses with the plain Type-1 adjoint where the original ran a least-squares
inversion, and on irregular sampling those are different transforms,
not one slightly less accurate than the other. `eps` is `nothing` for the exact direct sum, which has
no tolerance to honour.
"""
function plan_analysis end

"""
    make_scattered_plan(spectral, x, y, ms, T; period, solve, maxiter, rtol, eps,
                        nufft_nthreads) -> AbstractScatteringPlan

Build the scattered/nonuniform planar plan selected by `spectral` over points `(x, y)` and a uniform
mode grid of size `ms`. `SpectralBackends.DirectSumSpectralBackend` is the dependency-free exact
NUDFT; `FlowTransformBindings.FINUFFTBackend()` and `FlowTransformBindings.NonuniformFFTsBackend()`
select a nonuniform-FFT library; `SpectralBackends.AutoSpectralBackend` takes NonuniformFFTs when it is
loaded and FINUFFT when only it is, and the exact direct sum when neither is.

`nufft_nthreads` sets the library's own thread count (`0`, the default, takes `Threads.nthreads()`).
The direct sum accepts it and ignores it, as it does `eps`, so a caller can pass one set of options
without first knowing which backend it will get.
"""
make_scattered_plan(spectral::SB.AbstractDirectSumSpectralBackend, x, y, ms, ::Type{T};
                    period = nothing, solve::Bool = false, maxiter::Int = 100,
                    rtol::Real = default_solver_rtol(T, spectral, eps), damp::Real = 0,
                    eps = nothing, ntrans::Int = 1, nufft_nthreads::Int = 0) where {T} =
    DirectNUFFTPlan(x, y, ms, T; period, solve, maxiter, rtol, damp, ntrans)

make_scattered_plan(b::Union{FTB.FINUFFTBackend, FTB.NonuniformFFTsBackend}, x, y, ms, ::Type{T};
                    kwargs...) where {T} = nufft_scattered_plan(b, x, y, ms, T; kwargs...)

make_scattered_plan(t::SB.AbstractNUFFTSpectralBackend, x, y, ms, ::Type{T}; kwargs...) where {T} =
    throw(ArgumentError(
        "$(nameof(typeof(t))) names no NUFFT library; pass FlowTransformBindings.NonuniformFFTsBackend() " *
        "(`using NonuniformFFTs`) or FlowTransformBindings.FINUFFTBackend() (`using FINUFFT`)."))

function make_scattered_plan(::SB.AbstractAutoSpectralBackend, x, y, ms, ::Type{T}; kwargs...) where {T}
    lib = _nufft_library()
    lib === nothing && return DirectNUFFTPlan(x, y, ms, T; kwargs...)
    return nufft_scattered_plan(lib, x, y, ms, T; kwargs...)
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
# or an LSMR least-squares inversion when `solve`; `inverse_transform!` is the Type-2
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

# The exponential tables of the direct sums and the scratch they write through. It is also the operator
# of the plan's least-squares solve, Type-2 as `A` and Type-1 as `A†`, on stacks of one column: an
# `ms` matrix of modes and a length-`M` vector of values.
struct _NUDFTTables{T, EM <: AbstractMatrix{Complex{T}}, CV <: AbstractVector{Complex{T}}}
    ms::NTuple{2, Int}
    M::Int
    Ex::EM                  # (ms[1], M)  e^{-i f₁[k]·sx_n}
    Ey::EM                  # (M, ms[2])  e^{-i f₂[k]·sy_n}   (point index contiguous)
    Exc::EM                 # (ms[1], M)  conj(Ex)
    Eyc::EM                 # (ms[2], M)  conj(Ey)ᵀ
    Sbuf::EM                # (M, ms[2]) Type-1 scratch
    T1::EM                  # (ms[1], M) Type-2 scratch
    t::CV                   # (M) Type-2 of a solver iterate
    w::EM                   # (ms) Type-1 of a solver residual
end

FTB.lsmr_ncolumns(::_NUDFTTables) = 1
FTB.lsmr_allocate_domain(tab::_NUDFTTables) = fill!(similar(tab.w), 0)
FTB.lsmr_allocate_range(tab::_NUDFTTables) = fill!(similar(tab.t), 0)
FTB.lsmr_forward!(u, tab::_NUDFTTables, v, c, n) = FTB.colxpby!(u, _nudft_type2!(tab.t, tab, v), c, n, 1)
FTB.lsmr_adjoint!(v, tab::_NUDFTTables, u, c, n) = FTB.colxpby!(v, _nudft_type1!(tab.w, tab, u), c, n, 1)

struct DirectNUFFTPlan{T, TB <: _NUDFTTables{T}, CV <: AbstractVector{Complex{T}},
                       RV <: AbstractVector{T}, WS} <: AbstractScatteringPlan
    ms::NTuple{2, Int}
    M::Int
    invN::T                 # 1/prod(ms) — makes synthesis the ifft-convention inverse
    solve::Bool
    maxiter::Int
    rtol::T
    damp::T                 # Tikhonov λ; 0 unless the mode grid is over-specified
    sx::RV                  # (M) points on the 2π-periodic domain, retained so the plan can be
    sy::RV                  #     rebuilt elsewhere — see `plan_points`
    tab::TB
    cj::CV                  # (M) values buffer (shared by Type-1/Type-2)
    ls::WS                  # the solve's `FTB.LSMRWorkspace`, or `nothing` unless this plan solves
end

# `ntrans` is accepted so a caller can request a batch width without first asking which backend it
# will get, and ignored because direct summation transforms one field per call. The plan reports
# `batch_width == 1`, so nothing downstream feeds it a stack it cannot take. `eps` and
# `nufft_nthreads` are accepted and ignored for the same reason: they configure a fast library that
# this plan does not use.
function DirectNUFFTPlan(x::AbstractVector, y::AbstractVector, ms::NTuple{2, Int}, ::Type{T};
                         period = nothing, solve::Bool = false, maxiter::Int = 100,
                         rtol::Real = default_solver_rtol(T, SB.DirectSumSpectralBackend(), nothing),
                         damp::Real = 0, eps = nothing, ntrans::Int = 1,
                         nufft_nthreads::Int = 0) where {T}
    M = length(x)
    length(y) == M || throw(DimensionMismatch("x and y must have equal length"))
    warn_underdetermined(M, ms, solve, damp)
    xmin, ymin = T(minimum(x)), T(minimum(y))
    # Default period so a uniform 0:m-1 grid (span m-1) maps to the exact DFT nodes 2π·(0:m-1)/m.
    px = period === nothing ? (T(maximum(x)) - xmin) * ms[1] / (ms[1] - 1) : T(period[1])
    py = period === nothing ? (T(maximum(y)) - ymin) * ms[2] / (ms[2] - 1) : T(period[2])
    sx = T(2π) .* (T.(x) .- xmin) ./ px
    sy = T(2π) .* (T.(y) .- ymin) ./ py
    f1, f2 = _fftfreqs(ms[1]), _fftfreqs(ms[2])
    Ex = Complex{T}[cis(-f1[k] * sx[n]) for k in 1:ms[1], n in 1:M]
    Ey = Complex{T}[cis(-f2[k] * sy[n]) for n in 1:M, k in 1:ms[2]]
    tab = _NUDFTTables{T, Matrix{Complex{T}}, Vector{Complex{T}}}(
        ms, M, Ex, Ey, conj.(Ex), Matrix(conj.(transpose(Ey))),
        Matrix{Complex{T}}(undef, M, ms[2]), Matrix{Complex{T}}(undef, ms[1], M),
        Vector{Complex{T}}(undef, M), Matrix{Complex{T}}(undef, ms))
    ls = solve ? FTB.LSMRWorkspace(tab) : nothing
    return DirectNUFFTPlan{T, typeof(tab), Vector{Complex{T}}, Vector{T}, typeof(ls)}(
        ms, M, one(T) / prod(ms), solve, maxiter, T(rtol), T(damp), sx, sy, tab,
        Vector{Complex{T}}(undef, M), ls)
end

Base.show(io::IO, p::DirectNUFFTPlan{T}) where {T} =
    print(io, "DirectNUFFTPlan{", T, "}(ms=", p.ms, ", M=", p.M, ", solve=", p.solve, ")")

spectral_backend(::DirectNUFFTPlan) = SB.DirectSumSpectralBackend()

function task_local_plan(p::DirectNUFFTPlan{T, TB, CV, RV, WS}) where {T, TB, CV, RV, WS}
    t = p.tab
    tab = TB(t.ms, t.M, t.Ex, t.Ey, t.Exc, t.Eyc, similar(t.Sbuf), similar(t.T1), similar(t.t), similar(t.w))
    ls = p.ls === nothing ? nothing : FTB.LSMRWorkspace(tab)
    return DirectNUFFTPlan{T, TB, CV, RV, WS}(p.ms, p.M, p.invN, p.solve, p.maxiter, p.rtol, p.damp,
                                              p.sx, p.sy, tab, similar(p.cj), ls)
end

"""
    plan_points(plan) -> (x, y)

The scattered sample locations a nonuniform plan was built on, already mapped onto its `2π`-periodic
domain. This is what lets a transform be rebuilt on another process, where the plan itself cannot
travel. Rebuilding from these requires `period = (2π, 2π)`, since they are already scaled.
"""
plan_points(p::DirectNUFFTPlan) = (p.sx, p.sy)

# Direct summation is exact and single-threaded by construction, so it reports no tolerance and no
# thread count — both are `nothing`/`0`, the values its constructor ignores.
plan_analysis(p::DirectNUFFTPlan) =
    (solve = p.solve, maxiter = p.maxiter, rtol = p.rtol, damp = p.damp, eps = nothing,
     nufft_nthreads = 0)

# Type-1 (points → modes): X = Ex · (c ⊙ Ey), all preallocated.
function _nudft_type1!(X::AbstractMatrix, tab::_NUDFTTables, c::AbstractVector)
    @inbounds for k2 in 1:tab.ms[2], n in 1:tab.M
        tab.Sbuf[n, k2] = c[n] * tab.Ey[n, k2]
    end
    LinearAlgebra.mul!(X, tab.Ex, tab.Sbuf)
    return X
end

# Type-2 (modes → points): c_n = Σ_{k₁} Exc[k₁,n]·(X·Eyc)[k₁,n].
function _nudft_type2!(c::AbstractVector, tab::_NUDFTTables{T}, X::AbstractMatrix) where {T}
    LinearAlgebra.mul!(tab.T1, X, tab.Eyc)
    @inbounds for n in 1:tab.M
        acc = zero(Complex{T})
        for k1 in 1:tab.ms[1]
            acc += tab.Exc[k1, n] * tab.T1[k1, n]
        end
        c[n] = acc
    end
    return c
end

inverse_transform!(out_pts::AbstractVector, plan::DirectNUFFTPlan, Xmodes::AbstractMatrix) =
    (_nudft_type2!(plan.cj, plan.tab, Xmodes); @. out_pts = plan.cj * plan.invN; out_pts)

function forward_transform!(Xmodes::AbstractMatrix, plan::DirectNUFFTPlan, x_pts::AbstractVector)
    if plan.solve
        _lsmr_solve_nudft!(Xmodes, plan, x_pts)
    else
        copyto!(plan.cj, x_pts)
        _nudft_type1!(Xmodes, plan.tab, plan.cj)
    end
    return Xmodes
end

# Least-squares inversion: modes `f` with `A f ≈ N·x`, `A` = Type-2, so that synthesis (Type-2 scaled
# by `invN = 1/N`) reproduces the samples. The solve fits `A f̃ = x` and scales by `N` after, which keeps
# every quantity at the field's own magnitude (`N = 4·10⁴` at `ms = 200²`, whose square is near
# `floatmax(Float32)`) and the reported residual in the field's units. At `cond(A) = 1/eps` the smallest
# singular direction carries nothing the precision can represent, so that is `conlim`.
function _lsmr_solve_nudft!(f::AbstractMatrix, plan::DirectNUFFTPlan{T},
                            x_pts::AbstractVector) where {T}
    FTB.lsmr!(f, plan.tab, x_pts, plan.ls; atol = plan.rtol, btol = plan.rtol, rtol = 0,
              damp = plan.damp, conlim = inv(Base.eps(T)), maxiter = plan.maxiter)
    _check_solve(plan.ls, plan.M, plan.ms, plan.rtol, plan.maxiter)
    f .*= one(T) / plan.invN
    return f
end

include("NUFFTScattering.jl")

end # module Plans
