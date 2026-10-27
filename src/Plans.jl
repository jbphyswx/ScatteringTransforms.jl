module Plans

# A plan implements two in-place primitives over the leading `D` dimensions of its argument, leaving
# a trailing batch axis untouched (the region convention of `AbstractFFTs.plan_fft(A, 1:D)`):
#
#     forward_transform!(out, plan, x)   # x -> X̂, fft convention
#     inverse_transform!(out, plan, x)   # X̂ -> x, ifft convention, 1/N scaled
#
# The in-core plans are direct sums; FFTW, device and nonuniform-FFT plans come from extensions. A
# `SpectralBackends` tag selects which.

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

Whether `inverse_transform!(x, plan, x)` is valid, letting a transform alias its convolution output
with its multiply scratch. Default `false`: the direct-sum plan ping-pongs through its own scratch,
and over an odd number of axes its first pass writes into its input.
"""
inplace_inverse(::AbstractScatteringPlan) = false

"""
    forward_transform(plan, x) -> X̂
    inverse_transform(plan, x) -> x

Non-mutating counterparts of [`forward_transform!`](@ref) and [`inverse_transform!`](@ref), for the
differentiable `scattering(st, x)`. They allocate their output and mutate neither the plan's scratch
nor `x`. [`DirectSumPlan`](@ref) contracts each axis against its twiddle table in scalar loops, so
it accepts any element type, `ForwardDiff.Dual` included, and every AD backend traces it.
"""
function forward_transform end

"""
    inverse_transform(plan, x) -> x

See [`forward_transform`](@ref).
"""
function inverse_transform end

"""
    spectral_backend(plan) -> SpectralBackends.AbstractSpectralBackend

The spectral-backend tag that rebuilds `plan` on another process. A plan that a tag cannot rebuild,
such as a device-resident FFT plan, throws.
"""
spectral_backend(plan::AbstractScatteringPlan) = throw(ArgumentError(
    "no spectral backend tag is registered for $(typeof(plan)), so a transform using it cannot be " *
    "rebuilt from a serialisable spec — construct the transform on each worker explicitly."))

"""
    close_plan!(plan) -> nothing

Release the foreign-library resources `plan` owns without waiting for garbage collection. A no-op by
default, and safe to call more than once. Whoever builds a plan for a task closes it when the task
ends; cuFINUFFT plans are released only by this.
"""
close_plan!(::Any) = nothing

"""
    per_task_nthreads(requested) -> Int

Thread count for a library plan that [`task_local_plan`](@ref) builds inside a task: `requested`
when positive, else `1`, since the tasks already occupy the cores.
"""
per_task_nthreads(requested::Integer) = requested > 0 ? Int(requested) : 1

"""
    AnalysisNotConverged <: Exception

Thrown when an iterative analysis has no usable answer: a non-finite residual, a condition estimate
past what the precision represents, or a solve that must converge and did not.

The scattered planar solve returns at `maxiter` without throwing: LSMR decreases `‖r‖` and `‖A†r‖`
monotonically, so the truncated iterate is the best one reached.
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

Stopping tolerance for the scattered least-squares solve, set by the accuracy of the transform under
it.

The direct sum evaluates an exact adjoint pair, so the target is `sqrt(Base.eps(T))`. A fast NUFFT's
type-1 and type-2 are adjoints only to its tolerance `eps` (the library default for `T` when
`nothing`), so the target is `max(10·eps, sqrt(Base.eps(T)))`.
"""
default_solver_rtol(::Type{T}, ::SB.AbstractDirectSumSpectralBackend, eps) where {T} =
    T(sqrt(Base.eps(real(float(T)))))
default_solver_rtol(::Type{T}, ::SB.AbstractSpectralBackend, eps) where {T} =
    T(max(10 * Float64(eps === nothing ? FTB.default_tolerance(real(float(T))) : eps),
          sqrt(Base.eps(real(float(T))))))

"""
    warn_underdetermined(M, ms, solve, damp) -> nothing

Warn, once, that a solve has more modes than samples and no damping.

With `prod(ms) > M` at least `prod(ms) - M` mode directions are constrained by no sample, and LSMR
returns the minimum-norm iterate. Whether that is the answer wanted, or whether to reduce `ms`, add
samples, or set `damp` for Tikhonov regularisation, is the caller's choice; no `λ` is chosen for
them.
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

Number of co-located fields `plan` transforms per execution: `1` unless it was built for a batch. A
nonuniform plan's width is fixed at construction, so it transforms exactly that many fields per
execution.
"""
batch_width(::Any) = 1

"""
    task_local_plan(plan) -> plan

A plan equivalent to `plan` that is safe to use concurrently with it. Stateless plans (FFTW,
`AbstractFFTs`) return themselves; plans carrying mutable scratch return a copy that shares their
read-only tables and owns fresh scratch. Called once per task by the parallel backends.

A method that builds a new library plan takes its thread count from [`per_task_nthreads`](@ref), and
the caller must [`close_plan!`](@ref) that plan when the task ends.
"""
task_local_plan(plan::AbstractScatteringPlan) = plan

"""
    with_fft_nthreads(f, n) -> f()

Run `f`, which builds plans, under FFTW.jl's planner lock with FFTW's planner thread count set to
`n` and restored after (`FFTW.set_num_threads(f, n)`); without FFTW loaded, just call `f()`.

FFTW's planner state is process-global and shared by every library that plans on libfftw3, and a
plan keeps the thread count it was built with, so every plan builder here sets the count through
this. The fallback takes `args...` so the FFTW extension's two-argument method is more specific and
adds a method; redefining this one is an error during precompilation.
"""
with_fft_nthreads(f, args...) = f()

"""
    fftw_plan(T, dims; batch, planning, fft_nthreads) -> AbstractScatteringPlan

Build the FFTW-backed plan, a method the FFTW extension defines; this fallback throws an
`ArgumentError` naming `using FFTW`.
"""
fftw_plan(args...; kwargs...) = throw(ArgumentError(
    "FFTSpectralBackend requires the FFTW extension. Run `using FFTW`."))

"""
    abstractffts_plan(dummy_device_array; region = 1:ndims(dummy_device_array), fft_nthreads = 1)

FFT plan for the array type of `dummy_device_array`, built through `AbstractFFTs.plan_fft` and
`plan_ifft`, so a `CuArray` gives a cuFFT plan and a `ROCArray` a rocFFT plan. `region` selects the
transformed dimensions, e.g. `(1, 2)` for a `(Ny, Nx, B)` stack. The KernelAbstractions extension
defines the method.
"""
function abstractffts_plan end

"""
    available(spectral) -> Bool

Whether the extension implementing `spectral` is loaded: `true` from that extension, so the transform
`AutoSpectralBackend` takes follows from the loaded methods.
"""
available(::SB.AbstractSpectralBackend) = false

# The NUFFT library `AutoSpectralBackend` takes: NonuniformFFTs when it is loaded, FINUFFT when only it
# is; `nothing` when neither is.
_nufft_library() =
    FTB.is_available(FTB.NonuniformFFTsBackend()) ? FTB.NonuniformFFTsBackend() :
    FTB.is_available(FTB.FINUFFTBackend()) ? FTB.FINUFFTBackend() : nothing

"""
    make_plan(spectral, T, dims; batch = nothing, kwargs...) -> AbstractScatteringPlan

Build the spectral plan selected by `spectral` for arrays whose leading dimensions are `dims` and
whose element type is `Complex{T}`: arrays of size `dims` for `batch = nothing`, and `(dims…, batch)`
stacks for an integer `batch`.

`SpectralBackends.DirectSumSpectralBackend` is the dependency-free in-core default;
`SpectralBackends.FFTSpectralBackend` requires `using FFTW`;
`SpectralBackends.AutoSpectralBackend` takes the FFTW fast path when its extension is loaded and
otherwise the in-core direct sum.
"""
make_plan(::SB.AbstractDirectSumSpectralBackend, ::Type{T}, dims;
          batch::Union{Nothing,Int} = nothing, kwargs...) where {T} =
    DirectSumPlan(T, dims; batch = batch)

make_plan(::SB.AbstractFFTSpectralBackend, ::Type{T}, dims;
          batch::Union{Nothing,Int} = nothing, kwargs...) where {T} =
    fftw_plan(T, dims; batch = batch, kwargs...)

make_plan(::SB.AbstractAutoSpectralBackend, ::Type{T}, dims;
          batch::Union{Nothing,Int} = nothing, kwargs...) where {T} =
    available(SB.FFTSpectralBackend()) ? fftw_plan(T, dims; batch = batch, kwargs...) :
    DirectSumPlan(T, dims; batch = batch)

# The trailing batch axis's length: `1` for a plan over single arrays.
batch_length(::Nothing) = 1
function batch_length(batch::Int)
    batch >= 1 || throw(ArgumentError("batch is the length of the trailing batch axis, at least 1; got $batch"))
    return batch
end

"""
    plan_like(plan, prototype) -> plan′
    plan_like(plan, prototype, D; kwargs...) -> plan′

A plan of the same kind as `plan`, sized for arrays like `prototype`, for a cascade that needs plans
at resolutions chosen after the transform was built. The three-argument form transforms the leading
`D` axes of `prototype` and leaves its trailing stack axis alone, with `kwargs` going to the plan
builder. A device plan has no spectral-backend tag, so its method plans on `prototype`'s array type.
"""
plan_like(plan::AbstractScatteringPlan, prototype::AbstractArray) =
    make_plan(spectral_backend(plan), real(eltype(prototype)), size(prototype))
plan_like(plan::AbstractScatteringPlan, prototype::AbstractArray, D::Int; kwargs...) =
    make_plan(spectral_backend(plan), real(eltype(prototype)), ntuple(d -> size(prototype, d), D);
              batch = size(prototype, D + 1), kwargs...)

"""
    plan_analysis(plan) -> (; solve, maxiter, rtol, damp, eps, nufft_nthreads)

How a scattered-planar plan turns samples into modes, as the keyword arguments its constructor takes.

A plan rebuilt on the same points needs these too: on irregular sampling the type-1 adjoint and the
least-squares inversion (`solve = true`) are different transforms. `eps` is `nothing` for the exact
direct sum.
"""
function plan_analysis end

"""
    make_scattered_plan(spectral, x, y, ms, T; period, origin = (0, 0), solve, maxiter, rtol, damp,
                        eps, ntrans, nufft_nthreads) -> AbstractScatteringPlan

Build the scattered/nonuniform planar plan selected by `spectral` over points `(x, y)` and a uniform
mode grid of size `ms`. `SpectralBackends.DirectSumSpectralBackend` is the dependency-free exact
NUDFT; `FlowTransformBindings.FINUFFTBackend()` and `FlowTransformBindings.NonuniformFFTsBackend()`
select a nonuniform-FFT library; `SpectralBackends.AutoSpectralBackend` takes NonuniformFFTs when it is
loaded and FINUFFT when only it is, and the exact direct sum when neither is.

`period` is the domain's length along each axis and `origin` its lower corner, each a number for both
axes or a 2-tuple. A point lies at `2π·mod(x - origin, period)/period` on the plan's `2π`-periodic
domain, so mode `k` has wavelength `period/k` whatever the point set.

`ntrans` is the batch width and `nufft_nthreads` the library's own thread count (`0`, the default,
takes `Threads.nthreads()`). The exact direct sum has no tolerance and runs on one thread, so it
accepts `eps` and `nufft_nthreads` and ignores them.
"""
make_scattered_plan(spectral::SB.AbstractDirectSumSpectralBackend, x, y, ms, ::Type{T};
                    period, origin = (0, 0), solve::Bool = false, maxiter::Int = 100,
                    eps = nothing, rtol::Real = default_solver_rtol(T, spectral, eps),
                    damp::Real = 0, ntrans::Int = 1, nufft_nthreads::Int = 0) where {T} =
    DirectNUFFTPlan(x, y, ms, T; period, origin, solve, maxiter, rtol, damp, ntrans)

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
# In-core direct-summation DFT plan: `O(N²)` per axis and `O(N)` memory, the reference the fast
# plans are checked against.
# ---------------------------------------------------------------------------

"""
    DirectSumPlan{T,V,D,S}

Direct-summation DFT plan: evaluates `X_k = Σ_n x_n e^{-2πi kn/N}` (and its inverse) by direct
summation, separably over the leading `D` dimensions, leaving a trailing batch axis of length
`nbatch` untouched. Memory is `O(N)`: a per-axis table of roots of unity and its conjugate.
`scratch` is `nothing` for `D == 1` and a full-size complex array otherwise.
"""
struct DirectSumPlan{T, V <: AbstractVector{Complex{T}}, D, S} <: AbstractScatteringPlan
    twiddle::NTuple{D, V}       # twiddle[d][m+1] = exp(-2πi m / N_d)
    twiddle_conj::NTuple{D, V}  # its conjugate, for the inverse
    dims::NTuple{D, Int}        # transformed (leading) dimensions
    nbatch::Int                 # length of the trailing batch axis
    scratch::S
    invscale::T                 # 1 / prod(dims)
    # Takes `T,V,D,S` explicitly, since a `D = 0` tuple binds none of them, and checks that each
    # twiddle table has its axis length.
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
    DirectSumPlan(T, dims; batch = nothing) -> DirectSumPlan

Build a direct-summation DFT plan over leading dimensions `dims` (an `Int` or `NTuple{D,Int}`), with a
trailing batch axis of length `batch` when it is an integer.
"""
function DirectSumPlan(::Type{T}, dims::NTuple{D, Int}; batch::Union{Nothing,Int} = nothing) where {T, D}
    nbatch = batch_length(batch)
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
# The twiddle exponent advances by `k < n` per step, so it wraps at most once and a compare-subtract
# keeps it in range. `nb == 1` (the leading axis) accumulates in a register; `nb > 1` accumulates
# into the contiguous run of `nb` lanes.
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
# Non-mutating direct-sum transforms for autodiff. The contraction against the twiddle table stays in
# scalar loops: Enzyme has no derivative rule for complex `gemm`.
# ---------------------------------------------------------------------------

# Contract dimension `d` of `A` against the DFT kernel from `tw`. The 1↔d transposition is an
# involution, so the same permutation restores the layout.
function _dft_along(tw::AbstractVector, n::Int, A::AbstractArray, d::Int)
    perm = ntuple(i -> i == 1 ? d : (i == d ? 1 : i), ndims(A))
    Ap = d == 1 ? A : permutedims(A, perm)
    sz = size(Ap)
    M = reshape(Ap, n, :)
    # An explicit loop: a comprehension's generator closes over both the constant twiddle table and
    # the active input, which Enzyme's static activity analysis cannot separate.
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
# Exact direct-summation NUDFT for scattered planar points, `O(M·prod(ms))`, with the contract of
# `NUFFTScatteringPlan`: `forward_transform!` is the type-1 adjoint (points → mode grid), or an LSMR
# least-squares inversion when `solve`; `inverse_transform!` is the type-2 synthesis (modes → points)
# scaled by `1/prod(ms)`. Modes are in FFT order, so on a uniform `0:m-1` grid these are exactly
# `fft`/`ifft` and the lattice matches the wavelet bank's `fftfreq` layout.
#
# Separated per axis, with `s` the coordinates on the 2π-periodic domain (`_fold`) and `f_d` the
# FFT-ordered integer frequencies:
#   Type-1:  X[k₁,k₂] = Σ_n c_n · e^{-i f₁[k₁]·sx_n} · e^{-i f₂[k₂]·sy_n}
#   Type-2:  c_n      = Σ_{k₁,k₂} X[k₁,k₂] · e^{+i f₁[k₁]·sx_n} · e^{+i f₂[k₂]·sy_n}
#
# `Ex`/`Exc` are stored `(ms₁, M)` and `Ey`/`Eyc` `(M, ms₂)`/`(ms₂, M)`, each in the layout its
# consumer reads contiguously. A plan `B` wide applies the tables to each of `B` columns in turn.
# ---------------------------------------------------------------------------

# A per-axis setting given as one number for both axes or as a 2-tuple.
_axis_pair(v::Real) = (v, v)
_axis_pair(v::Tuple{Real, Real}) = v
_axis_pair(v) = throw(ArgumentError("expected a number or a 2-tuple of numbers, one per axis; got $v"))

# `2π·mod(x - o, P)/P` for each point: its coordinate on the 2π-periodic domain a scattered plan
# transforms on, for a domain of length `P` along this axis whose lower corner is at `o`.
function _fold(x::AbstractVector, P::Real, o::Real, ::Type{T}) where {T}
    p = T(P)
    (isfinite(p) && p > 0) || throw(ArgumentError(
        "period must be the domain's length along each axis, positive and finite; got $P"))
    isfinite(o) || throw(ArgumentError("origin must be finite; got $o"))
    s = T(2π) / p
    oT = T(o)
    return s .* mod.(T.(x) .- oT, p)
end

# The points of a scattered plan on its 2π-periodic domain.
function _folded_points(x, y, period, origin, ::Type{T}) where {T}
    length(y) == length(x) || throw(DimensionMismatch("x and y must have equal length"))
    P, o = _axis_pair(period), _axis_pair(origin)
    return _fold(x, P[1], o[1], T), _fold(y, P[2], o[2], T)
end

# The exponential tables of the direct sums and the scratch they write through. It is also the operator
# of the plan's least-squares solve, Type-2 as `A` and Type-1 as `A†`, on stacks of `B` columns: an
# `(ms…, B)` array of modes and an `(M, B)` array of values.
struct _NUDFTTables{T, EM <: AbstractMatrix{Complex{T}}, TA <: AbstractMatrix{Complex{T}},
                    WA <: AbstractArray{Complex{T}, 3}}
    ms::NTuple{2, Int}
    M::Int
    B::Int
    Ex::EM                  # (ms[1], M)  e^{-i f₁[k]·sx_n}
    Ey::EM                  # (M, ms[2])  e^{-i f₂[k]·sy_n}   (point index contiguous)
    Exc::EM                 # (ms[1], M)  conj(Ex)
    Eyc::EM                 # (ms[2], M)  conj(Ey)ᵀ
    Sbuf::EM                # (M, ms[2]) Type-1 scratch
    T1::EM                  # (ms[1], M) Type-2 scratch
    t::TA                   # (M, B) Type-2 of a solver iterate
    w::WA                   # (ms…, B) Type-1 of a solver residual
end

FTB.lsmr_ncolumns(tab::_NUDFTTables) = tab.B
FTB.lsmr_allocate_domain(tab::_NUDFTTables) = fill!(similar(tab.w), 0)
FTB.lsmr_allocate_range(tab::_NUDFTTables) = fill!(similar(tab.t), 0)

function FTB.lsmr_forward!(u, tab::_NUDFTTables, v, c, n)
    @inbounds for k in 1:n
        _nudft_type2!(view(tab.t, :, k), tab, view(v, :, :, k))
    end
    return FTB.colxpby!(u, tab.t, c, n, tab.B)
end

function FTB.lsmr_adjoint!(v, tab::_NUDFTTables, u, c, n)
    @inbounds for k in 1:n
        _nudft_type1!(view(tab.w, :, :, k), tab, view(u, :, k))
    end
    return FTB.colxpby!(v, tab.w, c, n, tab.B)
end

"""
    DirectNUFFTPlan{T,TB,CV,RV,WS}

Exact direct-summation NUDFT between `M` scattered points and an `ms` mode grid, transforming
`batch_width(plan)` co-located fields per call. Memory is `O(M·(ms₁ + ms₂))`: one exponential table
per axis.
"""
struct DirectNUFFTPlan{T, TB <: _NUDFTTables{T}, CV <: AbstractVector{Complex{T}},
                       RV <: AbstractVector{T}, WS} <: AbstractScatteringPlan
    ms::NTuple{2, Int}
    M::Int
    invN::T                 # 1/prod(ms) — makes synthesis the ifft-convention inverse
    solve::Bool
    maxiter::Int
    rtol::T
    damp::T                 # Tikhonov λ
    sx::RV                  # (M) points on the 2π-periodic domain, retained so the plan can be
    sy::RV                  #     rebuilt elsewhere — see `plan_points`
    tab::TB
    cj::CV                  # (M) one column of values (shared by Type-1/Type-2)
    ls::WS                  # the solve's `FTB.LSMRWorkspace`, or `nothing` unless this plan solves
end

batch_width(p::DirectNUFFTPlan) = p.tab.B

# `eps` and `nufft_nthreads` configure a fast library; the exact direct sum has neither a tolerance nor
# a thread count.
function DirectNUFFTPlan(x::AbstractVector, y::AbstractVector, ms::NTuple{2, Int}, ::Type{T};
                         period, origin = (0, 0), solve::Bool = false, maxiter::Int = 100,
                         rtol::Real = default_solver_rtol(T, SB.DirectSumSpectralBackend(), nothing),
                         damp::Real = 0, eps = nothing, ntrans::Int = 1,
                         nufft_nthreads::Int = 0) where {T}
    M = length(x)
    ntrans >= 1 || throw(ArgumentError("ntrans must be at least 1; got $ntrans"))
    sx, sy = _folded_points(x, y, period, origin, T)
    warn_underdetermined(M, ms, solve, damp)
    f(k, m) = FTB.mode_frequency(FTB.FFTModes(), k, m)
    Ex = Complex{T}[cis(-f(k, ms[1]) * sx[n]) for k in 1:ms[1], n in 1:M]
    Ey = Complex{T}[cis(-f(k, ms[2]) * sy[n]) for n in 1:M, k in 1:ms[2]]
    tab = _NUDFTTables{T, Matrix{Complex{T}}, Matrix{Complex{T}}, Array{Complex{T}, 3}}(
        ms, M, ntrans, Ex, Ey, conj.(Ex), Matrix(conj.(transpose(Ey))),
        Matrix{Complex{T}}(undef, M, ms[2]), Matrix{Complex{T}}(undef, ms[1], M),
        Matrix{Complex{T}}(undef, M, ntrans), Array{Complex{T}, 3}(undef, ms..., ntrans))
    ls = solve ? FTB.LSMRWorkspace(tab) : nothing
    return DirectNUFFTPlan{T, typeof(tab), Vector{Complex{T}}, typeof(sx), typeof(ls)}(
        ms, M, one(T) / prod(ms), solve, maxiter, T(rtol), T(damp), sx, sy, tab,
        Vector{Complex{T}}(undef, M), ls)
end

Base.show(io::IO, p::DirectNUFFTPlan{T}) where {T} =
    print(io, "DirectNUFFTPlan{", T, "}(ms=", p.ms, ", M=", p.M, ", ntrans=", p.tab.B, ", solve=",
          p.solve, ")")

spectral_backend(::DirectNUFFTPlan) = SB.DirectSumSpectralBackend()

function task_local_plan(p::DirectNUFFTPlan{T, TB, CV, RV, WS}) where {T, TB, CV, RV, WS}
    t = p.tab
    tab = TB(t.ms, t.M, t.B, t.Ex, t.Ey, t.Exc, t.Eyc, similar(t.Sbuf), similar(t.T1), similar(t.t),
             similar(t.w))
    ls = p.ls === nothing ? nothing : FTB.LSMRWorkspace(tab)
    return DirectNUFFTPlan{T, TB, CV, RV, WS}(p.ms, p.M, p.invN, p.solve, p.maxiter, p.rtol, p.damp,
                                              p.sx, p.sy, tab, similar(p.cj), ls)
end

"""
    plan_points(plan) -> (x, y)

The scattered sample locations a nonuniform plan was built on, already mapped onto its `2π`-periodic
domain, from which a transform is rebuilt on another process. Rebuilding from these requires
`period = (2π, 2π)`, since they are already scaled.
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

# Column `k` of a stack of point values, `(M)` for one field or `(M, B)`, and of a stack of modes,
# `(ms)` or `(ms…, B)`.
@inline _pcol(A::AbstractVector, ::Int) = A
@inline _pcol(A::AbstractMatrix, k::Int) = view(A, :, k)
@inline _mcol(A::AbstractMatrix, ::Int) = A
@inline _mcol(A::AbstractArray{<:Any, 3}, k::Int) = view(A, :, :, k)

function _check_stacks(plan::DirectNUFFTPlan, modes::AbstractArray, pts::AbstractArray)
    B = plan.tab.B
    (size(pts, 1) == plan.M && length(pts) == plan.M * B &&
     (size(modes, 1), size(modes, 2)) == plan.ms && length(modes) == prod(plan.ms) * B &&
     ndims(pts) <= 2 && ndims(modes) <= 3) || throw(DimensionMismatch(
        "this plan transforms $B field(s) of $(plan.M) points on a $(plan.ms) mode grid; got " *
        "$(size(pts)) values and $(size(modes)) modes"))
    return nothing
end

function inverse_transform!(out_pts::AbstractArray, plan::DirectNUFFTPlan, Xmodes::AbstractArray)
    _check_stacks(plan, Xmodes, out_pts)
    for k in 1:plan.tab.B
        _nudft_type2!(plan.cj, plan.tab, _mcol(Xmodes, k))
        o = _pcol(out_pts, k)
        @. o = plan.cj * plan.invN
    end
    return out_pts
end

function forward_transform!(Xmodes::AbstractArray, plan::DirectNUFFTPlan, x_pts::AbstractArray)
    _check_stacks(plan, Xmodes, x_pts)
    if plan.solve
        _lsmr_solve_nudft!(Xmodes, plan, x_pts)
    else
        for k in 1:plan.tab.B
            copyto!(plan.cj, _pcol(x_pts, k))
            _nudft_type1!(_mcol(Xmodes, k), plan.tab, plan.cj)
        end
    end
    return Xmodes
end

# Least-squares inversion: modes `f` with `A f ≈ N·x`, `A` = type-2, so that synthesis (type-2 scaled
# by `invN = 1/N`) reproduces the samples. The solve fits `A f̃ = x` and scales by `N` after, keeping
# every quantity at the field's own magnitude and the reported residual in the field's units. At
# `cond(A) = 1/eps` the smallest singular direction carries nothing the precision can represent, so
# that is `conlim`.
function _lsmr_solve_nudft!(f::AbstractArray, plan::DirectNUFFTPlan{T},
                            x_pts::AbstractArray) where {T}
    FTB.lsmr!(f, plan.tab, x_pts, plan.ls; atol = plan.rtol, btol = plan.rtol, rtol = 0,
              damp = plan.damp, conlim = inv(Base.eps(T)), maxiter = plan.maxiter)
    _check_solve(plan.ls, plan.M, plan.ms, plan.rtol, plan.maxiter)
    f .*= one(T) / plan.invN
    return f
end

include("NUFFTScattering.jl")

end # module Plans
