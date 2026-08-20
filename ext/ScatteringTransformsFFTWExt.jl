module ScatteringTransformsFFTWExt

"""
    ScatteringTransformsFFTWExt — FFTW fast path

`O(N log N)` spectral plans backed by FFTW, supplying the sole method of
`ScatteringTransforms.Plans.fftw_plan`. Loaded by `using FFTW`; selected by
`spectral = SpectralBackends.FFTSpectralBackend()` (or `AutoSpectralBackend()`).

Plans cover the leading `length(dims)` axes of a `(dims..., nbatch)` array, so one plan serves both
a single field and a whole batch.
"""

using FFTW: FFTW
using LinearAlgebra: LinearAlgebra
using SpectralBackends: SpectralBackends as SB
using ScatteringTransforms: ScatteringTransforms as ST

"""
    FFTWScatteringPlan{T,FP,IP}

FFTW-backed spectral plan. Holds pre-planned forward/inverse transforms applied in place via `mul!`
(no allocation). Plan objects are concrete type parameters.
"""
struct FFTWScatteringPlan{T, D, FP, IP} <: ST.Plans.AbstractScatteringPlan
    fwd::FP
    inv::IP
    dims::NTuple{D, Int}
    nbatch::Int
end

# The default `show` of an FFTW plan calls `fftw_sprint_plan`, which can segfault — so every struct
# owning one prints a one-line summary instead of its fields.
Base.show(io::IO, p::FFTWScatteringPlan{T}) where {T} =
    print(io, "FFTWScatteringPlan{", T, "}(dims=", p.dims, ", nbatch=", p.nbatch, ")")
Base.show(io::IO, ::MIME"text/plain", p::FFTWScatteringPlan) = show(io, p)

ST.Plans.spectral_backend(::FFTWScatteringPlan) = SB.FFTSpectralBackend()

"""
    fftw_plan(T, dims; nbatch = 1, planning = FFTW.MEASURE, fft_nthreads = 1)

Plan complex transforms of element type `Complex{T}` over the leading `length(dims)` axes of a
`(dims..., nbatch)` array.

`planning` defaults to `FFTW.MEASURE` rather than `FFTW.ESTIMATE`: a scattering transform executes
its plan `O(nw + paths)` times per field and is built once, so measured planning pays for itself
immediately. Planning runs against scratch arrays, so no caller data is destroyed. Pass
`FFTW.ESTIMATE` when build time dominates (many short-lived transforms).

`fft_nthreads` is baked into the plan at construction (FFTW reads its global thread count when
planning). It defaults to `1` because the parallel backends parallelise over the batch and wavelet
axes themselves, and nesting FFTW's threads inside those would oversubscribe. Raise it only for a
single large field on `SerialBackend`.
"""
function ST.Plans.fftw_plan(::Type{T}, dims::NTuple{D, Int}; nbatch::Int = 1,
                            batched::Bool = false,
                            planning::Integer = FFTW.MEASURE,
                            fft_nthreads::Int = 1) where {T, D}
    nbatch >= 1 || throw(ArgumentError("nbatch must be positive, got $nbatch"))
    fft_nthreads >= 1 || throw(ArgumentError("fft_nthreads must be positive, got $fft_nthreads"))
    # An FFTW plan is bound to an exact array shape, and a batch of one is genuinely ambiguous: the
    # single-field path works on `dims` arrays while the batched path always carries the trailing
    # axis. `batched` says which, so a chunk size of 1 plans `(dims…, 1)` rather than `dims`.
    scratch = zeros(Complex{T}, (batched || nbatch > 1) ? (dims..., nbatch) : dims)
    region = 1:D
    return ST.Plans.with_fft_nthreads(fft_nthreads) do
        fwd = FFTW.plan_fft(scratch, region; flags = planning)
        inv = FFTW.plan_ifft(scratch, region; flags = planning)
        return FFTWScatteringPlan{T, D, typeof(fwd), typeof(inv)}(fwd, inv, dims, nbatch)
    end
end

# FFTW reads its global thread count when a plan is built and bakes it in, so the count only has to
# hold for the duration of construction.
# Asking for a single thread does not touch `set_num_threads` at all, because calling it is what puts
# FFTW.jl on its multi-threaded execution path — and that path allocates on *every* execution, about
# 4 KiB per transform, which a cascade pays once per wavelet and once per path. It only shows up when
# `Threads.nthreads() > 1`, and it survives a later `set_num_threads(1)`: once the setter has been
# called there is no way back.
#
# Skipping the call keeps a process that only ever asks for one thread on the non-allocating path. It
# is not a guarantee, since any other package can enable threading first — loading
# FastSphericalHarmonics raises the count to 4 by itself — which is why the allocation gates compare
# two batch widths rather than asserting zero.
function ST.Plans.with_fft_nthreads(f, n::Integer)
    n <= 1 && return f()
    prev = FFTW.get_num_threads()
    try
        FFTW.set_num_threads(n)
        return f()
    finally
        FFTW.set_num_threads(prev)
    end
end
ST.Plans.fftw_plan(::Type{T}, N::Int; kwargs...) where {T} = ST.Plans.fftw_plan(T, (N,); kwargs...)

# FFTW plans are stateless under the new-array `mul!` execution path, so tasks share one plan.
ST.Plans.task_local_plan(p::FFTWScatteringPlan) = p

ST.Plans.forward_transform!(out::AbstractArray, p::FFTWScatteringPlan, x::AbstractArray) =
    (LinearAlgebra.mul!(out, p.fwd, x); out)
ST.Plans.inverse_transform!(out::AbstractArray, p::FFTWScatteringPlan, x::AbstractArray) =
    (LinearAlgebra.mul!(out, p.inv, x); out)

# Non-mutating, autodiff-friendly fast path: `plan * x` allocates and is differentiable via the
# `AbstractFFTs` ChainRules (reverse-mode Mooncake/Zygote). The primal must be the planned eltype,
# so this serves reverse-mode synthesis; `Dual` inputs need the direct sum.
ST.Plans.forward_transform(p::FFTWScatteringPlan, x::AbstractArray) = p.fwd * x
ST.Plans.inverse_transform(p::FFTWScatteringPlan, x::AbstractArray) = p.inv * x

end # module ScatteringTransformsFFTWExt
