module ScatteringTransformsFFTWExt

using FFTW: FFTW
using LinearAlgebra: LinearAlgebra
using SpectralBackends: SpectralBackends as SB
using ScatteringTransforms: ScatteringTransforms as ST

"""
    FFTWScatteringPlan{T,D,FP,IP,IPI}

FFTW-backed spectral plan: forward and inverse FFTW plans executed through `mul!`, and an in-place
inverse for `out === x`.
"""
struct FFTWScatteringPlan{T, D, FP, IP, IPI} <: ST.Plans.AbstractScatteringPlan
    fwd::FP
    inv::IP
    inv!::IPI          # the same inverse, planned in place — see `inverse_transform!`
    dims::NTuple{D, Int}
    nbatch::Int
end

# The default `show` of an FFTW plan calls `fftw_sprint_plan`, which can segfault, so a struct owning
# one prints a one-line summary.
Base.show(io::IO, p::FFTWScatteringPlan{T}) where {T} =
    print(io, "FFTWScatteringPlan{", T, "}(dims=", p.dims, ", nbatch=", p.nbatch, ")")
Base.show(io::IO, ::MIME"text/plain", p::FFTWScatteringPlan) = show(io, p)

ST.Plans.spectral_backend(::FFTWScatteringPlan) = SB.FFTSpectralBackend()
ST.Plans.available(::SB.FFTSpectralBackend) = true

"""
    fftw_plan(T, dims; batch = nothing, planning = FFTW.MEASURE, fft_nthreads = 1)

Plan complex transforms of element type `Complex{T}` over the leading `length(dims)` axes of a
`dims` array for `batch = nothing`, or of a `(dims..., batch)` stack for an integer `batch`.

A transform executes its plan `O(nw + paths)` times per field, so `planning` defaults to
`FFTW.MEASURE`; pass `FFTW.ESTIMATE` when build time dominates. Planning runs on scratch arrays.
`fft_nthreads` is fixed in the plan; leave it at `1` under a parallel backend, which already
threads over the batch and wavelet axes.
"""
function ST.Plans.fftw_plan(::Type{T}, dims::NTuple{D, Int}; batch::Union{Nothing,Int} = nothing,
                            planning::Integer = FFTW.MEASURE,
                            fft_nthreads::Int = 1) where {T, D}
    nbatch = ST.Plans.batch_length(batch)
    fft_nthreads >= 1 || throw(ArgumentError("fft_nthreads must be positive, got $fft_nthreads"))
    # An FFTW plan is bound to an exact array shape, a stack's trailing axis included.
    scratch = zeros(Complex{T}, _stack_size(dims, batch))
    region = 1:D
    return FFTW.set_num_threads(fft_nthreads) do
        fwd = FFTW.plan_fft(scratch, region; flags = planning)
        inv = FFTW.plan_ifft(scratch, region; flags = planning)
        invp = FFTW.plan_ifft!(scratch, region; flags = planning)
        return FFTWScatteringPlan{T, D, typeof(fwd), typeof(inv), typeof(invp)}(
            fwd, inv, invp, dims, nbatch)
    end
end

_stack_size(dims::Tuple, ::Nothing) = dims
_stack_size(dims::Tuple, batch::Int) = (dims..., batch)

# A plan keeps the planner thread count it was built with, and FFTW.jl runs a multi-threaded plan's
# parallel loop on Julia tasks spawned per execution. The count is process-global and loading
# FastSphericalHarmonics raises it, so every plan names its own, `n == 1` included.
ST.Plans.with_fft_nthreads(f, n::Integer) = FFTW.set_num_threads(f, n)
ST.Plans.fftw_plan(::Type{T}, N::Int; kwargs...) where {T} = ST.Plans.fftw_plan(T, (N,); kwargs...)

# FFTW plans are stateless under the new-array `mul!` execution path, so tasks share one plan.
ST.Plans.task_local_plan(p::FFTWScatteringPlan) = p

ST.Plans.forward_transform!(out::AbstractArray, p::FFTWScatteringPlan, x::AbstractArray) =
    (LinearAlgebra.mul!(out, p.fwd, x); out)
# `mul!` on an out-of-place FFTW plan with aliased arguments is undefined, so `out === x` dispatches.
ST.Plans.inverse_transform!(out::AbstractArray, p::FFTWScatteringPlan, x::AbstractArray) =
    out === x ? (p.inv! * x) : (LinearAlgebra.mul!(out, p.inv, x); out)
ST.Plans.inplace_inverse(::FFTWScatteringPlan) = true

# Non-mutating transforms for reverse-mode AD through the `AbstractFFTs` ChainRules. The input must
# have the planned element type, so `Dual` inputs need the direct sum.
ST.Plans.forward_transform(p::FFTWScatteringPlan, x::AbstractArray) = p.fwd * x
ST.Plans.inverse_transform(p::FFTWScatteringPlan, x::AbstractArray) = p.inv * x

end # module ScatteringTransformsFFTWExt
