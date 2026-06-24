module ScatteringTransformsFFTWExt

"""
    ScatteringTransformsFFTWExt — FFTW fast path

Provides an `O(N log N)` spectral plan backed by FFTW, overriding the in-core direct-sum
default. Loaded automatically by `using FFTW`; selected by `spectral=:fftw` (or `:auto`).
"""

using FFTW: FFTW
using LinearAlgebra: LinearAlgebra
using ScatteringTransforms: ScatteringTransforms

const Plans = ScatteringTransforms.Plans

"""
    FFTWScatteringPlan{T,FP,IP}

FFTW-backed spectral plan. Holds pre-planned forward/inverse transforms applied in place via
`mul!` (no allocation). Plan objects are concrete type parameters.
"""
struct FFTWScatteringPlan{T, FP, IP} <: Plans.AbstractScatteringPlan
    fwd::FP
    inv::IP
end

function Plans.fftw_plan(::Type{T}, dims) where {T}
    dummy = zeros(Complex{T}, dims)
    fwd = FFTW.plan_fft(dummy)
    inv = FFTW.plan_ifft(dummy)
    return FFTWScatteringPlan{T, typeof(fwd), typeof(inv)}(fwd, inv)
end

Plans.forward_transform!(out::AbstractArray, p::FFTWScatteringPlan, x::AbstractArray) =
    (LinearAlgebra.mul!(out, p.fwd, x); out)
Plans.inverse_transform!(out::AbstractArray, p::FFTWScatteringPlan, x::AbstractArray) =
    (LinearAlgebra.mul!(out, p.inv, x); out)

end # module ScatteringTransformsFFTWExt
