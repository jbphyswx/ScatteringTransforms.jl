module ScatteringTransformsFFTWExt

"""
    ScatteringTransformsFFTWExt — FFTW fast path

Provides an `O(N log N)` spectral plan backed by FFTW, overriding the in-core direct-sum
default. Loaded automatically by `using FFTW`; selected by `spectral=:fftw` (or `:auto`).
"""

using FFTW: FFTW
using LinearAlgebra: LinearAlgebra
using ScatteringTransforms: ScatteringTransforms

"""
    FFTWScatteringPlan{T,FP,IP}

FFTW-backed spectral plan. Holds pre-planned forward/inverse transforms applied in place via
`mul!` (no allocation). Plan objects are concrete type parameters.
"""
struct FFTWScatteringPlan{T, FP, IP} <: ScatteringTransforms.Plans.AbstractScatteringPlan
    fwd::FP
    inv::IP
end

function ScatteringTransforms.Plans.fftw_plan(::Type{T}, dims) where {T}
    dummy = zeros(Complex{T}, dims)
    fwd = FFTW.plan_fft(dummy)
    inv = FFTW.plan_ifft(dummy)
    return FFTWScatteringPlan{T, typeof(fwd), typeof(inv)}(fwd, inv)
end

ScatteringTransforms.Plans.forward_transform!(out::AbstractArray, p::FFTWScatteringPlan, x::AbstractArray) =
    (LinearAlgebra.mul!(out, p.fwd, x); out)
ScatteringTransforms.Plans.inverse_transform!(out::AbstractArray, p::FFTWScatteringPlan, x::AbstractArray) =
    (LinearAlgebra.mul!(out, p.inv, x); out)

# Non-mutating, autodiff-friendly fast path: `plan * x` allocates and is differentiable via the
# `AbstractFFTs` ChainRules (reverse-mode Mooncake/Zygote). The primal must be `Complex{Float64}`
# (the planned eltype), so this serves reverse-mode synthesis; `Dual` inputs need the direct sum.
ScatteringTransforms.Plans.forward_transform(p::FFTWScatteringPlan, x::AbstractArray) = p.fwd * x
ScatteringTransforms.Plans.inverse_transform(p::FFTWScatteringPlan, x::AbstractArray) = p.inv * x

end # module ScatteringTransformsFFTWExt
