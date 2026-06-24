module ScatteringCore

"""
    ScatteringCore.jl — Core scattering transform operations

Implements the fundamental building blocks: FFT-based convolution,
modulus, and averaging operations.

Design: All `!` functions are zero-allocation. Non-`!` wrappers allocate
and delegate to the `!` versions. Spectral transforms go through the plan
interface (`Plans.inverse_transform!`), so the engine is agnostic to whether
the backing transform is the in-core direct sum, FFTW, CUFFT, etc.
"""

using LinearAlgebra: LinearAlgebra
using ..Plans: Plans

export wavelet_convolve, wavelet_convolve!
export apply_modulus, apply_modulus!, spatial_average
export ScatteringLayer

"""
    wavelet_convolve(signal_fft, filter_fft, plan)

Perform wavelet convolution via frequency-domain multiplication then inverse transform.
Allocates output. For zero-allocation hot paths, use `wavelet_convolve!`.
"""
function wavelet_convolve(signal_fft::AbstractArray,
                          filter_fft::AbstractArray,
                          plan)
    out = similar(signal_fft)
    buffer = signal_fft .* filter_fft
    Plans.inverse_transform!(out, plan, buffer)
    return out
end

"""
    wavelet_convolve!(out, signal_fft, filter_fft, plan, buffer)

Truly zero-allocation wavelet convolution.

Multiplies `signal_fft .* filter_fft` into `buffer` in-place, then applies the inverse spectral
transform via `Plans.inverse_transform!(out, plan, buffer)` — writing directly into `out`.

`out` and `buffer` must both be pre-allocated complex arrays of the same size.
"""
function wavelet_convolve!(out::AbstractArray,
                          signal_fft::AbstractArray,
                          filter_fft::AbstractArray,
                          plan,
                          buffer::AbstractArray)
    # In-place pointwise multiply: buffer = signal_fft .* filter_fft
    @inbounds @simd for i in eachindex(signal_fft, filter_fft, buffer)
        buffer[i] = signal_fft[i] * filter_fft[i]
    end
    Plans.inverse_transform!(out, plan, buffer)
    return out
end

"""
    apply_modulus(signal)

Apply complex modulus |·| to get envelope. Allocates output.
For zero-allocation hot paths, use `apply_modulus!`.
"""
function apply_modulus(signal::AbstractArray)
    out = similar(signal, real(eltype(signal)))
    apply_modulus!(out, signal)
    return out
end

"""
    apply_modulus!(out, signal)

In-place modulus. Stores |signal| in pre-allocated `out`. Zero allocation.
"""
function apply_modulus!(out::AbstractArray, signal::AbstractArray)
    @inbounds @simd for i in eachindex(out, signal)
        out[i] = abs(signal[i])
    end
    return out
end

"""
    spatial_average(signal::AbstractArray{T}) -> T

Compute spatial average (global mean) for translation invariance.
Type-stable: returns element type T.
"""
function spatial_average(signal::AbstractArray)
    return sum(signal) / length(signal)
end

"""
    ScatteringLayer{V<:AbstractVector{Int}}

Represents a layer in the scattering transform network.
"""
struct ScatteringLayer{V<:AbstractVector{Int}}
    order::Int          # 0, 1, 2, ...
    scale_indices::V    # Which scales are used (generic array type)
end

end # module ScatteringCore
