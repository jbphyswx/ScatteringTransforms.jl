module ScatteringCore

"""
    ScatteringCore.jl — Core scattering transform operations

Implements the fundamental building blocks: FFT-based convolution,
modulus, and averaging operations.
"""

using FFTW: FFTW
using LinearAlgebra: LinearAlgebra

export wavelet_convolve, wavelet_convolve!
export apply_modulus, apply_modulus!, spatial_average
export ScatteringLayer

"""
    wavelet_convolve(signal_fft, filter_fft, ifft_plan)

Perform wavelet convolution via frequency-domain multiplication.
Allocates output. For zero-allocation, use wavelet_convolve!.
"""
function wavelet_convolve(signal_fft::AbstractArray{Complex{T}}, 
                          filter_fft::AbstractArray{Complex{T}},
                          ifft_plan) where T<:Real
    # Frequency-domain multiplication = convolution
    filtered_fft = signal_fft .* filter_fft
    return ifft_plan * filtered_fft
end

"""
    wavelet_convolve!(out, signal_fft, filter_fft, ifft_plan, buffer)

In-place wavelet convolution with pre-allocated buffer.
Uses buffer for intermediate storage, writes IFFT result to out.
Note: FFTW IFFT plan still allocates internally, but we minimize other allocs.
"""
function wavelet_convolve!(out::AbstractArray{Complex{T}}, 
                          signal_fft::AbstractArray{Complex{T}}, 
                          filter_fft::AbstractArray{Complex{T}},
                          ifft_plan,
                          buffer::AbstractArray{Complex{T}}) where T<:Real
    # In-place multiplication into buffer
    @inbounds @simd for i in eachindex(signal_fft, filter_fft, buffer)
        buffer[i] = signal_fft[i] * filter_fft[i]
    end
    # IFFT - copy result to out (FFTW plan returns new array)
    result = ifft_plan * buffer
    @inbounds @simd for i in eachindex(out, result)
        out[i] = result[i]
    end
    return out
end

"""
    apply_modulus(signal)

Apply complex modulus |·| to get envelope. Allocates output.
For zero-allocation, use apply_modulus!.
"""
apply_modulus(signal::AbstractArray) = abs.(signal)

"""
    apply_modulus!(out, signal)

In-place modulus. Stores |signal| in pre-allocated out. Zero allocation.
"""
function apply_modulus!(out::AbstractArray{T}, signal::AbstractArray{Complex{T}}) where T<:Real
    @inbounds @simd for i in eachindex(out, signal)
        out[i] = abs(signal[i])
    end
    return out
end

function apply_modulus!(out::AbstractArray{T}, signal::AbstractArray{T}) where T<:Real
    @inbounds @simd for i in eachindex(out, signal)
        out[i] = abs(signal[i])
    end
    return out
end

"""
    spatial_average(signal)

Compute spatial average (global mean) for translation invariance.
"""
spatial_average(signal::AbstractArray) = sum(signal) / length(signal)

"""
    spatial_average(signal::AbstractArray{T}) where T<:Real -> T

Type-stable version that preserves element type.
"""
function spatial_average(signal::AbstractArray{T}) where T<:Real
    return sum(signal) / T(length(signal))
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
