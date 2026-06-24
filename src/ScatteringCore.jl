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
export scattering
export ScatteringLayer

"""
    scattering(st, x) -> ScatteringCoefficients

Non-mutating, allocation-tolerant, element-type-generic scattering transform — the
autodiff-friendly counterpart of the in-place callable `st(x)`. It composes the non-mutating
[`Plans.forward_transform`](@ref)/[`Plans.inverse_transform`](@ref) with broadcast modulus and
mean (no preallocated workspace, no `mul!`), so gradients flow through it via
DifferentiationInterface (Mooncake/Zygote/Enzyme) and it accepts `Dual`/`Float32` inputs. It
returns the same coefficient container as `st(x)` and matches it numerically. Methods are added
for the 1D/2D/3D transforms in their respective submodules.

Use `st(x)` (mutating, zero-alloc) for production forward passes; use `scattering(st, x)` when
you need to differentiate the forward map (e.g. gradient-descent synthesis).
"""
function scattering end

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
    # In-place pointwise multiply via broadcast: works on CPU Arrays AND GPU arrays
    # (fuses to a single kernel), avoiding scalar indexing.
    @. buffer = signal_fft * filter_fft
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
    # Broadcast: CPU + GPU compatible (no scalar indexing). On GPU arrays this fuses to one
    # kernel; the KernelAbstractions extension provides an explicit-kernel override.
    @. out = abs(signal)
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
