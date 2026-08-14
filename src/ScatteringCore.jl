module ScatteringCore

"""
    ScatteringCore.jl — Core scattering transform operations

Implements the fundamental building blocks: FFT-based convolution,
modulus, and averaging operations.

All `!` functions are zero-allocation; non-`!` wrappers allocate and delegate
to them. Spectral transforms go through the plan interface
(`Plans.inverse_transform!`), so the engine is agnostic to whether the backing
transform is the in-core direct sum, FFTW, CUFFT, etc.
"""

using LinearAlgebra: LinearAlgebra
using ..Plans: Plans

export wavelet_convolve, wavelet_convolve!
export apply_modulus, apply_modulus!, spatial_average
export scattering

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
    task_workspace(st) -> st′

A transform equivalent to `st` that shares its read-only parts — filter bank, path tree, work list —
but owns fresh buffers and a task-local spectral plan, so the two can run concurrently.

This is what lets a parallel backend give each task private scratch without duplicating the filter
bank, which dominates a transform's memory (for a 256×256 J=4 L=8 transform, 33 MiB of the 38 MiB).
Methods are defined per transform type.
"""
function task_workspace end

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
    # Broadcast: CPU + GPU compatible (no scalar indexing). On GPU arrays this fuses to a single
    # kernel automatically, so no explicit-kernel override is needed (and none is defined — a broad
    # `::AbstractArray` override would also capture CPU arrays).
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
    modulus_mean(signal) -> Real

`⟨|signal|⟩` in a single reduction. A scattering coefficient is the mean of a modulus, so the
modulus field itself is never needed unless a coarser scale consumes it — this is the leaf case,
which writes nothing.
"""
modulus_mean(signal::AbstractArray) = sum(abs, signal) / length(signal)

"""
    modulus_mean!(out, signal) -> Real

Write `|signal|` into `out` and return `⟨|signal|⟩`. Used where the modulus field *is* consumed
downstream; the generic method is two device-friendly passes, the CPU method fuses them into one.
"""
modulus_mean!(out::AbstractArray, signal::AbstractArray) =
    (@. out = abs(signal); sum(out) / length(out))

function modulus_mean!(out::Array{T}, signal::Array{Complex{T}}) where {T <: Real}
    acc = zero(T)
    @inbounds @simd for i in eachindex(out, signal)
        v = abs(signal[i])
        out[i] = v
        acc += v
    end
    return acc / length(out)
end

end # module ScatteringCore
