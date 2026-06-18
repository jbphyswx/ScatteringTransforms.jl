module ScatteringTransformsKernelAbstractionsExt

using KernelAbstractions: KernelAbstractions
using ScatteringTransforms: ScatteringTransforms

const KA = KernelAbstractions

"""
    _modulus_kernel!(out, signal)

KernelAbstractions kernel: computes out[i] = abs(signal[i]) element-wise.
Dispatches to the correct backend (CPU, CUDA, ROCm, Metal, etc.) based on
the array type of `out`.
"""
@KA.kernel function _modulus_kernel!(out, signal)
    i = KA.@index(Global, Linear)
    out[i] = abs(signal[i])
end

"""
    _complexify_kernel!(out, signal)

KernelAbstractions kernel: promotes real signal to complex in-place.
out[i] = Complex(signal[i], 0).
"""
@KA.kernel function _complexify_kernel!(out, signal)
    i = KA.@index(Global, Linear)
    out[i] = signal[i] + zero(eltype(out))
end

"""
    _pointwise_mul_kernel!(out, a, b)

KernelAbstractions kernel: out[i] = a[i] * b[i] element-wise.
Used as the multiply step in wavelet_convolve! on GPU arrays.
"""
@KA.kernel function _pointwise_mul_kernel!(out, a, b)
    i = KA.@index(Global, Linear)
    out[i] = a[i] * b[i]
end

"""
    ScatteringTransforms.ScatteringCore.apply_modulus!(out::AbstractArray{T}, signal::AbstractArray{Complex{T}})

KernelAbstractions-accelerated override for non-CPU array types.
Falls back to the SIMD loop for CPU arrays.
"""
function ScatteringTransforms.ScatteringCore.apply_modulus!(
    out::AbstractArray{T},
    signal::AbstractArray{Complex{T}},
) where T<:Real
    # Detect backend from output array type
    backend = KA.get_backend(out)
    kernel! = _modulus_kernel!(backend)
    kernel!(out, signal; ndrange=length(out))
    KA.synchronize(backend)
    return out
end

"""
    ScatteringTransforms.ScatteringCore.apply_modulus!(out::AbstractArray{T}, signal::AbstractArray{T})

KernelAbstractions-accelerated override for real arrays.
"""
function ScatteringTransforms.ScatteringCore.apply_modulus!(
    out::AbstractArray{T},
    signal::AbstractArray{T},
) where T<:Real
    backend = KA.get_backend(out)
    kernel! = _modulus_kernel!(backend)
    kernel!(out, signal; ndrange=length(out))
    KA.synchronize(backend)
    return out
end

end # module ScatteringTransformsKernelAbstractionsExt