module ScatteringTransformsKernelAbstractionsExt

# NOTE (issue #3): this extension is intended to hold the vendor-neutral (KernelAbstractions) GPU
# execution path for the scattering transform — dispatched on GPUBackend/KA.Backend, modelled on
# StructureFunctions' GPUExt. That path is NOT yet implemented; the device kernels below are the
# building blocks for it. Do NOT override core elementwise ops (e.g. `apply_modulus!`) by
# `::AbstractArray`: dispatch is by argument type, so such an override hijacks CPU calls the instant
# KernelAbstractions is loaded anywhere in the session (see #3). GPU today is provided by
# ScatteringTransformsCUDAExt (CuArray + cuFFT); CPU/GPU elementwise ops use the base broadcast in
# ScatteringCore, which is already correct on both.

using KernelAbstractions: KernelAbstractions

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

end # module ScatteringTransformsKernelAbstractionsExt