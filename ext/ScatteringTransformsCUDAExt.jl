module ScatteringTransformsCUDAExt

using CUDA: CUDA
using LinearAlgebra: LinearAlgebra
using ScatteringTransforms: ScatteringTransforms

const Plans = ScatteringTransforms.Plans

"""
    CUFFTScatteringPlan{T,FP,IP}

CUFFT-backed spectral plan for GPU arrays, implementing the `Plans` interface. (Full GPU
correctness of the surrounding hot loops is completed in the parallel/GPU phase.)
"""
struct CUFFTScatteringPlan{T, FP, IP} <: Plans.AbstractScatteringPlan
    fwd::FP
    inv::IP
end

function _cufft_plan(::Type{T}, dims) where {T}
    dummy = CUDA.zeros(Complex{T}, dims)
    fwd = CUDA.CUFFT.plan_fft(dummy)
    inv = CUDA.CUFFT.plan_ifft(dummy)
    return CUFFTScatteringPlan{T, typeof(fwd), typeof(inv)}(fwd, inv)
end

Plans.forward_transform!(out::AbstractArray, p::CUFFTScatteringPlan, x::AbstractArray) =
    (LinearAlgebra.mul!(out, p.fwd, x); out)
Plans.inverse_transform!(out::AbstractArray, p::CUFFTScatteringPlan, x::AbstractArray) =
    (LinearAlgebra.mul!(out, p.inv, x); out)

"""
    gpu_filter_bank1d(fb::FilterBanks.FilterBank1D{T}) -> FilterBank1D{T, CuVector{Complex{T}}}

Transfer a CPU `FilterBank1D` to GPU. Returns a new `FilterBank1D` where
all wavelet and averaging filter arrays are `CuVector`.
"""
function ScatteringTransforms.FilterBanks.FilterBank1D{T}(
    fb::ScatteringTransforms.FilterBanks.FilterBank1D{T},
    ::CUDA.CuDevice,
) where T<:Real
    cu_wavelets = [CUDA.CuVector{Complex{T}}(ψ) for ψ in fb.wavelets]
    cu_avg      = CUDA.CuVector{Complex{T}}(fb.averaging)
    V = eltype(cu_wavelets)
    return ScatteringTransforms.FilterBanks.FilterBank1D{T, V}(cu_wavelets, cu_avg, fb.meta, fb.J, fb.Q)
end

"""
    gpu_filter_bank2d(fb::FilterBanks.FilterBank2D{T}) -> FilterBank2D{T, CuMatrix{Complex{T}}}

Transfer a CPU `FilterBank2D` to GPU.
"""
function ScatteringTransforms.FilterBanks.FilterBank2D{T}(
    fb::ScatteringTransforms.FilterBanks.FilterBank2D{T},
    ::CUDA.CuDevice,
) where T<:Real
    cu_wavelets = [CUDA.CuMatrix{Complex{T}}(ψ) for ψ in fb.wavelets]
    cu_avg      = CUDA.CuMatrix{Complex{T}}(fb.averaging)
    M = typeof(cu_avg)
    return ScatteringTransforms.FilterBanks.FilterBank2D{T, M}(cu_wavelets, cu_avg, fb.meta, fb.J, fb.L)
end

"""
    ScatteringTransform1D(N, J; ..., device=CUDA.device())

GPU constructor for `ScatteringTransform1D`. Builds the filter bank on GPU,
allocates CuArray workspace buffers, and uses CUDA.CUFFT plans.

# Example
```julia
using CUDA, ScatteringTransforms
st_gpu = ScatteringTransforms.ScatteringTransform1D(1024, 6; T=Float32, device=CUDA.device())
signal_gpu = CUDA.randn(Float32, 1024)
coeffs = st_gpu(signal_gpu)
```
"""
function ScatteringTransforms.Scattering1D.ScatteringTransform1D(
    N::Int, J::Int, device::CUDA.CuDevice;
    Q::Int=1, max_order::Int=2, T::Type=Float32,
)
    # Build CPU filter bank, then transfer to GPU
    cpu_fb = ScatteringTransforms.FilterBanks.build_filter_bank1d(N, J; Q=Q, T=T)
    cu_wavelets = [CUDA.CuVector{Complex{T}}(ψ) for ψ in cpu_fb.wavelets]
    cu_avg = CUDA.CuVector{Complex{T}}(cpu_fb.averaging)
    V_type = typeof(cu_avg)
    filter_bank = ScatteringTransforms.FilterBanks.FilterBank1D{T, V_type}(
        cu_wavelets, cu_avg, cpu_fb.meta, cpu_fb.J, cpu_fb.Q
    )
    tree = ScatteringTransforms.PathGraph.build_tree([m.j_eff for m in filter_bank.meta], max_order)

    # GPU spectral plan via CUFFT
    plan = _cufft_plan(T, N)

    num_w = length(filter_bank.wavelets)
    buffer_input = CUDA.zeros(Complex{T}, N)
    buffer_conv  = CUDA.zeros(Complex{T}, N)
    buffer_mod   = CUDA.zeros(T, N)

    if max_order >= 2
        U1_buffers     = [CUDA.zeros(T, N) for _ in 1:num_w]
        U1_fft_buffers = [CUDA.zeros(Complex{T}, N) for _ in 1:num_w]
    else
        U1_buffers     = CUDA.CuVector{T}[]
        U1_fft_buffers = CUDA.CuVector{Complex{T}}[]
    end

    buffer_signal_fft = CUDA.zeros(Complex{T}, N)
    M_type = typeof(buffer_conv)
    R_type = typeof(buffer_mod)
    return ScatteringTransforms.Scattering1D.ScatteringTransform1D{
        T, M_type, R_type, typeof(plan), typeof(tree)}(
        filter_bank, tree, max_order, plan,
        buffer_input, buffer_signal_fft, buffer_conv, buffer_mod,
        U1_buffers, U1_fft_buffers,
    )
end

"""
    ScatteringTransform2D(N, J; ..., device=CUDA.device())

GPU constructor for `ScatteringTransform2D`. Builds filter bank on GPU,
allocates CuMatrix workspace buffers, and uses CUDA.CUFFT 2D plans.

# Example
```julia
using CUDA, ScatteringTransforms
st2d_gpu = ScatteringTransforms.ScatteringTransform2D((128,128), 3; L=8, T=Float32, device=CUDA.device())
image_gpu = CUDA.randn(Float32, 128, 128)
coeffs = st2d_gpu(image_gpu)
```
"""
function ScatteringTransforms.Scattering2D.ScatteringTransform2D(
    N::NTuple{2,Int}, J::Int, device::CUDA.CuDevice;
    L::Int=8, max_order::Int=2, T::Type=Float32,
)
    cpu_fb = ScatteringTransforms.FilterBanks.build_filter_bank2d(N, J; L=L, T=T)
    cu_wavelets = [CUDA.CuMatrix{Complex{T}}(ψ) for ψ in cpu_fb.wavelets]
    cu_avg = CUDA.CuMatrix{Complex{T}}(cpu_fb.averaging)
    M_fb = typeof(cu_avg)
    filter_bank = ScatteringTransforms.FilterBanks.FilterBank2D{T, M_fb}(
        cu_wavelets, cu_avg, cpu_fb.meta, cpu_fb.J, cpu_fb.L
    )
    tree = ScatteringTransforms.PathGraph.build_tree([m.j_eff for m in filter_bank.meta], max_order)

    plan = _cufft_plan(T, N)

    num_w = length(filter_bank.wavelets)
    buffer_input = CUDA.zeros(Complex{T}, N)
    buffer_conv  = CUDA.zeros(Complex{T}, N)
    buffer_mod   = CUDA.zeros(T, N)

    if max_order >= 2
        U1_buffers     = [CUDA.zeros(T, N) for _ in 1:num_w]
        U1_fft_buffers = [CUDA.zeros(Complex{T}, N) for _ in 1:num_w]
    else
        U1_buffers     = CUDA.CuMatrix{T}[]
        U1_fft_buffers = CUDA.CuMatrix{Complex{T}}[]
    end

    buffer_signal_fft = CUDA.zeros(Complex{T}, N)
    M_type = typeof(buffer_conv)
    R_type = typeof(buffer_mod)
    return ScatteringTransforms.Scattering2D.ScatteringTransform2D{
        T, M_type, R_type, typeof(plan), typeof(tree)}(
        filter_bank, tree, max_order, plan,
        buffer_input, buffer_signal_fft, buffer_conv, buffer_mod,
        U1_buffers, U1_fft_buffers,
    )
end

"""
    ScatteringTransform3D(N, J, device::CUDA.CuDevice; n_orient=6, max_order=2, T=Float32)

GPU constructor for the 3D volumetric scattering transform (CuArray filter bank + workspace,
CUFFT plan).
"""
function ScatteringTransforms.Scattering3D.ScatteringTransform3D(
    N::NTuple{3,Int}, J::Int, device::CUDA.CuDevice;
    n_orient::Int=6, max_order::Int=2, T::Type=Float32,
)
    cpu_fb = ScatteringTransforms.FilterBanks.build_filter_bank3d(N, J; n_orient=n_orient, T=T)
    cu_wavelets = [CUDA.CuArray{Complex{T},3}(ψ) for ψ in cpu_fb.wavelets]
    cu_avg = CUDA.CuArray{Complex{T},3}(cpu_fb.averaging)
    A_fb = typeof(cu_avg)
    filter_bank = ScatteringTransforms.FilterBanks.FilterBank3D{T, A_fb}(
        cu_wavelets, cu_avg, cpu_fb.meta, cpu_fb.J, cpu_fb.n_orient
    )
    tree = ScatteringTransforms.PathGraph.build_tree([m.j_eff for m in filter_bank.meta], max_order)

    plan = _cufft_plan(T, N)

    num_w = length(filter_bank.wavelets)
    buffer_input = CUDA.zeros(Complex{T}, N)
    buffer_conv  = CUDA.zeros(Complex{T}, N)
    buffer_mod   = CUDA.zeros(T, N)

    if max_order >= 2
        U1_buffers     = [CUDA.zeros(T, N) for _ in 1:num_w]
        U1_fft_buffers = [CUDA.zeros(Complex{T}, N) for _ in 1:num_w]
    else
        U1_buffers     = CUDA.CuArray{T,3}[]
        U1_fft_buffers = CUDA.CuArray{Complex{T},3}[]
    end

    buffer_signal_fft = CUDA.zeros(Complex{T}, N)
    M_type = typeof(buffer_conv)
    R_type = typeof(buffer_mod)
    return ScatteringTransforms.Scattering3D.ScatteringTransform3D{
        T, M_type, R_type, typeof(plan), typeof(tree)}(
        filter_bank, tree, max_order, plan,
        buffer_input, buffer_signal_fft, buffer_conv, buffer_mod,
        U1_buffers, U1_fft_buffers,
    )
end

end # module ScatteringTransformsCUDAExt