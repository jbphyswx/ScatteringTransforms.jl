module Plans

"""
    Plans.jl — Spectral transform plans

The scattering transform needs a forward/inverse spectral transform to convolve with the
frequency-domain wavelets. Following the ecosystem convention, the core ships a slow but
dependency-free default — a **direct-summation DFT** — and fast paths live in extensions (FFTW
for uniform sampling, CUFFT on GPU, …).

A plan implements two in-place primitives:

    forward_transform!(out, plan, x)   # x -> X̂   (fft convention)
    inverse_transform!(out, plan, x)   # X̂ -> x   (ifft convention, 1/N scaled)

so the transform engine never references FFTW/CUDA directly.
"""

using LinearAlgebra: LinearAlgebra

export AbstractScatteringPlan, DirectSumPlan, forward_transform!, inverse_transform!, make_plan
export forward_transform, inverse_transform
export AbstractSpectralBackend, DirectSumBackend, FFTBackend, AutoSpectral

"""
    AbstractScatteringPlan

Supertype for spectral transform plans. A plan implements [`forward_transform!`](@ref) and
[`inverse_transform!`](@ref); the in-core default is [`DirectSumPlan`](@ref), with fast paths
(FFTW, CUFFT) provided by extensions.
"""
abstract type AbstractScatteringPlan end

"""
    forward_transform!(out, plan, x) -> out

In-place forward (fft-convention) spectral transform. Methods provided by concrete plans.
"""
function forward_transform! end

"""
    inverse_transform!(out, plan, x) -> out

In-place inverse (ifft-convention, `1/N`-scaled) spectral transform.
"""
function inverse_transform! end

"""
    forward_transform(plan, x) -> X̂
    inverse_transform(plan, x) -> x

Non-mutating, allocating, element-type-generic spectral transforms — the autodiff-friendly
counterparts of the in-place `forward_transform!`/`inverse_transform!`. They never touch the
plan's preallocated (`Complex{Float64}`-pinned) scratch buffers and never mutate their inputs,
so they accept `ForwardDiff.Dual`/`Float32`/etc. inputs and are differentiable by reverse-mode
backends (Mooncake/Zygote). Used by the non-mutating `scattering(st, x)` synthesis path; the
in-place `!` versions remain the production hot path.

The in-core [`DirectSumPlan`](@ref) implements these as dense per-axis DFT matrix-multiplies
(`W*x`) — differentiable by *every* AD backend with no special rules, but `O(N²)`; the FFTW
extension provides an `O(N log N)` method (`plan * x`, differentiable via `AbstractFFTs`
ChainRules). For large synthesis problems, load `using FFTW`.
"""
function forward_transform end

"""
    inverse_transform(plan, x) -> x

See [`forward_transform`](@ref).
"""
function inverse_transform end

"""
    fftw_plan(T, dims)

Fast-path plan constructor. Declaration only — the FFTW extension provides the sole method
(so there is no core method to overwrite). Call via [`make_plan`](@ref), which guards on the
extension being loaded.
"""
function fftw_plan end

"""
    abstractffts_plan(dummy_device_array; region=1:ndims(dummy_device_array))

Vendor-neutral device FFT plan constructor. Declaration only — the sole method is provided by the
KernelAbstractions extension, which builds forward/inverse plans via `AbstractFFTs.plan_fft` /
`plan_ifft` on `dummy_device_array`. Because those dispatch on the array *type*, the same builder
yields cuFFT (`CuArray`), rocFFT (`ROCArray`), or FFTW (plain `Array`) plans — so the GPU scattering
path is device-agnostic. `region` selects the transformed dimensions (e.g. `(1, 2)` for a batched
`(Ny, Nx, B)` stack, leaving the batch axis untouched).
"""
function abstractffts_plan end

"""
    AbstractSpectralBackend

Selects which spectral transform a transform uses: [`DirectSumBackend`](@ref) (in-core, always
available), [`FFTBackend`](@ref) (FFTW fast path, requires `using FFTW`), or [`AutoSpectral`](@ref)
(FFTW if loaded, else the direct sum). Dispatched on the *type* (not a Symbol) so the choice is
explicit and specializing.
"""
abstract type AbstractSpectralBackend end

"In-core direct-summation DFT (dependency-free, `O(N²)`)."
struct DirectSumBackend <: AbstractSpectralBackend end

"FFTW fast path (`O(N log N)`); requires the FFTW extension (`using FFTW`)."
struct FFTBackend <: AbstractSpectralBackend end

"Use the FFTW fast path if its extension is loaded, otherwise the in-core direct sum."
struct AutoSpectral <: AbstractSpectralBackend end

_have_fftw() = Base.get_extension(parentmodule(@__MODULE__), :ScatteringTransformsFFTWExt) !== nothing

"""
    make_plan(spectral::AbstractSpectralBackend, T, dims) -> AbstractScatteringPlan

Build the spectral plan selected by `spectral` for arrays of size `dims` and element type `T`.
"""
make_plan(::DirectSumBackend, ::Type{T}, dims) where {T} = DirectSumPlan(T, dims)
function make_plan(::FFTBackend, ::Type{T}, dims) where {T}
    _have_fftw() || throw(ArgumentError("FFTBackend requires the FFTW extension. Run `using FFTW`."))
    return fftw_plan(T, dims)
end
make_plan(::AutoSpectral, ::Type{T}, dims) where {T} =
    _have_fftw() ? fftw_plan(T, dims) : DirectSumPlan(T, dims)

# ---------------------------------------------------------------------------
# In-core direct-summation DFT plan (slow but dependency-free default)
# ---------------------------------------------------------------------------

"""
    DirectSumPlan{T,V,D,S}

Direct-summation DFT plan: evaluates `X_k = Σ_n x_n e^{-2πi kn/N}` (and its inverse) by direct
summation, separably per dimension. `O(N²)` per dimension but only `O(N)` memory (a per-axis
table of roots of unity) — no `O(N²)` matrix. Slow relative to FFTW, but correct, allocation-
free in the hot path, and dependency-free. `scratch` is `nothing` for 1D, a complex array for
`D ≥ 2`. Containers stay parametric.
"""
struct DirectSumPlan{T, V<:AbstractVector{Complex{T}}, D, S} <: AbstractScatteringPlan
    twiddle::NTuple{D, V}   # twiddle[d][m+1] = exp(-2πi m / N_d), m in 0:N_d-1
    dims::NTuple{D, Int}
    scratch::S
end

function _twiddles(::Type{T}, N::Int) where {T}
    tw = Vector{Complex{T}}(undef, N)
    @inbounds for m in 0:(N - 1)
        tw[m + 1] = cis(-2 * T(π) * T(m) / T(N))
    end
    return tw
end

"""
    DirectSumPlan(T, dims) -> DirectSumPlan

Build a direct-summation DFT plan for an array of size `dims` (an `Int` or `NTuple{D,Int}`).
"""
function DirectSumPlan(::Type{T}, dims::NTuple{D, Int}) where {T, D}
    tw = ntuple(d -> _twiddles(T, dims[d]), D)
    scratch = D == 1 ? nothing : zeros(Complex{T}, dims)
    return DirectSumPlan{T, Vector{Complex{T}}, D, typeof(scratch)}(tw, dims, scratch)
end
DirectSumPlan(::Type{T}, N::Int) where {T} = DirectSumPlan(T, (N,))

# Core 1D direct-sum kernel: out[k] = Σ_n x[n] * tw[((k-1)(n-1) mod N)+1], optionally conjugated
# (inverse) and scaled by 1/N. Operates on length-N vectors (or strided views).
@inline function _dft1!(out, x, tw::AbstractVector{Complex{T}}, N::Int, inverse::Bool) where {T}
    invN = inverse ? inv(T(N)) : one(T)
    @inbounds for k in 1:N
        acc = zero(Complex{T})
        for n in 1:N
            w = tw[((k - 1) * (n - 1)) % N + 1]
            acc += x[n] * (inverse ? conj(w) : w)
        end
        out[k] = acc * invN
    end
    return out
end

# 1D
forward_transform!(out::AbstractVector, p::DirectSumPlan{T,V,1}, x::AbstractVector) where {T,V} =
    _dft1!(out, x, p.twiddle[1], p.dims[1], false)
inverse_transform!(out::AbstractVector, p::DirectSumPlan{T,V,1}, x::AbstractVector) where {T,V} =
    _dft1!(out, x, p.twiddle[1], p.dims[1], true)

# 2D separable: transform each column (dim 1) then each row (dim 2), via `scratch`.
function _dft2!(out, x, p::DirectSumPlan{T,V,2}, inverse::Bool) where {T,V}
    Ny, Nx = p.dims
    sc = p.scratch
    @inbounds for jx in 1:Nx
        _dft1!(view(sc, :, jx), view(x, :, jx), p.twiddle[1], Ny, inverse)
    end
    @inbounds for jy in 1:Ny
        _dft1!(view(out, jy, :), view(sc, jy, :), p.twiddle[2], Nx, inverse)
    end
    return out
end
forward_transform!(out::AbstractMatrix, p::DirectSumPlan{T,V,2}, x::AbstractMatrix) where {T,V} =
    _dft2!(out, x, p, false)
inverse_transform!(out::AbstractMatrix, p::DirectSumPlan{T,V,2}, x::AbstractMatrix) where {T,V} =
    _dft2!(out, x, p, true)

# 3D separable: ping-pong between `out` and `scratch` so the result lands in `out`
# (A → out [dim1], out → scratch [dim2], scratch → out [dim3]).
function _dft3!(out, x, p::DirectSumPlan{T,V,3}, inverse::Bool) where {T,V}
    n1, n2, n3 = p.dims
    sc = p.scratch
    @inbounds for j3 in 1:n3, j2 in 1:n2
        _dft1!(view(out, :, j2, j3), view(x, :, j2, j3), p.twiddle[1], n1, inverse)
    end
    @inbounds for j3 in 1:n3, j1 in 1:n1
        _dft1!(view(sc, j1, :, j3), view(out, j1, :, j3), p.twiddle[2], n2, inverse)
    end
    @inbounds for j2 in 1:n2, j1 in 1:n1
        _dft1!(view(out, j1, j2, :), view(sc, j1, j2, :), p.twiddle[3], n3, inverse)
    end
    return out
end
forward_transform!(out::AbstractArray{<:Any,3}, p::DirectSumPlan{T,V,3}, x::AbstractArray{<:Any,3}) where {T,V} =
    _dft3!(out, x, p, false)
inverse_transform!(out::AbstractArray{<:Any,3}, p::DirectSumPlan{T,V,3}, x::AbstractArray{<:Any,3}) where {T,V} =
    _dft3!(out, x, p, true)

# ---------------------------------------------------------------------------
# Non-mutating, autodiff-friendly direct-sum transforms (dense DFT matrix per axis).
# Built from the read-only twiddle table each call; slow (O(N²)) but pure and differentiable.
# ---------------------------------------------------------------------------

# Dense per-axis DFT matrix W[k,n] = exp(∓2πi (k-1)(n-1)/N) from the twiddle table (no 1/N here;
# the inverse scaling is applied once over all axes by the caller). W is symmetric in (k,n).
function _dft_matrix(tw::AbstractVector{Complex{T}}, N::Int, inverse::Bool) where {T}
    return [inverse ? conj(tw[((k - 1) * (n - 1)) % N + 1]) : tw[((k - 1) * (n - 1)) % N + 1]
            for k in 1:N, n in 1:N]
end

# Apply a 1D operator matrix F along dimension `d` of a 3D array via permute/reshape/matmul.
# (2,1,3) and (3,2,1) are involutions, so the same perm restores the layout.
function _apply_along(F::AbstractMatrix, A::AbstractArray{<:Any,3}, d::Int)
    if d == 1
        sz = size(A)
        return reshape(F * reshape(A, sz[1], :), sz...)
    end
    perm = d == 2 ? (2, 1, 3) : (3, 2, 1)
    Ap = permutedims(A, perm)
    sz = size(Ap)
    R = reshape(F * reshape(Ap, sz[1], :), sz...)
    return permutedims(R, perm)
end

function forward_transform(p::DirectSumPlan{T,V,1}, x::AbstractVector) where {T,V}
    return _dft_matrix(p.twiddle[1], p.dims[1], false) * x
end
function inverse_transform(p::DirectSumPlan{T,V,1}, x::AbstractVector) where {T,V}
    return (_dft_matrix(p.twiddle[1], p.dims[1], true) * x) .* inv(T(p.dims[1]))
end

function forward_transform(p::DirectSumPlan{T,V,2}, x::AbstractMatrix) where {T,V}
    F1 = _dft_matrix(p.twiddle[1], p.dims[1], false)
    F2 = _dft_matrix(p.twiddle[2], p.dims[2], false)
    return F1 * x * transpose(F2)
end
function inverse_transform(p::DirectSumPlan{T,V,2}, x::AbstractMatrix) where {T,V}
    F1 = _dft_matrix(p.twiddle[1], p.dims[1], true)
    F2 = _dft_matrix(p.twiddle[2], p.dims[2], true)
    return (F1 * x * transpose(F2)) .* inv(T(p.dims[1] * p.dims[2]))
end

function forward_transform(p::DirectSumPlan{T,V,3}, x::AbstractArray{<:Any,3}) where {T,V}
    y = x
    for d in 1:3
        y = _apply_along(_dft_matrix(p.twiddle[d], p.dims[d], false), y, d)
    end
    return y
end
function inverse_transform(p::DirectSumPlan{T,V,3}, x::AbstractArray{<:Any,3}) where {T,V}
    y = x
    for d in 1:3
        y = _apply_along(_dft_matrix(p.twiddle[d], p.dims[d], true), y, d)
    end
    return y .* inv(T(prod(p.dims)))
end

end # module Plans
