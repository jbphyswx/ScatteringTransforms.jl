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
    fftw_plan(T, dims)

Fast-path plan constructor. Declaration only — the FFTW extension provides the sole method
(so there is no core method to overwrite). Call via [`make_plan`](@ref), which guards on the
extension being loaded.
"""
function fftw_plan end

"""
    make_plan(spectral, T, dims) -> AbstractScatteringPlan

Select a spectral plan. `spectral` is one of:
- `:direct` — the in-core `DirectSumPlan` (always available, slow);
- `:fftw`   — the FFTW fast path (requires `using FFTW`);
- `:auto`   — FFTW if its extension is loaded, otherwise the direct sum.
"""
function make_plan(spectral::Symbol, ::Type{T}, dims) where {T}
    spectral === :direct && return DirectSumPlan(T, dims)
    have_fftw = Base.get_extension(parentmodule(@__MODULE__), :ScatteringTransformsFFTWExt) !== nothing
    if spectral === :fftw
        have_fftw || throw(ArgumentError("spectral=:fftw requires the FFTW extension. Run `using FFTW`."))
        return fftw_plan(T, dims)
    elseif spectral === :auto
        return have_fftw ? fftw_plan(T, dims) : DirectSumPlan(T, dims)
    else
        throw(ArgumentError("unknown spectral=$(repr(spectral)) (use :auto, :direct, or :fftw)"))
    end
end

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

end # module Plans
