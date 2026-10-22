module ScatteringCore

using LinearAlgebra: LinearAlgebra
using ..Plans: Plans

export wavelet_convolve, wavelet_convolve!
export apply_modulus, apply_modulus!, spatial_average
export scattering

"""
    scattering(st, x) -> coefficient container

The coefficients of `st(x)`, computed without mutation so the forward map can be differentiated
(as `synthesize` does) and takes any element type, such as `ForwardDiff.Dual`.

Every intermediate comes from the non-mutating [`Plans.forward_transform`](@ref) and
[`Plans.inverse_transform`](@ref) and broadcasts, so each call allocates; `st(x)` is the in-place
forward pass. Methods are defined for the 1D, 2D and 3D transforms.
"""
function scattering end

"""
    task_workspace(st) -> st′

A transform equivalent to `st` that shares its read-only parts — filter bank, path tree, work list —
but owns fresh buffers and a task-local spectral plan, so the two can run concurrently. Methods are
defined per transform type.
"""
function task_workspace end

"""
    wavelet_convolve(signal_fft, filter_fft, plan) -> AbstractArray

The inverse transform of `signal_fft .* filter_fft` through `plan`, in a new array;
[`wavelet_convolve!`](@ref) writes in place.
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

Wavelet convolution without allocation: multiplies `signal_fft .* filter_fft` into `buffer`, then
applies the inverse spectral transform via `Plans.inverse_transform!(out, plan, buffer)`.

`out` and `buffer` must both be pre-allocated complex arrays of the same size.
"""
function wavelet_convolve!(out::AbstractArray,
                          signal_fft::AbstractArray,
                          filter_fft::AbstractArray,
                          plan,
                          buffer::AbstractArray)
    @. buffer = signal_fft * filter_fft
    Plans.inverse_transform!(out, plan, buffer)
    return out
end

"""
    _modulus(z)

`|z|` for the scattering nonlinearity, computed as `sqrt(abs2(z))`.

It agrees with `abs(z)` to one ulp while `abs2(z)` neither overflows nor underflows, i.e. for
`sqrt(floatmin) < |z| < sqrt(floatmax)`. A wavelet coefficient is `O(‖x‖)`, so a field near those
magnitudes must be rescaled before transforming.
"""
@inline _modulus(z::Number) = sqrt(abs2(z))

"""
    apply_modulus(signal) -> AbstractArray

`|signal|` elementwise, in a new real array; [`apply_modulus!`](@ref) writes in place.
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
    @. out = _modulus(signal)
    return out
end

"""
    spatial_average(signal) -> Number

The global mean of `signal`.
"""
function spatial_average(signal::AbstractArray)
    return sum(signal) / length(signal)
end

"""
    modulus_mean(signal) -> Real

`⟨|signal|⟩` in one reduction, writing no modulus field: the cascade's leaf case, where no coarser
scale reads it.
"""
modulus_mean(signal::AbstractArray) = sum(_modulus, signal) / length(signal)

"""
    modulus_mean!(out, signal) -> Real

Write `|signal|` into `out`, real or complex, and return `⟨|signal|⟩`: the case where a coarser
scale reads the modulus field.
"""
modulus_mean!(out::AbstractArray, signal::AbstractArray) =
    (@. out = _modulus(signal); sum(out) / length(out))

function modulus_mean!(out::Array{T}, signal::Array{Complex{T}}) where {T <: Real}
    acc = zero(T)
    @inbounds @simd for i in eachindex(out, signal)
        v = _modulus(signal[i])
        out[i] = v
        acc += v
    end
    return acc / length(out)
end

# `|·|` written straight into a complex destination: the next cascade step transforms `U₁`, which
# takes a complex input.
function modulus_mean!(out::Array{Complex{T}}, signal::Array{Complex{T}}) where {T <: Real}
    acc = zero(T)
    @inbounds @simd for i in eachindex(out, signal)
        v = _modulus(signal[i])
        out[i] = v
        acc += v
    end
    return acc / length(out)
end

# ---------------------------------------------------------------------------
# Periodization: decimating a transform's output without transforming at full size
# ---------------------------------------------------------------------------

"""
    periodize_mul!(dst, src, filt, r) -> dst
    periodize!(dst, src, r) -> dst

`dst[k] = (1/∏r) Σ_m src[k + m∘n] * filt[k + m∘n]`, where `n = size(dst)` and `m` runs over the
`∏r` alias blocks: the spectrum of `(src ⋆ filt)` decimated by `r` per axis. For any spectrum `Ŷ`,

    ifft_N(Ŷ)[1:r:end] == ifft_M(periodize_r(Ŷ)),   M = N ./ r

with the `1/∏r` supplying the normalisation `ifft_M` lacks. Indices are 0-based on unshifted grids,
so bin `k` of `dst` gathers bins `k, k+n, k+2n, …` of `src`; a singleton `filt` axis broadcasts.
Both are broadcasts only, so they run on device arrays.
"""
function periodize_mul!(dst::AbstractArray{<:Any, D}, src::AbstractArray{<:Any, D},
                        filt::AbstractArray{<:Any, D}, r::NTuple{D, Int}) where {D}
    if all(isone, r)
        @. dst = src * filt
        return dst
    end
    n = size(dst)
    s = one(real(eltype(dst))) / prod(r)
    aliases = CartesianIndices(r)
    # A singleton filter axis broadcasts, so a `(spatial…, 1)` filter meets a `(spatial…, B)` stack
    # whose batch axis carries `r = 1`.
    fsingle = ntuple(d -> size(filt, d) == 1, D)
    blockof(M) = ntuple(d -> ((M[d] - 1) * n[d] + 1):(M[d] * n[d]), D)
    fblockof(M) = ntuple(d -> fsingle[d] ? (1:1) : (((M[d] - 1) * n[d] + 1):(M[d] * n[d])), D)
    @inbounds begin
        M = first(aliases)
        sv, fv = view(src, blockof(M)...), view(filt, fblockof(M)...)
        @. dst = sv * fv * s
        for M in Iterators.drop(aliases, 1)
            sv, fv = view(src, blockof(M)...), view(filt, fblockof(M)...)
            @. dst += sv * fv * s
        end
    end
    return dst
end

function periodize!(dst::AbstractArray{<:Any, D}, src::AbstractArray{<:Any, D},
                    r::NTuple{D, Int}) where {D}
    if all(isone, r)
        copyto!(dst, src)
        return dst
    end
    n = size(dst)
    s = one(real(eltype(dst))) / prod(r)
    aliases = CartesianIndices(r)
    blockof(M) = ntuple(d -> ((M[d] - 1) * n[d] + 1):(M[d] * n[d]), D)
    @inbounds begin
        blk = blockof(first(aliases))
        sv = view(src, blk...)
        @. dst = sv * s
        for M in Iterators.drop(aliases, 1)
            blk = blockof(M)
            sv = view(src, blk...)
            @. dst += sv * s
        end
    end
    return dst
end

"""
    periodize_mul2!(dst, src, f1, f2, r) -> dst

[`periodize_mul!`](@ref) with two filters, `dst[k] = (1/∏r) Σ_m src[·] * f1[·] * f2[·]`, in one
pass: the monogenic band-pass `R_d · ψ_j` is a product of two full-resolution filters.
"""
function periodize_mul2!(dst::AbstractArray{<:Any, D}, src::AbstractArray{<:Any, D},
                         f1::AbstractArray{<:Any, D}, f2::AbstractArray{<:Any, D},
                         r::NTuple{D, Int}) where {D}
    if all(isone, r)
        @. dst = src * f1 * f2
        return dst
    end
    n = size(dst)
    s = one(real(eltype(dst))) / prod(r)
    aliases = CartesianIndices(r)
    blockof(M) = ntuple(d -> ((M[d] - 1) * n[d] + 1):(M[d] * n[d]), D)
    @inbounds begin
        blk = blockof(first(aliases))
        sv, av, bv = view(src, blk...), view(f1, blk...), view(f2, blk...)
        @. dst = sv * av * bv * s
        for M in Iterators.drop(aliases, 1)
            blk = blockof(M)
            sv, av, bv = view(src, blk...), view(f1, blk...), view(f2, blk...)
            @. dst += sv * av * bv * s
        end
    end
    return dst
end

"""
    periodize(src, r) -> dst

Allocating form of `periodize!` (see [`periodize_mul!`](@ref)), for the non-mutating forward that
autodiff differentiates; it writes nothing in place.
"""
function periodize(src::AbstractArray{<:Any, D}, r::NTuple{D, Int}) where {D}
    all(isone, r) && return src
    n = ntuple(d -> size(src, d) ÷ r[d], D)
    blockof(M) = ntuple(d -> ((M[d] - 1) * n[d] + 1):(M[d] * n[d]), D)
    aliases = CartesianIndices(r)
    # Copying the first block and adding the rest out of place gives the accumulator one concrete
    # type.
    acc = src[blockof(first(aliases))...]
    for M in Iterators.drop(aliases, 1)
        acc = acc + view(src, blockof(M)...)
    end
    return acc .* (one(real(eltype(src))) / prod(r))
end

"""
    periodize_filter!(dst, src, r) -> dst

A filter periodized onto a coarser grid: `dst[k] = Σ_m src[k + m∘n]`, with no `1/∏r`, which
[`periodize_mul!`](@ref) applies to the product being decimated.

The periodized filter keeps the original's frequencies. A Morlet evaluated afresh on the `N/r` grid
is a different wavelet: its profile is a function of normalised angular frequency, so its peak lands
at bin `(N/r)·k0/2π`, an octave lower per factor of two.
"""
function periodize_filter!(dst::AbstractArray{<:Any, D}, src::AbstractArray{<:Any, D},
                           r::NTuple{D, Int}) where {D}
    if all(isone, r)
        copyto!(dst, src)
        return dst
    end
    n = size(dst)
    aliases = CartesianIndices(r)
    blockof(M) = ntuple(d -> ((M[d] - 1) * n[d] + 1):(M[d] * n[d]), D)
    @inbounds begin
        blk = blockof(first(aliases))
        sv = view(src, blk...)
        copyto!(dst, sv)
        for M in Iterators.drop(aliases, 1)
            blk = blockof(M)
            sv = view(src, blk...)
            @. dst += sv
        end
    end
    return dst
end

end # module ScatteringCore
