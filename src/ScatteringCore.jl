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
bank, which dominates a transform's memory (for a 256×256 J=4 L=8 transform, 16.5 MiB of the 21.5 MiB).
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
    _modulus(z)

`|z|` for the scattering nonlinearity, as `sqrt(abs2(z))` rather than `abs(z)`.

`Base.abs(::Complex)` is `hypot`, which rescales by the larger component to stay exact across the
whole exponent range. That costs a branch and a division per element, and the modulus is 34-46% of
a cascade's runtime: `hypot` measures 5.1x (N=1024) to 9.2x (N=256) slower than the direct form,
worth 1.22x (N=1024) to 1.62x (N=256) on the whole 2D cascade.

The two agree to one ulp — 2.2e-16 relative over samples spanning 1.7e-12 to 2.7e+11. The range
`hypot` buys back is not reachable here: `abs2` overflows only for `|z| > sqrt(floatmax)` (1.3e154
in Float64, 1.8e19 in Float32) and underflows below `sqrt(floatmin)`, while a wavelet coefficient
is `O(‖x‖)`. A field near those magnitudes must be rescaled before transforming.
"""
@inline _modulus(z::Number) = sqrt(abs2(z))

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
    @. out = _modulus(signal)
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
modulus_mean(signal::AbstractArray) = sum(_modulus, signal) / length(signal)

"""
    modulus_mean!(out, signal) -> Real

Write `|signal|` into `out` and return `⟨|signal|⟩`. Used where the modulus field *is* consumed
downstream; the generic method is two device-friendly passes, the CPU method fuses them into one.
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

# `|·|` straight into a complex destination. The cascade's next step is the transform of `U₁`, whose
# input must be complex, so writing the modulus real and widening it afterwards costs two extra
# sweeps of the field — a write and a read — for a value that is real only in transit.
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

`dst[k] = (1/∏r) Σ_m src[k + m∘n] * filt[k + m∘n]`, where `n = size(dst)` and `m` runs over
`∏r` alias blocks — the spectrum of `(src ⋆ filt)` decimated by `r` per axis.

This is the whole point of the periodized cascade. For **any** spectrum `Ŷ`, with no band-limit
assumption whatsoever,

    ifft_N(Ŷ)[1:r:end] == ifft_M(periodize_r(Ŷ)),   M = N ./ r

so the decimated samples of a full-size inverse transform *are* the small inverse transform of the
periodized spectrum. Transforming at full size and keeping every `r`-th sample is wasted work, not
extra accuracy. The `1/∏r` is the decimation's own factor: an inverse normalised by `1/length` on a
grid `∏r` times smaller supplies `∏r` too little.

`r` is per axis because a single scalar cannot serve every grid — `(128, 100)` admits `r = (8, 4)`
but no scalar `8`. It also fits oriented wavelets, whose support is anisotropic.

Indices are raw and 0-based, which is what makes the alias stride plain addition: these plans are
unshifted, so bin `k` of the small grid gathers `k, k+n, k+2n, …` of the large one. The multiply is
fused in and the `1/∏r` folded into each block, so the whole thing is one pass over `src` and one
write of `dst` per alias — broadcast throughout, so it carries to a device unchanged.
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
    # A singleton filter axis is broadcast rather than sliced, so a `(spatial…, 1)` filter meets a
    # `(spatial…, B)` stack: the batch axis carries `r = 1` and every alias block spans all of it.
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

[`periodize_mul!`](@ref) with two filters: `dst[k] = (1/∏r) Σ_m src[·] * f1[·] * f2[·]`.

The monogenic amplitude band-passes by `R_d · ψ_j`, a product of two full-resolution filters. Forming
that product into scratch first and then periodizing would sweep the full grid twice; fusing it costs
one pass, exactly as the single-filter form does.
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

Allocating [`periodize_mul!`](@ref), for the non-mutating autodiff forward: no `dst` to write into and
no in-place writes for a reverse-mode backend to trip over.
"""
function periodize(src::AbstractArray{<:Any, D}, r::NTuple{D, Int}) where {D}
    all(isone, r) && return src
    n = ntuple(d -> size(src, d) ÷ r[d], D)
    blockof(M) = ntuple(d -> ((M[d] - 1) * n[d] + 1):(M[d] * n[d]), D)
    aliases = CartesianIndices(r)
    # The first block is copied, not viewed, and every later one is added out of place, so the
    # accumulator has one concrete type. A `mapreduce` over views returns the lone view unreduced
    # when there is a single alias, which infers as a union.
    acc = src[blockof(first(aliases))...]
    for M in Iterators.drop(aliases, 1)
        acc = acc + view(src, blockof(M)...)
    end
    return acc .* (one(real(eltype(src))) / prod(r))
end

"""
    periodize_filter!(dst, src, r) -> dst

A filter periodized onto a coarser grid: `dst[k] = Σ_m src[k + m∘n]`, with **no** `1/∏r`.

Not interchangeable with *rebuilding* the filter on the coarse grid. The Morlet profile is a function
of normalised angular frequency — `kx = 2π·fftfreq(n,i-1)` and `k0 = 3π/(4·2^j)` are both independent
of `n` — so evaluating it on a grid of size `n = N/r` stretches the frequency axis by `r` and lands
the peak at bin `(N/r)·k0/2π` instead of `N·k0/2π`: one octave too low per factor of two, i.e. a
different wavelet. Periodizing the original filter is what makes the undecimated case reproduce the
full cascade exactly.

No scale factor here because the `1/∏r` belongs to the *product* being decimated, not to the filter;
applying it in both places would double-count.
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
