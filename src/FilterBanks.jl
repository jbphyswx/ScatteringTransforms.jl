module FilterBanks

"""
    FilterBanks.jl — Construct dyadic filter banks for scattering transforms

Creates complete filter bank structures with wavelets at multiple scales
and orientations (for 2D), plus averaging (scaling) filters.
"""

using LinearAlgebra: LinearAlgebra

# Import Filters submodule
using ..Filters: Filters

export FilterBank1D, FilterBank2D, FilterBank3D
export build_filter_bank1d, build_filter_bank2d, build_filter_bank3d
export WaveletMeta

# Inline fftfreq for a single 0-indexed bin — no allocation.
# Mirrors FFTW.fftfreq(N)[k+1].
@inline _fftfreq(N::Int, k::Int) = k < (N + 1) ÷ 2 ? k / N : (k - N) / N

"""
    _complement_lowpass(wavelets) -> averaging filter φ

Build the scaling function (low-pass averaging filter) as the *complement* of the wavelet
energy: `|φ(ω)|² = max(0, 1 − Σⱼ|ψⱼ(ω)|²)`. This makes the Littlewood–Paley sum
`Σⱼ|ψⱼ|² + |φ|² ≡ 1` a (near) tight frame, so the transform is non-expansive (no frequency is
amplified). The DC bin is pinned to `φ(0)=1` exactly (the wavelets are zero-mean there), which
keeps the localized-field spatial mean equal to the globally-averaged coefficient.
Works for 1D/2D/3D filter arrays.
"""
function _complement_lowpass(wavelets::AbstractVector{A}) where {T, A<:AbstractArray{Complex{T}}}
    ϕ = similar(first(wavelets))
    @inbounds for i in eachindex(ϕ)
        s = zero(T)
        for ψ in wavelets
            s += abs2(ψ[i])
        end
        ϕ[i] = Complex{T}(sqrt(max(zero(T), one(T) - s)))
    end
    ϕ[firstindex(ϕ)] = one(Complex{T})   # exact DC = 1 (preserves mean ⇔ averaged-coefficient)
    return ϕ
end

"""
    _tight_frame_lowpass!(wavelets) -> averaging filter φ

Globally rescale `wavelets` (in place) so that `max_ω Σⱼ|ψⱼ(ω)|² = 1` — making the transform
non-expansive — then return the complement low-pass φ, giving a tight frame with
Littlewood–Paley sum `Σⱼ|ψⱼ|² + |φ|² ≡ 1`. The rescale is a single global constant, so it does
not change the *relative* coefficient structure.
"""
function _tight_frame_lowpass!(wavelets::AbstractVector{A}) where {T, A<:AbstractArray{Complex{T}}}
    maxs = zero(T)
    @inbounds for i in eachindex(first(wavelets))
        s = zero(T)
        for ψ in wavelets
            s += abs2(ψ[i])
        end
        maxs = max(maxs, s)
    end
    if maxs > zero(T)
        c = inv(sqrt(maxs))
        for ψ in wavelets
            ψ .*= c
        end
    end
    return _complement_lowpass(wavelets)
end

"""
    WaveletMeta{T}

Concrete per-wavelet metadata: a struct rather than a `NamedTuple`, so the container stays
concretely typed.

# Fields
- `scale::Int`: octave index `j`
- `q::Int`: sub-octave index within the octave (1D, `0..Q-1`); `0` for 2D
- `orient::Int`: orientation index `l` (2D, `0..L-1`); `0` for 1D
- `j_eff::T`: effective log-scale used to order paths. `j + q/Q` in 1D, `T(j)` in 2D. The
  second-order admissibility constraint is `j_eff(child) > j_eff(parent)` (frequency strictly
  decreasing) — which for 2D means *scale strictly increasing over all orientation pairs*.
- `center_freq::T`: wavelet center frequency
- `theta::T`: orientation angle in radians (2D); `0` for 1D
"""
struct WaveletMeta{T}
    scale::Int
    q::Int
    orient::Int
    j_eff::T
    center_freq::T
    theta::T
end

"""
    FilterBank1D{T,V,W,MV}

Complete 1D filter bank for the scattering transform. Every container is a type parameter (no
hardcoded `Vector`): `V` the per-filter array type (CPU/GPU/static/…), `W` the wavelet
collection, `MV` the metadata collection.

# Fields
- `wavelets::W`: wavelet filters in the Fourier domain (`W<:AbstractVector{V}`)
- `averaging::V`: low-pass averaging (scaling) filter
- `meta::MV`: per-wavelet `WaveletMeta`
- `J::Int`: number of octaves (scales)
- `Q::Int`: wavelets per octave
"""
struct FilterBank1D{T, V<:AbstractVector{Complex{T}}, W<:AbstractVector{V}, MV<:AbstractVector{WaveletMeta{T}}}
    wavelets::W
    averaging::V
    meta::MV
    J::Int
    Q::Int
end

"""
    build_filter_bank1d(N::Int, J::Int; Q::Int=1) -> FilterBank1D

Build a 1D Morlet filter bank with dyadic scales.

# Arguments
- `N::Int`: Signal length (FFT size)
- `J::Int`: Maximum scale (number of octaves)
- `Q::Int`: Wavelets per octave (default 1 for dyadic, 8 for high Q)

# Returns
- `FilterBank1D`: Complete filter bank with J scales
"""
function build_filter_bank1d(::Type{T}, N::Int, J::Int; Q::Int=1) where {T<:Real}
    # Create first wavelet to get the array type
    morlet = Filters.Morlet1D{T}(N, 0; Q=Q)
    ψ_sample = Filters.frequency_response(morlet)
    V = typeof(ψ_sample)
    
    wavelets = Vector{V}(undef, 0)
    meta = Vector{WaveletMeta{T}}(undef, 0)

    for j in 0:J-1
        for q in 0:Q-1
            effective_j = j + q / Q

            morlet = Filters.Morlet1D{T}(N, j * Q + q; Q=Q)
            ψ = Filters.frequency_response(morlet)

            push!(wavelets, ψ)
            push!(meta, WaveletMeta{T}(j, q, 0, T(effective_j), morlet.center_freq, zero(T)))
        end
    end
    
    # Low-pass = complement of the wavelet energy (tight-frame Littlewood-Paley ≈ 1)
    ϕ = _tight_frame_lowpass!(wavelets)

    return FilterBank1D(wavelets, ϕ, meta, J, Q)
end
build_filter_bank1d(N::Int, J::Int; kwargs...) = build_filter_bank1d(Float64, N, J; kwargs...)

"""
    FilterBank2D{T,M,W,MV}

Complete 2D filter bank with oriented wavelets. Containers are type parameters (no hardcoded
`Vector`): `M` the per-filter matrix type, `W` the wavelet collection, `MV` the metadata
collection.

# Fields
- `wavelets::W`: oriented wavelet filters (`W<:AbstractVector{M}`)
- `averaging::M`: low-pass averaging filter
- `meta::MV`: per-wavelet `WaveletMeta`
- `J::Int`: number of scales
- `L::Int`: number of orientations
"""
struct FilterBank2D{T, M<:AbstractMatrix{Complex{T}}, W<:AbstractVector{M}, MV<:AbstractVector{WaveletMeta{T}}}
    wavelets::W
    averaging::M
    meta::MV
    J::Int
    L::Int
end

"""
    build_filter_bank2d(N::NTuple{2,Int}, J::Int; L::Int=8) -> FilterBank2D

Build a 2D oriented Morlet filter bank.

# Arguments
- `N::NTuple{2,Int}`: Image dimensions (Ny, Nx)
- `J::Int`: Number of dyadic scales
- `L::Int`: Number of orientations (default 8, evenly spaced)

# Returns
- `FilterBank2D`: Complete 2D filter bank
"""
function build_filter_bank2d(::Type{T}, N::NTuple{2,Int}, J::Int; L::Int=8) where {T<:Real}
    # Create sample wavelet to get matrix type
    morlet = Filters.Morlet2D{T}(N, 0, 0.0; L=L)
    ψ_sample = Filters.frequency_response(morlet)
    M = typeof(ψ_sample)
    
    wavelets = Vector{M}(undef, 0)
    meta = Vector{WaveletMeta{T}}(undef, 0)

    for j in 0:J-1
        for l in 0:L-1
            theta = T(π) * l / L

            morlet = Filters.Morlet2D{T}(N, j, theta; L=L)
            ψ = Filters.frequency_response(morlet)

            push!(wavelets, ψ)
            # j_eff = T(j): same-scale (different-orientation) pairs share j_eff and are therefore
            # NOT admissible as second-order paths; only strictly coarser scales are.
            push!(meta, WaveletMeta{T}(j, 0, l, T(j), morlet.center_freq, theta))
        end
    end
    
    # Low-pass = complement of the wavelet energy (tight-frame Littlewood-Paley ≈ 1)
    ϕ = _tight_frame_lowpass!(wavelets)

    return FilterBank2D(wavelets, ϕ, meta, J, L)
end
build_filter_bank2d(N::NTuple{2,Int}, J::Int; kwargs...) = build_filter_bank2d(Float64, N, J; kwargs...)

"""
    FilterBank3D{T,A<:AbstractArray{Complex{T},3}}

Complete 3D oriented Morlet filter bank: `J` scales × `n_orient` sphere directions, plus a
low-pass averaging filter.
"""
struct FilterBank3D{T, A<:AbstractArray{Complex{T},3}, W<:AbstractVector{A}, MV<:AbstractVector{WaveletMeta{T}}}
    wavelets::W
    averaging::A
    meta::MV
    J::Int
    n_orient::Int
end

"""
    build_filter_bank3d(N::NTuple{3,Int}, J::Int; n_orient::Int=6, T=Float64) -> FilterBank3D

Build a 3D oriented Morlet filter bank with `J` dyadic scales and `n_orient` near-uniform
orientations on the sphere (Fibonacci spiral).
"""
function build_filter_bank3d(::Type{T}, N::NTuple{3,Int}, J::Int; n_orient::Int=6) where {T<:Real}
    dirs = Filters.fibonacci_directions(n_orient, T)
    morlet = Filters.Morlet3D{T}(N, 0, dirs[1])
    ψ_sample = Filters.frequency_response(morlet)
    A = typeof(ψ_sample)

    wavelets = Vector{A}(undef, 0)
    meta = Vector{WaveletMeta{T}}(undef, 0)
    for j in 0:(J - 1)
        for (o, d) in enumerate(dirs)
            morlet = Filters.Morlet3D{T}(N, j, d)
            push!(wavelets, Filters.frequency_response(morlet))
            push!(meta, WaveletMeta{T}(j, 0, o - 1, T(j), morlet.center_freq, zero(T)))
        end
    end
    # Low-pass = complement of the wavelet energy (tight-frame Littlewood-Paley ≈ 1)
    ϕ = _tight_frame_lowpass!(wavelets)
    return FilterBank3D(wavelets, ϕ, meta, J, n_orient)
end
build_filter_bank3d(N::NTuple{3,Int}, J::Int; kwargs...) = build_filter_bank3d(Float64, N, J; kwargs...)

end # module FilterBanks
