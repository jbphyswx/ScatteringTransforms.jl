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
export averaging_filter
export WaveletMeta

# Inline fftfreq for a single 0-indexed bin — no allocation.
# Mirrors FFTW.fftfreq(N)[k+1].
@inline _fftfreq(N::Int, k::Int) = k < (N + 1) ÷ 2 ? k / N : (k - N) / N

"""
    WaveletMeta{T}

Concrete per-wavelet metadata (replaces the abstract `Vector{NamedTuple}`).

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
    FilterBank1D{T,V<:AbstractVector{Complex{T}}}

Complete 1D filter bank for scattering transform.

# Type Parameters
- `T`: Real element type (Float32, Float64, etc.)
- `V`: Wavelet vector type (allows CPU/GPU arrays)

# Fields
- `wavelets::Vector{V}`: Wavelet filters in Fourier domain
- `averaging::V`: Low-pass averaging (scaling) filter
- `meta::Vector{NamedTuple}`: Metadata for each wavelet
- `J::Int`: Number of octaves (scales)
- `Q::Int`: Number of wavelets per octave
"""
struct FilterBank1D{T,V<:AbstractVector{Complex{T}}}
    wavelets::Vector{V}
    averaging::V
    meta::Vector{WaveletMeta{T}}
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
function build_filter_bank1d(N::Int, J::Int; Q::Int=1, T::Type{<:Real}=Float64)
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
    
    # Build averaging filter with same element type
    ϕ = averaging_filter(N, J, T)
    
    return FilterBank1D{T,V}(wavelets, ϕ, meta, J, Q)
end

"""
    averaging_filter(N::Int, J::Int, ::Type{T}=Float64) -> Vector{Complex{T}}

Build low-pass averaging filter (father wavelet / scaling function).
Element type T allows Float32/Float64.
"""
function averaging_filter(N::Int, J::Int, ::Type{T}=Float64) where {T<:Real}
    xi_J  = T(0.5) / T(2.0)^(J - 1)
    sigma = xi_J * T(0.8)
    inv2  = inv(T(2))
    inv_s = inv(sigma)
    
    ϕ = Vector{Complex{T}}(undef, N)
    @inbounds for i in 1:N
        ω = T(_fftfreq(N, i - 1))
        ϕ[i] = Complex{T}(exp(-(ω * inv_s)^2 * inv2))
    end
    return ϕ
end

"""
    FilterBank2D{T,M<:AbstractMatrix{Complex{T}}}

Complete 2D filter bank with oriented wavelets.

# Type Parameters
- `T`: Real element type
- `M`: Matrix type for wavelets (allows CPU/GPU arrays)

# Fields
- `wavelets::Vector{M}`: Wavelets indexed by [scale_index]
- `averaging::M`: Low-pass averaging filter
- `meta::Vector{NamedTuple}`: Metadata
- `J::Int`: Number of scales
- `L::Int`: Number of orientations
"""
struct FilterBank2D{T,M<:AbstractMatrix{Complex{T}}}
    wavelets::Vector{M}
    averaging::M
    meta::Vector{WaveletMeta{T}}
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
function build_filter_bank2d(N::NTuple{2,Int}, J::Int; L::Int=8, T::Type{<:Real}=Float64)
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
    
    # 2D averaging filter with same element type
    ϕ = averaging_filter2d(N, J, T)
    
    return FilterBank2D{T,M}(wavelets, ϕ, meta, J, L)
end

"""
    averaging_filter2d(N::NTuple{2,Int}, J::Int, ::Type{T}=Float64) -> Matrix{Complex{T}}

Build 2D low-pass averaging filter.
"""
function averaging_filter2d(N::NTuple{2,Int}, J::Int, ::Type{T}=Float64) where {T<:Real}
    Ny, Nx = N
    xi_J  = T(0.5) / T(2.0)^(J - 1)
    sigma = xi_J * T(0.8)
    inv2  = inv(T(2))
    inv_s = inv(sigma)
    
    ϕ = Matrix{Complex{T}}(undef, Ny, Nx)
    @inbounds for ix in 1:Nx
        kx = T(_fftfreq(Nx, ix - 1))
        for iy in 1:Ny
            ky = T(_fftfreq(Ny, iy - 1))
            k  = sqrt(kx^2 + ky^2)
            ϕ[iy, ix] = Complex{T}(exp(-(k * inv_s)^2 * inv2))
        end
    end
    return ϕ
end

"""
    FilterBank3D{T,A<:AbstractArray{Complex{T},3}}

Complete 3D oriented Morlet filter bank: `J` scales × `n_orient` sphere directions, plus a
low-pass averaging filter.
"""
struct FilterBank3D{T,A<:AbstractArray{Complex{T},3}}
    wavelets::Vector{A}
    averaging::A
    meta::Vector{WaveletMeta{T}}
    J::Int
    n_orient::Int
end

"""
    build_filter_bank3d(N::NTuple{3,Int}, J::Int; n_orient::Int=6, T=Float64) -> FilterBank3D

Build a 3D oriented Morlet filter bank with `J` dyadic scales and `n_orient` near-uniform
orientations on the sphere (Fibonacci spiral).
"""
function build_filter_bank3d(N::NTuple{3,Int}, J::Int; n_orient::Int=6, T::Type{<:Real}=Float64)
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
    ϕ = averaging_filter3d(N, J, T)
    return FilterBank3D{T,A}(wavelets, ϕ, meta, J, n_orient)
end

"""
    averaging_filter3d(N::NTuple{3,Int}, J::Int, ::Type{T}=Float64) -> Array{Complex{T},3}

Build a radially-symmetric 3D low-pass averaging filter.
"""
function averaging_filter3d(N::NTuple{3,Int}, J::Int, ::Type{T}=Float64) where {T<:Real}
    Nz, Ny, Nx = N
    xi_J  = T(0.5) / T(2.0)^(J - 1)
    sigma = xi_J * T(0.8)
    inv2  = inv(T(2))
    inv_s = inv(sigma)

    ϕ = Array{Complex{T},3}(undef, Nz, Ny, Nx)
    @inbounds for ix in 1:Nx
        kx = T(_fftfreq(Nx, ix - 1))
        for iy in 1:Ny
            ky = T(_fftfreq(Ny, iy - 1))
            for iz in 1:Nz
                kz = T(_fftfreq(Nz, iz - 1))
                k = sqrt(kx^2 + ky^2 + kz^2)
                ϕ[iz, iy, ix] = Complex{T}(exp(-(k * inv_s)^2 * inv2))
            end
        end
    end
    return ϕ
end

end # module FilterBanks
