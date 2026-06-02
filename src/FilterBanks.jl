module FilterBanks

"""
    FilterBanks.jl — Construct dyadic filter banks for scattering transforms

Creates complete filter bank structures with wavelets at multiple scales
and orientations (for 2D), plus averaging (scaling) filters.
"""

using FFTW: FFTW
using LinearAlgebra: LinearAlgebra

# Import Filters submodule
using ..Filters: Filters

export FilterBank1D, FilterBank2D
export build_filter_bank1d, build_filter_bank2d
export averaging_filter

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
    meta::Vector{NamedTuple}
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
function build_filter_bank1d(N::Int, J::Int; Q::Int=1, T::Type=Float64)
    # Create first wavelet to get the array type
    morlet = Filters.Morlet1D{T}(N, 0; Q=Q)
    ψ_sample = Filters.frequency_response(morlet)
    V = typeof(ψ_sample)
    
    wavelets = Vector{V}(undef, 0)
    meta = Vector{NamedTuple}(undef, 0)
    
    for j in 0:J-1
        for q in 0:Q-1
            effective_j = j + q / Q
            
            morlet = Filters.Morlet1D{T}(N, floor(Int, effective_j); Q=Q)
            ψ = Filters.frequency_response(morlet)
            
            push!(wavelets, ψ)
            push!(meta, (scale=j, q=q, j_eff=effective_j, 
                        center_freq=morlet.center_freq))
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
function averaging_filter(N::Int, J::Int, ::Type{T}=Float64) where T<:Real
    ω = FFTW.fftfreq(N) .* T(2π) .* N
    
    # Cutoff at scale 2^(J-1)
    scale = T(2.0)^(J-1)
    sigma = T(0.8) * scale
    
    # Gaussian lowpass
    ϕ = exp.(-(ω .* sigma).^2 ./ T(2))
    
    return complex.(ϕ)
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
    meta::Vector{NamedTuple}
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
function build_filter_bank2d(N::NTuple{2,Int}, J::Int; L::Int=8, T::Type=Float64)
    # Create sample wavelet to get matrix type
    morlet = Filters.Morlet2D{T}(N, 0, 0.0; L=L)
    ψ_sample = Filters.frequency_response(morlet)
    M = typeof(ψ_sample)
    
    wavelets = Vector{M}(undef, 0)
    meta = Vector{NamedTuple}(undef, 0)
    
    for j in 0:J-1
        for l in 0:L-1
            theta = T(π) * l / L
            
            morlet = Filters.Morlet2D{T}(N, j, theta; L=L)
            ψ = Filters.frequency_response(morlet)
            
            push!(wavelets, ψ)
            push!(meta, (scale=j, orient=l, theta=theta,
                        center_freq=morlet.center_freq))
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
function averaging_filter2d(N::NTuple{2,Int}, J::Int, ::Type{T}=Float64) where T<:Real
    Ny, Nx = N
    
    kx = FFTW.fftfreq(Nx) .* T(2π) .* Nx
    ky = FFTW.fftfreq(Ny) .* T(2π) .* Ny
    
    # Meshgrid
    KX = [kx[j] for i in 1:Ny, j in 1:Nx]
    KY = [ky[i] for i in 1:Ny, j in 1:Nx]
    
    # Radial frequency
    K = sqrt.(KX.^2 .+ KY.^2)
    
    # Cutoff at largest scale
    scale = T(2.0)^(J-1)
    sigma = T(0.8) * scale
    
    # Gaussian lowpass
    ϕ = exp.(-(K .* sigma).^2 ./ T(2))
    
    return complex.(ϕ)
end

end # module FilterBanks
