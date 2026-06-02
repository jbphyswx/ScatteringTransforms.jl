module Filters

"""
    Filters.jl — Frequency-domain wavelet filter definitions

Implements Morlet wavelets in the frequency domain for FFT-based convolutions.
"""

using FFTW: FFTW
using LinearAlgebra: LinearAlgebra

export Morlet1D, Morlet2D
export frequency_response, gaussian_window, plane_wave

"""
    Morlet1D{T<:Real}

1D Morlet wavelet in frequency domain.

The Morlet wavelet is a complex sinusoid modulated by a Gaussian:
    ψ(x) = (1/√|Σ|) exp(-x²/(2σ²)) (exp(i k₀ x) - β)

where β = exp(-σ²k₀²/2) ensures zero mean (admissibility condition).

In frequency domain:
    Ψ(ω) = exp(-(ω-k₀)²σ²/2) - β exp(-ω²σ²/2)

# Type Parameters
- `T`: Element type (Float32, Float64, etc.)

# Fields
- `center_freq::T`: Center frequency k₀
- `bandwidth::T`: Standard deviation σ of Gaussian envelope  
- `beta::T`: Correction factor for zero mean
- `N::Int`: Filter length (FFT size)
"""
struct Morlet1D{T<:Real}
    center_freq::T
    bandwidth::T
    beta::T
    N::Int
    
    function Morlet1D{T}(N::Int, j::Real; Q::Int=1, r::T=T(sqrt(0.5))) where T<:Real
        # Center frequency: xi = 0.5 * 2^(-j/Q) in normalized frequency [0, 1]
        xi = T(0.5) / (T(2.0)^(j / Q))
        
        # Bandwidth: sigma = xi * (1 - 2^(-1/Q)) / (1 + 2^(-1/Q)) / sqrt(2*log(1/r))
        # This ensures proper coverage of frequency axis (from Lostanlen 2017, Kymatio)
        factor = T(1.0) / (T(2.0)^(T(1.0) / Q))
        term1 = (T(1.0) - factor) / (T(1.0) + factor)
        term2 = T(1.0) / Base.sqrt(T(2.0) * Base.log(T(1.0) / r))
        sigma = xi * term1 * term2  # Bandwidth proportional to center frequency
        
        # β ensures zero mean (wavelet admissibility)
        # The wavelet is: Ψ(ω) = G(ω - ξ) - β·G(ω) where G is Gaussian
        β = Base.exp(-(sigma * xi)^2 / T(2))
        
        new{T}(xi, sigma, β, N)
    end
end

# Convenience constructor - defaults to Float64 for backward compatibility
Morlet1D(N::Int, j::Real; kwargs...) = Morlet1D{Float64}(N, j; kwargs...)

"""
    frequency_response(m::Morlet1D{T}) -> Vector{Complex{T}}

Compute the frequency response Ψ(ω) of a 1D Morlet wavelet.

Returns a length-N vector with the Fourier-domain filter coefficients.
The response is analytic (zero for negative frequencies) for proper
wavelet transform. Element type matches the wavelet's precision.
"""
function frequency_response(m::Morlet1D{T}) where T<:Real
    N = m.N
    # Normalized frequency grid [0, 0.5] for positive frequencies
    ω = FFTW.fftfreq(N)
    
    # Compute Gaussians: G(ω - ξ) and G(ω)
    gabor = exp.(-((ω .- m.center_freq) ./ m.bandwidth).^2 ./ T(2))
    lowpass = exp.(-(ω ./ m.bandwidth).^2 ./ T(2))
    
    # Compute kappa numerically (like kymatio) to ensure Ψ(0) = 0
    # Find index closest to ω=0
    zero_idx = argmin(abs.(ω))
    kappa = gabor[zero_idx] / lowpass[zero_idx]
    
    # Morlet wavelet: G(ω-ξ) - kappa * G(ω)
    Ψ = gabor .- kappa .* lowpass
    
    # Make analytic: zero out negative frequencies
    neg_idx = ω .< 0
    Ψ[neg_idx] .= 0
    
    return complex.(Ψ)
end

"""
    Morlet2D{T<:Real}

2D oriented Morlet wavelet in frequency domain.

The 2D Morlet wavelet is created by taking a 1D Morlet and rotating it
to angle θ, with elliptical Gaussian envelope controlled by elongation.

# Type Parameters
- `T`: Element type (Float32, Float64, etc.)

# Fields
- `center_freq::T`: Center wavenumber |k₀|
- `bandwidth_x::T`: Bandwidth along major axis  
- `bandwidth_y::T`: Bandwidth along minor axis (controls elongation)
- `theta::T`: Orientation angle in radians
- `beta::T`: Correction factor
- `N::NTuple{2,Int}`: Filter dimensions (Ny, Nx)
"""
struct Morlet2D{T<:Real}
    center_freq::T
    bandwidth_x::T
    bandwidth_y::T
    theta::T
    beta::T
    N::NTuple{2,Int}
    
    function Morlet2D{T}(N::NTuple{2,Int}, j::Int, theta::Real; 
                         L::Int=8, Q::Int=1, sigma0::T=T(0.8),
                         elongation::T=T(4.0)) where T<:Real
        # Dyadic scale
        scale = T(2.0)^j
        sigma_x = sigma0 * scale
        sigma_y = sigma_x / elongation  # Elongated wavelet
        
        # Center frequency
        k0 = T(3π) / (T(4) * scale)
        
        # β for zero mean
        β = exp(-(sigma_x * k0)^2 / T(2))
        
        new{T}(k0, sigma_x, sigma_y, T(theta), β, N)
    end
end

# Convenience constructor - defaults to Float64
Morlet2D(N::NTuple{2,Int}, j::Int, theta::Real; kwargs...) = 
    Morlet2D{Float64}(N, j, theta; kwargs...)

"""
    frequency_response(m::Morlet2D{T}) -> Matrix{Complex{T}}

Compute the 2D frequency response Ψ(kx, ky) of an oriented Morlet wavelet.
Element type matches the wavelet's precision.
"""
function frequency_response(m::Morlet2D{T}) where T<:Real
    Ny, Nx = m.N
    
    # 2D frequency grid
    kx = FFTW.fftfreq(Nx) .* T(2π) .* Nx
    ky = FFTW.fftfreq(Ny) .* T(2π) .* Ny
    
    # Create 2D meshgrid
    KX = [kx[j] for i in 1:Ny, j in 1:Nx]
    KY = [ky[i] for i in 1:Ny, j in 1:Nx]
    
    # Rotate coordinates by -theta to align with wavelet orientation
    ct = cos(m.theta)
    st = sin(m.theta)
    KX_rot = KX .* ct .+ KY .* st  # Along wavelet direction
    KY_rot = -KX .* st .+ KY .* ct  # Perpendicular to wavelet
    
    # Distance from center frequency in rotated frame
    dkx = KX_rot .- m.center_freq
    dky = KY_rot
    
    # 2D Gaussian envelope (elliptical)
    envelope = exp.(-(dkx ./ m.bandwidth_x).^2 ./ T(2) .-
                     (dky ./ m.bandwidth_y).^2 ./ T(2))
    
    # Centered Gaussian for β correction
    centered = exp.(-(KX_rot ./ m.bandwidth_x).^2 ./ T(2) .-
                     (KY_rot ./ m.bandwidth_y).^2 ./ T(2))
    
    Ψ = envelope .- m.beta .* centered
    
    # Make analytic: zero out region where k · direction < 0
    neg_idx = KX_rot .< 0
    Ψ[neg_idx] .= 0
    
    return complex.(Ψ)  # Returns Matrix{Complex{T}}
end

"""
    gaussian_window(N::Int, sigma::Real) -> Vector{Float64}

Create a Gaussian window in real space.
"""
function gaussian_window(N::Int, sigma::Real)
    x = range(-N÷2, N÷2 - 1, length=N)
    return exp.(-x.^2 ./ (2 .* sigma.^2))
end

"""
    plane_wave(N::Int, k::Real) -> Vector{ComplexF64}

Create a complex plane wave exp(i k x) for testing.
"""
function plane_wave(N::Int, k::Real)
    x = range(0, 2π * (1 - 1/N), length=N)
    return exp.(im .* k .* x)
end

end # module Filters
