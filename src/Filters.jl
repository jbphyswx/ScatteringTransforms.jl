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
# Inline fftfreq for a single bin k (0-indexed) of length N.
# Equivalent to FFTW.fftfreq(N)[k+1]. No allocation.
# For even N: bins 0..N÷2-1 are positive, bins N÷2..N-1 are negative (Nyquist goes negative).
# For odd N: bins 0..(N-1)÷2 are positive, rest negative.
@inline _fftfreq(N::Int, k::Int) = k < (N + 1) ÷ 2 ? k / N : (k - N) / N

function frequency_response(m::Morlet1D{T}) where T<:Real
    N = m.N
    ξ = m.center_freq
    σ = m.bandwidth
    inv2 = inv(T(2))
    
    Ψ = Vector{Complex{T}}(undef, N)
    
    # kappa: ratio at ω=0 (bin 0). gabor(0)=exp(-(xi/sigma)^2/2), lowpass(0)=1
    kappa = exp(-(ξ / σ)^2 * inv2)
    
    @inbounds for i in 1:N
        ω = T(_fftfreq(N, i - 1))
        if ω < 0
            Ψ[i] = zero(Complex{T})
        else
            g  = exp(-((ω - ξ) / σ)^2 * inv2)
            lp = exp(-(ω / σ)^2 * inv2)
            Ψ[i] = Complex{T}(g - kappa * lp)
        end
    end
    
    return Ψ
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
    k0  = m.center_freq
    σx  = m.bandwidth_x
    σy  = m.bandwidth_y
    β   = m.beta
    ct  = T(cos(m.theta))
    st  = T(sin(m.theta))
    inv2 = inv(T(2))
    σx2_div2 = σx^2 * inv2
    σy2_div2 = σy^2 * inv2
    
    Ψ = Matrix{Complex{T}}(undef, Ny, Nx)
    
    @inbounds for ix in 1:Nx
        kx = T(_fftfreq(Nx, ix - 1)) * T(2π)
        for iy in 1:Ny
            ky = T(_fftfreq(Ny, iy - 1)) * T(2π)
            # Rotate
            kxr =  kx * ct + ky * st
            kyr = -kx * st + ky * ct
            if kxr < 0
                Ψ[iy, ix] = zero(Complex{T})
            else
                dkx = kxr - k0
                env = exp(-dkx^2 * σx2_div2 - kyr^2 * σy2_div2)
                cen = exp(-kxr^2 * σx2_div2 - kyr^2 * σy2_div2)
                Ψ[iy, ix] = Complex{T}(env - β * cen)
            end
        end
    end
    
    return Ψ
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
