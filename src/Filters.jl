module Filters

"""
    Filters.jl — Frequency-domain wavelet filter definitions

Implements Morlet wavelets in the frequency domain for FFT-based convolutions.
"""

export Morlet1D, Morlet2D, Morlet3D
export frequency_response, frequency_response!
export gaussian_lowpass, gaussian_lowpass!
export fibonacci_directions

"""
    Morlet1D{T<:Real}

1D Morlet wavelet in frequency domain, over normalized frequency `ω ∈ [0, ½]`:

    Ψ(ω) = exp(-(ω-ξ)²/2σ²) - κ exp(-ω²/2σ²),   κ = exp(-(ξ/σ)²/2)

`σ` here is a **frequency-domain** width (unlike `Morlet2D`/`Morlet3D`, whose `σ` is a real-space
width), so the zero-mean term is `κ`, evaluated in [`frequency_response!`](@ref) — `Ψ(0) = 1·κ - κ·1`
after normalising, which is what makes the wavelet admissible. The filter is analytic: zero for
`ω < 0`.

# Type Parameters
- `T`: Element type (Float32, Float64, etc.)

# Fields
- `center_freq::T`: Center frequency ξ
- `bandwidth::T`: Frequency-domain width σ
- `N::Int`: Filter length (FFT size)
"""
struct Morlet1D{T<:Real}
    center_freq::T
    bandwidth::T
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

        new{T}(xi, sigma, N)
    end
end

# Convenience constructor - defaults to Float64
Morlet1D(N::Int, j::Real; kwargs...) = Morlet1D{Float64}(N, j; kwargs...)

# Inline fftfreq for a single bin k (0-indexed) of length N.
# Equivalent to FFTW.fftfreq(N)[k+1]. No allocation.
# For even N: bins 0..N÷2-1 are positive, bins N÷2..N-1 are negative (Nyquist goes negative).
# For odd N: bins 0..(N-1)÷2 are positive, rest negative.
@inline _fftfreq(N::Int, k::Int) = k < (N + 1) ÷ 2 ? k / N : (k - N) / N

"""
    frequency_response(m::Morlet1D{T}) -> Vector{T}

Compute the frequency response Ψ(ω) of a 1D Morlet wavelet.

Returns a length-N vector with the Fourier-domain filter coefficients.
The response is analytic (zero for negative frequencies) for proper
wavelet transform. Element type matches the wavelet's precision.
"""
frequency_response(m::Morlet1D{T}) where {T<:Real} =
    frequency_response!(Vector{T}(undef, m.N), m)

"""
    frequency_response!(Ψ, m) -> Ψ

Write the response into `Ψ` instead of allocating it. A bank that evaluates its filters on demand
rather than storing them needs exactly one array, and this is what lets it reuse that array.
"""
function frequency_response!(Ψ::AbstractVector{T}, m::Morlet1D{T}) where T<:Real
    N = m.N
    ξ = m.center_freq
    σ = m.bandwidth
    inv2 = inv(T(2))
    length(Ψ) == N || throw(DimensionMismatch("Ψ must have length $N"))
    
    # kappa: ratio at ω=0 (bin 0). gabor(0)=exp(-(xi/sigma)^2/2), lowpass(0)=1
    kappa = exp(-(ξ / σ)^2 * inv2)
    
    @inbounds for i in 1:N
        ω = T(_fftfreq(N, i - 1))
        if ω < 0
            Ψ[i] = zero(T)
        else
            g  = exp(-((ω - ξ) / σ)^2 * inv2)
            lp = exp(-(ω / σ)^2 * inv2)
            Ψ[i] = g - kappa * lp
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
    frequency_response(m::Morlet2D{T}) -> Matrix{T}

Compute the 2D frequency response Ψ(kx, ky) of an oriented Morlet wavelet.
Element type matches the wavelet's precision.
"""
frequency_response(m::Morlet2D{T}) where {T<:Real} =
    frequency_response!(Matrix{T}(undef, m.N), m)

function frequency_response!(Ψ::AbstractMatrix{T}, m::Morlet2D{T}) where T<:Real
    Ny, Nx = m.N
    size(Ψ) == m.N || throw(DimensionMismatch("Ψ must have size $(m.N)"))
    k0  = m.center_freq
    σx  = m.bandwidth_x
    σy  = m.bandwidth_y
    β   = m.beta
    ct  = T(cos(m.theta))
    st  = T(sin(m.theta))
    inv2 = inv(T(2))
    σx2_div2 = σx^2 * inv2
    σy2_div2 = σy^2 * inv2
    
    @inbounds for ix in 1:Nx
        kx = T(_fftfreq(Nx, ix - 1)) * T(2π)
        for iy in 1:Ny
            ky = T(_fftfreq(Ny, iy - 1)) * T(2π)
            # Rotate
            kxr =  kx * ct + ky * st
            kyr = -kx * st + ky * ct
            if kxr < 0
                Ψ[iy, ix] = zero(T)
            else
                dkx = kxr - k0
                env = exp(-dkx^2 * σx2_div2 - kyr^2 * σy2_div2)
                cen = exp(-kxr^2 * σx2_div2 - kyr^2 * σy2_div2)
                Ψ[iy, ix] = env - β * cen
            end
        end
    end
    
    return Ψ
end

"""
    Morlet3D{T<:Real}

3D oriented Morlet wavelet in the frequency domain, a bump centered at `k₀ n̂` for a unit
direction `n̂` on the sphere, with an anisotropic Gaussian envelope (std `σ∥` along `n̂`,
`σ⊥ = σ∥/elongation` perpendicular) and analytic on the half-space `k·n̂ ≥ 0`.

# Fields
- `center_freq::T`: `|k₀|`
- `sigma_par::T`, `sigma_perp::T`: real-space envelope widths along / perpendicular to `n̂`
- `direction::NTuple{3,T}`: unit orientation `n̂`
- `beta::T`: zero-mean correction
- `N::NTuple{3,Int}`: grid dimensions
"""
struct Morlet3D{T<:Real}
    center_freq::T
    sigma_par::T
    sigma_perp::T
    direction::NTuple{3,T}
    beta::T
    N::NTuple{3,Int}

    function Morlet3D{T}(N::NTuple{3,Int}, j::Int, direction::NTuple{3,Real};
                         sigma0::T=T(0.8), elongation::T=T(4.0)) where T<:Real
        scale = T(2.0)^j
        sigma_par = sigma0 * scale
        sigma_perp = sigma_par / elongation
        k0 = T(3π) / (T(4) * scale)
        # normalize the direction
        nx, ny, nz = T(direction[1]), T(direction[2]), T(direction[3])
        nrm = Base.sqrt(nx^2 + ny^2 + nz^2)
        n̂ = (nx / nrm, ny / nrm, nz / nrm)
        β = exp(-(sigma_par * k0)^2 / T(2))
        new{T}(k0, sigma_par, sigma_perp, n̂, β, N)
    end
end

Morlet3D(N::NTuple{3,Int}, j::Int, direction; kwargs...) =
    Morlet3D{Float64}(N, j, direction; kwargs...)

"""
    frequency_response(m::Morlet3D{T}) -> Array{T,3}

3D frequency response `Ψ(kx,ky,kz)` of an oriented Morlet wavelet.
"""
frequency_response(m::Morlet3D{T}) where {T<:Real} =
    frequency_response!(Array{T,3}(undef, m.N), m)

function frequency_response!(Ψ::AbstractArray{T,3}, m::Morlet3D{T}) where T<:Real
    Nz, Ny, Nx = m.N
    k0 = m.center_freq
    σpar2 = m.sigma_par^2
    σperp2 = m.sigma_perp^2
    nx, ny, nz = m.direction
    β = m.beta
    inv2 = inv(T(2))

    size(Ψ) == m.N || throw(DimensionMismatch("Ψ must have size $(m.N)"))
    @inbounds for ix in 1:Nx
        kx = T(_fftfreq(Nx, ix - 1)) * T(2π)
        for iy in 1:Ny
            ky = T(_fftfreq(Ny, iy - 1)) * T(2π)
            for iz in 1:Nz
                kz = T(_fftfreq(Nz, iz - 1)) * T(2π)
                kpar = kx * nx + ky * ny + kz * nz
                if kpar < 0
                    Ψ[iz, iy, ix] = zero(T)
                else
                    kperp2 = kx^2 + ky^2 + kz^2 - kpar^2
                    env = exp(-((kpar - k0)^2 * σpar2 + kperp2 * σperp2) * inv2)
                    cen = exp(-(kpar^2 * σpar2 + kperp2 * σperp2) * inv2)
                    Ψ[iz, iy, ix] = env - β * cen
                end
            end
        end
    end
    return Ψ
end

"""
    gaussian_lowpass!(φ, σ) -> φ
    gaussian_lowpass(T, dims, J; sigma0=0.8) -> Array{T,D}

Isotropic Gaussian low-pass `φ̂(k) = exp(-|k|²σ²/2)` on the FFT grid of `dims`, with `k` the angular
frequency (`2π·fftfreq` per axis) and `σ = sigma0·2^J` a real-space width. The scaling function of
the localized (Mallat) transform `S_p = (U_p ⋆ φ_J) ↓ s`.

Two properties are load-bearing. `φ̂(0) = 1` makes the convolution preserve a field's mean; and `φ̂`
must vanish on the subsampling lattice `{m·N/s, m ≠ 0}`, since that is exactly the condition for
`⟨S_p⟩` to survive decimation and still equal the scattering coefficient `⟨U_p⟩`. The lattice starts
at `|k| = 2π/s`, so at the default `s = 2^(J-1)` the largest surviving term is `exp(-(4π·sigma0)²/2)`
— `1e-22` at `sigma0 = 0.8`. That condition is the requirement; the Gaussian family and `sigma0` are
a choice with wide margin.

Distinct from a filter bank's `averaging`, the tight-frame complement `√(max(0, 1-Σ|ψ|²))`: every `ψ̂`
here is analytic, so that one is identically `1` across the analytic complement and fails the lattice
condition outright.
"""
function gaussian_lowpass!(φ::AbstractArray{T, D}, σ::Real) where {T <: Real, D}
    n = size(φ)
    h = T(σ)^2 / 2
    @inbounds for I in CartesianIndices(φ)
        s = zero(T)
        for d in 1:D
            k = T(2π) * T(_fftfreq(n[d], I[d] - 1))
            s += k * k
        end
        φ[I] = exp(-s * h)
    end
    return φ
end

gaussian_lowpass(::Type{T}, dims::NTuple{D, Int}, J::Integer; sigma0::Real = 0.8) where {T <: Real, D} =
    gaussian_lowpass!(Array{T, D}(undef, dims), T(sigma0) * T(2)^J)

"""
    fibonacci_directions(n, ::Type{T}=Float64) -> Vector{NTuple{3,T}}

`n` near-uniform unit directions on the sphere (Fibonacci spiral), used as 3D wavelet
orientations.
"""
function fibonacci_directions(n::Int, ::Type{T}=Float64) where {T<:Real}
    ϕ = T(π) * (T(3) - Base.sqrt(T(5)))   # golden angle
    dirs = Vector{NTuple{3,T}}(undef, n)
    @inbounds for i in 0:(n - 1)
        z = T(1) - T(2) * (T(i) + T(0.5)) / T(n)
        r = Base.sqrt(max(zero(T), T(1) - z^2))
        θ = ϕ * T(i)
        dirs[i + 1] = (r * Base.cos(θ), r * Base.sin(θ), z)
    end
    return dirs
end

end # module Filters
