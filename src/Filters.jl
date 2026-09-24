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

`σ` is a frequency-domain width. `κ` sets `Ψ(0) = 0`, so the wavelet has zero mean. The filter is
analytic: zero for `ω < 0`.

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

        # Bandwidth: sigma = xi * (1 - 2^(-1/Q)) / (1 + 2^(-1/Q)) / sqrt(2*log(1/r)), which puts the
        # crossing of adjacent wavelets at `r` of their peak.
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

# Whether bin `k` (0-indexed) of an `N`-point axis is the Nyquist bin, which stands for both ±½.
@inline _isnyquist(N::Int, k::Int) = iseven(N) && k == N ÷ 2

"""
    _alias_rms(f, n, I) -> value

`f` at the normalized frequencies of bin `I` of an `n`-point grid, `f` taking an `NTuple` of them.
On an even axis's Nyquist bin `−½` and `+½` are one bin, so there the value is the root mean square
of `f` over every sign of those coordinates. A real field's Littlewood–Paley sum at that bin is then
the mean of the continuous sum over its aliases: an analytic wavelet centred at `+½` keeps the bin.
"""
function _alias_rms(f::F, n::NTuple{D,Int}, I::CartesianIndex{D}) where {F,D}
    fr = ntuple(d -> _fftfreq(n[d], I[d] - 1), Val(D))
    nyq = ntuple(d -> _isnyquist(n[d], I[d] - 1), Val(D))
    v = f(fr)
    any(nyq) || return v
    acc = v * v
    cnt = 1
    for m in 1:(2^D - 1)
        all(d -> (m >> (d - 1)) & 1 == 0 || nyq[d], 1:D) || continue
        w = f(ntuple(d -> (m >> (d - 1)) & 1 == 1 ? -fr[d] : fr[d], Val(D)))
        acc += w * w
        cnt += 1
    end
    return sqrt(acc / cnt)
end

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

Write the response into `Ψ`.
"""
function frequency_response!(Ψ::AbstractVector{T}, m::Morlet1D{T}) where T<:Real
    N = m.N
    ξ = m.center_freq
    σ = m.bandwidth
    inv2 = inv(T(2))
    length(Ψ) == N || throw(DimensionMismatch("Ψ must have length $N"))
    
    # kappa: ratio at ω=0 (bin 0). gabor(0)=exp(-(xi/sigma)^2/2), lowpass(0)=1
    kappa = exp(-(ξ / σ)^2 * inv2)
    resp(f) = (ω = T(f[1]); ω < 0 ? zero(T) :
               exp(-((ω - ξ) / σ)^2 * inv2) - kappa * exp(-(ω / σ)^2 * inv2))
    @inbounds for I in CartesianIndices(Ψ)
        Ψ[I] = _alias_rms(resp, (N,), I)
    end
    return Ψ
end

"""
    Morlet2D{T<:Real}

2D oriented Morlet wavelet in the frequency domain, `k` the angular wavenumber:

    Ψ(k) = exp(-((k∥ - k₀)²σ∥² + k⊥²σ⊥²)/2) - β exp(-(k∥²σ∥² + k⊥²σ⊥²)/2),   β = exp(-(k₀σ∥)²/2)

for `k∥ ≥ 0` and zero otherwise, with `k∥ = kx cos θ + ky sin θ`, `k⊥ = -kx sin θ + ky cos θ`. The
widths are real-space: `σ∥ = sigma0·2ʲ`, `σ⊥ = σ∥·L/4`, and `k₀ = 3π/(4·2ʲ)`. The envelope's angular
standard deviation at `|k| = k₀` is then `4/(L·k₀σ∥)`, which at the default `sigma0` is `0.68·π/L`: a
fixed fraction of the spacing `π/L` of the `L` orientations.

# Fields
- `center_freq::T`: `k₀`
- `sigma_par::T`, `sigma_perp::T`: real-space widths along and across the orientation
- `theta::T`: orientation angle in radians, from the second array axis
- `beta::T`: zero-mean correction
- `N::NTuple{2,Int}`: filter dimensions `(Ny, Nx)`
"""
struct Morlet2D{T<:Real}
    center_freq::T
    sigma_par::T
    sigma_perp::T
    theta::T
    beta::T
    N::NTuple{2,Int}

    function Morlet2D{T}(N::NTuple{2,Int}, j::Int, theta::Real;
                         L::Int=8, sigma0::T=T(0.8)) where T<:Real
        L >= 1 || throw(ArgumentError("L must be at least 1; got $L"))
        scale = T(2)^j
        σpar = sigma0 * scale
        σperp = σpar * T(L) / 4
        k0 = T(3π) / (T(4) * scale)
        β = exp(-(σpar * k0)^2 / T(2))
        new{T}(k0, σpar, σperp, T(theta), β, N)
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
    size(Ψ) == m.N || throw(DimensionMismatch("Ψ must have size $(m.N)"))
    k0  = m.center_freq
    σx  = m.sigma_par
    σy  = m.sigma_perp
    β   = m.beta
    ct  = T(cos(m.theta))
    st  = T(sin(m.theta))
    inv2 = inv(T(2))
    σx2_div2 = σx^2 * inv2
    σy2_div2 = σy^2 * inv2
    # `f = (fy, fx)`, the grid's axis order.
    function resp(f)
        kx = T(f[2]) * T(2π)
        ky = T(f[1]) * T(2π)
        kxr =  kx * ct + ky * st
        kyr = -kx * st + ky * ct
        kxr < 0 && return zero(T)
        return exp(-(kxr - k0)^2 * σx2_div2 - kyr^2 * σy2_div2) -
               β * exp(-kxr^2 * σx2_div2 - kyr^2 * σy2_div2)
    end
    @inbounds for I in CartesianIndices(Ψ)
        Ψ[I] = _alias_rms(resp, m.N, I)
    end
    return Ψ
end

"""
    Morlet3D{T<:Real}

3D oriented Morlet wavelet in the frequency domain, a bump centered at `k₀ n̂` for a unit
direction `n̂`, analytic on the half-space `k·n̂ ≥ 0`: [`Morlet2D`](@ref)'s envelope with `k∥ = k·n̂`
and `k⊥ = |k − k∥ n̂|`. The real-space widths are `σ∥ = sigma0·2ʲ` and `σ⊥ = σ∥·√(π·n_orient)/8`,
the 2D rule `σ⊥ = σ∥·π/(4Δθ)` at the spacing `Δθ = √(4π/n_orient)` of `n_orient` directions spread
over the sphere.

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
                         n_orient::Int=6, sigma0::T=T(0.8)) where T<:Real
        n_orient >= 1 || throw(ArgumentError("n_orient must be at least 1; got $n_orient"))
        scale = T(2)^j
        sigma_par = sigma0 * scale
        sigma_perp = sigma_par * sqrt(T(π) * n_orient) / 8
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
    k0 = m.center_freq
    σpar2 = m.sigma_par^2
    σperp2 = m.sigma_perp^2
    nx, ny, nz = m.direction
    β = m.beta
    inv2 = inv(T(2))

    size(Ψ) == m.N || throw(DimensionMismatch("Ψ must have size $(m.N)"))
    # `f = (fz, fy, fx)`, the grid's axis order.
    function resp(f)
        kx = T(f[3]) * T(2π)
        ky = T(f[2]) * T(2π)
        kz = T(f[1]) * T(2π)
        kpar = kx * nx + ky * ny + kz * nz
        kpar < 0 && return zero(T)
        kperp2 = kx^2 + ky^2 + kz^2 - kpar^2
        return exp(-((kpar - k0)^2 * σpar2 + kperp2 * σperp2) * inv2) -
               β * exp(-(kpar^2 * σpar2 + kperp2 * σperp2) * inv2)
    end
    @inbounds for I in CartesianIndices(Ψ)
        Ψ[I] = _alias_rms(resp, m.N, I)
    end
    return Ψ
end

"""
    gaussian_lowpass!(φ, σ) -> φ
    gaussian_lowpass(T, dims, J; sigma0=0.8) -> Array{T,D}

Isotropic Gaussian low-pass `φ̂(k) = exp(-|k|²σ²/2)` on the FFT grid of `dims`, with `k` the angular
frequency (`2π·fftfreq` per axis) and `σ = sigma0·2^J` a real-space width: the averaging filter
`φ_J` of every filter bank, and of the localized transform `S_p = (U_p ⋆ φ_J) ↓ s`.

`φ̂(0) = 1`, so averaging preserves a field's mean. Decimation by `s` keeps `⟨S_p⟩ = ⟨U_p⟩` when `φ̂`
vanishes on the lattice `{2πm/s : m ≠ 0}`; at `s = 2^(J-1)` its largest value there is
`exp(-(4π·sigma0)²/2)`, `1e-22` at `sigma0 = 0.8`.
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
