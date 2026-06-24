module Monogenic

"""
    Monogenic.jl — Monogenic (Riesz) wavelet scattering for 1D / 2D / 3D

The monogenic signal is the natural higher-dimensional generalization of the analytic signal: it
pairs an **isotropic** band-pass field with its **Riesz transform** components and recovers a
local *amplitude*, *phase*, and *orientation* at every point (Felsberg & Sommer 2001; Unser et
al. 2009). Where the oriented-Morlet scattering transform fixes a discrete set of orientations
and takes the analytic modulus, monogenic scattering uses a single rotation-covariant
nonlinearity — the **monogenic amplitude** — so orientation is recovered continuously rather than
quantized.

For a real field `x`, an isotropic band-pass `ψ_j` (radial in frequency, real and zero-mean), and
the Riesz multipliers `R_d(k) = -i k_d/|k|` (`d = 1…D`):

- band-pass component   `m₀ = x ⋆ ψ_j`               (real)
- Riesz components      `m_d = x ⋆ (R_d ψ_j)`         (real, `d = 1…D`)
- **monogenic amplitude** `A_j = √(m₀² + Σ_d m_d²)`   — the rotation-covariant envelope.

`A_j` plays the role the analytic modulus `|x ⋆ ψ_λ|` plays in ordinary scattering, and the
transform cascades it exactly as before (path graph = strictly increasing scale). The isotropic
bank is a tight frame (`Σ_j |ψ̂_j|² + |φ̂|² ≡ 1`), and the Riesz multipliers satisfy the
partition `Σ_d |R_d(k)|² = 1` off-DC, so no frequency is amplified.
"""

using ..Filters: Filters
using ..FilterBanks: FilterBanks
using ..Plans: Plans
using ..PathGraph: PathGraph
using ..Coefficients: Coefficients
using ..ScatteringCore: ScatteringCore

export MonogenicFilterBank, MonogenicScattering
export build_monogenic_bank, riesz_multipliers, monogenic_amplitude, monogenic_components

# Inline fftfreq for bin k (0-indexed), length N — normalized frequency in cycles/sample.
@inline _fftfreq(N::Int, k::Int) = k < (N + 1) ÷ 2 ? k / N : (k - N) / N

# Radial (isotropic) band-pass response over a D-dim frequency grid: a Gaussian bump on |k| at
# center ξ with width σ, minus a low-pass term pinning the DC bin to zero (admissibility).
function _radial_bandpass(dims::NTuple{D,Int}, ξ::T, σ::T) where {D,T}
    κ = exp(-(ξ / σ)^2 / 2)
    ψ = Array{Complex{T},D}(undef, dims)
    @inbounds for I in CartesianIndices(dims)
        kk = zero(T)
        for d in 1:D
            f = T(_fftfreq(dims[d], I[d] - 1))
            kk += f * f
        end
        kn = sqrt(kk)
        g = exp(-((kn - ξ) / σ)^2 / 2)
        lp = exp(-(kn / σ)^2 / 2)
        ψ[I] = Complex{T}(g - κ * lp)
    end
    return ψ
end

"""
    riesz_multipliers(dims, ::Type{T}=Float64) -> NTuple{D, Array{Complex{T},D}}

The `D` Riesz-transform frequency multipliers `R_d(k) = -i k_d/|k|` (with `R_d(0)=0`) over a grid
of size `dims`. Scale-free (a ratio of frequencies), so one set serves every wavelet scale. They
satisfy `Σ_d |R_d(k)|² = 1` off the DC bin.
"""
function riesz_multipliers(dims::NTuple{D,Int}, ::Type{T}=Float64) where {D,T}
    R = ntuple(_ -> Array{Complex{T},D}(undef, dims), D)
    @inbounds for I in CartesianIndices(dims)
        kk = zero(T)
        ks = ntuple(d -> T(_fftfreq(dims[d], I[d] - 1)), D)
        for d in 1:D
            kk += ks[d]^2
        end
        kn = sqrt(kk)
        for d in 1:D
            R[d][I] = kn == zero(T) ? zero(Complex{T}) : Complex{T}(zero(T), -ks[d] / kn)
        end
    end
    return R
end

"""
    MonogenicFilterBank{D,T,A,W,R,MV}

Isotropic band-pass wavelets `wavelets` (one per scale/sub-octave), the `D` scale-free Riesz
multipliers `riesz`, and the complementary low-pass `averaging`, forming a tight frame. Every
container is a type parameter.
"""
struct MonogenicFilterBank{D, T, A<:AbstractArray{Complex{T},D}, W<:AbstractVector{A},
                           R<:NTuple{D,A}, MV<:AbstractVector{FilterBanks.WaveletMeta{T}}}
    wavelets::W
    riesz::R
    averaging::A
    meta::MV
    J::Int
    Q::Int
end

"""
    build_monogenic_bank(dims::NTuple{D,Int}, J; Q=1, T=Float64) -> MonogenicFilterBank

Build a `D`-dimensional isotropic Morlet-style monogenic filter bank: `J` octaves × `Q`
sub-octaves of radial band-pass wavelets (center frequency `ξ_j = ξ₀·2^{-j/Q}`, widths from the
Lostanlen/Kymatio rule, reusing [`Filters.Morlet1D`](@ref) for the radial profile), the Riesz
multipliers, and the tight-frame complementary low-pass.
"""
function build_monogenic_bank(dims::NTuple{D,Int}, J::Int; Q::Int=1,
                              T::Type{<:Real}=Float64) where {D}
    A = Array{Complex{T},D}
    wavelets = Vector{A}(undef, 0)
    meta = Vector{FilterBanks.WaveletMeta{T}}(undef, 0)
    for j in 0:(J - 1)
        for q in 0:(Q - 1)
            # Reuse the 1D Morlet radial profile (normalized-frequency ξ, σ).
            m = Filters.Morlet1D{T}(1, j * Q + q; Q=Q)
            push!(wavelets, _radial_bandpass(dims, m.center_freq, m.bandwidth))
            push!(meta, FilterBanks.WaveletMeta{T}(j, q, 0, T(j + q / Q), m.center_freq, zero(T)))
        end
    end
    ϕ = FilterBanks._tight_frame_lowpass!(wavelets)
    R = riesz_multipliers(dims, T)
    return MonogenicFilterBank{D,T,A,typeof(wavelets),typeof(R),typeof(meta)}(
        wavelets, R, ϕ, meta, J, Q)
end

"""
    monogenic_amplitude(m0, riesz_components) -> A

Monogenic amplitude `A = √(m₀² + Σ_d m_d²)` from the band-pass field `m0` and the tuple/vector of
Riesz component fields. Broadcast, so it is CPU/GPU/autodiff-generic.
"""
function monogenic_amplitude(m0::AbstractArray, riesz)
    acc = m0 .^ 2
    for m in riesz
        acc = acc .+ m .^ 2
    end
    return sqrt.(acc)
end

"""
    MonogenicScattering{D,T,FB,Tree,P,V}

Monogenic scattering transform on a `D`-dimensional grid: an isotropic [`MonogenicFilterBank`](@ref),
the scattering path tree (strictly-increasing scale), a spectral `plan`, and reusable workspace
buffers.
"""
struct MonogenicScattering{D, T, FB<:MonogenicFilterBank, Tree<:PathGraph.ScatteringTree,
                           P<:Plans.AbstractScatteringPlan,
                           CB<:AbstractArray{Complex{T},D}, RB<:AbstractArray{T,D}, UF<:AbstractVector}
    filter_bank::FB
    tree::Tree
    max_order::Int
    plan::P
    dims::NTuple{D,Int}
    buffer_input::CB        # complex promotion of input
    buffer_signal_fft::CB   # preserved signal FFT
    buffer_freq::CB         # pointwise-multiply scratch (inside the amplitude kernel)
    buffer_conv::CB         # inverse-transform output
    amp::RB                 # monogenic-amplitude accumulator
    U1_fft::UF              # per-scale FFT of the first-order amplitude field (S2)
end

"""
    MonogenicScattering(dims, J; Q=1, max_order=2, T=Float64, spectral=AutoSpectral())

Construct a monogenic scattering transform for fields of size `dims` (`Int` or `NTuple{D,Int}`).
"""
function MonogenicScattering(dims::NTuple{D,Int}, J::Int; Q::Int=1, max_order::Int=2,
                             T::Type=Float64,
                             spectral::Plans.AbstractSpectralBackend=Plans.AutoSpectral()) where {D}
    fb = build_monogenic_bank(dims, J; Q=Q, T=T)
    tree = PathGraph.build_tree([m.j_eff for m in fb.meta], max_order)
    plan = Plans.make_plan(spectral, T, dims)
    cb() = zeros(Complex{T}, dims)
    nw = length(fb.wavelets)
    U1_fft = max_order >= 2 ? [cb() for _ in 1:nw] : Array{Complex{T},D}[]
    return MonogenicScattering{D,T,typeof(fb),typeof(tree),typeof(plan),
                               typeof(cb()),typeof(zeros(T, dims)),typeof(U1_fft)}(
        fb, tree, max_order, plan, dims, cb(), cb(), cb(), cb(), zeros(T, dims), U1_fft)
end
MonogenicScattering(N::Int, J::Int; kwargs...) = MonogenicScattering((N,), J; kwargs...)

# In-place monogenic amplitude of the field whose FFT is `Xf`, band-passed by wavelet `ψ`.
# Writes √(m₀² + Σ_d m_d²) into `dst`. Reuses buffer_freq/buffer_conv; leaves Xf intact.
function _monogenic_amplitude!(dst, st::MonogenicScattering{D}, Xf, ψ) where {D}
    plan = st.plan
    R = st.filter_bank.riesz
    @. st.buffer_freq = Xf * ψ
    Plans.inverse_transform!(st.buffer_conv, plan, st.buffer_freq)
    @. dst = real(st.buffer_conv)^2
    for d in 1:D
        @. st.buffer_freq = Xf * R[d] * ψ
        Plans.inverse_transform!(st.buffer_conv, plan, st.buffer_freq)
        @. dst += real(st.buffer_conv)^2
    end
    @. dst = sqrt(dst)
    return dst
end

"""
    (st::MonogenicScattering)(x) -> ScatteringCoefficients1D

Apply the monogenic scattering transform; returns averaged coefficients in a
`ScatteringCoefficients1D` (`S0` mean, `S1` over scales, `S2` over scale pairs).
"""
function (st::MonogenicScattering{D})(x::AbstractArray) where {D}
    T = eltype(st.amp)
    n = length(st.filter_bank.wavelets)
    coeffs = Coefficients.ScatteringCoefficients1D(n, T; compute_S2=st.max_order >= 2)

    @. st.buffer_input = complex(x)
    Plans.forward_transform!(st.buffer_signal_fft, st.plan, st.buffer_input)
    Xf = st.buffer_signal_fft

    nw = length(st.filter_bank.wavelets)
    S1 = coeffs.S1
    @inbounds for (j, ψ) in enumerate(st.filter_bank.wavelets)
        _monogenic_amplitude!(st.amp, st, Xf, ψ)
        S1[j] = ScatteringCore.spatial_average(st.amp)
        # For S2: cache the FFT of each first-order amplitude field (re-analysis input).
        if st.max_order >= 2
            @. st.buffer_input = complex(st.amp)
            Plans.forward_transform!(st.U1_fft[j], st.plan, st.buffer_input)
        end
    end

    if st.max_order >= 2 && length(st.tree.by_order) >= 3
        S2 = coeffs.S2
        @inbounds for p in PathGraph.order_range(st.tree, 2)
            idx = PathGraph.path_indices(st.tree, p)
            j1, j2 = idx[1], idx[2]
            # Re-analyze the cached order-1 amplitude (FFT) at the coarser scale j2.
            _monogenic_amplitude!(st.amp, st, st.U1_fft[j1], st.filter_bank.wavelets[j2])
            S2[j1, j2] = ScatteringCore.spatial_average(st.amp)
        end
    end

    S0 = ScatteringCore.spatial_average(x)
    return Coefficients.update_S0(coeffs, S0)
end

"""
    monogenic_components(st, x, j) -> (; bandpass, riesz, amplitude, phase)

The monogenic decomposition of `x` band-passed by isotropic wavelet `j` (1-based): the band-pass
field `bandpass = x ⋆ ψ_j`, the `D` Riesz component fields `riesz`, the monogenic `amplitude`
`√(bandpass² + Σ|riesz|²)`, and the local monogenic `phase = atan(‖riesz‖, bandpass)`. The Riesz
vector's direction gives the local orientation (e.g. `atan(riesz[2], riesz[1])` in 2D).
"""
function monogenic_components(st::MonogenicScattering{D}, x::AbstractArray, j::Integer) where {D}
    plan = st.plan
    fb = st.filter_bank
    ψ = fb.wavelets[j]
    Xf = Plans.forward_transform(plan, complex.(x))
    bandpass = real.(Plans.inverse_transform(plan, Xf .* ψ))
    riesz = ntuple(d -> real.(Plans.inverse_transform(plan, Xf .* fb.riesz[d] .* ψ)), D)
    amplitude = monogenic_amplitude(bandpass, riesz)
    rnorm = sqrt.(sum(m -> m .^ 2, riesz))
    phase = atan.(rnorm, bandpass)
    return (; bandpass, riesz, amplitude, phase)
end

end # module Monogenic
