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
using SpectralBackends: SpectralBackends as SB
using ..PathGraph: PathGraph
using ..Coefficients: Coefficients
using ..ScatteringCore: ScatteringCore
using ..Cascade: Cascade

export MonogenicFilterBank, ComputedMonogenicFilterBank, MonogenicScattering
export build_monogenic_bank, riesz_multipliers, monogenic_amplitude, monogenic_components

# Inline fftfreq for bin k (0-indexed), length N — normalized frequency in cycles/sample.
@inline _fftfreq(N::Int, k::Int) = k < (N + 1) ÷ 2 ? k / N : (k - N) / N

# Radial (isotropic) band-pass response over a D-dim frequency grid: a Gaussian bump on |k| at
# center ξ with width σ, minus a low-pass term pinning the DC bin to zero (admissibility).
_radial_bandpass(dims::NTuple{D,Int}, ξ::T, σ::T) where {D,T} =
    _radial_bandpass!(Array{T,D}(undef, dims), ξ, σ)

function _radial_bandpass!(ψ::AbstractArray{T,D}, ξ::T, σ::T) where {D,T}
    κ = exp(-(ξ / σ)^2 / 2)
    dims = size(ψ)
    @inbounds for I in CartesianIndices(dims)
        kk = zero(T)
        for d in 1:D
            f = T(_fftfreq(dims[d], I[d] - 1))
            kk += f * f
        end
        kn = sqrt(kk)
        g = exp(-((kn - ξ) / σ)^2 / 2)
        lp = exp(-(kn / σ)^2 / 2)
        ψ[I] = g - κ * lp
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

The wavelets and the low-pass are **real**: a radial band-pass is a real function of `|k|`. Only
the Riesz multipliers `R_d(k) = -i k_d/|k|` are complex.
"""
struct MonogenicFilterBank{D, T, A<:AbstractArray{T,D}, W<:AbstractVector{A},
                           R<:NTuple{D,AbstractArray{Complex{T},D}},
                           MV<:AbstractVector{FilterBanks.WaveletMeta{T}}}
    wavelets::W
    riesz::R
    averaging::A
    meta::MV
    J::Int
    Q::Int
end

"""
    build_monogenic_bank([T=Float64,] dims::NTuple{D,Int}, J; Q=1) -> MonogenicFilterBank

Build a `D`-dimensional isotropic Morlet-style monogenic filter bank: `J` octaves × `Q`
sub-octaves of radial band-pass wavelets (center frequency `ξ_j = ξ₀·2^{-j/Q}`, widths from the
Lostanlen/Kymatio rule, reusing [`Filters.Morlet1D`](@ref) for the radial profile), the Riesz
multipliers, and the tight-frame complementary low-pass.
"""
build_monogenic_bank(dims::NTuple{D,Int}, J::Int; kwargs...) where {D} =
    build_monogenic_bank(Float64, dims, J; kwargs...)

function build_monogenic_bank(::Type{T}, dims::NTuple{D,Int}, J::Int; Q::Int=1) where {T<:Real,D}
    A = Array{T,D}
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
    ComputedMonogenicFilterBank{D,T,A,R,MV}

The monogenic bank with its radial band-passes evaluated on demand instead of stored — the
counterpart of [`FilterBanks.ComputedFilterBank2D`](@ref). The Riesz multipliers stay stored: there
are `D` of them however many scales the bank has, so they do not grow with `J`.
"""
struct ComputedMonogenicFilterBank{D, T, A<:AbstractArray{T,D},
                                   R<:NTuple{D,AbstractArray{Complex{T},D}},
                                   MV<:AbstractVector{FilterBanks.WaveletMeta{T}},
                                   PV<:AbstractVector{NTuple{2,T}}}
    profiles::PV   # (ξ, σ) per wavelet — everything `_radial_bandpass` needs
    rescale::T
    riesz::R
    averaging::A
    scratch::A
    meta::MV
    J::Int
    Q::Int
end

const AnyMonogenicBank = Union{MonogenicFilterBank, ComputedMonogenicFilterBank}

FilterBanks.nwavelets(fb::MonogenicFilterBank) = length(fb.wavelets)
FilterBanks.nwavelets(fb::ComputedMonogenicFilterBank) = length(fb.profiles)
FilterBanks.filter_at(fb::MonogenicFilterBank, j::Integer) = fb.wavelets[j]
function FilterBanks.filter_at(fb::ComputedMonogenicFilterBank, j::Integer)
    ξ, σ = fb.profiles[j]
    _radial_bandpass!(fb.scratch, ξ, σ)
    fb.scratch .*= fb.rescale
    return fb.scratch
end
FilterBanks.iscomputed(::MonogenicFilterBank) = false
FilterBanks.iscomputed(::ComputedMonogenicFilterBank) = true
FilterBanks.task_bank(fb::MonogenicFilterBank) = fb
FilterBanks.task_bank(fb::ComputedMonogenicFilterBank) =
    ComputedMonogenicFilterBank(fb.profiles, fb.rescale, fb.riesz, fb.averaging,
                                similar(fb.scratch), fb.meta, fb.J, fb.Q)

build_monogenic_bank(::Type{T}, dims::NTuple{D,Int}, J::Int, ::Val{true};
                     Q::Int=1) where {T<:Real,D} = build_monogenic_bank(T, dims, J; Q=Q)

function build_monogenic_bank(::Type{T}, dims::NTuple{D,Int}, J::Int, ::Val{false};
                              Q::Int=1) where {T<:Real,D}
    profiles = NTuple{2,T}[]
    meta = FilterBanks.WaveletMeta{T}[]
    for j in 0:(J - 1), q in 0:(Q - 1)
        m = Filters.Morlet1D{T}(1, j * Q + q; Q=Q)
        push!(profiles, (m.center_freq, m.bandwidth))
        push!(meta, FilterBanks.WaveletMeta{T}(j, q, 0, T(j + q / Q), m.center_freq, zero(T)))
    end
    scratch = Array{T,D}(undef, dims)
    acc = zeros(T, dims)
    for (ξ, σ) in profiles
        _radial_bandpass!(scratch, ξ, σ)
        @. acc += abs2(scratch)
    end
    mx = maximum(acc)
    c = mx > zero(T) ? inv(sqrt(mx)) : one(T)
    ϕ = Array{T,D}(undef, dims)
    @. ϕ = sqrt(max(zero(T), one(T) - acc * c^2))
    ϕ[firstindex(ϕ)] = one(T)
    R = riesz_multipliers(dims, T)
    return ComputedMonogenicFilterBank(profiles, c, R, ϕ, scratch, meta, J, Q)
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
    MonogenicWorkspace{T,D}

The periodized cascade plus what the monogenic amplitude needs on top of it: one real accumulator
per resolution, and the Riesz-weighted band-passes `R_d ψ_j` periodized to each *parent* resolution.

`R_d ψ_j` is stored as one periodized filter rather than two, because periodizing the factors
separately is a different function from periodizing their product. Level 1 stores nothing: there
`r = 1`, so the two full-resolution filters fuse directly in the multiply
([`ScatteringCore.periodize_mul2!`](@ref)).
"""
struct MonogenicWorkspace{T, D, PW <: Cascade.PeriodizedWorkspace{T, D},
                          AV <: AbstractVector{<:AbstractArray{T, D}}, FV <: AbstractVector}
    pw::PW
    amp::AV        # per level: the amplitude accumulator at that resolution
    rfilters::FV   # rfilters[l][j][d]; empty at level 1 and wherever `j` is not applied at `l`
end

"""
    MonogenicScattering{D,T,FB,Tree,P,G,CB,MW}

Monogenic scattering transform on a `D`-dimensional grid: an isotropic [`MonogenicFilterBank`](@ref),
the scattering path tree (strictly-increasing scale), a spectral `plan`, and the periodized
[`MonogenicWorkspace`](@ref) the cascade runs in.
"""
struct MonogenicScattering{D, T, FB<:AnyMonogenicBank, Tree<:PathGraph.ScatteringTree,
                           P<:Plans.AbstractScatteringPlan, G<:AbstractVector,
                           CB<:AbstractArray{Complex{T},D}, MW<:MonogenicWorkspace{T,D}}
    filter_bank::FB
    tree::Tree
    groups::G               # (j1, children, path ids), longest-first — the cascade's work list
    max_order::Int
    plan::P
    dims::NTuple{D,Int}
    buffer_signal_fft::CB   # preserved signal FFT, at full resolution
    mw::MW
end

# `R_d ψ_j` periodized onto each parent resolution. Only the pairs the cascade actually applies are
# built, so the storage follows the same geometric decay the working buffers do.
function _monogenic_workspace(fb, groups, dims::NTuple{D, Int}, ::Type{T}, α::Int,
                              spectral) where {T, D}
    pw = Cascade.build(fb, groups, dims, T, α, spectral)
    nw = FilterBanks.nwavelets(fb)
    R = fb.riesz
    amp = [zeros(T, ntuple(d -> dims[d] ÷ r[d], D)) for r in pw.resolutions]
    empty_r = ntuple(_ -> zeros(Complex{T}, ntuple(_ -> 0, D)), D)
    scratch = Array{Complex{T}, D}(undef, dims)
    rfilters = map(eachindex(pw.resolutions)) do l
        v = fill(empty_r, nw)
        l == 1 && return v
        rl = pw.resolutions[l]
        rd = ntuple(d -> dims[d] ÷ rl[d], D)
        for j in 1:nw
            isempty(pw.filters[l][j]) && continue        # `j` is never applied at this level
            ψ = FilterBanks.filter_at(fb, j)
            v[j] = ntuple(D) do d
                @. scratch = R[d] * ψ
                pf = zeros(Complex{T}, rd)
                ScatteringCore.periodize_filter!(pf, scratch, rl)
                pf
            end
        end
        v
    end
    return MonogenicWorkspace{T, D, typeof(pw), typeof(amp), typeof(rfilters)}(pw, amp, rfilters)
end

"""
    MonogenicScattering([T=Float64,] dims, J; Q=1, max_order=2, spectral=AutoSpectralBackend())

Construct a monogenic scattering transform for fields of size `dims` (`Int` or `NTuple{D,Int}`).
The element type is positional, as for `zeros(T, …)`; omit it for `Float64`.
"""
function MonogenicScattering(::Type{T}, dims::NTuple{D,Int}, J::Int; Q::Int=1, max_order::Int=2,
                             cache::Bool=true, oversampling::Int=J,
                             spectral::SB.AbstractSpectralBackend=SB.AutoSpectralBackend()) where {T,D}
    fb = build_monogenic_bank(T, dims, J, Val(cache); Q=Q)
    tree = PathGraph.build_tree([m.j_eff for m in fb.meta], max_order)
    plan = Plans.make_plan(spectral, T, dims)
    nw = FilterBanks.nwavelets(fb)
    groups = max_order >= 2 ? PathGraph.order2_groups(tree, nw) :
             [(j, Int[], Int[]) for j in 1:nw]
    mw = _monogenic_workspace(fb, groups, dims, T, oversampling, spectral)
    return MonogenicScattering{D,T,typeof(fb),typeof(tree),typeof(plan),typeof(groups),
                               typeof(mw.pw.work[1]),typeof(mw)}(
        fb, tree, groups, max_order, plan, dims, zeros(Complex{T}, dims), mw)
end
MonogenicScattering(::Type{T}, N::Int, J::Int; kwargs...) where {T} =
    MonogenicScattering(T, (N,), J; kwargs...)
MonogenicScattering(dims::NTuple{D,Int}, J::Int; kwargs...) where {D} =
    MonogenicScattering(Float64, dims, J; kwargs...)
MonogenicScattering(N::Int, J::Int; kwargs...) = MonogenicScattering(Float64, (N,), J; kwargs...)

# `√(m₀² + Σ_d m_d²)` into `dst`, from the full-resolution spectrum `Xf` band-passed by wavelet `j`
# and produced on level `l`'s grid. Both filters are full resolution, so the Riesz weighting fuses
# into the multiply and nothing is materialised. Leaves `Xf` intact.
function _amp_from_full!(dst, mw::MonogenicWorkspace{T, D}, fb, Xf, j::Int, l::Int) where {T, D}
    pw = mw.pw
    r = pw.resolutions[l]
    R = fb.riesz
    w, o = pw.work[l], pw.out[l]
    ψ = Cascade._filter(pw, fb, 1, pw.filters[1], j)
    ScatteringCore.periodize_mul!(w, Xf, ψ, r)
    Plans.inverse_transform!(o, pw.plans[l], w)
    @. dst = real(o)^2
    for d in 1:D
        ScatteringCore.periodize_mul2!(w, Xf, R[d], ψ, r)
        Plans.inverse_transform!(o, pw.plans[l], w)
        @. dst += real(o)^2
    end
    @. dst = sqrt(dst)
    return dst
end

# The same, from a first-order amplitude's spectrum `u1f` living on level `l1`, produced on `l2`.
# Here the band-pass has already been periodized to `l1`, so the Riesz-weighted one must be the
# periodized *product* — except at `l1 == 1`, where the factors are still full resolution and fuse.
function _amp_from_parent!(dst, mw::MonogenicWorkspace{T, D}, fb, u1f, j::Int, l1::Int,
                           l2::Int) where {T, D}
    pw = mw.pw
    rr = ntuple(d -> pw.resolutions[l2][d] ÷ pw.resolutions[l1][d], D)
    w, o = pw.work[l2], pw.out[l2]
    ScatteringCore.periodize_mul!(w, u1f, Cascade._filter(pw, fb, l1, pw.filters[l1], j), rr)
    Plans.inverse_transform!(o, pw.plans[l2], w)
    @. dst = real(o)^2
    if l1 == 1
        R = fb.riesz
        ψ = Cascade._filter(pw, fb, 1, pw.filters[1], j)
        for d in 1:D
            ScatteringCore.periodize_mul2!(w, u1f, R[d], ψ, rr)
            Plans.inverse_transform!(o, pw.plans[l2], w)
            @. dst += real(o)^2
        end
    else
        rf = mw.rfilters[l1][j]
        for d in 1:D
            ScatteringCore.periodize_mul!(w, u1f, rf[d], rr)
            Plans.inverse_transform!(o, pw.plans[l2], w)
            @. dst += real(o)^2
        end
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
    mw = st.mw
    pw = mw.pw
    T = eltype(first(mw.amp))
    fb = st.filter_bank
    n = FilterBanks.nwavelets(fb)
    coeffs = Coefficients.ScatteringCoefficients1D(n, T; compute_S2=st.max_order >= 2)

    c0 = pw.work[1]
    @. c0 = complex(x)
    Plans.forward_transform!(st.buffer_signal_fft, st.plan, c0)
    Xf = st.buffer_signal_fft

    S1 = coeffs.S1
    S2 = coeffs.S2
    isempty(S2) || fill!(S2, zero(eltype(S2)))
    @inbounds for (j1, children, _) in st.groups
        l1 = pw.level[j1]
        a1 = mw.amp[l1]
        _amp_from_full!(a1, mw, fb, Xf, j1, l1)
        S1[j1] = ScatteringCore.spatial_average(a1)
        isempty(children) && continue
        # `a1` is copied out before any child can reuse its level's accumulator.
        w1 = pw.work[l1]
        @. w1 = complex(a1)
        u1f = pw.spec[l1]
        Plans.forward_transform!(u1f, pw.plans[l1], w1)
        for j2 in children
            l2 = pw.level[j2]
            a2 = mw.amp[l2]
            _amp_from_parent!(a2, mw, fb, u1f, j2, l1, l2)
            S2[j1, j2] = ScatteringCore.spatial_average(a2)
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
    ψ = FilterBanks.filter_at(fb, j)
    Xf = Plans.forward_transform(plan, complex.(x))
    bandpass = real.(Plans.inverse_transform(plan, Xf .* ψ))
    riesz = ntuple(d -> real.(Plans.inverse_transform(plan, Xf .* fb.riesz[d] .* ψ)), D)
    amplitude = monogenic_amplitude(bandpass, riesz)
    rnorm = sqrt.(sum(m -> m .^ 2, riesz))
    phase = atan.(rnorm, bandpass)
    return (; bandpass, riesz, amplitude, phase)
end

end # module Monogenic
