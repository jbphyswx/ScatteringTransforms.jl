module ScatteringTransformsNUFSHTExt

"""
    ScatteringTransformsNUFSHTExt — spherical scattering on S² (scattered points)

Scattering transform for scalar fields sampled at scattered points on the sphere, via NUFSHT.

The previous implementation used a **brick-wall** band-pass in ℓ (zeroing modes outside
`[ℓ_lo, ℓ_hi]`), which rings in real space and is not a wavelet. This version uses NUFSHT's
**smooth Gaussian spectral transfer** `GaussianTransfer(σ²) = exp(-σ²ℓ(ℓ+1)/2)` (a heat-kernel
low-pass) and forms band-pass wavelets as **differences of Gaussians** across dyadic scales —
smooth, localized needlet-like bands. `nusht_filter!` applies a transfer as
points → (bandlimited SH) → points in one call, so no manual mode bookkeeping is needed.

Coefficients (globally averaged, matching the gridded transforms):
- `S0 = ⟨field⟩`
- `S1[j] = ⟨|field ⋆ ψ_j|⟩`
- `S2[j1,j2] = ⟨||field ⋆ ψ_{j1}| ⋆ ψ_{j2}|⟩` for strictly coarser `j2 < j1`

with `ψ_j` the difference-of-Gaussians band-pass between dyadic cutoffs `ℓ_{j-1} < ℓ_j`.
"""

using NUFSHT: NUFSHT
using ScatteringTransforms: ScatteringTransforms

const ST = ScatteringTransforms

"""
    SphericalScattering{T,P}

Spherical scattering transform. `sigma2[k+1]` is the Gaussian-transfer variance for the dyadic
low-pass cutoff `ℓ_k = lmax / 2^(J-k)`, `k = 0..J`; band-pass wavelet `j` is
`lowpass(ℓ_j) − lowpass(ℓ_{j-1})`.
"""
struct SphericalScattering{T,P}
    lmax::Int
    J::Int
    max_order::Int
    plan::P
    sigma2::Vector{T}   # length J+1, σ² for cutoffs ℓ_0 … ℓ_J
    M::Int              # number of sample points
end

# σ² so the Gaussian transfer is e^{-1/2} at degree ℓ (i.e. ℓ marks the low-pass roll-off).
_sigma2_for_cutoff(ℓ::Real, ::Type{T}) where {T} = ℓ <= 0 ? T(Inf) : T(1) / (T(ℓ) * (T(ℓ) + 1))

function ST.spherical_scattering(pts_theta::AbstractVector{T}, pts_phi::AbstractVector{T},
                                 lmax::Int, J::Int; max_order::Int = 2) where {T<:Real}
    plan = NUFSHT.make_plan(pts_theta, pts_phi, lmax; T = T)
    # dyadic cutoffs ℓ_k = lmax / 2^(J-k), k = 0..J  (ℓ_J = lmax)
    sigma2 = T[_sigma2_for_cutoff(lmax / 2.0^(J - k), T) for k in 0:J]
    return SphericalScattering{T, typeof(plan)}(lmax, J, max_order, plan, sigma2, length(pts_theta))
end

# Smooth band-pass wavelet j applied to `field`: lowpass(ℓ_j) − lowpass(ℓ_{j-1}), into `band`.
# Uses `hi`, `lo` as scratch (all length M).
function _bandpass!(band::AbstractVector{T}, field::AbstractVector{T}, j::Int,
                    st::SphericalScattering{T}, hi::AbstractVector{T}, lo::AbstractVector{T}) where {T}
    NUFSHT.nusht_filter!(hi, field, NUFSHT.GaussianTransfer(st.sigma2[j + 1]), st.plan)  # lowpass ℓ_j
    NUFSHT.nusht_filter!(lo, field, NUFSHT.GaussianTransfer(st.sigma2[j]),     st.plan)  # lowpass ℓ_{j-1}
    @. band = hi - lo
    return band
end

"""
    (st::SphericalScattering)(field) -> (; S0, S1, S2)

Apply the spherical scattering transform to a scalar `field` sampled at the plan's points.
"""
function (st::SphericalScattering{T})(field::AbstractVector{T}) where {T}
    J, M = st.J, st.M
    S0 = sum(field) / T(M)
    S1 = zeros(T, J)
    S2 = zeros(T, J, J)

    hi = Vector{T}(undef, M)
    lo = Vector{T}(undef, M)
    band = Vector{T}(undef, M)
    U1 = [Vector{T}(undef, M) for _ in 1:J]

    # first order
    for j in 1:J
        _bandpass!(band, field, j, st, hi, lo)
        @. U1[j] = abs(band)
        S1[j] = sum(U1[j]) / T(M)
    end

    # second order over strictly coarser scales (j2 < j1)
    if st.max_order >= 2
        for j1 in 1:J, j2 in 1:(j1 - 1)
            _bandpass!(band, U1[j1], j2, st, hi, lo)
            @. band = abs(band)
            S2[j1, j2] = sum(band) / T(M)
        end
    end

    return (S0 = S0, S1 = S1, S2 = S2)
end

# ─────────────────────────────────────────────────────────────────────────────
# Spherical MONOGENIC scattering (Part E)
#
# Monogenic amplitude per scale:  A_j = √(U⁰_j² + |U^R_j|²),  where
#   U⁰_j   = difference-of-Gaussians band-pass of the field   (spin-0, scalar)
#   U^R_j  = spin-1 Riesz field  R = ð∘(−Δ_S)^{-1/2}  applied to U⁰_j.
# The Riesz operator is SH-diagonal with eigenvalue r_ℓ = 1 (ℓ≥1; 0 at ℓ=0), since
# ð contributes √(ℓ(ℓ+1)) and (−Δ_S)^{-1/2} contributes 1/√(ℓ(ℓ+1)).
#
# KEY: the Riesz *energy* needs only spin-0 transforms. With g_j = (−Δ_S)^{-1/2} U⁰_j,
#   |U^R_j|² = |ð g_j|² = |∇_S g_j|²,
# and the surface-gradient magnitude follows from the Bochner/product identity
#   |∇_S g|² = ½ Δ_S(g²) − g · Δ_S g                    (Δ_S diagonal: â_ℓm ↦ −ℓ(ℓ+1)â_ℓm).
# So spin-1 synthesis is NOT required for the scattering amplitude — only the spin-0
# `nusht_filter!` with per-ℓ multipliers (exact in exact arithmetic). Pointwise orientation/phase
# on S² WOULD need the actual spin-1 field; see jbphyswx/NUFSHT.jl#1.
# ─────────────────────────────────────────────────────────────────────────────

# Generic per-degree spectral multiplier h(ℓ): NUFSHT's apply_transfer!/nusht_filter! dispatch
# on `kernel_transfer(filter, ℓ)`, so any object with this method is a valid transfer.
struct _FnTransfer{F}
    f::F
end
NUFSHT.kernel_transfer(t::_FnTransfer, ℓ) = t.f(ℓ)

# Difference-of-Gaussians band b_j(ℓ) = G(σ²_hi) − G(σ²_lo), G(σ²) = exp(−σ²ℓ(ℓ+1)/2).
# b_j(0) = 0 automatically (both Gaussians are 1 at ℓ=0).
@inline _bj(ℓ, σ²hi, σ²lo) = exp(-σ²hi * ℓ * (ℓ + 1) / 2) - exp(-σ²lo * ℓ * (ℓ + 1) / 2)

_band_transfer(σ²hi, σ²lo) = _FnTransfer(ℓ -> _bj(ℓ, σ²hi, σ²lo))
# g = (−Δ_S)^{-1/2} of the band:  h(ℓ) = b_j(ℓ)/√(ℓ(ℓ+1)).
_riesz_potential_transfer(σ²hi, σ²lo) =
    _FnTransfer(ℓ -> ℓ == 0 ? 0.0 : _bj(ℓ, σ²hi, σ²lo) / sqrt(ℓ * (ℓ + 1)))
# Δ_S g = −ℓ(ℓ+1)·ĝ:  h(ℓ) = −√(ℓ(ℓ+1))·b_j(ℓ).
_riesz_laplacian_transfer(σ²hi, σ²lo) =
    _FnTransfer(ℓ -> ℓ == 0 ? 0.0 : -sqrt(ℓ * (ℓ + 1)) * _bj(ℓ, σ²hi, σ²lo))
const _LAPLACIAN = _FnTransfer(ℓ -> -ℓ * (ℓ + 1))

"""
    SphericalMonogenicScattering{T,P}

Spherical monogenic scattering transform. Shares the dyadic difference-of-Gaussians bands of
[`SphericalScattering`](@ref) but replaces the analytic modulus with the spherical monogenic
amplitude `A_j = √(U⁰_j² + |∇_S g_j|²)`.
"""
struct SphericalMonogenicScattering{T,P}
    lmax::Int
    J::Int
    max_order::Int
    plan::P
    sigma2::Vector{T}
    M::Int
end

function ST.spherical_monogenic_scattering(pts_theta::AbstractVector{T}, pts_phi::AbstractVector{T},
                                           lmax::Int, J::Int; max_order::Int = 2) where {T<:Real}
    plan = NUFSHT.make_plan(pts_theta, pts_phi, lmax; T = T)
    sigma2 = T[_sigma2_for_cutoff(lmax / 2.0^(J - k), T) for k in 0:J]
    return SphericalMonogenicScattering{T, typeof(plan)}(lmax, J, max_order, plan, sigma2,
                                                         length(pts_theta))
end

# Monogenic amplitude field of `field` band-passed at scale `j`, written into `amp`.
# `w` is a NamedTuple of length-M scratch vectors (band, g, lapg, g2, lapg2).
function _spherical_monogenic_amplitude!(amp, st::SphericalMonogenicScattering{T},
                                         field, j::Int, w) where {T}
    σ²hi = st.sigma2[j + 1]
    σ²lo = st.sigma2[j]
    NUFSHT.nusht_filter!(w.band, field, _band_transfer(σ²hi, σ²lo), st.plan)             # U⁰
    NUFSHT.nusht_filter!(w.g,    field, _riesz_potential_transfer(σ²hi, σ²lo), st.plan)  # g = (−Δ)^{-1/2}U⁰
    NUFSHT.nusht_filter!(w.lapg, field, _riesz_laplacian_transfer(σ²hi, σ²lo), st.plan)  # Δ_S g
    @. w.g2 = w.g^2
    NUFSHT.nusht_filter!(w.lapg2, w.g2, _LAPLACIAN, st.plan)                             # Δ_S(g²)
    # |∇_S g|² = ½ Δ_S(g²) − g Δ_S g  (clamp tiny negatives from finite-lmax/adjoint error)
    @. amp = sqrt(w.band^2 + max(zero(T), T(0.5) * w.lapg2 - w.g * w.lapg))
    return amp
end

"""
    (st::SphericalMonogenicScattering)(field) -> (; S0, S1, S2)

Apply the spherical monogenic scattering transform to a scalar `field` sampled at the plan's
points. `S1[j] = ⟨A_j⟩`, `S2[j1,j2] = ⟨A_{j2}[A_{j1}]⟩` for strictly coarser `j2 < j1`.
"""
function (st::SphericalMonogenicScattering{T})(field::AbstractVector{T}) where {T}
    J, M = st.J, st.M
    S0 = sum(field) / T(M)
    S1 = zeros(T, J)
    S2 = zeros(T, J, J)
    w = (band = Vector{T}(undef, M), g = Vector{T}(undef, M), lapg = Vector{T}(undef, M),
         g2 = Vector{T}(undef, M), lapg2 = Vector{T}(undef, M))
    U1 = [Vector{T}(undef, M) for _ in 1:J]

    for j in 1:J
        _spherical_monogenic_amplitude!(U1[j], st, field, j, w)
        S1[j] = sum(U1[j]) / T(M)
    end

    if st.max_order >= 2
        amp = Vector{T}(undef, M)
        for j1 in 1:J, j2 in 1:(j1 - 1)
            _spherical_monogenic_amplitude!(amp, st, U1[j1], j2, w)
            S2[j1, j2] = sum(amp) / T(M)
        end
    end

    return (S0 = S0, S1 = S1, S2 = S2)
end

end # module ScatteringTransformsNUFSHTExt
