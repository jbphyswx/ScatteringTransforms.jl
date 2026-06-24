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

end # module ScatteringTransformsNUFSHTExt
