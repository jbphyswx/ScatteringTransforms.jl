module ScatteringTransformsNUFSHTExt

using NUFSHT: NUFSHT
using ScatteringTransforms: ScatteringTransforms
using LinearAlgebra: LinearAlgebra
using FastSphericalHarmonics: FastSphericalHarmonics

"""
    ScatteringTransformSphere{T}

Spherical scattering transform using NUFSHT.

Applies a cascade of spherical harmonic filter bank convolutions (band-pass
filtering in `lmax` space) and pointwise modulus operations to compute
translation-invariant scattering coefficients for scalar fields on S².

# Fields
- `lmax::Int`: Maximum spherical harmonic degree (sets spatial resolution)
- `J::Int`: Number of dyadic scales
- `max_order::Int`: Maximum scattering order (1 or 2)
- `plan::NUFSHT.NUSHTplan`: Pre-planned NUFSHT plan (synthesis: coeffs→values)
- `band_limits::Vector{Int}`: Band-pass cutoffs `[l0, l1, ..., lJ]`
"""
struct ScatteringTransformSphere{T<:Real}
    lmax::Int
    J::Int
    max_order::Int
    plan::NUFSHT.NUSHTplan
    band_limits::Vector{Int}
end

"""
    ScatteringTransformSphere(pts_theta, pts_phi, lmax, J; max_order=2, T=Float64)

Construct a spherical scattering transform for `M` scattered points on S².

- `pts_theta`: colatitude θ ∈ [0,π], length M
- `pts_phi`: longitude φ ∈ [0,2π), length M
- `lmax`: maximum spherical harmonic degree
- `J`: number of dyadic scales (band-pass shells)
"""
function ScatteringTransformSphere(
    pts_theta::AbstractVector{T},
    pts_phi::AbstractVector{T},
    lmax::Int,
    J::Int;
    max_order::Int = 2,
) where T<:Real
    plan = NUFSHT.make_plan(pts_theta, pts_phi, lmax; T=T)
    # Dyadic band limits: l_j = round(lmax / 2^(J-j)) for j = 0..J
    band_limits = [round(Int, lmax / 2^(J - j)) for j in 0:J]
    band_limits[end] = lmax  # ensure last is exact
    return ScatteringTransformSphere{T}(lmax, J, max_order, plan, band_limits)
end

"""
    _sph_bandpass_filter!(coeffs_out, coeffs_in, l_lo, l_hi, lmax)

Zero out all modes outside the band [l_lo, l_hi] in a coefficient array.
`coeffs_in` and `coeffs_out` are (lmax+1)×(2lmax+1) arrays (NUFSHT layout).
"""
function _sph_bandpass_filter!(
    coeffs_out::AbstractMatrix{Complex{T}},
    coeffs_in::AbstractMatrix{Complex{T}},
    l_lo::Int,
    l_hi::Int,
    lmax::Int,
) where T<:Real
    fill!(coeffs_out, zero(Complex{T}))
    @inbounds for l in l_lo:l_hi
        for m in -l:l
            idx = FastSphericalHarmonics.sph_mode(l, m)
            coeffs_out[idx] = coeffs_in[idx]
        end
    end
    return coeffs_out
end

"""
    (st::ScatteringTransformSphere{T})(field, pts_theta, pts_phi) -> NamedTuple

Apply spherical scattering transform to a scalar field sampled at scattered
points `(pts_theta, pts_phi)` on S².

Returns `(S0, S1, S2)`:
- `S0::T`: global mean (0th order)
- `S1::Vector{T}`: length J, band-averaged energy per scale
- `S2::Matrix{T}`: J×J, cross-scale energy (upper triangular)
"""
function (st::ScatteringTransformSphere{T})(
    field::AbstractVector{T},
    pts_theta::AbstractVector{T},
    pts_phi::AbstractVector{T},
) where T<:Real
    lmax = st.lmax
    J    = st.J
    Nθ   = lmax + 1
    Nφ   = 2 * lmax + 1
    M    = length(field)
    
    # Pre-allocate coefficient and value buffers
    # nusht_type1! takes real input, nusht_type2! writes to real output
    coeffs_full = zeros(Complex{T}, Nθ, Nφ)
    coeffs_band = zeros(Complex{T}, Nθ, Nφ)
    values_band = zeros(T, M)
    
    # Analysis: field → spherical harmonic coefficients (type-1 NUFSHT)
    # nusht_type1! accepts real field values
    NUFSHT.nusht_type1!(coeffs_full, field, st.plan)
    
    l0m0 = FastSphericalHarmonics.sph_mode(0, 0)
    S0 = real(coeffs_full[l0m0]) * T(√(4π))
    
    S1 = zeros(T, J)
    S2 = zeros(T, J, J)
    
    # U1[j] = |field ★ ψ_j| (modulus of band-pass filtered field, at scattered pts)
    U1 = [zeros(T, M) for _ in 1:J]
    
    for j in 1:J
        l_lo = j == 1 ? 0 : st.band_limits[j-1] + 1
        l_hi = st.band_limits[j]
        
        _sph_bandpass_filter!(coeffs_band, coeffs_full, l_lo, l_hi, lmax)
        
        # Synthesis: band-pass coefficients → real values at scattered points (type-2 NUFSHT)
        NUFSHT.nusht_type2!(values_band, coeffs_band, st.plan)
        
        @inbounds for i in 1:M
            U1[j][i] = abs(values_band[i])
        end
        
        S1[j] = sum(U1[j]) / T(M)
    end
    
    if st.max_order >= 2
        for j1 in 1:J
            coeffs_u1 = zeros(Complex{T}, Nθ, Nφ)
            NUFSHT.nusht_type1!(coeffs_u1, U1[j1], st.plan)
            
            for j2 in (j1+1):J
                l_lo = j2 == 1 ? 0 : st.band_limits[j2-1] + 1
                l_hi = st.band_limits[j2]
                
                _sph_bandpass_filter!(coeffs_band, coeffs_u1, l_lo, l_hi, lmax)
                NUFSHT.nusht_type2!(values_band, coeffs_band, st.plan)
                
                @inbounds for i in 1:M
                    values_band[i] = abs(values_band[i])
                end
                
                S2[j1, j2] = sum(values_band) / T(M)
            end
        end
    end
    
    return (S0=S0, S1=S1, S2=S2)
end

"""
    scattering_transform_sphere!(S1, S2, st, field, pts_theta, pts_phi) -> S0

In-place spherical scattering transform. Fills pre-allocated `S1` and `S2`.
Returns S0 (scalar, zero-alloc for arrays via dispatch).
"""
function scattering_transform_sphere!(
    S1::AbstractVector{T},
    S2::AbstractMatrix{T},
    st::ScatteringTransformSphere{T},
    field::AbstractVector{T},
    pts_theta::AbstractVector{T},
    pts_phi::AbstractVector{T},
) where T<:Real
    result = st(field, pts_theta, pts_phi)
    S1 .= result.S1
    S2 .= result.S2
    return result.S0
end

end # module ScatteringTransformsNUFSHTExt
