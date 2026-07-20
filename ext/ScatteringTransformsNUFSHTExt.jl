module ScatteringTransformsNUFSHTExt

"""
    ScatteringTransformsNUFSHTExt — scattered-point spherical scattering on S² (NUFSHT backend)

Provides the **scattered-sphere** backend for the shared spherical scattering core
(`SphericalCore`): a scalar field sampled at arbitrary points `(θ, φ)` on S², analysed/synthesised
via NUFSHT. The DoG band-pass bank, the S0/S1/S2 cascade, and the spin-0 Bochner monogenic amplitude
all live in `SphericalCore`; this extension only supplies

  * a plan wrapper `NUSHTSphericalPlan` over `NUFSHT.NUSHTplan`;
  * `sphere_coeffs` (exact analysis via `nusht_solve!`) and `sphere_apply!` (per-degree multiply via
    `apply_transfer!` + synthesis via `nusht_type2!`);
  * `sphere_mean` (unweighted sample mean over the scattered points);

plus the constructors `spherical_scattering` / `spherical_monogenic_scattering`.

Analysis uses NUFSHT's **exact inverse** (`nusht_solve!`), not the adjoint (`nusht_type1!`): on
scattered points the adjoint mis-scales the coefficients degree-dependently, so the exact inverse is
needed for correct absolute magnitudes. The `SphericalCore` cascade
analyses each field once and reuses its coefficients across all bands, so the (iterative) solve runs
once per field rather than once per band.
"""

using NUFSHT: NUFSHT
using ScatteringTransforms: ScatteringTransforms

const ST = ScatteringTransforms
const SC = ST.SphericalCore

# ---------------------------------------------------------------------------
# Scattered-sphere plan + the two SphericalCore interface methods
# ---------------------------------------------------------------------------

"""
    NUSHTSphericalPlan{P,V,T} <: SphericalCore.AbstractSphericalPlan

Scattered-point spherical plan wrapping a `NUFSHT.NUSHTplan`, the sample count `M`, the band limit
`lmax`, the scattered points `(theta, phi)` (retained so spin-weighted plans can be built on the same
points for `spherical_monogenic_components`), and the CG-inversion tolerance/iteration cap used by
[`sphere_coeffs`](@ref). Analysis uses NUFSHT's **exact inversion** (`nusht_solve!`), not the adjoint:
on scattered points the adjoint mis-scales the coefficients degree-dependently, so the exact inversion
is needed for correct absolute magnitudes. Accurate inversion needs the sampling to resolve the band
limit, i.e. roughly `M ≳ (lmax+1)²` well-distributed points.
"""
struct NUSHTSphericalPlan{P, V<:AbstractVector, T<:Real} <: SC.AbstractSphericalPlan
    plan::P
    M::Int
    lmax::Int
    theta::V
    phi::V
    rtol::T
    maxiter::Int
end

# Generic per-degree spectral multiplier h(ℓ): NUFSHT's apply_transfer! dispatches on
# `kernel_transfer(filter, ℓ)`, so any object with this method is a valid transfer.
struct _FnTransfer{F}
    f::F
end
NUFSHT.kernel_transfer(t::_FnTransfer, ℓ) = t.f(ℓ)

# Analysis: the TRUE spherical-harmonic coefficients of `field` via exact CG inversion.
function SC.sphere_coeffs(plan::NUSHTSphericalPlan, field::AbstractVector)
    C = zero(plan.plan.C)
    NUFSHT.nusht_solve!(C, field, plan.plan; rtol = plan.rtol, maxiter = plan.maxiter)
    return C
end

# Apply the per-degree multiplier `h(ℓ)` to a copy of the coefficients and synthesise at the points.
function SC.sphere_apply!(out::AbstractVector, plan::NUSHTSphericalPlan, C, h)
    C2 = copy(C)
    NUFSHT.apply_transfer!(C2, _FnTransfer(h), plan.lmax)
    NUFSHT.nusht_type2!(out, C2, plan.plan)
    return out
end

# Unweighted sample mean over the (quasi-uniform) scattered points ≈ the spherical average.
SC.sphere_mean(plan::NUSHTSphericalPlan, field::AbstractVector) = sum(field) / plan.M

# ---------------------------------------------------------------------------
# Constructors (scattered-sphere entry points declared in core)
# ---------------------------------------------------------------------------

_nusht_plan(pts_theta::AbstractVector{T}, pts_phi::AbstractVector{T}, lmax::Int,
            rtol::Real, maxiter::Int) where {T<:Real} =
    NUSHTSphericalPlan(NUFSHT.make_plan(pts_theta, pts_phi, lmax; T = T), length(pts_theta), lmax,
                       collect(T, pts_theta), collect(T, pts_phi), T(rtol), maxiter)

function ST.spherical_scattering(pts_theta::AbstractVector{T}, pts_phi::AbstractVector{T},
                                 lmax::Int, J::Int; max_order::Int = 2,
                                 rtol::Real = 1.0e-8, maxiter::Int = 500) where {T<:Real}
    plan = _nusht_plan(pts_theta, pts_phi, lmax, rtol, maxiter)
    return SC.SphericalScattering{T, typeof(plan)}(lmax, J, max_order, plan, SC.dog_sigma2(lmax, J, T))
end

function ST.spherical_monogenic_scattering(pts_theta::AbstractVector{T}, pts_phi::AbstractVector{T},
                                           lmax::Int, J::Int; max_order::Int = 2,
                                           rtol::Real = 1.0e-8, maxiter::Int = 500) where {T<:Real}
    plan = _nusht_plan(pts_theta, pts_phi, lmax, rtol, maxiter)
    return SC.SphericalMonogenicScattering{T, typeof(plan)}(lmax, J, max_order, plan,
                                                            SC.dog_sigma2(lmax, J, T))
end

# ---------------------------------------------------------------------------
# Pointwise spherical monogenic decomposition — the S² analogue of the planar `monogenic_components`,
# using NUFSHT's spin-weighted scattered synthesis.
#
# For band-pass `U⁰_j = b_j(ℓ)·a_ℓm` (spin-0), the spin-1 Riesz field is
#   U^R_j = ð∘(−Δ_S)^{-1/2} U⁰_j,   coeffs = √(ℓ(ℓ+1))·(1/√(ℓ(ℓ+1)))·b_j(ℓ)·a_ℓm = b_j(ℓ)·a_ℓm  (ℓ≥1).
# So the SAME coefficient array `sf = b_j(ℓ)·a` synthesised at spin-0 gives the band-pass field and at
# spin-1 gives the complex tangent Riesz field `U = u_θ + i u_φ`. The scalar amplitude √(U⁰²+|U^R|²)
# agrees (up to the SHT's accuracy) with the spin-0 Bochner identity used by the scattering cascade —
# validated in the tests, which also check the spin-1 synthesis against the closed-form `sYlm`.
# ---------------------------------------------------------------------------

function ST.spherical_monogenic_components(st::SC.SphericalMonogenicScattering, field::AbstractVector, j::Int)
    p = st.plan
    lmax = st.lmax
    T = eltype(p.theta)
    # Complex spin-0 coefficients of the field in NUFSHT's dense spin layout (exact CG inversion,
    # so the band-pass and Riesz fields below share one consistent set of coefficients).
    p0 = NUFSHT.make_spin_plan(p.theta, p.phi, lmax, 0; T = T)
    p1 = NUFSHT.make_spin_plan(p.theta, p.phi, lmax, 1; T = T)
    a = zeros(Complex{T}, lmax + 1, 2lmax + 1)
    NUFSHT.nusht_solve_spin!(a, Complex{T}.(field), p0)

    # sf = b_j(ℓ)·a  (b_j(0)=0, so the ℓ=0 term vanishes as the Riesz operator requires).
    bfn = SC.band_multiplier(st.sigma2[j + 1], st.sigma2[j])
    sf = similar(a)
    @inbounds for ℓ in 0:lmax
        bl = bfn(ℓ)
        for m in -ℓ:ℓ
            idx = NUFSHT.spin_coeff_index(ℓ, m, lmax)
            sf[idx] = bl * a[idx]
        end
    end

    M = p.M
    U0 = Vector{Complex{T}}(undef, M)
    UR = Vector{Complex{T}}(undef, M)
    NUFSHT.nusht_type2_spin!(U0, sf, p0)          # spin-0 synthesis → band-pass (real up to error)
    NUFSHT.nusht_type2_spin!(UR, sf, p1)          # spin-1 synthesis → Riesz tangent field u_θ + i u_φ

    bandpass = real.(U0)
    riesz = (real.(UR), imag.(UR))                # (u_θ, u_φ)
    rnorm = abs.(UR)
    amplitude = sqrt.(bandpass .^ 2 .+ rnorm .^ 2)
    phase = atan.(rnorm, bandpass)                # atan(‖riesz‖, bandpass)
    orientation = atan.(imag.(UR), real.(UR))     # tangent-vector direction on S²
    return (; bandpass, riesz, amplitude, phase, orientation)
end

end # module ScatteringTransformsNUFSHTExt
