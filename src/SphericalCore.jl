module SphericalCore

"""
    SphericalCore.jl — backend-agnostic spherical scattering core

Shared machinery for scattering on S², independent of *how* the spherical harmonic transform is
performed. Two backends plug in through a tiny interface (methods added by extensions):

  * scattered points  — `ScatteringTransformsNUFSHTExt` (NUFSHT, adjoint / CG-solve);
  * structured grid    — `ScatteringTransformsFastSphericalHarmonicsExt` (fast SHT on a CC grid).

The interface a backend `plan` must implement is just:

  * `sphere_coeffs(plan, field)`      — analyse `field` to its SH coefficients (exact inverse for
    scattered points, forward SHT for a structured grid);
  * `sphere_apply!(out, plan, C, h)`  — apply the per-degree multiplier `h(ℓ)` to a copy of `C` and
    synthesise into `out`;
  * `sphere_mean(plan, field)`        — the spherical average of `field` (unweighted sample mean for
    scattered points; quadrature-weighted integral for a structured grid).

Everything else — the dyadic difference-of-Gaussians band-pass bank, the S0/S1/S2 cascade (which
analyses each field once and reuses its coefficients across bands), and the spin-0 Bochner monogenic
amplitude — is pure, shared math built on those primitives. Pointwise
monogenic orientation/phase (`spherical_monogenic_components`) additionally needs a spin-1 synthesis
primitive, supplied by the NUFSHT extension.
"""

export AbstractSphericalPlan, SphericalScattering, SphericalMonogenicScattering
export sphere_coeffs, sphere_apply!, sphere_mean

# ---------------------------------------------------------------------------
# Backend interface — methods provided by the NUFSHT / FastSphericalHarmonics extensions.
#
# The band-pass filter is factored into *analyse once* (`sphere_coeffs`) + *apply-a-multiplier-and-
# synthesise* (`sphere_apply!`). This lets the cascade analyse each field a single time and reuse its
# coefficients across all bands (and, for the scattered backend, use an accurate inverse rather than
# the mis-scaled adjoint — analysing per band would multiply that solve cost by J).
# ---------------------------------------------------------------------------

"""
    AbstractSphericalPlan

Supertype for spherical spectral plans (scattered NUFSHT, structured SHT, …). A concrete plan must
implement [`sphere_coeffs`](@ref), [`sphere_apply!`](@ref), and [`sphere_mean`](@ref).
"""
abstract type AbstractSphericalPlan end

"""
    sphere_coeffs(plan, field) -> C

Spherical-harmonic **analysis**: the coefficients `C` of `field` (up to the plan's band limit). For a
structured grid this is the fast forward SHT; for scattered points it is the **exact** (least-squares /
CG) inversion — *not* the adjoint, which mis-scales the coefficients. Returned opaquely and consumed
only by [`sphere_apply!`](@ref)/`sphere_mean` on the same `plan`.
"""
function sphere_coeffs end

"""
    sphere_apply!(out, plan, C, h) -> out

Apply the per-degree multiplier `h(ℓ)` to (a copy of) the coefficients `C` and **synthesise** the
result into `out` — i.e. `out = Σ_{ℓm} h(ℓ) · C_{ℓm} · Y_{ℓm}`. Does not mutate `C`.
"""
function sphere_apply! end

"""
    sphere_mean(plan, field) -> scalar

Spherical average of `field` under the plan's sampling: an unweighted sample mean for (quasi-uniform)
scattered points, the exact quadrature integral for a structured grid. Provided by each backend.
"""
function sphere_mean end

# ---------------------------------------------------------------------------
# Dyadic difference-of-Gaussians band-pass bank (pure math).
# ---------------------------------------------------------------------------

# σ² so the Gaussian transfer is e^{-1/2} at degree ℓ (ℓ marks the low-pass roll-off).
_sigma2_for_cutoff(ℓ::Real, ::Type{T}) where {T} = ℓ <= 0 ? T(Inf) : T(1) / (T(ℓ) * (T(ℓ) + 1))

"""
    dog_sigma2(lmax, J, T) -> Vector{T}

Gaussian-transfer variances for the `J+1` dyadic low-pass cutoffs `ℓ_k = lmax / 2^(J-k)`, `k=0..J`
(`ℓ_J = lmax`). Band-pass wavelet `j` is `lowpass(ℓ_j) − lowpass(ℓ_{j-1})`.
"""
dog_sigma2(lmax::Integer, J::Integer, ::Type{T}) where {T} =
    T[_sigma2_for_cutoff(lmax / 2.0^(J - k), T) for k in 0:J]

# Heat-kernel low-pass transfer at degree ℓ: G(σ²) = exp(-σ² ℓ(ℓ+1) / 2).
@inline _gauss(ℓ, σ²) = exp(-σ² * ℓ * (ℓ + 1) / 2)

# Difference-of-Gaussians band b_j(ℓ) = G(σ²_hi) − G(σ²_lo); b_j(0) = 0 automatically.
@inline _bj(ℓ, σ²hi, σ²lo) = _gauss(ℓ, σ²hi) - _gauss(ℓ, σ²lo)

"Per-degree multiplier for the difference-of-Gaussians band-pass wavelet `j`."
band_multiplier(σ²hi, σ²lo) = ℓ -> _bj(ℓ, σ²hi, σ²lo)

# g = (−Δ_S)^{-1/2} of the band:  h(ℓ) = b_j(ℓ)/√(ℓ(ℓ+1)),  0 at ℓ=0.
riesz_potential_multiplier(σ²hi, σ²lo) = ℓ -> ℓ == 0 ? 0.0 : _bj(ℓ, σ²hi, σ²lo) / sqrt(ℓ * (ℓ + 1))
# Δ_S g = −ℓ(ℓ+1)·ĝ:  h(ℓ) = −√(ℓ(ℓ+1))·b_j(ℓ),  0 at ℓ=0.
riesz_laplacian_multiplier(σ²hi, σ²lo) = ℓ -> ℓ == 0 ? 0.0 : -sqrt(ℓ * (ℓ + 1)) * _bj(ℓ, σ²hi, σ²lo)
# Laplace–Beltrami:  Δ_S = ×(−ℓ(ℓ+1)).
laplacian_multiplier() = ℓ -> -ℓ * (ℓ + 1)

# ---------------------------------------------------------------------------
# Transforms (backend-generic; the plan supplies sphere_filter! / sphere_mean).
# ---------------------------------------------------------------------------

"""
    SphericalScattering{T,P}

Spherical scattering transform over a backend `plan::P`. `sigma2[k+1]` is the Gaussian-transfer
variance for the dyadic low-pass cutoff `ℓ_k` (`k=0..J`); band-pass wavelet `j` is
`lowpass(ℓ_j) − lowpass(ℓ_{j-1})`.
"""
struct SphericalScattering{T, P}
    lmax::Int
    J::Int
    max_order::Int
    plan::P
    sigma2::Vector{T}
end

"""
    (st::SphericalScattering)(field) -> (; S0, S1, S2)

Apply the spherical scattering transform to a scalar `field` sampled by the plan.
`S1[j] = ⟨|field ⋆ ψ_j|⟩`, `S2[j1,j2] = ⟨||field ⋆ ψ_{j1}| ⋆ ψ_{j2}|⟩` for strictly coarser `j2 < j1`.
"""
function (st::SphericalScattering{T})(field::AbstractArray) where {T}
    J = st.J
    S0 = sphere_mean(st.plan, field)
    S1 = zeros(T, J)
    S2 = zeros(T, J, J)
    band = similar(field)
    U1 = [similar(field) for _ in 1:J]

    C = sphere_coeffs(st.plan, field)                 # analyse the field once
    for j in 1:J
        sphere_apply!(band, st.plan, C, band_multiplier(st.sigma2[j + 1], st.sigma2[j]))
        @. U1[j] = abs(band)
        S1[j] = sphere_mean(st.plan, U1[j])
    end

    if st.max_order >= 2
        for j1 in 1:J
            C1 = sphere_coeffs(st.plan, U1[j1])       # analyse U1[j1] once, reuse across its children
            for j2 in 1:(j1 - 1)
                sphere_apply!(band, st.plan, C1, band_multiplier(st.sigma2[j2 + 1], st.sigma2[j2]))
                @. band = abs(band)
                S2[j1, j2] = sphere_mean(st.plan, band)
            end
        end
    end
    return (S0 = S0, S1 = S1, S2 = S2)
end

"""
    SphericalMonogenicScattering{T,P}

Spherical monogenic scattering: shares the dyadic difference-of-Gaussians bands of
[`SphericalScattering`](@ref) but replaces the analytic modulus with the spherical monogenic
amplitude `A_j = √(U⁰_j² + |∇_S g_j|²)` (spin-0 Bochner identity — see [`monogenic_amplitude!`](@ref)).
"""
struct SphericalMonogenicScattering{T, P}
    lmax::Int
    J::Int
    max_order::Int
    plan::P
    sigma2::Vector{T}
end

"""
    monogenic_amplitude!(amp, st, C, j, w) -> amp

Spherical monogenic amplitude at scale `j` of the field whose (already-computed) SH coefficients are
`C`, written into `amp`. `w` is a NamedTuple of scratch fields `(g, lapg, g2, lapg2)`.

Uses only spin-0 transforms via the Bochner/product identity: with `g_j = (−Δ_S)^{-1/2} U⁰_j`,

    |U^R_j|² = |∇_S g_j|² = ½ Δ_S(g_j²) − g_j · Δ_S g_j,

so `A_j = √(U⁰_j² + |∇_S g_j|²)`. The Riesz *vector* itself (needed for orientation/phase) requires
spin-1 synthesis and is handled separately by `spherical_monogenic_components`.
"""
function monogenic_amplitude!(amp::AbstractArray, st::SphericalMonogenicScattering{T},
                              C, j::Int, w) where {T}
    σ²hi = st.sigma2[j + 1]
    σ²lo = st.sigma2[j]
    sphere_apply!(amp,    st.plan, C, band_multiplier(σ²hi, σ²lo))              # U⁰ (into amp)
    sphere_apply!(w.g,    st.plan, C, riesz_potential_multiplier(σ²hi, σ²lo))   # g = (−Δ)^{-1/2}U⁰
    sphere_apply!(w.lapg, st.plan, C, riesz_laplacian_multiplier(σ²hi, σ²lo))   # Δ_S g
    @. w.g2 = w.g^2
    Cg2 = sphere_coeffs(st.plan, w.g2)                                          # re-analyse g²
    sphere_apply!(w.lapg2, st.plan, Cg2, laplacian_multiplier())               # Δ_S(g²)
    # |∇_S g|² = ½ Δ_S(g²) − g Δ_S g  (clamp tiny negatives from finite-lmax error)
    @. amp = sqrt(amp^2 + max(zero(T), T(0.5) * w.lapg2 - w.g * w.lapg))
    return amp
end

"""
    (st::SphericalMonogenicScattering)(field) -> (; S0, S1, S2)

Apply the spherical monogenic scattering transform. `S1[j] = ⟨A_j⟩`,
`S2[j1,j2] = ⟨A_{j2}[A_{j1}]⟩` for strictly coarser `j2 < j1`.
"""
function (st::SphericalMonogenicScattering{T})(field::AbstractArray) where {T}
    J = st.J
    S0 = sphere_mean(st.plan, field)
    S1 = zeros(T, J)
    S2 = zeros(T, J, J)
    w = (g = similar(field), lapg = similar(field), g2 = similar(field), lapg2 = similar(field))
    U1 = [similar(field) for _ in 1:J]

    C = sphere_coeffs(st.plan, field)                 # analyse the field once
    for j in 1:J
        monogenic_amplitude!(U1[j], st, C, j, w)
        S1[j] = sphere_mean(st.plan, U1[j])
    end

    if st.max_order >= 2
        amp = similar(field)
        for j1 in 1:J
            C1 = sphere_coeffs(st.plan, U1[j1])       # analyse U1[j1] once
            for j2 in 1:(j1 - 1)
                monogenic_amplitude!(amp, st, C1, j2, w)
                S2[j1, j2] = sphere_mean(st.plan, amp)
            end
        end
    end
    return (S0 = S0, S1 = S1, S2 = S2)
end

end # module SphericalCore
