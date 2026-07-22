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

using ..Plans: Plans
using LinearAlgebra: LinearAlgebra

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

# ---------------------------------------------------------------------------
# In-core dependency-free scattered-sphere plan: SH analysis/synthesis by direct real-Yℓm summation.
#
# The `O(M·(lmax+1)²)` fallback that lets `spherical_scattering` run with no external SH library — the
# spherical counterpart of the direct-sum planar plans. A real spherical-harmonic design matrix `Y`
# (M points × (lmax+1)² harmonics) is built by an associated-Legendre recurrence; analysis is a
# conjugate-gradient least-squares solve of the normal equations `Y'Y a = Y'field` (the accurate
# inverse, like NUFSHT's `nusht_solve!`, not the adjoint), and synthesis is `Σ_ℓm h(ℓ)·a_ℓm·Yℓm`.
# Columns are L2-normalized for conditioning; since the band multiplier `h(ℓ)` is constant within a
# degree, the per-degree filtering is invariant to that scaling.
# ---------------------------------------------------------------------------

# Fully-normalized associated Legendre P̄_ℓ^m(cosθ) for all 0≤m≤ℓ≤lmax at one point, via the standard
# stable recurrence (values stay O(1) — unlike the raw `(2m-1)!!` form, which overflows at large lmax).
# The overall per-(ℓ,m) normalization constant is immaterial here: it is absorbed by the column
# normalization of the design matrix and cancels in the per-degree band multiplier.
function _assoc_legendre!(P::AbstractMatrix{T}, x::Real, lmax::Int) where {T}
    xT = T(x)
    u = sqrt(max(zero(T), one(T) - xT^2))       # sinθ
    fill!(P, zero(T))
    P[1, 1] = one(T)                            # P̄_0^0 (arbitrary O(1) scale)
    @inbounds for m in 1:lmax                   # sectoral: P̄_m^m = -√((2m+1)/(2m))·sinθ·P̄_{m-1}^{m-1}
        P[m + 1, m + 1] = -sqrt((2m + one(T)) / (2m)) * u * P[m, m]
    end
    @inbounds for m in 0:lmax
        if m < lmax                             # subdiagonal: P̄_{m+1}^m = √(2m+3)·cosθ·P̄_m^m
            P[m + 2, m + 1] = sqrt(2m + T(3)) * xT * P[m + 1, m + 1]
        end
        for ℓ in (m + 2):lmax                   # three-term recurrence in ℓ
            a = sqrt((2ℓ - one(T)) * (2ℓ + one(T)) / ((ℓ - m) * (ℓ + m)))
            b = sqrt((2ℓ + one(T)) * (ℓ + m - one(T)) * (ℓ - m - one(T)) /
                     ((2ℓ - T(3)) * (ℓ + m) * (ℓ - m)))
            P[ℓ + 1, m + 1] = a * xT * P[ℓ, m + 1] - b * P[ℓ - 1, m + 1]
        end
    end
    return P
end

struct DirectSHTSphericalPlan{T, YM<:AbstractMatrix{T}, GM<:AbstractMatrix{T},
                              IV<:AbstractVector{Int}, VV<:AbstractVector{T}} <: AbstractSphericalPlan
    lmax::Int
    M::Int
    K::Int
    Y::YM            # (M, K) fully-normalized real-SH design matrix
    G::GM            # (K, K) Gram matrix Y'Y (SPD) for the least-squares analysis
    ldeg::IV         # (K) degree ℓ of each column
    theta::VV        # (M) colatitudes (retained for the spin-1 gradient synthesis)
    phi::VV          # (M) longitudes
    rtol::T
    maxiter::Int
    weights::VV      # (M) spatial-mean quadrature weights (sum to 1)
    rhs::VV          # (K) Y'field
    r::VV            # (K) CG residual
    p::VV            # (K) CG search direction
    Gp::VV           # (K) CG G·p
    ha::VV           # (K) synthesis scratch h(ℓ)·C
end

function DirectSHTSphericalPlan(θ::AbstractVector, φ::AbstractVector, lmax::Int, ::Type{T};
                                rtol::Real = 1.0e-8, maxiter::Int = 500, weights = nothing) where {T}
    M = length(θ)
    length(φ) == M || throw(DimensionMismatch("θ and φ must have equal length"))
    w = weights === nothing ? fill(one(T) / M, M) : (T.(weights) ./ sum(weights))
    θT, φT = collect(T, θ), collect(T, φ)
    K = (lmax + 1)^2
    Y = Matrix{T}(undef, M, K)
    ldeg = Vector{Int}(undef, K)
    P = Matrix{T}(undef, lmax + 1, lmax + 1)
    s2 = sqrt(T(2))
    @inbounds for n in 1:M
        _assoc_legendre!(P, cos(θT[n]), lmax)
        φn = φT[n]
        col = 0
        for ℓ in 0:lmax, m in -ℓ:ℓ
            col += 1
            am = abs(m)
            Pv = P[ℓ + 1, am + 1]
            Y[n, col] = m == 0 ? Pv : (m > 0 ? s2 * Pv * cos(m * φn) : s2 * Pv * sin(am * φn))
            ldeg[col] = ℓ
        end
    end
    G = transpose(Y) * Y
    return DirectSHTSphericalPlan{T, Matrix{T}, Matrix{T}, Vector{Int}, Vector{T}}(
        lmax, M, K, Y, G, ldeg, θT, φT, T(rtol), maxiter, w,
        Vector{T}(undef, K), Vector{T}(undef, K), Vector{T}(undef, K),
        Vector{T}(undef, K), Vector{T}(undef, K))
end

# CG on the SPD Gram matrix: solve G·a = rhs (the least-squares normal equations Y'Y a = Y'field).
function _cg_gram!(a::AbstractVector, plan::DirectSHTSphericalPlan{T}) where {T}
    fill!(a, zero(T))
    copyto!(plan.r, plan.rhs)
    copyto!(plan.p, plan.r)
    rsold = real(LinearAlgebra.dot(plan.r, plan.r))
    rs0 = rsold
    rs0 == 0 && return a
    @inbounds for _ in 1:plan.maxiter
        LinearAlgebra.mul!(plan.Gp, plan.G, plan.p)
        α = rsold / real(LinearAlgebra.dot(plan.p, plan.Gp))
        a .+= α .* plan.p
        plan.r .-= α .* plan.Gp
        rsnew = real(LinearAlgebra.dot(plan.r, plan.r))
        sqrt(rsnew) <= plan.rtol * sqrt(rs0) && break
        plan.p .= plan.r .+ (rsnew / rsold) .* plan.p
        rsold = rsnew
    end
    return a
end

# Analysis: least-squares SH coefficients of `field` (fresh array — the cascade holds several at once).
# `field` may be a length-M vector (scattered) or the (Nθ,Nφ) grid matrix (structured); `vec` flattens
# it in the same order the plan's points were built.
function sphere_coeffs(plan::DirectSHTSphericalPlan{T}, field::AbstractArray) where {T}
    LinearAlgebra.mul!(plan.rhs, transpose(plan.Y), vec(field))
    a = Vector{T}(undef, plan.K)
    _cg_gram!(a, plan)
    return a
end

# Synthesis: out = Σ_k h(ℓ_k)·C_k·Y_k  (apply the per-degree multiplier, then evaluate at the points).
function sphere_apply!(out::AbstractArray, plan::DirectSHTSphericalPlan{T}, C, h) where {T}
    @inbounds for k in 1:plan.K
        plan.ha[k] = T(h(plan.ldeg[k])) * C[k]
    end
    LinearAlgebra.mul!(vec(out), plan.Y, plan.ha)
    return out
end

# Spherical average under the plan's quadrature weights (uniform 1/M for scattered points; sinθ area
# weights for the structured grid).
sphere_mean(plan::DirectSHTSphericalPlan, field::AbstractArray) =
    LinearAlgebra.dot(plan.weights, vec(field))

# ---------------------------------------------------------------------------
# Spherical-plan backend seam (mirrors the planar `Plans.make_scattered_plan`).
# ---------------------------------------------------------------------------

"In-core exact direct-summation SH transform for scattered points on S² (dependency-free)."
struct DirectSHTBackend <: Plans.AbstractSpectralBackend end

"NUFSHT fast path for scattered points on S²; requires the NUFSHT extension (`using NUFSHT`)."
struct NUSHTBackend <: Plans.AbstractSpectralBackend end

_have_nufsht() = Base.get_extension(parentmodule(@__MODULE__), :ScatteringTransformsNUFSHTExt) !== nothing

"""
    nusht_spherical_plan(θ, φ, lmax, T; rtol, maxiter)

Fast-path scattered-sphere plan constructor. Declaration only — the NUFSHT extension provides the sole
method. Call via [`make_spherical_plan`](@ref), which guards on the extension being loaded.
"""
function nusht_spherical_plan end

"""
    make_spherical_plan(spectral, θ, φ, lmax, T; rtol, maxiter) -> AbstractSphericalPlan

Build the scattered-sphere plan selected by `spectral` over points `(θ, φ)` and band limit `lmax`.
[`DirectSHTBackend`](@ref) is the dependency-free default; [`NUSHTBackend`](@ref) (or
`Plans.AutoSpectral` once the NUFSHT extension is loaded) uses the NUFSHT fast path.
"""
function make_spherical_plan(::DirectSHTBackend, θ, φ, lmax, ::Type{T};
                             rtol::Real = 1.0e-8, maxiter::Int = 500) where {T}
    return DirectSHTSphericalPlan(θ, φ, lmax, T; rtol = rtol, maxiter = maxiter)
end
function make_spherical_plan(::NUSHTBackend, θ, φ, lmax, ::Type{T}; kwargs...) where {T}
    _have_nufsht() ||
        throw(ArgumentError("NUSHTBackend requires the NUFSHT extension. Run `using NUFSHT`."))
    return nusht_spherical_plan(θ, φ, lmax, T; kwargs...)
end
make_spherical_plan(::Plans.AutoSpectral, θ, φ, lmax, ::Type{T}; kwargs...) where {T} =
    _have_nufsht() ? nusht_spherical_plan(θ, φ, lmax, T; kwargs...) :
    make_spherical_plan(DirectSHTBackend(), θ, φ, lmax, T; kwargs...)

# ---------------------------------------------------------------------------
# Structured (uniform-grid) sphere: the equiangular (Fejér-midpoint) grid `θ_j = π(j-½)/N`,
# `φ_k = 2π(k-1)/(2N-1)`, `N = lmax+1` — the same grid FastSphericalHarmonics uses, computed here
# dependency-free so `structured_sphere_points` is backend-independent. The in-core default treats the
# grid as points for the direct SHT (with sinθ area weights for the spherical mean); FastSpherical
# harmonics is the fast exact path.
# ---------------------------------------------------------------------------

"""
    structured_grid(lmax, T) -> (Θ, Φ)

Colatitudes `Θ` (length `lmax+1`) and longitudes `Φ` (length `2lmax+1`) of the equiangular structured
grid. Matches `FastSphericalHarmonics.sph_points(lmax+1)`.
"""
function structured_grid(lmax::Int, ::Type{T}) where {T}
    N = lmax + 1
    Θ = T[π * (j - one(T) / 2) / N for j in 1:N]
    Φ = T[2π * (k - 1) / (2N - 1) for k in 1:(2N - 1)]
    return Θ, Φ
end

# Fejér first-rule quadrature weights for the midpoint colatitudes θ_j = π(j-½)/N: they integrate
# ∫₀^π g(cosθ) sinθ dθ = ∫₋₁¹ g dx exactly for band-limited g, so the spherical mean below is exact
# (not just the sinθ area approximation). Non-negative and sum to 2.
function _fejer1_weights(N::Int, ::Type{T}) where {T}
    w = Vector{T}(undef, N)
    @inbounds for j in 1:N
        θ = T(π) * (j - one(T) / 2) / N
        s = zero(T)
        for m in 1:(N ÷ 2)
            s += cos(2m * θ) / T(4 * m^2 - 1)
        end
        w[j] = (2 / T(N)) * (1 - 2s)
    end
    return w
end

# Direct SHT plan on the flattened structured grid (column-major, θ fastest — matching `vec` of the
# (Nθ,Nφ) field matrix), with exact Fejér quadrature weights for the spherical mean.
function _direct_structured_plan(lmax::Int, ::Type{T}; rtol::Real, maxiter::Int) where {T}
    Θ, Φ = structured_grid(lmax, T)
    Nθ, Nφ = length(Θ), length(Φ)
    wj = _fejer1_weights(Nθ, T)
    θf = vec(T[Θ[j] for j in 1:Nθ, _ in 1:Nφ])
    φf = vec(T[Φ[k] for _ in 1:Nθ, k in 1:Nφ])
    wf = vec(T[wj[j] for j in 1:Nθ, _ in 1:Nφ])   # DirectSHTSphericalPlan normalizes to sum 1
    return DirectSHTSphericalPlan(θf, φf, lmax, T; rtol = rtol, maxiter = maxiter, weights = wf)
end

"FastSphericalHarmonics fast path for the structured (uniform-grid) sphere; requires `using FastSphericalHarmonics`."
struct SHTBackend <: Plans.AbstractSpectralBackend end

_have_fsh() =
    Base.get_extension(parentmodule(@__MODULE__), :ScatteringTransformsFastSphericalHarmonicsExt) !== nothing

"""
    fsh_structured_plan(lmax, T)

Fast-path structured-sphere plan constructor. Declaration only — the FastSphericalHarmonics extension
provides the sole method. Call via [`make_structured_plan`](@ref), which guards on the extension.
"""
function fsh_structured_plan end

"""
    make_structured_plan(spectral, lmax, T; rtol, maxiter) -> AbstractSphericalPlan

Build the structured-sphere plan selected by `spectral`. [`DirectSHTBackend`](@ref) is the
dependency-free default (direct SHT on the grid); [`SHTBackend`](@ref) (or `Plans.AutoSpectral` once
the FastSphericalHarmonics extension is loaded) uses the fast exact SHT.
"""
function make_structured_plan(::DirectSHTBackend, lmax, ::Type{T};
                              rtol::Real = 1.0e-8, maxiter::Int = 500) where {T}
    return _direct_structured_plan(lmax, T; rtol = rtol, maxiter = maxiter)
end
function make_structured_plan(::SHTBackend, lmax, ::Type{T}; kwargs...) where {T}
    _have_fsh() || throw(ArgumentError(
        "SHTBackend requires the FastSphericalHarmonics extension. Run `using FastSphericalHarmonics`."))
    return fsh_structured_plan(lmax, T)
end
make_structured_plan(::Plans.AutoSpectral, lmax, ::Type{T}; kwargs...) where {T} =
    _have_fsh() ? fsh_structured_plan(lmax, T) : make_structured_plan(DirectSHTBackend(), lmax, T; kwargs...)

# ---------------------------------------------------------------------------
# Dependency-free spin-1 Riesz field for the pointwise monogenic decomposition.
#
# The Riesz tangent vector is the surface gradient of `g = (−Δ_S)^{-1/2} U⁰`: `u_θ = ∂_θ g`,
# `u_φ = (1/sinθ) ∂_φ g` (equivalently the spin-1 field `ð g`). Given `g`'s real-SH coefficients `gc`,
# the φ-derivative of the `cos/sin(mφ)` factor is analytic; the θ-derivative of the associated Legendre
# part is taken by a central finite difference of the recurrence (robust, convention-free, and accurate
# to ≈ cbrt(eps) for this diagnostic). This is the direct-plan counterpart of NUFSHT's spin-1 synthesis.
# ---------------------------------------------------------------------------
function _riesz_gradient(plan::DirectSHTSphericalPlan{T}, gc::AbstractVector) where {T}
    M, lmax = plan.M, plan.lmax
    h = cbrt(eps(T))
    s2 = sqrt(T(2))
    uθ = zeros(T, M)
    uφ = zeros(T, M)
    Pp = Matrix{T}(undef, lmax + 1, lmax + 1)
    Pm = Matrix{T}(undef, lmax + 1, lmax + 1)
    P0 = Matrix{T}(undef, lmax + 1, lmax + 1)
    @inbounds for n in 1:M
        θn, φn = plan.theta[n], plan.phi[n]
        _assoc_legendre!(Pp, cos(θn + h), lmax)
        _assoc_legendre!(Pm, cos(θn - h), lmax)
        _assoc_legendre!(P0, cos(θn), lmax)
        invs = one(T) / sin(θn)
        aθ = zero(T)
        aφ = zero(T)
        col = 0
        for ℓ in 0:lmax, m in -ℓ:ℓ
            col += 1
            am = abs(m)
            dP = (Pp[ℓ + 1, am + 1] - Pm[ℓ + 1, am + 1]) / (2h)      # ∂_θ P̄_ℓ^|m|
            if m == 0
                aθ += gc[col] * dP
            elseif m > 0
                aθ += gc[col] * s2 * dP * cos(m * φn)
                aφ += gc[col] * s2 * P0[ℓ + 1, am + 1] * (-m * sin(m * φn)) * invs
            else
                aθ += gc[col] * s2 * dP * sin(am * φn)
                aφ += gc[col] * s2 * P0[ℓ + 1, am + 1] * (am * cos(am * φn)) * invs
            end
        end
        uθ[n] = aθ
        uφ[n] = aφ
    end
    return uθ, uφ
end

# Pointwise monogenic decomposition on the direct SHT plan (spin-0 band-pass + spin-1 Riesz gradient).
function direct_monogenic_components(st::SphericalMonogenicScattering{<:Any, <:DirectSHTSphericalPlan},
                                     field::AbstractVector, j::Int)
    plan = st.plan
    σ²hi, σ²lo = st.sigma2[j + 1], st.sigma2[j]
    a = sphere_coeffs(plan, field)                                  # spin-0 SH coefficients
    U0 = similar(plan.theta)
    sphere_apply!(U0, plan, a, band_multiplier(σ²hi, σ²lo))         # band-pass U⁰ (real)
    rp = riesz_potential_multiplier(σ²hi, σ²lo)
    gc = [oftype(a[1], rp(plan.ldeg[k])) * a[k] for k in 1:plan.K]  # coeffs of g = (−Δ_S)^{-1/2} U⁰
    uθ, uφ = _riesz_gradient(plan, gc)
    # NUFSHT's spin-1 synthesis uses ð g = -(∂_θ + i/sinθ ∂_φ)g, so its Riesz vector is the negative of
    # the raw surface gradient; match that sign so the two backends give identical riesz/orientation.
    @. uθ = -uθ
    @. uφ = -uφ
    rnorm = sqrt.(uθ .^ 2 .+ uφ .^ 2)
    amplitude = sqrt.(U0 .^ 2 .+ rnorm .^ 2)
    phase = atan.(rnorm, U0)
    orientation = atan.(uφ, uθ)
    return (; bandpass = U0, riesz = (uθ, uφ), amplitude, phase, orientation)
end

end # module SphericalCore
