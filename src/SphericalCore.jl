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
export SphericalWorkspace, SphericalMonogenicWorkspace
export spherical_scattering!, spherical_monogenic_scattering!
export sphere_coeffs, sphere_coeffs!, sphere_coeffs_buffer, sphere_apply!, sphere_mean

using ..Plans: Plans
using LinearAlgebra: LinearAlgebra
using SpectralBackends: SpectralBackends as SB

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
    sphere_coeffs!(C, plan, field) -> C

In-place [`sphere_coeffs`](@ref): analyse `field` into the pre-allocated coefficient container `C`,
which must have come from `sphere_coeffs_buffer(plan)`. This is what lets the cascade re-analyse a
first-order field without allocating a coefficient vector per scale.
"""
function sphere_coeffs! end

"""
    sphere_coeffs_buffer(plan) -> C

A coefficient container of the right type and size for `plan`, suitable for [`sphere_coeffs!`](@ref).
Backends whose coefficients are not a plain vector (the FastSphericalHarmonics triangular layout, the
NUFSHT dense spin layout) return their own shape.
"""
function sphere_coeffs_buffer end

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

"""
    plan_points(plan) -> (θ, φ) or nothing

The sample locations a spherical plan was built on, or `nothing` when the plan is defined by its
band limit alone (a structured grid). This is what a transform needs to be rebuilt on another
process, where the plan itself cannot travel.
"""
function plan_points end

"""
    plan_weights(plan) -> weights or nothing

The quadrature weights the plan averages with, or `nothing` for a backend that takes the unweighted
sample mean.

These have to be carried, not re-derived: a structured grid's Fejér weights vary by a factor of ~5
across colatitudes, so rebuilding a plan on the same points without them silently substitutes a
uniform average and shifts every coefficient.
"""
function plan_weights end

"""
    plan_solver(plan) -> (; rtol, maxiter)

The iterative-solver settings a plan analyses with, so a rebuilt plan inverts to the same tolerance
rather than to whatever the constructor defaults to.
"""
function plan_solver end

"""
    batch_plan(plan, B) -> plan or nothing

An equivalent plan that transforms `B` co-located fields per call, or `nothing` if this backend has
no batched form. Same points, band limit, weights and solver settings — only the batch width differs.

A backend that returns a plan here lets [`spherical_scattering_batch!`](@ref) issue one transform per
cascade step for the whole stack instead of one per field. Returning `nothing` is not a defect; it
means the per-field loop is the only path, and callers fall back to it.
"""
batch_plan(::Any, ::Integer) = nothing

"""
    supports_batch(plan) -> Bool

Whether [`batch_plan`](@ref) can widen this plan. Asked before building anything, so a caller can
choose between the batched and per-field cascades without paying for a plan it may not use.
"""
supports_batch(::Any) = false

"""
    plan_nufft(plan) -> SpectralBackends.AbstractSpectralBackend

The NUFFT backend a scattered-sphere plan resolved to, or `AutoSpectralBackend()` for a plan that
runs none.

`Plans.spectral_backend` cannot express this: it reports `NUFSHTSpectralBackend` whether the
transform is driven by FINUFFT, NonuniformFFTs, or direct summation. Carrying it separately is what
lets a rebuilt transform — on a distributed worker, say — run the same transform as the original
instead of silently re-resolving to whatever that process happens to have loaded.
"""
plan_nufft(::Any) = SB.AutoSpectralBackend()

"""
    AnalysisNotConverged <: Exception

Thrown when an iterative spherical analysis stops at `maxiter` still above its tolerance.

Such a solve does not return an imprecise answer, it returns a meaningless one — conjugate gradients
on the normal equations can grow without bound, so the coefficients may exceed the field by many
orders of magnitude. Returning them silently would propagate that into every coefficient downstream,
so the analysis refuses instead.
"""
struct AnalysisNotConverged <: Exception
    residual::Float64
    rtol::Float64
    iters::Int
    maxiter::Int
    ntrans::Int
end

function Base.showerror(io::IO, e::AnalysisNotConverged)
    print(io, "AnalysisNotConverged: spherical analysis reached relative residual ", e.residual,
          " after ", e.iters, " of ", e.maxiter, " iterations, against rtol = ", e.rtol,
          " (ntrans = ", e.ntrans, "). ")
    return print(io, "The sampling may not resolve the band limit — accurate analysis needs roughly ",
                 "M ≳ (lmax+1)² well-distributed points — or `maxiter` may be too small.")
end

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
    SphericalScattering{T,P,V}

Spherical scattering transform over a backend `plan::P`. `sigma2[k+1]` is the Gaussian-transfer
variance for the dyadic low-pass cutoff `ℓ_k` (`k=0..J`); band-pass wavelet `j` is
`lowpass(ℓ_j) − lowpass(ℓ_{j-1})`.
"""
struct SphericalScattering{T, P, V <: AbstractVector{T}}
    lmax::Int
    J::Int
    max_order::Int
    plan::P
    sigma2::V
end

"""
    task_local(st) -> st

A copy of the spherical transform safe to run concurrently with the original: the design matrix,
Gram factor and point set are read-only and shared, while the plan's analysis/synthesis scratch is
duplicated by `Plans.task_local_plan`. Analysis writes through that scratch, so tasks sharing one
plan would overwrite each other's coefficients.
"""
task_local(st::SphericalScattering) =
    SphericalScattering(st.lmax, st.J, st.max_order, Plans.task_local_plan(st.plan), st.sigma2)

"""
    SphericalWorkspace{A,C}

Scratch for one spherical cascade: the band-pass output `band`, the current first-order field `u1`,
and two coefficient containers — `C` for the input field, `C1` for the first-order field being
re-analysed. Two is all the cascade ever needs, because it finishes every child of a scale before
starting the next.

Build one with `SphericalWorkspace(st, field)` and reuse it across calls; a task that transforms
concurrently needs its own (and its own `Plans.task_local_plan` of the spherical plan).
"""
struct SphericalWorkspace{A, C}
    band::A
    u1::A
    C::C
    C1::C
end

SphericalWorkspace(st, field::AbstractArray) =
    SphericalWorkspace(similar(field), similar(field),
                       sphere_coeffs_buffer(st.plan), sphere_coeffs_buffer(st.plan))

"""
    (st::SphericalScattering)(field) -> (; S0, S1, S2)

Apply the spherical scattering transform to a scalar `field` sampled by the plan.
`S1[j] = ⟨|field ⋆ ψ_j|⟩`, `S2[j1,j2] = ⟨||field ⋆ ψ_{j1}| ⋆ ψ_{j2}|⟩` for strictly coarser `j2 < j1`.
Allocates a workspace per call; use [`spherical_scattering!`](@ref) to reuse one.
"""
function (st::SphericalScattering{T})(field::AbstractArray) where {T}
    ws = SphericalWorkspace(st, field)
    return spherical_scattering!(zeros(T, st.J), zeros(T, st.J, st.J), st, ws, field)
end

"""
    spherical_scattering!(S1, S2, st, ws, field) -> (; S0, S1, S2)

In-place spherical scattering into pre-allocated `S1`/`S2` using the workspace `ws`
(see [`SphericalWorkspace`](@ref)). Allocation-free once `ws` exists.

Grouped by first-order scale so only one first-order field and one coefficient vector are live at a
time: scale `j1` is band-passed, averaged into `S1[j1]`, analysed once, and then consumed by every
coarser `j2 < j1` before the next `j1` begins.
"""
function spherical_scattering!(S1::AbstractVector, S2::AbstractMatrix,
                               st::SphericalScattering, ws::SphericalWorkspace,
                               field::AbstractArray)
    J = st.J
    S0 = sphere_mean(st.plan, field)
    isempty(S2) || fill!(S2, zero(eltype(S2)))
    sphere_coeffs!(ws.C, st.plan, field)                  # analyse the field once
    for j1 in 1:J
        sphere_apply!(ws.band, st.plan, ws.C, band_multiplier(st.sigma2[j1 + 1], st.sigma2[j1]))
        @. ws.u1 = abs(ws.band)
        S1[j1] = sphere_mean(st.plan, ws.u1)
        (st.max_order >= 2 && j1 > 1) || continue
        sphere_coeffs!(ws.C1, st.plan, ws.u1)             # analyse U1[j1] once, reuse across children
        for j2 in 1:(j1 - 1)
            sphere_apply!(ws.band, st.plan, ws.C1, band_multiplier(st.sigma2[j2 + 1], st.sigma2[j2]))
            @. ws.band = abs(ws.band)
            S2[j1, j2] = sphere_mean(st.plan, ws.band)
        end
    end
    return (S0 = S0, S1 = S1, S2 = S2)
end

"""
    spherical_scattering_batch!(S1, S2, st, ws, X) -> (; S0, S1, S2)

Cascade over a stack of `B` fields sampled at the same points: `X` is `(M, B)`, `S1` is `(J, B)` and
`S2` is `(J, J, B)`. Every transform covers all `B` fields in one call, so the per-call setup is paid
once per cascade step rather than `B` times — `st.plan` must have been built with a matching batch
size for that to hold.
"""
function spherical_scattering_batch!(S1::AbstractMatrix, S2::AbstractArray,
                                     st::SphericalScattering, ws::SphericalWorkspace,
                                     X::AbstractMatrix)
    J = st.J
    S0 = sphere_mean(st.plan, X)
    isempty(S2) || fill!(S2, zero(eltype(S2)))
    sphere_coeffs!(ws.C, st.plan, X)
    for j1 in 1:J
        sphere_apply!(ws.band, st.plan, ws.C, band_multiplier(st.sigma2[j1 + 1], st.sigma2[j1]))
        @. ws.u1 = abs(ws.band)
        S1[j1, :] .= sphere_mean(st.plan, ws.u1)
        (st.max_order >= 2 && j1 > 1) || continue
        sphere_coeffs!(ws.C1, st.plan, ws.u1)
        for j2 in 1:(j1 - 1)
            sphere_apply!(ws.band, st.plan, ws.C1, band_multiplier(st.sigma2[j2 + 1], st.sigma2[j2]))
            @. ws.band = abs(ws.band)
            S2[j1, j2, :] .= sphere_mean(st.plan, ws.band)
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
struct SphericalMonogenicScattering{T, P, V <: AbstractVector{T}}
    lmax::Int
    J::Int
    max_order::Int
    plan::P
    sigma2::V
end

task_local(st::SphericalMonogenicScattering) =
    SphericalMonogenicScattering(st.lmax, st.J, st.max_order, Plans.task_local_plan(st.plan),
                                 st.sigma2)

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
    sphere_coeffs!(w.Cg2, st.plan, w.g2)                                        # re-analyse g²
    sphere_apply!(w.lapg2, st.plan, w.Cg2, laplacian_multiplier())             # Δ_S(g²)
    # |∇_S g|² = ½ Δ_S(g²) − g Δ_S g  (clamp tiny negatives from finite-lmax error)
    @. amp = sqrt(amp^2 + max(zero(T), T(0.5) * w.lapg2 - w.g * w.lapg))
    return amp
end

"""
    SphericalMonogenicWorkspace{A,C,S}

Scratch for one spherical monogenic cascade: the current amplitude field `u1`, a second `amp` for
the order-2 amplitudes, two coefficient containers, and `scratch` — the `(g, lapg, g2, lapg2)`
fields the Bochner identity needs (see [`monogenic_amplitude!`](@ref)).
"""
struct SphericalMonogenicWorkspace{A, C, S}
    u1::A
    amp::A
    C::C
    C1::C
    scratch::S
end

SphericalMonogenicWorkspace(st, field::AbstractArray) = SphericalMonogenicWorkspace(
    similar(field), similar(field),
    sphere_coeffs_buffer(st.plan), sphere_coeffs_buffer(st.plan),
    (g = similar(field), lapg = similar(field), g2 = similar(field), lapg2 = similar(field),
     Cg2 = sphere_coeffs_buffer(st.plan)))

"""
    (st::SphericalMonogenicScattering)(field) -> (; S0, S1, S2)

Apply the spherical monogenic scattering transform. `S1[j] = ⟨A_j⟩`,
`S2[j1,j2] = ⟨A_{j2}[A_{j1}]⟩` for strictly coarser `j2 < j1`. Allocates a workspace per call; use
[`spherical_monogenic_scattering!`](@ref) to reuse one.
"""
function (st::SphericalMonogenicScattering{T})(field::AbstractArray) where {T}
    ws = SphericalMonogenicWorkspace(st, field)
    return spherical_monogenic_scattering!(zeros(T, st.J), zeros(T, st.J, st.J), st, ws, field)
end

"""
    spherical_monogenic_scattering!(S1, S2, st, ws, field) -> (; S0, S1, S2)

In-place spherical monogenic scattering — the monogenic counterpart of
[`spherical_scattering!`](@ref), grouped by first-order scale so one amplitude field is live at a
time rather than all `J`.
"""
function spherical_monogenic_scattering!(S1::AbstractVector, S2::AbstractMatrix,
                                         st::SphericalMonogenicScattering,
                                         ws::SphericalMonogenicWorkspace, field::AbstractArray)
    J = st.J
    S0 = sphere_mean(st.plan, field)
    isempty(S2) || fill!(S2, zero(eltype(S2)))
    sphere_coeffs!(ws.C, st.plan, field)                  # analyse the field once
    for j1 in 1:J
        monogenic_amplitude!(ws.u1, st, ws.C, j1, ws.scratch)
        S1[j1] = sphere_mean(st.plan, ws.u1)
        (st.max_order >= 2 && j1 > 1) || continue
        sphere_coeffs!(ws.C1, st.plan, ws.u1)             # analyse U1[j1] once, reuse across children
        for j2 in 1:(j1 - 1)
            monogenic_amplitude!(ws.amp, st, ws.C1, j2, ws.scratch)
            S2[j1, j2] = sphere_mean(st.plan, ws.amp)
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

# `Y`, `G`, `ldeg`, the point set and the quadrature weights are read-only and shared; the CG and
# synthesis scratch is per-task, because analysis writes its result through it.
plan_points(p::DirectSHTSphericalPlan) = (p.theta, p.phi)
plan_weights(p::DirectSHTSphericalPlan) = p.weights
plan_solver(p::DirectSHTSphericalPlan) = (rtol = p.rtol, maxiter = p.maxiter)
Plans.spectral_backend(::DirectSHTSphericalPlan) = SB.DirectSumSpectralBackend()

Plans.task_local_plan(p::DirectSHTSphericalPlan) = DirectSHTSphericalPlan(
    p.lmax, p.M, p.K, p.Y, p.G, p.ldeg, p.theta, p.phi, p.rtol, p.maxiter, p.weights,
    similar(p.rhs), similar(p.r), similar(p.p), similar(p.Gp), similar(p.ha))

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
    return sphere_coeffs!(Vector{T}(undef, plan.K), plan, field)
end

function sphere_coeffs!(a::AbstractVector, plan::DirectSHTSphericalPlan, field::AbstractArray)
    LinearAlgebra.mul!(plan.rhs, transpose(plan.Y), vec(field))
    _cg_gram!(a, plan)
    return a
end

sphere_coeffs_buffer(plan::DirectSHTSphericalPlan{T}) where {T} = Vector{T}(undef, plan.K)

# Synthesis: out = Σ_k h(ℓ_k)·C_k·Y_k  (apply the per-degree multiplier, then evaluate at the points).
#
# `h` is constant within a degree, and the design matrix's columns are laid out `(ℓ, m = -ℓ:ℓ)` in
# order, so walking that layout evaluates `h` once per degree — `lmax+1` times rather than
# `(lmax+1)²`. `h` is a difference of exponentials, so this replaces `2(lmax+1)²` transcendental
# calls per band with `2(lmax+1)`: measured 6.10 µs -> 0.38 µs at lmax=24, 23.4 µs -> 0.86 µs at
# lmax=48. It is a small share of the call, which the `Y * ha` product below dominates.
function sphere_apply!(out::AbstractArray, plan::DirectSHTSphericalPlan{T}, C, h) where {T}
    @inbounds begin
        k = 1
        for l in 0:plan.lmax
            hl = T(h(l))
            for _ in -l:l
                plan.ha[k] = hl * C[k]
                k += 1
            end
        end
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

# Only `Auto*` needs to ask what is loaded; an explicitly named backend dispatches straight to the
# extension's builder, or to that builder's throwing stub.
_have_nufsht() = Base.get_extension(parentmodule(@__MODULE__), :ScatteringTransformsNUFSHTExt) !== nothing

"""
    nusht_spherical_plan(θ, φ, lmax, T; rtol, maxiter)

Fast-path scattered-sphere plan constructor. The real method lives in the NUFSHT extension; this is
its throwing stub.
"""
nusht_spherical_plan(args...; kwargs...) = throw(ArgumentError(
    "NUFSHTSpectralBackend requires the NUFSHT extension. Run `using NUFSHT`."))

"""
    default_rtol(spectral) -> Real

Conjugate-gradient tolerance for the scattered-sphere analysis solve under `spectral`.

The solve cannot resolve the field better than the transform it is built on, so the useful tolerance
is backend-specific. In-core direct summation evaluates `Y` exactly, so the solve tolerance *is* the
analysis accuracy and `1e-8` buys real digits. NUFSHT analysis instead floors at its own
approximation error — measured against the exact in-core transform, ~2e-4 at `lmax = 8`, `M = 500`
and ~2e-5 at `lmax = 16`, `M = 1500` — and `1e-4` reaches that floor, matching `1e-8`'s accuracy in
up to 3× less time. `1e-2` does not: it lands about twice as far off.
"""
default_rtol(::SB.AbstractSpectralBackend) = 1.0e-8
default_rtol(::SB.AbstractNUFSHTSpectralBackend) = 1.0e-4
default_rtol(::SB.AbstractAutoSpectralBackend) =
    _have_nufsht() ? default_rtol(SB.NUFSHTSpectralBackend()) : default_rtol(SB.DirectSumSpectralBackend())

"""
    make_spherical_plan(spectral, θ, φ, lmax, T; rtol, maxiter) -> AbstractSphericalPlan

Build the scattered-sphere plan selected by `spectral` over points `(θ, φ)` and band limit `lmax`.
`SpectralBackends.DirectSumSpectralBackend` is the dependency-free default;
`SpectralBackends.NUFSHTSpectralBackend` (or `SpectralBackends.AutoSpectralBackend` once the NUFSHT
extension is loaded) uses the NUFSHT fast path.
"""
# The in-core plan transforms one field per call, so `ntrans` is accepted and ignored rather than
# rejected: a caller asking for a batch still gets correct results, just not the batched transform.
function make_spherical_plan(s::SB.AbstractDirectSumSpectralBackend, θ, φ, lmax, ::Type{T};
                             rtol::Real = default_rtol(s), maxiter::Int = 500, weights = nothing,
                             ntrans::Int = 1,
                             nufft::SB.AbstractSpectralBackend = SB.AutoSpectralBackend()) where {T}
    # Refused rather than ignored: this plan evaluates `Y` directly and runs no NUFFT, so honouring a
    # named one is impossible and silently dropping it would misreport what the transform does.
    nufft isa SB.AbstractAutoSpectralBackend || throw(ArgumentError(
        "the in-core spherical plan performs no NUFFT, so `nufft = $nufft` cannot be honoured; " *
        "pass spectral = SpectralBackends.NUFSHTSpectralBackend() for a NUFFT-backed transform."))
    return DirectSHTSphericalPlan(θ, φ, lmax, T; rtol = rtol, maxiter = maxiter, weights = weights)
end
# NUFSHT takes the unweighted sample mean, so it has no quadrature to accept; `weights` is dropped
# rather than silently ignored further down.
make_spherical_plan(::SB.AbstractNUFSHTSpectralBackend, θ, φ, lmax, ::Type{T};
                    weights = nothing, kwargs...) where {T} =
    nusht_spherical_plan(θ, φ, lmax, T; kwargs...)
make_spherical_plan(::SB.AbstractAutoSpectralBackend, θ, φ, lmax, ::Type{T}; kwargs...) where {T} =
    _have_nufsht() ? make_spherical_plan(SB.NUFSHTSpectralBackend(), θ, φ, lmax, T; kwargs...) :
    make_spherical_plan(SB.DirectSumSpectralBackend(), θ, φ, lmax, T; kwargs...)

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

_have_fsh() =
    Base.get_extension(parentmodule(@__MODULE__), :ScatteringTransformsFastSphericalHarmonicsExt) !== nothing

"""
    fsh_structured_plan(lmax, T)

Fast-path structured-sphere plan constructor. The real method lives in the FastSphericalHarmonics
extension; this is its throwing stub.
"""
fsh_structured_plan(args...; kwargs...) = throw(ArgumentError(
    "FSHTSpectralBackend requires the FastSphericalHarmonics extension. " *
    "Run `using FastSphericalHarmonics`."))

"""
    make_structured_plan(spectral, lmax, T; rtol, maxiter) -> AbstractSphericalPlan

Build the structured-sphere plan selected by `spectral`.
`SpectralBackends.DirectSumSpectralBackend` is the dependency-free default (direct SHT on the grid);
`SpectralBackends.FSHTSpectralBackend` (or `SpectralBackends.AutoSpectralBackend` once the
FastSphericalHarmonics extension is loaded) uses the fast exact SHT.
"""
function make_structured_plan(::SB.AbstractDirectSumSpectralBackend, lmax, ::Type{T};
                              rtol::Real = 1.0e-8, maxiter::Int = 500) where {T}
    return _direct_structured_plan(lmax, T; rtol = rtol, maxiter = maxiter)
end
make_structured_plan(::SB.AbstractFSHTSpectralBackend, lmax, ::Type{T}; kwargs...) where {T} =
    fsh_structured_plan(lmax, T)
make_structured_plan(::SB.AbstractAutoSpectralBackend, lmax, ::Type{T}; kwargs...) where {T} =
    _have_fsh() ? fsh_structured_plan(lmax, T) :
    make_structured_plan(SB.DirectSumSpectralBackend(), lmax, T; kwargs...)

# ---------------------------------------------------------------------------
# Dependency-free spin-1 Riesz field for the pointwise monogenic decomposition.
#
# The Riesz tangent vector is the surface gradient of `g = (−Δ_S)^{-1/2} U⁰`: `u_θ = ∂_θ g`,
# `u_φ = (1/sinθ) ∂_φ g` (equivalently the spin-1 field `ð g`). Given `g`'s real-SH coefficients `gc`,
# the φ-derivative of the `cos/sin(mφ)` factor is analytic, and so is the θ-derivative of the
# associated Legendre part — see `_dtheta_pbar`. This is the direct-plan counterpart of NUFSHT's
# spin-1 synthesis.
# ---------------------------------------------------------------------------

# ∂_θ P̄_ℓ^m from same-degree neighbours:
#   ∂_θ P̄_ℓ^m = ½[ √((ℓ−m)(ℓ+m+1))·P̄_ℓ^{m+1} − √((ℓ+m)(ℓ−m+1))·P̄_ℓ^{m−1} ]
# with `P̄_ℓ^{−1} = −P̄_ℓ^{1}` (the normalised form of `P_ℓ^{−m} = (−1)^m (ℓ−m)!/(ℓ+m)!·P_ℓ^m`), which
# collapses the `m = 0` case to `+√(ℓ(ℓ+1))·P̄_ℓ^1`. The sign is opposite the form usually quoted for
# functions without the Condon–Shortley phase, which `_assoc_legendre!` carries in its sectoral step.
#
# The identity is homogeneous in the normalisation, so it holds for `_assoc_legendre!`'s output as
# written — that routine's arbitrary overall scale cancels. One Legendre recurrence per point instead
# of the three a central difference needs, and exact rather than accurate to `cbrt(eps)`.
@inline function _dtheta_pbar(P::AbstractMatrix{T}, ℓ::Int, m::Int) where {T}
    up = (m + 1 > ℓ) ? zero(T) : sqrt(T((ℓ - m) * (ℓ + m + 1))) * P[ℓ + 1, m + 2]
    dn = m == 0 ? -sqrt(T(ℓ * (ℓ + 1))) * (ℓ >= 1 ? P[ℓ + 1, 2] : zero(T)) :
         sqrt(T((ℓ + m) * (ℓ - m + 1))) * P[ℓ + 1, m]
    return (up - dn) / 2
end

function _riesz_gradient(plan::DirectSHTSphericalPlan{T}, gc::AbstractVector) where {T}
    M, lmax = plan.M, plan.lmax
    s2 = sqrt(T(2))
    uθ = zeros(T, M)
    uφ = zeros(T, M)
    P0 = Matrix{T}(undef, lmax + 1, lmax + 1)
    @inbounds for n in 1:M
        θn, φn = plan.theta[n], plan.phi[n]
        _assoc_legendre!(P0, cos(θn), lmax)
        invs = one(T) / sin(θn)
        aθ = zero(T)
        aφ = zero(T)
        col = 0
        for ℓ in 0:lmax, m in -ℓ:ℓ
            col += 1
            am = abs(m)
            dP = _dtheta_pbar(P0, ℓ, am)                             # ∂_θ P̄_ℓ^|m|
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
