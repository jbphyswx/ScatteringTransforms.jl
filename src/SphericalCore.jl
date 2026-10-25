module SphericalCore

# Scattering on S², independent of how the spherical-harmonic transform is performed. A plan
# implements analysis (`sphere_coeffs`, `sphere_coeffs!`), per-degree multiplier and synthesis
# (`sphere_apply!`) and the spherical mean (`sphere_mean`); the difference-of-Gaussians bank and the
# S0/S1/S2 cascade are built on those, and the monogenic amplitude also uses `sphere_riesz_energy!`.
# The in-core plans are defined here; the NUFSHT (scattered points) and FastSphericalHarmonics
# (structured grid) extensions supply fast ones.

export AbstractSphericalPlan, SphericalScattering, SphericalMonogenicScattering
export SphericalWorkspace, SphericalMonogenicWorkspace
export spherical_scattering!, spherical_monogenic_scattering!
export sphere_coeffs, sphere_coeffs!, sphere_coeffs_buffer, sphere_apply!, sphere_mean

using ..Plans: Plans
using LinearAlgebra: LinearAlgebra
using SpectralBackends: SpectralBackends as SB
using FlowTransformBindings: FlowTransformBindings as FTB

# ---------------------------------------------------------------------------
# Backend interface. Analysis (`sphere_coeffs`) is separate from multiplier-and-synthesis
# (`sphere_apply!`), so the cascade analyses each field once and reuses its coefficients across every
# band.
# ---------------------------------------------------------------------------

"""
    AbstractSphericalPlan

Supertype for spherical spectral plans (scattered NUFSHT, structured SHT, …). A concrete plan must
implement [`sphere_coeffs`](@ref), [`sphere_apply!`](@ref), and [`sphere_mean`](@ref).
"""
abstract type AbstractSphericalPlan end

"""
    sphere_coeffs(plan, field) -> C

Spherical-harmonic analysis: the coefficients `C` of `field` up to the plan's band limit. For a
structured grid this is the forward SHT; for scattered points it is the least-squares inversion of
synthesis. `C` is opaque and valid only with the same `plan`'s interface functions, such as
[`sphere_apply!`](@ref).
"""
function sphere_coeffs end

"""
    sphere_coeffs!(C, plan, field) -> C

In-place [`sphere_coeffs`](@ref): analyse `field` into the pre-allocated coefficient container `C`,
which must have come from `sphere_coeffs_buffer(plan)`.
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

Spherical average of `field` under the plan's quadrature: for scattered points the sample mean, or
the weighted mean when the plan carries weights; for a structured grid the exact quadrature
integral. Provided by each backend.
"""
function sphere_mean end

"""
    sphere_riesz_energy!(out, plan, C, h, scratch) -> out

`|∇_S g|²` at the plan's samples for `g = Σ_{ℓm} h(ℓ)·C_{ℓm}·Y_{ℓm}`, exact for coefficients of
degree `≤ lmax`. `scratch` comes from [`riesz_scratch`](@ref). Does not mutate `C`.
"""
function sphere_riesz_energy! end

"""
    riesz_scratch(plan, field) -> scratch

The scratch [`sphere_riesz_energy!`](@ref) needs for `plan` and fields shaped like `field`.
"""
function riesz_scratch end

# Real `sph_mode` layout (FastSphericalHarmonics, NUFSHT): `(ℓ, m)` sits at row `ℓ - |m| + 1`, column
# `2|m| + (m ≥ 0)`, over orthonormal real harmonics with `cos(mφ)` for `m > 0`, `sin(|m|φ)` for `m < 0`
# and no Condon–Shortley phase.
@inline _sph_slot(ℓ::Int, m::Int) = (ℓ - abs(m) + 1, 2abs(m) + (m >= 0))

sph_layout_riesz_scratch(plan, field::AbstractArray, lmax::Int) =
    (D = sphere_coeffs_buffer(plan), f = similar(field),
     a = Vector{Complex{real(eltype(field))}}(undef, 2lmax + 1),
     b = Vector{Complex{real(eltype(field))}}(undef, 2lmax + 1))

"""
    sph_layout_riesz_energy!(out, plan, C, h, lmax, scratch) -> out

[`sphere_riesz_energy!`](@ref) for coefficients in the real `sph_mode` layout, by three scalar
syntheses through `sphere_apply!`. With `L = -i r×∇`, a real `g` has `|∇_S g|² = |L₊g|² + (∂_φ g)²`,
and both operators act within each degree: `L₊Y_{ℓm} = √((ℓ-m)(ℓ+m+1)) Y_{ℓ,m+1}` on the
Condon–Shortley complex harmonics, and `∂_φ` maps `cos(mφ) ↦ -m sin(mφ)`, `sin(mφ) ↦ m cos(mφ)`.
`L₊g = P + iQ` is synthesised as its real and imaginary parts.
"""
function sph_layout_riesz_energy!(out, plan, C, h, lmax::Int, scr)
    D, f, a, b = scr.D, scr.f, scr.a, scr.b
    T = real(eltype(f))
    s2 = sqrt(T(2))
    one_ = Returns(one(T))
    # ∂_φ g.
    fill!(D, zero(eltype(D)))
    @inbounds for ℓ in 1:lmax
        hl = T(h(ℓ))
        for m in 1:ℓ
            D[_sph_slot(ℓ, m)...] = m * hl * C[_sph_slot(ℓ, -m)...]
            D[_sph_slot(ℓ, -m)...] = -m * hl * C[_sph_slot(ℓ, m)...]
        end
    end
    sphere_apply!(f, plan, D, one_)
    @. out = f^2
    # Re and Im of L₊g.
    for imagpart in (false, true)
        fill!(D, zero(eltype(D)))
        @inbounds for ℓ in 1:lmax
            _raise!(a, b, C, T(h(ℓ)), ℓ, s2)
            for m in 0:ℓ
                sg = isodd(m) ? -one(T) : one(T)
                bm, bn = b[ℓ + 1 + m], b[ℓ + 1 - m]
                # Coefficients of the real fields (L₊g ± conj(L₊g))/2 and /2i.
                v = imagpart ? (bm - sg * conj(bn)) / (2im) : (bm + sg * conj(bn)) / 2
                if m == 0
                    D[_sph_slot(ℓ, 0)...] = real(v)
                else
                    ap = sg * v
                    D[_sph_slot(ℓ, m)...] = s2 * real(ap)
                    D[_sph_slot(ℓ, -m)...] = -s2 * imag(ap)
                end
            end
        end
        sphere_apply!(f, plan, D, one_)
        @. out += f^2
    end
    return out
end

# `b = L₊a` on degree `ℓ`, with `a` the Condon–Shortley coefficients of `hl·g`: from the real
# coefficients, `a_m = (-1)^m (c_m - i c_{-m})/√2` and `a_{-m} = (c_m + i c_{-m})/√2` for `m > 0`.
# Index `m + ℓ + 1` holds order `m`.
@inline function _raise!(a, b, C, hl::T, ℓ::Int, s2::T) where {T}
    @inbounds begin
        a[ℓ + 1] = complex(hl * C[_sph_slot(ℓ, 0)...])
        for m in 1:ℓ
            ap = complex(hl * C[_sph_slot(ℓ, m)...], -hl * C[_sph_slot(ℓ, -m)...]) / s2
            a[ℓ + 1 + m] = isodd(m) ? -ap : ap
            a[ℓ + 1 - m] = conj(ap)
        end
        b[1] = zero(eltype(b))
        for m in -ℓ:(ℓ - 1)
            b[m + ℓ + 2] = sqrt(T((ℓ - m) * (ℓ + m + 1))) * a[m + ℓ + 1]
        end
    end
    return b
end

"""
    sphere_plan_at(plan, lmax) -> plan or nothing

A plan of the same kind at the lower band limit `lmax`, or `nothing` when this backend cannot be
narrowed, in which case every band is synthesised at the full band limit. This is the spherical
form of decimation: band-pass wavelet `j` is confined to degrees `≲ ℓ_j`.
"""
sphere_plan_at(::Any, ::Integer) = nothing

"""
    sphere_lmax(plan) -> Int

The band limit of `plan`. Every backend that defines [`sphere_plan_at`](@ref) defines this.
"""
function sphere_lmax end

"""
    sphere_restrict!(Cdst, pdst, Csrc, psrc) -> Cdst

Copy the spherical-harmonic coefficients of degree `≤ pdst`'s band limit out of `Csrc` (laid out for
`psrc`) into `Cdst` (laid out for `pdst`), zeroing anything `Cdst` holds beyond them. Coefficient
layout is backend-specific, so each backend defines this.
"""
function sphere_restrict! end

"""
    sphere_field_buffer(plan) -> field

A grid container of the right type and size for `plan`, for a band synthesised at that plan's band
limit. Defaults to [`sphere_coeffs_buffer`](@ref), which is the same shape on backends whose
coefficients and samples share a layout.
"""
sphere_field_buffer(plan) = sphere_coeffs_buffer(plan)

"""
    band_lmax(lmax, J, j, headroom) -> Int

Band limit that carries wavelet `j`: `headroom · ℓ_j`, clamped to `lmax`; `headroom = Inf` keeps
`lmax`.

Band `j` is a difference of Gaussians whose transfer above its cutoff is
`exp(-ℓ(ℓ+1) / 2ℓ_j(ℓ_j+1))`: `0.135` at `2ℓ_j`, `0.011` at `3ℓ_j`, `3.4e-4` at `4ℓ_j`, the
largest value a given `headroom` discards.
"""
function band_lmax(lmax::Integer, J::Integer, j::Integer, headroom::Real)
    isfinite(headroom) || return Int(lmax)
    ℓj = lmax / 2.0^(J - j)
    return min(Int(lmax), max(1, ceil(Int, headroom * ℓj)))
end

"""
    plan_points(plan) -> (θ, φ) or nothing

The sample locations a spherical plan was built on, from which a transform is rebuilt on another
process, or `nothing` when the plan is defined by its band limit alone (a structured grid).
"""
function plan_points end

"""
    plan_weights(plan) -> weights or nothing

The quadrature weights the plan averages with, or `nothing` for a backend that takes the unweighted
sample mean or whose quadrature follows from its grid. A plan rebuilt on the same points needs these
too: the in-core structured plan's Fejér weights are not uniform.
"""
function plan_weights end

"""
    plan_solver(plan) -> (; rtol, maxiter, solve)

The analysis settings of a scattered-sphere plan, as its constructor takes them: `solve`, whether it
fits the coefficients by least squares or takes the weighted adjoint, and the fit's tolerances. `nothing`
for a plan whose analysis is an exact transform.
"""
function plan_solver end

"""
    batch_plan(plan, B) -> plan or nothing

An equivalent plan that transforms `B` co-located fields per call, for
[`spherical_scattering_batch!`](@ref), or `nothing` if this backend has no batched form. Only the
batch width differs from `plan`.

Returns `plan` itself when `Plans.batch_width(plan) == B`, and otherwise a plan the caller owns and
closes with `Plans.close_plan!`. Concurrent tasks use [`task_local_batch_plan`](@ref).
"""
batch_plan(::Any, ::Integer) = nothing

"""
    task_local_batch_plan(plan, B) -> plan or nothing

A `B`-wide plan safe to apply while `plan` is applied in another task, owned by the caller, or
`nothing` when the backend has no batched form.

A spherical plan's analysis solves through its own scratch, so tasks cannot share one. At `plan`'s
own width this is `Plans.task_local_plan(plan)`, and at any other width [`batch_plan`](@ref) builds a
fresh one.
"""
function task_local_batch_plan(plan, B::Integer)
    supports_batch(plan) || return nothing
    return Plans.batch_width(plan) == B ? Plans.task_local_plan(plan) : batch_plan(plan, B)
end

"""
    supports_batch(plan) -> Bool

Whether [`batch_plan`](@ref) can widen this plan, answered without building one.
"""
supports_batch(::Any) = false

"""
    plan_nufft(plan) -> SpectralBackends.AbstractSpectralBackend

The NUFFT backend a scattered-sphere plan resolved to, or `AutoSpectralBackend()` for a plan that
runs none. `Plans.spectral_backend` reports `NUFSHTSpectralBackend` whichever NUFFT runs underneath,
so a transform rebuilt on another process needs this too to run the same one.
"""
plan_nufft(::Any) = SB.AutoSpectralBackend()

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

# g = (−Δ_S)^{-1/2} of the band:  h(ℓ) = b_j(ℓ)/√(ℓ(ℓ+1)),  0 at ℓ=0, in the variances' type.
riesz_potential_multiplier(σ²hi::T, σ²lo::T) where {T} =
    ℓ -> ℓ == 0 ? zero(T) : _bj(ℓ, σ²hi, σ²lo) / sqrt(T(ℓ * (ℓ + 1)))

# ---------------------------------------------------------------------------
# Transforms, generic over the plan interface above
# ---------------------------------------------------------------------------

"""
    SphericalScattering{T,P,V,BV}

Spherical scattering transform over a backend `plan::P`. `sigma2[k+1]` is the Gaussian-transfer
variance for the dyadic low-pass cutoff `ℓ_k` (`k=0..J`); band-pass wavelet `j` is
`lowpass(ℓ_j) − lowpass(ℓ_{j-1})`. `bands` holds one narrowed plan per scale
([`band_plans`](@ref)), or `nothing`.
"""
struct SphericalScattering{T, P, V <: AbstractVector{T}, BV}
    lmax::Int
    J::Int
    max_order::Int
    plan::P
    sigma2::V
    # Per-scale narrowed plans, or `nothing` when the backend cannot narrow (or the caller asked for
    # no narrowing). `nothing` runs the full-band-limit cascade unchanged.
    bands::BV
end

SphericalScattering(lmax::Int, J::Int, max_order::Int, plan, sigma2) =
    SphericalScattering(lmax, J, max_order, plan, sigma2, nothing)

Base.show(io::IO, st::SphericalScattering{T}) where {T} =
    print(io, "SphericalScattering{", T, "}(lmax=", st.lmax, ", J=", st.J, ", max_order=", st.max_order,
          ", ", st.plan, st.bands === nothing ? "" : ", per-scale band limits", ")")

"""
    band_plans(plan, lmax, ls) -> Vector or nothing

One plan per scale, scale `j` at band limit `ls[j]`, or `nothing` if this backend cannot narrow or
no band needs narrowing. A scale at the full band limit `lmax` reuses `plan`.
"""
function band_plans(plan, lmax::Integer, ls::AbstractVector{<:Integer})
    all(>=(lmax), ls) && return nothing
    bands = map(l -> l >= lmax ? plan : sphere_plan_at(plan, l), ls)
    return any(isnothing, bands) ? nothing : bands
end

"""
    task_local(st) -> st

A copy of `st` safe to run concurrently with it: the plan's read-only tables are shared and its
analysis and synthesis scratch duplicated by `Plans.task_local_plan`. Release it with
[`close_task_local!`](@ref).
"""
function task_local(st::SphericalScattering)
    plan = Plans.task_local_plan(st.plan)
    # Each narrowed plan carries its own synthesis scratch, so a task needs its own — except where
    # the scale reuses `st.plan` itself, which must go on aliasing the copy.
    bands = st.bands === nothing ? nothing :
            map(p -> p === st.plan ? plan : Plans.task_local_plan(p), st.bands)
    return SphericalScattering(st.lmax, st.J, st.max_order, plan, st.sigma2, bands)
end

"""
    close_task_local!(stl, st) -> nothing

`Plans.close_plan!` every plan [`task_local`](@ref)`(st)` built for `stl`; plans `stl` shares with
`st` stay open.
"""
function close_task_local!(stl::SphericalScattering, st::SphericalScattering)
    _close_unshared!(stl.plan, st.plan)
    stl.bands === nothing || foreach(_close_unshared!, stl.bands, st.bands)
    return nothing
end

_close_unshared!(p, shared) = (p === shared || Plans.close_plan!(p); nothing)

"""
    SphericalWorkspace{A,C,AV,CV}

Scratch for one spherical cascade: the band-pass output `band`, the current first-order field `u1`,
and two coefficient containers, `C` for the input field and `C1` for the first-order field being
re-analysed. Two suffice, since the cascade finishes every child of a scale before starting the
next. `bband`, `bu1`, `bC` and `bC1` are the same per scale at its narrowed band limit, and empty
when the transform has no `bands`.

Build one with `SphericalWorkspace(st, field)` and reuse it across calls; a task that transforms
concurrently needs its own, with a [`task_local`](@ref) transform.
"""
struct SphericalWorkspace{A, C, AV, CV}
    band::A
    u1::A
    C::C
    C1::C
    bband::AV
    bu1::AV
    bC::CV
    bC1::CV
end

function SphericalWorkspace(st, field::AbstractArray)
    C() = sphere_coeffs_buffer(st.plan)
    if st.bands === nothing
        e = similar(field, 0)
        ec = [C()][1:0]
        return SphericalWorkspace(similar(field), similar(field), C(), C(),
                                  typeof(e)[], typeof(e)[], ec, ec)
    end
    bband = [sphere_field_buffer(p) for p in st.bands]
    bu1 = [sphere_field_buffer(p) for p in st.bands]
    bC = [sphere_coeffs_buffer(p) for p in st.bands]
    bC1 = [sphere_coeffs_buffer(p) for p in st.bands]
    return SphericalWorkspace(similar(field), similar(field), C(), C(), bband, bu1, bC, bC1)
end

Base.show(io::IO, ws::SphericalWorkspace) =
    print(io, "SphericalWorkspace(fields of size ", size(ws.band), ", ", length(ws.bband), " band buffers)")

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
    st.bands === nothing || return _banded_cascade!(S1, S2, st, ws, S0)
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

# The same cascade with every band synthesised at its own band limit. `ws.C` already holds the
# field's coefficients at the full band limit.
#
# Scale `j2 < j1` reuses its own scale's buffers, which its first-order pass has already finished
# with — the cascade visits `j1` in increasing order, so those are dead by the time a child needs
# them.
function _banded_cascade!(S1, S2, st::SphericalScattering, ws::SphericalWorkspace, S0)
    bands = st.bands
    for j1 in 1:st.J
        p1 = bands[j1]
        h1 = band_multiplier(st.sigma2[j1 + 1], st.sigma2[j1])
        sphere_restrict!(ws.bC[j1], p1, ws.C, st.plan)
        sphere_apply!(ws.bband[j1], p1, ws.bC[j1], h1)
        @. ws.bu1[j1] = abs(ws.bband[j1])
        S1[j1] = sphere_mean(p1, ws.bu1[j1])
        (st.max_order >= 2 && j1 > 1) || continue
        sphere_coeffs!(ws.bC1[j1], p1, ws.bu1[j1])
        for j2 in 1:(j1 - 1)
            p2 = bands[j2]
            h2 = band_multiplier(st.sigma2[j2 + 1], st.sigma2[j2])
            sphere_restrict!(ws.bC[j2], p2, ws.bC1[j1], p1)
            sphere_apply!(ws.bband[j2], p2, ws.bC[j2], h2)
            @. ws.bband[j2] = abs(ws.bband[j2])
            S2[j1, j2] = sphere_mean(p2, ws.bband[j2])
        end
    end
    return (S0 = S0, S1 = S1, S2 = S2)
end

"""
    spherical_scattering_batch!(S1, S2, st, ws, X) -> (; S0, S1, S2)

Cascade over a stack of `B` fields sampled at the same points: `X` is `(M, B)`, `S1` is `(J, B)` and
`S2` is `(J, J, B)`. Every transform covers all `B` fields in one call; `st.plan` must have batch
width `B`.
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
    SphericalMonogenicScattering{T,P,V}

Spherical monogenic scattering: shares the dyadic difference-of-Gaussians bands of
[`SphericalScattering`](@ref) but replaces the analytic modulus with the spherical monogenic
amplitude `A_j = √(U⁰_j² + |∇_S g_j|²)` (see [`monogenic_amplitude!`](@ref)).
"""
struct SphericalMonogenicScattering{T, P, V <: AbstractVector{T}}
    lmax::Int
    J::Int
    max_order::Int
    plan::P
    sigma2::V
end

Base.show(io::IO, st::SphericalMonogenicScattering{T}) where {T} =
    print(io, "SphericalMonogenicScattering{", T, "}(lmax=", st.lmax, ", J=", st.J, ", max_order=",
          st.max_order, ", ", st.plan, ")")

task_local(st::SphericalMonogenicScattering) =
    SphericalMonogenicScattering(st.lmax, st.J, st.max_order, Plans.task_local_plan(st.plan),
                                 st.sigma2)

close_task_local!(stl::SphericalMonogenicScattering, st::SphericalMonogenicScattering) =
    _close_unshared!(stl.plan, st.plan)

"""
    monogenic_amplitude!(amp, st, C, j, w) -> amp

Spherical monogenic amplitude at scale `j` of the field whose (already-computed) SH coefficients are
`C`, written into `amp`. `w` holds the Riesz energy field `r2` and the plan's `riesz` scratch.

With `g_j = (−Δ_S)^{-1/2} U⁰_j` the spin-1 Riesz field has `|U^R_j| = |∇_S g_j|`, so
`A_j = √(U⁰_j² + |∇_S g_j|²)`, the energy evaluated by [`sphere_riesz_energy!`](@ref). The Riesz
*vector* (orientation/phase) is `spherical_monogenic_components`.
"""
function monogenic_amplitude!(amp::AbstractArray, st::SphericalMonogenicScattering,
                              C, j::Int, w)
    σ²hi = st.sigma2[j + 1]
    σ²lo = st.sigma2[j]
    sphere_apply!(amp, st.plan, C, band_multiplier(σ²hi, σ²lo))                         # U⁰
    sphere_riesz_energy!(w.r2, st.plan, C, riesz_potential_multiplier(σ²hi, σ²lo), w.riesz)
    @. amp = sqrt(amp^2 + w.r2)
    return amp
end

"""
    SphericalMonogenicWorkspace{A,C,S}

Scratch for one spherical monogenic cascade: the current amplitude field `u1`, a second `amp` for
the order-2 amplitudes, two coefficient containers, and `scratch` — the Riesz energy field and the
plan's [`riesz_scratch`](@ref) (see [`monogenic_amplitude!`](@ref)).
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
    (r2 = similar(field), riesz = riesz_scratch(st.plan, field)))

Base.show(io::IO, ws::SphericalMonogenicWorkspace) =
    print(io, "SphericalMonogenicWorkspace(fields of size ", size(ws.u1), ")")

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

In-place spherical monogenic scattering, the monogenic counterpart of
[`spherical_scattering!`](@ref), grouped by first-order scale with one amplitude field live at a
time.
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
# In-core scattered-sphere plan: `O(M·(lmax+1)²)` per transform, `O(M + (lmax+1)²)` memory. A field's
# real spherical-harmonic coefficients, laid out `(ℓ, m = -ℓ:ℓ)` in order, are fitted by LSMR to the
# synthesis
#
#     f(θ, φ) = Σ_ℓ P̄_ℓ^0(cosθ) a_ℓ0 + √2 Σ_ℓ Σ_{m=1}^{ℓ} P̄_ℓ^m(cosθ) (a_ℓm cos mφ + a_ℓ,-m sin mφ),
#
# summed directly at the points. Points of one colatitude form a ring and share its Legendre values:
# per ring the order sums `Σ_ℓ P̄_ℓ^m a_ℓ,±m` cost `O((lmax+1)²)`, and per point the sum over orders
# `O(lmax)`, with `cos mφ, sin mφ` by rotation from `cos φ, sin φ`.
# ---------------------------------------------------------------------------

# `P̄_ℓ^m = √((2ℓ+1)(ℓ−m)!/(ℓ+m)!)·P_ℓ^m`, with the Condon–Shortley phase, has the recurrences
# `P̄_m^m = sectoral[m]·sinθ·P̄_{m-1}^{m-1}`, `P̄_{m+1}^m = subdiagonal[m+1]·cosθ·P̄_m^m` and
# `P̄_ℓ^m = a[ℓ+1, m+1]·cosθ·P̄_{ℓ-1}^m − b[ℓ+1, m+1]·P̄_{ℓ-2}^m` for `ℓ ≥ m + 2`, whose values stay
# `O(1)`; these are their coefficients up to a band limit.
struct _LegendreTable{T, V <: AbstractVector{T}, M <: AbstractMatrix{T}}
    lmax::Int
    sectoral::V
    subdiagonal::V
    a::M
    b::M
end

function _LegendreTable(::Type{T}, lmax::Int) where {T}
    sectoral = T[-sqrt((2m + one(T)) / (2m)) for m in 1:lmax]
    subdiagonal = T[sqrt(2m + T(3)) for m in 0:(lmax - 1)]
    a = zeros(T, lmax + 1, lmax + 1)
    b = zeros(T, lmax + 1, lmax + 1)
    for m in 0:lmax, ℓ in (m + 2):lmax
        a[ℓ + 1, m + 1] = sqrt((2ℓ - one(T)) * (2ℓ + one(T)) / ((ℓ - m) * (ℓ + m)))
        b[ℓ + 1, m + 1] = sqrt((2ℓ + one(T)) * (ℓ + m - one(T)) * (ℓ - m - one(T)) /
                               ((2ℓ - T(3)) * (ℓ + m) * (ℓ - m)))
    end
    return _LegendreTable(lmax, sectoral, subdiagonal, a, b)
end

# `P[ℓ+1, m+1] = P̄_ℓ^m` for `0 ≤ m ≤ ℓ ≤ lmax` at `x = cosθ`, `u = sinθ`; entries with `ℓ < m` are left
# as they were. The overall scale is arbitrary, since analysis and synthesis share it.
function _assoc_legendre!(P::AbstractMatrix{T}, tab::_LegendreTable, x::T, u::T) where {T}
    lmax = tab.lmax
    @inbounds begin
        P[1, 1] = one(T)
        for m in 1:lmax
            P[m + 1, m + 1] = tab.sectoral[m] * u * P[m, m]
        end
        for m in 0:(lmax - 1)
            P[m + 2, m + 1] = tab.subdiagonal[m + 1] * x * P[m + 1, m + 1]
            for ℓ in (m + 2):lmax
                P[ℓ + 1, m + 1] = tab.a[ℓ + 1, m + 1] * x * P[ℓ, m + 1] -
                                  tab.b[ℓ + 1, m + 1] * P[ℓ - 1, m + 1]
            end
        end
    end
    return P
end

# The coefficient slot of `(ℓ, m)`.
@inline _lm(ℓ::Int, m::Int) = ℓ * ℓ + ℓ + m + 1

# A direct plan's points grouped by colatitude: ring `r` holds the points `pts[ptr[r]:ptr[r+1]-1]`, at
# `cosθ = rcos[r]`, `sinθ = rsin[r]`; `cφ`, `sφ` hold each point's `cos φ`, `sin φ`.
struct _SHTRings{T, V <: AbstractVector{T}, I <: AbstractVector{Int}}
    rcos::V
    rsin::V
    ptr::I
    pts::I
    cφ::V
    sφ::V
end

function _SHTRings(θ::Vector{T}, φ::Vector{T}) where {T}
    order = sortperm(θ)
    ptr = Int[]
    rcos, rsin = T[], T[]
    for t in eachindex(order)
        n = order[t]
        if t == 1 || θ[n] != θ[order[t - 1]]
            push!(ptr, t)
            s, c = sincos(θ[n])
            push!(rsin, s)
            push!(rcos, c)
        end
    end
    push!(ptr, length(order) + 1)
    sφ, cφ = similar(φ), similar(φ)
    @inbounds for n in eachindex(φ)
        sφ[n], cφ[n] = sincos(φ[n])
    end
    return _SHTRings(rcos, rsin, ptr, order, cφ, sφ)
end

_nrings(r::_SHTRings) = length(r.ptr) - 1

# What a direct plan's transforms write through, per task: the Legendre values of the current ring,
# its order sums for each of `B` columns (`(B, lmax+1)`, column-contiguous), and one point's sum per
# column.
struct _SHTScratch{T, M <: AbstractMatrix{T}, V <: AbstractVector{T}}
    P::M
    gc::M
    gs::M
    acc::V
end

_SHTScratch(::Type{T}, lmax::Int, B::Int) where {T} =
    _SHTScratch(Matrix{T}(undef, lmax + 1, lmax + 1), Matrix{T}(undef, B, lmax + 1),
                Matrix{T}(undef, B, lmax + 1), Vector{T}(undef, B))

# `f[:, b] = Σ_ℓm a[_lm(ℓ, m), b]·Y_ℓm` at every point, for the first `nb` of the `(lmax+1)²`-long
# columns of `a`, into the `M`-long columns of `f`. Both are indexed linearly, so a field may have any
# shape holding its `M` values.
function _sht_synthesize!(f, rings::_SHTRings{T}, tab::_LegendreTable, s::_SHTScratch,
                          a, nb::Int) where {T}
    lmax = tab.lmax
    M = length(rings.pts)
    K = (lmax + 1)^2
    P, gc, gs, acc = s.P, s.gc, s.gs, s.acc
    s2 = sqrt(T(2))
    @inbounds for r in 1:_nrings(rings)
        _assoc_legendre!(P, tab, rings.rcos[r], rings.rsin[r])
        for b in 1:nb
            o = (b - 1) * K
            c = zero(T)
            for ℓ in 0:lmax
                c += P[ℓ + 1, 1] * a[o + _lm(ℓ, 0)]
            end
            gc[b, 1] = c
        end
        for m in 1:lmax, b in 1:nb
            o = (b - 1) * K
            c = zero(T)
            sn = zero(T)
            for ℓ in m:lmax
                p = P[ℓ + 1, m + 1]
                c += p * a[o + _lm(ℓ, m)]
                sn += p * a[o + _lm(ℓ, -m)]
            end
            gc[b, m + 1] = s2 * c
            gs[b, m + 1] = s2 * sn
        end
        for t in rings.ptr[r]:(rings.ptr[r + 1] - 1)
            n = rings.pts[t]
            c1, s1 = rings.cφ[n], rings.sφ[n]
            for b in 1:nb
                acc[b] = gc[b, 1]
            end
            cm, sm = one(T), zero(T)
            for m in 1:lmax
                cm, sm = cm * c1 - sm * s1, sm * c1 + cm * s1
                for b in 1:nb
                    acc[b] += cm * gc[b, m + 1] + sm * gs[b, m + 1]
                end
            end
            for b in 1:nb
                f[(b - 1) * M + n] = acc[b]
            end
        end
    end
    return f
end

# The adjoint of `_sht_synthesize!`: `v[_lm(ℓ, m), b] = Σ_n u[n, b]·Y_ℓm(n)`, the same sums in the
# transposed order.
function _sht_adjoint!(v, rings::_SHTRings{T}, tab::_LegendreTable, s::_SHTScratch,
                       u, nb::Int) where {T}
    lmax = tab.lmax
    M = length(rings.pts)
    K = (lmax + 1)^2
    P, hc, hs = s.P, s.gc, s.gs
    s2 = sqrt(T(2))
    @inbounds begin
        for i in 1:(K * nb)
            v[i] = zero(eltype(v))
        end
        for r in 1:_nrings(rings)
            for m in 0:lmax, b in 1:nb
                hc[b, m + 1] = zero(T)
                hs[b, m + 1] = zero(T)
            end
            for t in rings.ptr[r]:(rings.ptr[r + 1] - 1)
                n = rings.pts[t]
                c1, s1 = rings.cφ[n], rings.sφ[n]
                for b in 1:nb
                    hc[b, 1] += u[(b - 1) * M + n]
                end
                cm, sm = one(T), zero(T)
                for m in 1:lmax
                    cm, sm = cm * c1 - sm * s1, sm * c1 + cm * s1
                    for b in 1:nb
                        x = u[(b - 1) * M + n]
                        hc[b, m + 1] += cm * x
                        hs[b, m + 1] += sm * x
                    end
                end
            end
            _assoc_legendre!(P, tab, rings.rcos[r], rings.rsin[r])
            for b in 1:nb
                o = (b - 1) * K
                c = hc[b, 1]
                for ℓ in 0:lmax
                    v[o + _lm(ℓ, 0)] += P[ℓ + 1, 1] * c
                end
            end
            for m in 1:lmax, b in 1:nb
                o = (b - 1) * K
                c, sn = s2 * hc[b, m + 1], s2 * hs[b, m + 1]
                for ℓ in m:lmax
                    p = P[ℓ + 1, m + 1]
                    v[o + _lm(ℓ, m)] += p * c
                    v[o + _lm(ℓ, -m)] += p * sn
                end
            end
        end
    end
    return v
end

# The synthesis as the operator of the least-squares analysis, on `B` columns: `t` holds `A v` and `w`
# holds `A† u`.
struct _SHTOperator{R, LT, S, TA, WA}
    rings::R
    tab::LT
    scr::S
    B::Int
    t::TA
    w::WA
end

FTB.lsmr_ncolumns(op::_SHTOperator) = op.B
FTB.lsmr_allocate_domain(op::_SHTOperator) = fill!(similar(op.w), 0)
FTB.lsmr_allocate_range(op::_SHTOperator) = fill!(similar(op.t), 0)
FTB.lsmr_forward!(u, op::_SHTOperator, v, c, n) =
    FTB.colxpby!(u, _sht_synthesize!(op.t, op.rings, op.tab, op.scr, v, n), c, n, op.B)
FTB.lsmr_adjoint!(v, op::_SHTOperator, u, c, n) =
    FTB.colxpby!(v, _sht_adjoint!(op.w, op.rings, op.tab, op.scr, u, n), c, n, op.B)

"""
    DirectSHTSphericalPlan{T,B}

The in-core scattered-sphere plan over `M` points at band limit `lmax`, transforming `B` co-located
fields per call. Analysis is the least-squares fit of the real spherical harmonics of degree
`≤ lmax` by LSMR (`FlowTransformBindings.lsmr!`), weighted by the quadrature weights `w`, which sum to 1:
`min Σⱼ wⱼ (fⱼ − Σ a_ℓm Y_ℓm(xⱼ))²`. With `solve = false` it is the weighted adjoint
`a_ℓm = Σⱼ wⱼ fⱼ Y_ℓm(xⱼ)` over harmonics of mean square 1. Synthesis is their direct sum at the points;
neither forms the `M × (lmax+1)²` matrix of harmonics, so memory is `O(M + (lmax+1)²)`.
"""
struct DirectSHTSphericalPlan{T, B, VV <: AbstractVector{T}, R, LT, OP, FO, WS, HA} <: AbstractSphericalPlan
    lmax::Int
    M::Int
    K::Int
    theta::VV        # (M) colatitudes
    phi::VV          # (M) longitudes
    weights::VV      # (M) spatial-mean quadrature weights, summing to 1
    rtol::T
    maxiter::Int
    solve::Bool      # the least-squares fit, or the weighted adjoint
    rings::R         # read-only, shared with every copy
    tab::LT          # read-only, shared with every copy
    op::OP           # the transforms' operator and scratch
    fit::FO          # `op` under the weights, the operator the fit runs on
    ls::WS           # the fit's `FTB.LSMRWorkspace`
    ha::HA           # (K[, B]) coefficients times a degree multiplier, for `sphere_apply!`
end

function _direct_sht_plan(θ::V, φ::V, w::V, rtol::T, maxiter::Int, solve::Bool, rings, tab, lmax::Int,
                          B::Int) where {T, V <: AbstractVector{T}}
    M, K = length(θ), (lmax + 1)^2
    op = _SHTOperator(rings, tab, _SHTScratch(T, lmax, B), B, Matrix{T}(undef, M, B),
                      Matrix{T}(undef, K, B))
    fit = FTB.RangeWeighted(op, w)
    ls = FTB.LSMRWorkspace(fit)
    ha = B == 1 ? Vector{T}(undef, K) : Matrix{T}(undef, K, B)
    return DirectSHTSphericalPlan{T, B, V, typeof(rings), typeof(tab), typeof(op), typeof(fit), typeof(ls),
                                  typeof(ha)}(lmax, M, K, θ, φ, w, rtol, maxiter, solve, rings, tab, op, fit, ls, ha)
end

function DirectSHTSphericalPlan(θ::AbstractVector, φ::AbstractVector, lmax::Int, ::Type{T};
                                rtol::Real = 1.0e-8, maxiter::Int = 500, weights = nothing,
                                ntrans::Int = 1, solve::Bool = true) where {T}
    M = length(θ)
    length(φ) == M || throw(DimensionMismatch("θ and φ must have equal length"))
    lmax >= 0 || throw(ArgumentError("lmax must be nonnegative; got $lmax"))
    ntrans >= 1 || throw(ArgumentError("ntrans must be at least 1; got $ntrans"))
    # Host arrays: the transforms visit the points one at a time.
    θT, φT = collect(T, θ), collect(T, φ)
    w = weights === nothing ? fill(one(T) / M, M) : collect(T, T.(weights) ./ sum(weights))
    return _direct_sht_plan(θT, φT, w, T(rtol), maxiter, solve, _SHTRings(θT, φT), _LegendreTable(T, lmax),
                            lmax, ntrans)
end

Base.show(io::IO, p::DirectSHTSphericalPlan{T, B}) where {T, B} =
    print(io, "DirectSHTSphericalPlan{", T, "}(lmax=", p.lmax, ", M=", p.M, ", ntrans=", B, ")")

# The point set, the quadrature weights and the tables are read-only and shared; a task gets its own
# operator scratch and solver workspace, which analysis writes its result through.
plan_points(p::DirectSHTSphericalPlan) = (p.theta, p.phi)
plan_weights(p::DirectSHTSphericalPlan) = p.weights
plan_solver(p::DirectSHTSphericalPlan) = (rtol = p.rtol, maxiter = p.maxiter, solve = p.solve)
Plans.spectral_backend(::DirectSHTSphericalPlan) = SB.DirectSumSpectralBackend()
Plans.batch_width(::DirectSHTSphericalPlan{T, B}) where {T, B} = B

Plans.task_local_plan(p::DirectSHTSphericalPlan{T, B}) where {T, B} =
    _direct_sht_plan(p.theta, p.phi, p.weights, p.rtol, p.maxiter, p.solve, p.rings, p.tab, p.lmax, B)

supports_batch(::DirectSHTSphericalPlan) = true
batch_plan(p::DirectSHTSphericalPlan{T, B}, n::Integer) where {T, B} =
    n == B ? p : _direct_sht_plan(p.theta, p.phi, p.weights, p.rtol, p.maxiter, p.solve, p.rings, p.tab,
                                  p.lmax, Int(n))

# Analysis of `field`, one column per field of a `(M, B)` stack. A field may be a length-`M` vector
# (scattered) or the `(Nθ, Nφ)` grid matrix (structured), read in the order the plan's points were
# built. The fit's LSMR stops on `‖A†r‖ ≤ rtol ‖A†b‖` and returns its best iterate at `maxiter`.
sphere_coeffs(plan::DirectSHTSphericalPlan, field::AbstractArray) =
    sphere_coeffs!(sphere_coeffs_buffer(plan), plan, field)

function sphere_coeffs!(a::AbstractArray, plan::DirectSHTSphericalPlan{T, B}, field::AbstractArray) where {T, B}
    if plan.solve
        FTB.lsmr!(a, plan.fit, field, plan.ls; rtol = plan.rtol, maxiter = plan.maxiter,
                  conlim = inv(eps(T)))
        check_sphere_solve(plan.ls, plan.M, plan.lmax, plan.rtol, plan.maxiter)
        return a
    end
    # The adjoint of `w ⊙ f`, through the fit's range buffer, which the adjoint does not use; a point of
    # zero weight contributes nothing whatever `f` holds there.
    t, M = plan.op.t, plan.M
    @inbounds for b in 1:B, n in 1:M
        i = (b - 1) * M + n
        wn = plan.weights[n]
        t[i] = iszero(wn) ? zero(T) : wn * field[i]
    end
    return _sht_adjoint!(a, plan.rings, plan.tab, plan.op.scr, t, B)
end

sphere_coeffs_buffer(plan::DirectSHTSphericalPlan{T, B}) where {T, B} =
    B == 1 ? Vector{T}(undef, plan.K) : Matrix{T}(undef, plan.K, B)

# Synthesis: out = Σ_k h(ℓ_k)·C_k·Y_k for each column. The coefficients are laid out `(ℓ, m = -ℓ:ℓ)` in
# order, so `h` is evaluated once per degree.
function sphere_apply!(out::AbstractArray, plan::DirectSHTSphericalPlan{T, B}, C, h) where {T, B}
    K = plan.K
    @inbounds for l in 0:plan.lmax
        hl = T(h(l))
        for b in 1:B
            o = (b - 1) * K
            for k in (o + l * l + 1):(o + (l + 1) * (l + 1))
                plan.ha[k] = hl * C[k]
            end
        end
    end
    _sht_synthesize!(out, plan.rings, plan.tab, plan.op.scr, plan.ha, B)
    return out
end

# Spherical average under the plan's quadrature weights: uniform `1/M` by default for scattered
# points, Fejér weights on the structured grid. A plan `B > 1` wide averages each column of an
# `(M, B)` stack.
sphere_mean(plan::DirectSHTSphericalPlan{T, B}, field::AbstractArray) where {T, B} =
    B == 1 ? LinearAlgebra.dot(plan.weights, vec(field)) :
             transpose(reshape(field, plan.M, B)) * plan.weights

# Throw `Plans.AnalysisNotConverged` when a spherical analysis has no usable answer for some column: a
# condition estimate at `conlim` or `1/eps` (`FTB.LSMR_CONDITION`), or a residual that is not finite. A
# solve that stopped at `maxiter` keeps its best iterate.
function check_sphere_solve(ws::FTB.LSMRWorkspace, M::Integer, lmax::Integer, rtol::Real,
                            maxiter::Integer)
    @inbounds for k in eachindex(ws.status)
        (ws.status[k] == FTB.LSMR_CONDITION || !isfinite(ws.normr[k])) &&
            _refuse_sphere_solve(ws, k, M, lmax, rtol, maxiter)
    end
    return nothing
end

@noinline _refuse_sphere_solve(ws::FTB.LSMRWorkspace, k::Int, M::Integer, lmax::Integer, rtol::Real,
                               maxiter::Integer) =
    throw(Plans.AnalysisNotConverged(ws.residual[k], rtol, ws.iterations[k], maxiter,
        "cond(A) ≈ $(Float32(ws.condA[k])), stopped on $(ws.status[k]). The $M points do not " *
        "determine the $((lmax + 1)^2) coefficients of band limit $lmax: supply roughly " *
        "M ≳ (lmax+1)² well-distributed points, or lower lmax."))

# ---------------------------------------------------------------------------
# Scattered-sphere plan selection
# ---------------------------------------------------------------------------

# Whether the NUFSHT extension is loaded; consulted only by `AutoSpectralBackend`.
_have_nufsht() = Base.get_extension(parentmodule(@__MODULE__), :ScatteringTransformsNUFSHTExt) !== nothing

"""
    nusht_spherical_plan(θ, φ, lmax, T; rtol, maxiter, weights, ntrans, spin, nufft, solve)

Scattered-sphere plan on NUFSHT, a method the NUFSHT extension defines; this fallback throws an
`ArgumentError` naming `using NUFSHT`.
"""
nusht_spherical_plan(args...; kwargs...) = throw(ArgumentError(
    "NUFSHTSpectralBackend requires the NUFSHT extension. Run `using NUFSHT`."))

"""
    default_rtol(spectral) -> Real

Tolerance for the scattered-sphere analysis solve under `spectral`: `1e-8` for the in-core direct
sum, which evaluates the harmonics exactly, and `1e-4` for NUFSHT, whose analysis is accurate only to
its own approximation error.
"""
default_rtol(::SB.AbstractSpectralBackend) = 1.0e-8
default_rtol(::SB.AbstractNUFSHTSpectralBackend) = 1.0e-4
default_rtol(::SB.AbstractAutoSpectralBackend) =
    _have_nufsht() ? default_rtol(SB.NUFSHTSpectralBackend()) : default_rtol(SB.DirectSumSpectralBackend())

"""
    plan_spin(plan) -> (spin-0 plan, spin-1 plan) or nothing

The spin-weighted plan pair a scattered-sphere plan carries for
[`spherical_monogenic_components`](@ref ScatteringTransforms.spherical_monogenic_components), or
`nothing` for a plan built without one. Only the monogenic constructors build it. The pair is
working state, so concurrent use needs `Plans.task_local_plan`.
"""
plan_spin(::Any) = nothing

"""
    make_spherical_plan(spectral, θ, φ, lmax, T; rtol, maxiter, weights, ntrans, spin, nufft, solve)
        -> AbstractSphericalPlan

Build the scattered-sphere plan selected by `spectral` over points `(θ, φ)` and band limit `lmax`.
`SpectralBackends.DirectSumSpectralBackend` is the dependency-free default;
`SpectralBackends.NUFSHTSpectralBackend` (or `SpectralBackends.AutoSpectralBackend` once the NUFSHT
extension is loaded) uses the NUFSHT fast path.

`ntrans = B` builds a plan that transforms `B` co-located fields per call, and `weights` are the
quadrature weights of the spherical mean. `solve = true` analyses by the least-squares fit weighted by
them, `false` by the adjoint weighted by them (uniform without them). `spin = true` asks for what
[`spherical_monogenic_components`](@ref ScatteringTransforms.spherical_monogenic_components) needs:
the NUFSHT plan builds its spin pair (see [`plan_spin`](@ref)), and the in-core plan, which takes the
surface gradient of `(−Δ_S)^{-1/2}U⁰`, needs nothing more. The in-core plan runs no NUFFT and throws
for a named `nufft`.
"""
function make_spherical_plan(s::SB.AbstractDirectSumSpectralBackend, θ, φ, lmax, ::Type{T};
                             rtol::Real = default_rtol(s), maxiter::Int = 500, weights = nothing,
                             ntrans::Int = 1, spin::Bool = false,
                             nufft::SB.AbstractSpectralBackend = SB.AutoSpectralBackend(),
                             solve::Bool = true) where {T}
    nufft isa SB.AbstractAutoSpectralBackend || throw(ArgumentError(
        "the in-core spherical plan performs no NUFFT, so `nufft = $nufft` cannot be honoured; " *
        "pass transform = SpectralBackends.NUFSHTSpectralBackend() for a NUFFT-backed transform."))
    return DirectSHTSphericalPlan(θ, φ, lmax, T; rtol = rtol, maxiter = maxiter, weights = weights,
                                  ntrans = ntrans, solve = solve)
end
make_spherical_plan(::SB.AbstractNUFSHTSpectralBackend, θ, φ, lmax, ::Type{T}; kwargs...) where {T} =
    nusht_spherical_plan(θ, φ, lmax, T; kwargs...)
make_spherical_plan(::SB.AbstractAutoSpectralBackend, θ, φ, lmax, ::Type{T}; kwargs...) where {T} =
    _have_nufsht() ? make_spherical_plan(SB.NUFSHTSpectralBackend(), θ, φ, lmax, T; kwargs...) :
    make_spherical_plan(SB.DirectSumSpectralBackend(), θ, φ, lmax, T; kwargs...)

# ---------------------------------------------------------------------------
# Structured sphere: the equiangular midpoint grid `θ_j = π(j-½)/N`, `φ_k = 2π(k-1)/(2N-1)`,
# `N = lmax+1`, which is FastSphericalHarmonics' grid. The in-core plan treats it as points for the
# direct SHT, with Fejér quadrature weights for the spherical mean.
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
# ∫₀^π g(cosθ) sinθ dθ = ∫₋₁¹ g dx exactly for band-limited g, so the spherical mean below is exact.
# Non-negative and sum to 2.
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

Structured-sphere plan on FastSphericalHarmonics, a method the FastSphericalHarmonics extension
defines; this fallback throws an `ArgumentError` naming `using FastSphericalHarmonics`.
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
# Spin-1 Riesz field on the in-core plan, for the pointwise monogenic decomposition.
#
# The Riesz tangent vector is the surface gradient of `g = (−Δ_S)^{-1/2} U⁰`: `u_θ = ∂_θ g`,
# `u_φ = (1/sinθ) ∂_φ g`. Given `g`'s real-SH coefficients, the φ-derivative of the `cos/sin(mφ)`
# factor is analytic, and so is the θ-derivative of the associated Legendre part (`_dtheta_pbar`).
# Like the synthesis, both run ring by ring: per ring the order sums of `P̄` and of its derivative, per
# point the sum over orders.
# ---------------------------------------------------------------------------

# `−∂_θ P̄_ℓ^m` from same-degree neighbours:
#   ½[ √((ℓ+m)(ℓ−m+1))·P̄_ℓ^{m−1} − √((ℓ−m)(ℓ+m+1))·P̄_ℓ^{m+1} ],
# the identity for `∂_θ P_ℓ^m` without the Condon–Shortley phase, which `P̄` carries, so it gives the
# negated derivative. `P̄_ℓ^{−1} = −P̄_ℓ^{1}` (the normalised form of
# `P_ℓ^{−m} = (−1)^m (ℓ−m)!/(ℓ+m)!·P_ℓ^m`) collapses the `m = 0` case to `−√(ℓ(ℓ+1))·P̄_ℓ^1`. The
# identity is linear in `P̄`, so `_assoc_legendre!`'s arbitrary overall scale cancels.
@inline function _dtheta_pbar(P::AbstractMatrix{T}, ℓ::Int, m::Int) where {T}
    up = (m + 1 > ℓ) ? zero(T) : sqrt(T((ℓ - m) * (ℓ + m + 1))) * P[ℓ + 1, m + 2]
    dn = m == 0 ? -sqrt(T(ℓ * (ℓ + 1))) * (ℓ >= 1 ? P[ℓ + 1, 2] : zero(T)) :
         sqrt(T((ℓ + m) * (ℓ - m + 1))) * P[ℓ + 1, m]
    return (dn - up) / 2
end

riesz_scratch(plan::DirectSHTSphericalPlan{T}, ::AbstractArray) where {T} =
    (gc = Vector{T}(undef, plan.K), P = Matrix{T}(undef, plan.lmax + 1, plan.lmax + 1),
     dc = Vector{T}(undef, plan.lmax + 1), ds = Vector{T}(undef, plan.lmax + 1),
     pc = Vector{T}(undef, plan.lmax + 1), ps = Vector{T}(undef, plan.lmax + 1))

# `scr.gc = h(ℓ)·C` in the `(ℓ, m = -ℓ:ℓ)` coefficient order.
function _scaled_coeffs!(scr, plan::DirectSHTSphericalPlan{T}, C, h) where {T}
    @inbounds for l in 0:plan.lmax
        hl = T(h(l))
        for k in (l * l + 1):((l + 1) * (l + 1))
            scr.gc[k] = hl * C[k]
        end
    end
    return scr.gc
end

# `store(n, Gθ, Gφ)` at every point `n`, with `(Gθ, Gφ) = (−∂_θ g, −∂_φ g / sinθ)` for the field `g`
# whose coefficients are `scr.gc`.
function _each_gradient(store::F, plan::DirectSHTSphericalPlan{T}, scr) where {F, T}
    lmax = plan.lmax
    rings = plan.rings
    P, g = scr.P, scr.gc
    dc, ds, pc, ps = scr.dc, scr.ds, scr.pc, scr.ps
    s2 = sqrt(T(2))
    @inbounds for r in 1:_nrings(rings)
        _assoc_legendre!(P, plan.tab, rings.rcos[r], rings.rsin[r])
        for m in 0:lmax
            sdc = zero(T)
            sds = zero(T)
            spc = zero(T)
            sps = zero(T)
            for ℓ in m:lmax
                dP = _dtheta_pbar(P, ℓ, m)
                p = P[ℓ + 1, m + 1]
                gp, gm = g[_lm(ℓ, m)], g[_lm(ℓ, -m)]
                sdc += dP * gp
                sds += dP * gm
                spc += p * gp
                sps += p * gm
            end
            dc[m + 1], ds[m + 1], pc[m + 1], ps[m + 1] = sdc, sds, spc, sps
        end
        invs = one(T) / rings.rsin[r]
        for t in rings.ptr[r]:(rings.ptr[r + 1] - 1)
            n = rings.pts[t]
            c1, s1 = rings.cφ[n], rings.sφ[n]
            gθ = dc[1]
            gφ = zero(T)
            cm, sm = one(T), zero(T)
            for m in 1:lmax
                cm, sm = cm * c1 - sm * s1, sm * c1 + cm * s1
                gθ += s2 * (cm * dc[m + 1] + sm * ds[m + 1])
                gφ += s2 * m * (sm * pc[m + 1] - cm * ps[m + 1])
            end
            store(n, gθ, gφ * invs)
        end
    end
    return nothing
end

function sphere_riesz_energy!(out::AbstractArray, plan::DirectSHTSphericalPlan, C, h, scr)
    _scaled_coeffs!(scr, plan, C, h)
    _each_gradient((n, gθ, gφ) -> (@inbounds out[n] = gθ^2 + gφ^2), plan, scr)
    return out
end

# Pointwise monogenic decomposition on the direct SHT plan (spin-0 band-pass + spin-1 Riesz gradient).
function direct_monogenic_components(st::SphericalMonogenicScattering{<:Any, <:DirectSHTSphericalPlan},
                                     field::AbstractVector, j::Int)
    plan = st.plan
    σ²hi, σ²lo = st.sigma2[j + 1], st.sigma2[j]
    a = sphere_coeffs(plan, field)                                  # spin-0 SH coefficients
    U0 = similar(plan.theta)
    sphere_apply!(U0, plan, a, band_multiplier(σ²hi, σ²lo))         # band-pass U⁰ (real)
    # Coefficients of g = (−Δ_S)^{-1/2} U⁰, then its surface gradient `(∂_θ g, ∂_φ g / sinθ)` at each
    # point, the negation of what `_each_gradient` stores.
    scr = riesz_scratch(plan, field)
    _scaled_coeffs!(scr, plan, a, riesz_potential_multiplier(σ²hi, σ²lo))
    uθ, uφ = similar(plan.theta), similar(plan.theta)
    _each_gradient((n, gθ, gφ) -> (@inbounds (uθ[n] = -gθ; uφ[n] = -gφ)), plan, scr)
    rnorm = sqrt.(uθ .^ 2 .+ uφ .^ 2)
    amplitude = sqrt.(U0 .^ 2 .+ rnorm .^ 2)
    phase = atan.(rnorm, U0)
    orientation = atan.(uφ, uθ)
    return (; bandpass = U0, riesz = (uθ, uφ), amplitude, phase, orientation)
end

end # module SphericalCore
