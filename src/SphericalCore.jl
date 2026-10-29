module SphericalCore

# Scattering on S², independent of how the spherical-harmonic transform is performed. A plan
# implements analysis (`sphere_coeffs`, `sphere_coeffs!`), per-degree multiplier and synthesis
# (`sphere_apply!`) and the spherical mean (`sphere_mean`); the difference-of-Gaussians bank and the
# S0/S1/S2 cascade are built on those, and the monogenic amplitude also uses `sphere_riesz_energy!`.
# A scattered transform analyses its samples once and runs the cascade on a plan of the same kind on
# the Gauss–Legendre grid of its band limit (`grid_plan`). The in-core plans are defined here; the
# NUFSHT (scattered points) and FastSphericalHarmonics (structured grid) extensions supply fast ones.

export AbstractSphericalPlan, SphericalScatteringTransform, SphericalMonogenicScatteringTransform
export SphericalWorkspace, SphericalMonogenicWorkspace
export spherical_scattering!, spherical_monogenic_scattering!
export sphere_coeffs, sphere_coeffs!, sphere_coeffs_buffer, sphere_apply!, sphere_mean

using ..Plans: Plans
using ..Coefficients: Coefficients
using ..ScatteringCore: ScatteringCore
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
structured grid this is the forward SHT; for scattered points it is the least-squares fit or the
weighted adjoint. `C` is opaque and valid only with the interface functions of `plan` or of a plan of
its kind at its band limit, such as [`sphere_apply!`](@ref).
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
    (D = sphere_coeffs_buffer(plan), f = similar(field))

"""
    sph_layout_riesz_energy!(out, plan, C, h, lmax, scratch) -> out

[`sphere_riesz_energy!`](@ref) for coefficients in the real `sph_mode` layout, by three scalar
syntheses through `sphere_apply!`. With `L = -i r×∇`, a real `g` has `|∇_S g|² = |L₊g|² + (∂_φ g)²`,
and both operators act within each degree: `L₊Y_{ℓm} = √((ℓ-m)(ℓ+m+1)) Y_{ℓ,m+1}` on the
Condon–Shortley complex harmonics, and `∂_φ` maps `cos(mφ) ↦ -m sin(mφ)`, `sin(mφ) ↦ m cos(mφ)`.
`L₊g = P + iQ` is synthesised as its real and imaginary parts.
"""
function sph_layout_riesz_energy!(out, plan, C, h, lmax::Int, scr)
    D, f = scr.D, scr.f
    fill!(out, zero(eltype(out)))
    for component in 0:2
        _sph_derivative!(D, C, h, lmax, component)
        sphere_apply!(f, plan, D, Returns(one(real(eltype(f)))))
        @. out += f^2
    end
    return out
end

_sph_batch(::CartesianIndex{2}) = 1
_sph_batch(I::CartesianIndex{3}) = I[3]

@inline function _complex_harmonic(C, ℓ, m, b, ::Type{T}) where {T}
    abs(m) <= ℓ || return zero(Complex{T})
    m == 0 && return complex(C[_sph_slot(ℓ, 0)..., b])
    a = complex(C[_sph_slot(ℓ, abs(m))..., b], -C[_sph_slot(ℓ, -abs(m))..., b]) / sqrt(T(2))
    return m < 0 ? conj(a) : (isodd(m) ? -a : a)
end

@inline function _raised_harmonic(C, ℓ, m, b, ::Type{T}) where {T}
    m <= -ℓ && return zero(Complex{T})
    return sqrt(T((ℓ - m + 1) * (ℓ + m))) * _complex_harmonic(C, ℓ, m - 1, b, T)
end

@inline function _sph_derivative_entry(C, I, h, lmax, component)
    T = real(eltype(C))
    am = I[2] ÷ 2
    ℓ = I[1] + am - 1
    ℓ <= lmax || return zero(T)
    b = _sph_batch(I)
    m = isodd(I[2]) ? am : -am
    hl = T(h(ℓ))
    component == 0 && return m * hl * C[_sph_slot(ℓ, -m)..., b]
    sg = isodd(am) ? -one(T) : one(T)
    bp = _raised_harmonic(C, ℓ, am, b, T)
    bn = _raised_harmonic(C, ℓ, -am, b, T)
    v = component == 1 ? (bp + sg * conj(bn)) / 2 : (bp - sg * conj(bn)) / (2im)
    am == 0 && return hl * real(v)
    return hl * sqrt(T(2)) * sg * (m > 0 ? real(v) : -imag(v))
end

function _sph_derivative!(D, C, h, lmax, component)
    D .= _sph_derivative_entry.(Ref(C), CartesianIndices(D), Ref(h), lmax, component)
    return D
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
    grid_plan(plan, L) -> plan or nothing

A plan of `plan`'s kind, band limit and batch width on the Gauss–Legendre grid of band limit `L ≥ lmax`
([`gauss_legendre_grid`](@ref)), analysing by that grid's quadrature, or `nothing` for a backend with
none. Its coefficients are laid out as `plan`'s. Its analysis of a field holding degrees up to
`2L + 1 − lmax` is exact. A scattered transform analyses its samples with `plan` and runs every later
step of its cascade on this one.
"""
grid_plan(::Any, ::Integer) = nothing

sphere_npoints(plan) = length(first(plan_points(plan)))

"""
    grid_band(grid) -> Int

The band limit `L` of a [`grid_plan`](@ref)'s Gauss–Legendre grid, from its `(L + 1)(2L + 1)` nodes.
"""
grid_band(grid) = (isqrt(1 + 8 * sphere_npoints(grid)) - 3) ÷ 4

"""
    coeffs_mean(plan, C, b = 1)

The spherical mean of the field whose coefficients under `plan` are field `b` of `C`.
"""
function coeffs_mean end

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
    ℓj = lmax / 2.0^(j - 1)
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
    plan_solver(plan) -> (; rtol, maxiter, analysis)

The analysis policy and fit tolerances of a scattered-sphere plan. `nothing`
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

Gaussian-transfer variances for cutoffs `ℓ_k = lmax / 2^k`, `k=0..J`.
Band `j` is `lowpass(ℓ_{j-1}) − lowpass(ℓ_j)`, with scale 1 finest.
"""
dog_sigma2(lmax::Integer, J::Integer, ::Type{T}) where {T} =
    T[_sigma2_for_cutoff(lmax / 2.0^k, T) for k in 0:J]

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
    SphericalScatteringTransform

Spherical scattering transform over a backend `plan::P`. `sigma2[k+1]` is the Gaussian-transfer
variance for the dyadic low-pass cutoff `ℓ_k` (`k=0..J`); band-pass wavelet `j` is
`lowpass(ℓ_{j-1}) − lowpass(ℓ_j)`. `bands` holds one narrowed plan per scale
([`band_plans`](@ref)), or `nothing`. `grid` is the plan the cascade runs on after `plan` analyses the
field ([`grid_plan`](@ref)), or `nothing` for a transform whose cascade runs on `plan`.
"""
struct SphericalScatteringTransform{T, P, V <: AbstractVector{T}, BV, G, WS, W} <: ScatteringCore.AbstractScatteringTransform
    lmax::Int
    J::Int
    max_order::Int
    plan::P
    sigma2::V
    # Per-scale narrowed plans, or `nothing` when the backend cannot narrow (or the caller asked for
    # no narrowing). `nothing` runs the full-band-limit cascade unchanged.
    bands::BV
    grid::G
    ws::WS
    workspaces::W
end

function SphericalScatteringTransform(lmax::Int, J::Int, max_order::Int, plan, sigma2, bands, grid)
    info = (; plan, bands, grid)
    ws = SphericalWorkspace(info, sphere_field_buffer(plan))
    return SphericalScatteringTransform(lmax, J, max_order, plan, sigma2, bands, grid, ws,
                                ScatteringCore.Workspaces(SphericalScatteringTransform))
end

SphericalScatteringTransform(lmax::Int, J::Int, max_order::Int, plan, sigma2, bands = nothing) =
    SphericalScatteringTransform(lmax, J, max_order, plan, sigma2, bands, nothing)

Base.show(io::IO, st::SphericalScatteringTransform{T}) where {T} =
    print(io, "SphericalScatteringTransform{", T, "}(lmax=", st.lmax, ", J=", st.J, ", max_order=", st.max_order,
          ", ", st.plan, st.bands === nothing ? "" : ", per-scale band limits",
          st.grid === nothing ? "" : ", cascade on the Gauss–Legendre grid", ")")

# The plan the cascade runs on once the field is analysed.
_cascade_plan(st) = st.grid === nothing ? st.plan : st.grid

# A band field of the cascade, shaped like `field` on the plan's own samples, else on the grid's nodes.
_band_buffer(st, field::AbstractArray) =
    st.grid === nothing ? similar(field) :
    similar(field, sphere_npoints(st.grid), Base.tail(size(field))...)

# `S0`: the plan's quadrature of the samples, or for a transform with a grid the mean of the field its
# analysis `C` recovers. `X` is a `(M, B)` stack.
_field_mean(st, C, field) = st.grid === nothing ? sphere_mean(st.plan, field) : coeffs_mean(st.grid, C)
function _field_means!(out, st, C, X::AbstractMatrix)
    st.grid === nothing ? sphere_means!(out, st.plan, X) : coeffs_means!(out, st.grid, C)
    return out
end

function sphere_means! end
function coeffs_means! end

_on_grid(_, ::Nothing) = nothing
_on_grid(f, grid) = f(grid)

"""
    batch_transform(st, B) -> st

`st` over `B` co-located fields per call, its plan and grid through [`batch_plan`](@ref): each is
`st`'s own at its own width and otherwise built for the copy, which [`close_task_local!`](@ref)
releases.
"""
batch_transform(st::SphericalScatteringTransform, B::Integer) =
    SphericalScatteringTransform(st.lmax, st.J, st.max_order, batch_plan(st.plan, B), st.sigma2, nothing,
                        _on_grid(g -> batch_plan(g, B), st.grid))

"""
    task_local_batch(st, B) -> st

A copy of `st` over `B` co-located fields per call, safe to run while `st` runs in another task: its
plan and grid through [`task_local_batch_plan`](@ref). Release it with [`close_task_local!`](@ref).
"""
task_local_batch(st::SphericalScatteringTransform, B::Integer) =
    SphericalScatteringTransform(st.lmax, st.J, st.max_order, task_local_batch_plan(st.plan, B), st.sigma2,
                        nothing, _on_grid(g -> task_local_batch_plan(g, B), st.grid))

"""
    band_plans(plan, lmax, ls) -> Vector or nothing

One plan per scale, scale `j` at band limit `ls[j]`, or `nothing` if this backend cannot narrow or
no band needs narrowing. A scale at the full band limit `lmax` reuses `plan`.
"""
band_plans(plan, lmax::Integer, ::Nothing) = nothing
band_plans(plan, lmax::Integer, ls::AbstractVector{<:Integer}) = nothing

"""
    task_local(st) -> st

A copy of `st` safe to run concurrently with it: the plan's read-only tables are shared and its
analysis and synthesis scratch duplicated by `Plans.task_local_plan`. Release it with
[`close_task_local!`](@ref).
"""
function task_local(st::SphericalScatteringTransform)
    plan = Plans.task_local_plan(st.plan)
    # Each narrowed plan carries its own synthesis scratch, so a task needs its own — except where
    # the scale reuses `st.plan` itself, which must go on aliasing the copy.
    bands = st.bands === nothing ? nothing :
            map(p -> p === st.plan ? plan : Plans.task_local_plan(p), st.bands)
    return SphericalScatteringTransform(st.lmax, st.J, st.max_order, plan, st.sigma2, bands,
                               _on_grid(Plans.task_local_plan, st.grid))
end

"""
    close_task_local!(stl, st) -> nothing

`Plans.close_plan!` every plan [`task_local`](@ref)`(st)`, [`batch_transform`](@ref) or
[`task_local_batch`](@ref) built for `stl`; plans `stl` shares with `st` stay open.
"""
function close_task_local!(stl::SphericalScatteringTransform, st::SphericalScatteringTransform)
    _close_unshared!(stl.plan, st.plan)
    stl.bands === nothing || foreach(_close_unshared!, stl.bands, st.bands)
    stl.grid === nothing || _close_unshared!(stl.grid, st.grid)
    return nothing
end

_close_unshared!(p, shared) = (p === shared || Plans.close_plan!(p); nothing)

"""
    SphericalWorkspace{A,C,AV,CV}

Scratch for one spherical cascade: the band-pass output `band`, the current first-order field `u1`,
both on the samples the cascade runs on (the transform's grid, where it has one), and two coefficient
containers, `C` for the input field and `C1` for the first-order field being re-analysed. Two
suffice, since the cascade finishes every child of a scale before starting the next. `bband`, `bu1`,
`bC` and `bC1` are the same per scale at its narrowed band limit, and empty when the transform has no
`bands`.

Build one with `SphericalWorkspace(st, field)` and reuse it across calls; a task that transforms
concurrently needs its own, with a [`task_local`](@ref) transform.
"""
struct SphericalWorkspace{A, C, AV, CV, V}
    band::A
    u1::A
    C::C
    C1::C
    bband::AV
    bu1::AV
    bC::CV
    bC1::CV
    S0::V
    means::V
end

function SphericalWorkspace(st, field::AbstractArray)
    C() = sphere_coeffs_buffer(st.plan)
    F() = _band_buffer(st, field)
    means = similar(field, real(eltype(field)), Plans.batch_width(st.plan))
    if st.bands === nothing
        e = similar(field, 0)
        ec = [C()][1:0]
        return SphericalWorkspace(F(), F(), C(), C(), typeof(e)[], typeof(e)[], ec, ec, similar(means), means)
    end
    bband = [sphere_field_buffer(p) for p in st.bands]
    bu1 = [sphere_field_buffer(p) for p in st.bands]
    bC = [sphere_coeffs_buffer(p) for p in st.bands]
    bC1 = [sphere_coeffs_buffer(p) for p in st.bands]
    return SphericalWorkspace(F(), F(), C(), C(), bband, bu1, bC, bC1, similar(means), means)
end

Base.show(io::IO, ws::SphericalWorkspace) =
    print(io, "SphericalWorkspace(fields of size ", size(ws.band), ", ", length(ws.bband), " band buffers)")

"""
    (st::SphericalScatteringTransform)(field) -> ScatteringCoefficients1D

Apply the spherical scattering transform to a scalar `field` sampled by the plan.
`S1[j] = ⟨|field ⋆ ψ_j|⟩`, `S2[j1,j2] = ⟨||field ⋆ ψ_{j1}| ⋆ ψ_{j2}|⟩` for strictly coarser `j2 > j1`.
With a `grid`, the means are its quadrature of the field the analysis recovers, and `S0` that field's
mean. The transform retains its workspace; each call allocates coefficient storage.
"""
function (st::SphericalScatteringTransform{T})(field::AbstractArray) where {T}
    S2 = st.max_order >= 2 ? zeros(T, st.J, st.J) : Matrix{T}(undef, 0, 0)
    return spherical_scattering!(zeros(T, st.J), S2, st, st.ws, field)
end

"""
    spherical_scattering!(S1, S2, st, ws, field) -> ScatteringCoefficients1D

In-place spherical scattering into pre-allocated `S1`/`S2` using the workspace `ws`
(see [`SphericalWorkspace`](@ref)). The returned container uses the supplied arrays.

Grouped by first-order scale so only one first-order field and one coefficient vector are live at a
time: scale `j1` is band-passed, averaged into `S1[j1]`, analysed once, and then consumed by every
coarser `j2 > j1` before the next `j1` begins.
"""
function spherical_scattering!(S1::AbstractVector, S2::AbstractMatrix,
                               st::SphericalScatteringTransform, ws::SphericalWorkspace,
                               field::AbstractArray)
    J = st.J
    isempty(S2) || fill!(S2, zero(eltype(S2)))
    sphere_coeffs!(ws.C, st.plan, field)                  # analyse the field once
    S0 = _field_mean(st, ws.C, field)
    st.bands === nothing || return _banded_cascade!(S1, S2, st, ws, S0)
    g = _cascade_plan(st)
    for j1 in 1:J
        sphere_apply!(ws.band, g, ws.C, band_multiplier(st.sigma2[j1], st.sigma2[j1 + 1]))
        @. ws.u1 = abs(ws.band)
        S1[j1] = sphere_mean(g, ws.u1)
        (st.max_order >= 2 && j1 < J) || continue
        sphere_coeffs!(ws.C1, g, ws.u1)                   # analyse U1[j1] once, reuse across children
        for j2 in (j1 + 1):J
            sphere_apply!(ws.band, g, ws.C1, band_multiplier(st.sigma2[j2], st.sigma2[j2 + 1]))
            @. ws.band = abs(ws.band)
            S2[j1, j2] = sphere_mean(g, ws.band)
        end
    end
    return Coefficients.ScatteringCoefficients1D(S1, S2; S0)
end

# The same cascade with every band synthesised at its own band limit. `ws.C` already holds the
# field's coefficients at the full band limit.
#
# Each child uses its scale's buffers while the parent's modulus stays in `bu1[j1]`.
function _banded_cascade!(S1, S2, st::SphericalScatteringTransform, ws::SphericalWorkspace, S0)
    bands = st.bands
    for j1 in 1:st.J
        p1 = bands[j1]
        h1 = band_multiplier(st.sigma2[j1], st.sigma2[j1 + 1])
        sphere_restrict!(ws.bC[j1], p1, ws.C, st.plan)
        sphere_apply!(ws.bband[j1], p1, ws.bC[j1], h1)
        @. ws.bu1[j1] = abs(ws.bband[j1])
        S1[j1] = sphere_mean(p1, ws.bu1[j1])
        (st.max_order >= 2 && j1 < st.J) || continue
        sphere_coeffs!(ws.bC1[j1], p1, ws.bu1[j1])
        for j2 in (j1 + 1):st.J
            p2 = bands[j2]
            h2 = band_multiplier(st.sigma2[j2], st.sigma2[j2 + 1])
            sphere_restrict!(ws.bC[j2], p2, ws.bC1[j1], p1)
            sphere_apply!(ws.bband[j2], p2, ws.bC[j2], h2)
            @. ws.bband[j2] = abs(ws.bband[j2])
            S2[j1, j2] = sphere_mean(p2, ws.bband[j2])
        end
    end
    return Coefficients.ScatteringCoefficients1D(S1, S2; S0)
end

"""
    spherical_scattering_batch!(S1, S2, st, ws, X) -> (; S0, S1, S2)

Cascade over a stack of `B` fields sampled at the same points: `X` is `(M, B)`, `S1` is `(J, B)` and
`S2` is `(J, J, B)`. Every transform covers all `B` fields in one call; `st.plan`, and its grid, must
have batch width `B` ([`batch_transform`](@ref)).
"""
function spherical_scattering_batch!(S1::AbstractMatrix, S2::AbstractArray,
                                     st::SphericalScatteringTransform, ws::SphericalWorkspace,
                                     X::AbstractMatrix)
    J = st.J
    isempty(S2) || fill!(S2, zero(eltype(S2)))
    sphere_coeffs!(ws.C, st.plan, X)
    S0 = _field_means!(ws.S0, st, ws.C, X)
    g = _cascade_plan(st)
    for j1 in 1:J
        sphere_apply!(ws.band, g, ws.C, band_multiplier(st.sigma2[j1], st.sigma2[j1 + 1]))
        @. ws.u1 = abs(ws.band)
        sphere_means!(ws.means, g, ws.u1)
        view(S1, j1, :) .= ws.means
        (st.max_order >= 2 && j1 < J) || continue
        sphere_coeffs!(ws.C1, g, ws.u1)
        for j2 in (j1 + 1):J
            sphere_apply!(ws.band, g, ws.C1, band_multiplier(st.sigma2[j2], st.sigma2[j2 + 1]))
            @. ws.band = abs(ws.band)
            sphere_means!(ws.means, g, ws.band)
            view(S2, j1, j2, :) .= ws.means
        end
    end
    return (S0 = S0, S1 = S1, S2 = S2)
end


"""
    SphericalMonogenicScatteringTransform

Spherical monogenic scattering: shares the dyadic difference-of-Gaussians bands of
[`SphericalScatteringTransform`](@ref) but replaces the analytic modulus with the spherical monogenic
amplitude `A_j = √(U⁰_j² + |∇_S g_j|²)` (see [`monogenic_amplitude!`](@ref)). `grid` is as in
[`SphericalScatteringTransform`](@ref).
"""
struct SphericalMonogenicScatteringTransform{T, P, V <: AbstractVector{T}, G, WS, W} <: ScatteringCore.AbstractScatteringTransform
    lmax::Int
    J::Int
    max_order::Int
    plan::P
    sigma2::V
    grid::G
    ws::WS
    workspaces::W
end

function SphericalMonogenicScatteringTransform(lmax::Int, J::Int, max_order::Int, plan, sigma2, grid)
    ws = SphericalMonogenicWorkspace((; plan, grid), sphere_field_buffer(plan))
    return SphericalMonogenicScatteringTransform(lmax, J, max_order, plan, sigma2, grid, ws,
                                         ScatteringCore.Workspaces(SphericalMonogenicScatteringTransform))
end

SphericalMonogenicScatteringTransform(lmax::Int, J::Int, max_order::Int, plan, sigma2) =
    SphericalMonogenicScatteringTransform(lmax, J, max_order, plan, sigma2, nothing)

Base.show(io::IO, st::SphericalMonogenicScatteringTransform{T}) where {T} =
    print(io, "SphericalMonogenicScatteringTransform{", T, "}(lmax=", st.lmax, ", J=", st.J, ", max_order=",
          st.max_order, ", ", st.plan,
          st.grid === nothing ? "" : ", cascade on the Gauss–Legendre grid", ")")

task_local(st::SphericalMonogenicScatteringTransform) =
    SphericalMonogenicScatteringTransform(st.lmax, st.J, st.max_order, Plans.task_local_plan(st.plan),
                                 st.sigma2, _on_grid(Plans.task_local_plan, st.grid))

function close_task_local!(stl::SphericalMonogenicScatteringTransform, st::SphericalMonogenicScatteringTransform)
    _close_unshared!(stl.plan, st.plan)
    stl.grid === nothing || _close_unshared!(stl.grid, st.grid)
    return nothing
end

ScatteringCore.task_workspace(st::SphericalScatteringTransform) = task_local(st)
ScatteringCore.task_workspace(st::SphericalMonogenicScatteringTransform) = task_local(st)

"""
    monogenic_amplitude!(amp, st, C, j, w) -> amp

Spherical monogenic amplitude at scale `j` of the field whose (already-computed) SH coefficients are
`C`, written into `amp` on the samples the cascade runs on. `w` holds the Riesz energy field `r2`
and that plan's `riesz` scratch.

With `g_j = (−Δ_S)^{-1/2} U⁰_j` the spin-1 Riesz field has `|U^R_j| = |∇_S g_j|`, so
`A_j = √(U⁰_j² + |∇_S g_j|²)`, the energy evaluated by [`sphere_riesz_energy!`](@ref). The Riesz
*vector* (orientation/phase) is `spherical_monogenic_components`.
"""
function monogenic_amplitude!(amp::AbstractArray, st::SphericalMonogenicScatteringTransform,
                              C, j::Int, w)
    σ²hi = st.sigma2[j]
    σ²lo = st.sigma2[j + 1]
    g = _cascade_plan(st)
    sphere_apply!(amp, g, C, band_multiplier(σ²hi, σ²lo))                               # U⁰
    sphere_riesz_energy!(w.r2, g, C, riesz_potential_multiplier(σ²hi, σ²lo), w.riesz)
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

function SphericalMonogenicWorkspace(st, field::AbstractArray)
    r2 = _band_buffer(st, field)
    return SphericalMonogenicWorkspace(_band_buffer(st, field), _band_buffer(st, field),
                                       sphere_coeffs_buffer(st.plan), sphere_coeffs_buffer(st.plan),
                                       (r2 = r2, riesz = riesz_scratch(_cascade_plan(st), r2)))
end

Base.show(io::IO, ws::SphericalMonogenicWorkspace) =
    print(io, "SphericalMonogenicWorkspace(fields of size ", size(ws.u1), ")")

"""
    (st::SphericalMonogenicScatteringTransform)(field) -> ScatteringCoefficients1D

Apply the spherical monogenic scattering transform. `S1[j] = ⟨A_j⟩`,
`S2[j1,j2] = ⟨A_{j2}[A_{j1}]⟩` for strictly coarser `j2 > j1`.
The transform retains its workspace; each call allocates coefficient storage.
"""
function (st::SphericalMonogenicScatteringTransform{T})(field::AbstractArray) where {T}
    S2 = st.max_order >= 2 ? zeros(T, st.J, st.J) : Matrix{T}(undef, 0, 0)
    return spherical_monogenic_scattering!(zeros(T, st.J), S2, st, st.ws, field)
end

"""
    spherical_monogenic_scattering!(S1, S2, st, ws, field) -> ScatteringCoefficients1D

In-place spherical monogenic scattering, the monogenic counterpart of
[`spherical_scattering!`](@ref), grouped by first-order scale with one amplitude field live at a
time.
"""
function spherical_monogenic_scattering!(S1::AbstractVector, S2::AbstractMatrix,
                                         st::SphericalMonogenicScatteringTransform,
                                         ws::SphericalMonogenicWorkspace, field::AbstractArray)
    J = st.J
    isempty(S2) || fill!(S2, zero(eltype(S2)))
    sphere_coeffs!(ws.C, st.plan, field)                  # analyse the field once
    S0 = _field_mean(st, ws.C, field)
    g = _cascade_plan(st)
    for j1 in 1:J
        monogenic_amplitude!(ws.u1, st, ws.C, j1, ws.scratch)
        S1[j1] = sphere_mean(g, ws.u1)
        (st.max_order >= 2 && j1 < J) || continue
        sphere_coeffs!(ws.C1, g, ws.u1)                   # analyse U1[j1] once, reuse across children
        for j2 in (j1 + 1):J
            monogenic_amplitude!(ws.amp, st, ws.C1, j2, ws.scratch)
            S2[j1, j2] = sphere_mean(g, ws.amp)
        end
    end
    return Coefficients.ScatteringCoefficients1D(S1, S2; S0)
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

struct _SHTRecurrence{V, M, N}
    lmax::Int
    starts::V
    steps::M
    norm::N
end

function _SHTRecurrence(::Type{T}, lmax::Int) where {T}
    S = promote_type(T, Float64)
    starts = [FTB.wigner_d_start(S, m, 0) for m in 0:lmax]
    steps = Matrix{NTuple{5,T}}(undef, lmax + 1, lmax + 1)
    for m in 0:lmax, ℓ in m:(lmax - 1)
        steps[ℓ + 1, m + 1] = map(T, FTB.wigner_d_coeffs(S, ℓ, m, 0))
    end
    return _SHTRecurrence(lmax, starts, steps, T[sqrt(T(2ℓ + 1)) for ℓ in 0:lmax])
end

# P̄_ℓ^m(θ) = √(2ℓ+1) d^ℓ_{m0}(θ), with the Condon–Shortley phase.
function _latitude_values!(P::AbstractMatrix{T}, tab::_SHTRecurrence, θ) where {T}
    S = typeof(first(tab.starts).lC)
    sh, ch = sincos(S(θ) / 2)
    lc, ls = log2(ch), log2(sh)
    pos = θ <= T(π) / 2
    u = T(2 * (pos ? sh : ch)^2)
    up, dn = FTB.wigner_d_bounds(T)
    span = FTB.wigner_d_span(T)
    @inbounds for m in 0:tab.lmax
        y, k = FTB.wigner_d_start_value(T, tab.starts[m + 1], lc, ls)
        e = y
        for ℓ in m:tab.lmax
            P[ℓ + 1, m + 1] = tab.norm[ℓ + 1] * ldexp(y, span * Int(k))
            ℓ == tab.lmax && break
            α, _, γ, cp, cm = tab.steps[ℓ + 1, m + 1]
            sa, c, sg, sign = FTB.wigner_d_signed(pos, α, cp, cm, γ)
            y, e = FTB.wigner_d_step(sa, c, sg, sign, u, y, e)
            y, e, k = FTB.wigner_d_rescale(y, e, k, up, dn)
        end
    end
    return P
end

# The coefficient slot of `(ℓ, m)`.
@inline _lm(ℓ::Int, m::Int) = ℓ * ℓ + ℓ + m + 1

# A direct plan's points grouped by colatitude: ring `r` holds the points `pts[ptr[r]:ptr[r+1]-1]`, at
# `cosθ = rcos[r]`, `sinθ = rsin[r]`; `cφ`, `sφ` hold each point's `cos φ`, `sin φ`.
struct _SHTRings{T, V <: AbstractVector{T}, I <: AbstractVector{Int}}
    theta::V
    rsin::V
    ptr::I
    pts::I
    cφ::V
    sφ::V
end

function _SHTRings(θ::Vector{T}, φ::Vector{T}) where {T}
    order = sortperm(θ)
    ptr = Int[]
    theta, rsin = T[], T[]
    for t in eachindex(order)
        n = order[t]
        if t == 1 || θ[n] != θ[order[t - 1]]
            push!(ptr, t)
            push!(rsin, sin(θ[n]))
            push!(theta, θ[n])
        end
    end
    push!(ptr, length(order) + 1)
    sφ, cφ = similar(φ), similar(φ)
    @inbounds for n in eachindex(φ)
        sφ[n], cφ[n] = sincos(φ[n])
    end
    return _SHTRings(theta, rsin, ptr, order, cφ, sφ)
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
function _sht_synthesize!(f, rings::_SHTRings{T}, tab::_SHTRecurrence, s::_SHTScratch,
                          a, nb::Int) where {T}
    lmax = tab.lmax
    M = length(rings.pts)
    K = (lmax + 1)^2
    P, gc, gs, acc = s.P, s.gc, s.gs, s.acc
    s2 = sqrt(T(2))
    @inbounds for r in 1:_nrings(rings)
        _latitude_values!(P, tab, rings.theta[r])
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
function _sht_adjoint!(v, rings::_SHTRings{T}, tab::_SHTRecurrence, s::_SHTScratch,
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
            _latitude_values!(P, tab, rings.theta[r])
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
    sampling::R
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
    FTB.colxpby!(u, _sht_synthesize!(op.t, op.sampling, op.tab, op.scr, v, n), c, n, op.B)
FTB.lsmr_adjoint!(v, op::_SHTOperator, u, c, n) =
    FTB.colxpby!(v, _sht_adjoint!(op.w, op.sampling, op.tab, op.scr, u, n), c, n, op.B)

_sht_scratch(::Type{T}, lmax, B, ::_SHTRings) where {T} = _SHTScratch(T, lmax, B)

"""
    DirectSHTSphericalPlan

The in-core scattered-sphere plan over `M` points at band limit `lmax`, transforming `B` co-located
fields per call. Analysis is the least-squares fit of the real spherical harmonics of degree
`≤ lmax` by LSMR (`FlowTransformBindings.lsmr!`), weighted by the quadrature weights `w`, which sum to 1:
`min Σⱼ wⱼ (fⱼ − Σ a_ℓm Y_ℓm(xⱼ))²`. With `analysis = Plans.WeightedAdjoint()` it is the weighted adjoint
`a_ℓm = Σⱼ wⱼ fⱼ Y_ℓm(xⱼ)` over harmonics of mean square 1. Synthesis is their direct sum at the points;
neither forms the `M × (lmax+1)²` matrix of harmonics, so memory is `O(M + (lmax+1)²)`.
"""
struct DirectSHTSphericalPlan{T, VV <: AbstractVector{T}, A, R, LT, OP, FO, WS, HA} <: AbstractSphericalPlan
    lmax::Int
    M::Int
    K::Int
    B::Int
    theta::VV        # (M) colatitudes
    phi::VV          # (M) longitudes
    weights::VV      # (M) spatial-mean quadrature weights, summing to 1
    rtol::T
    maxiter::Int
    analysis::A
    sampling::R
    tab::LT          # read-only, shared with every copy
    op::OP           # the transforms' operator and scratch
    fit::FO          # `op` under the weights, the operator the fit runs on
    ls::WS           # the fit's `FTB.LSMRWorkspace`
    ha::HA           # (K, B) coefficients times a degree multiplier
end

function _direct_sht_plan(θ::V, φ::V, w::V, rtol::T, maxiter::Int, analysis::Plans.AbstractAnalysis, sampling, tab, lmax::Int,
                          B::Int) where {T, V <: AbstractVector{T}}
    M, K = length(θ), (lmax + 1)^2
    op = _SHTOperator(sampling, tab, _sht_scratch(T, lmax, B, sampling), B, Matrix{T}(undef, M, B),
                      Matrix{T}(undef, K, B))
    fit = FTB.RangeWeighted(op, w)
    ls = Plans._analysis_workspace(analysis, fit)
    ha = Matrix{T}(undef, K, B)
    return DirectSHTSphericalPlan{T, V, typeof(analysis), typeof(sampling), typeof(tab), typeof(op), typeof(fit), typeof(ls),
                                  typeof(ha)}(lmax, M, K, B, θ, φ, w, rtol, maxiter, analysis, sampling, tab, op, fit, ls, ha)
end

function DirectSHTSphericalPlan(θ::AbstractVector, φ::AbstractVector, lmax::Int, ::Type{T};
                                rtol::Real = 1.0e-8, maxiter::Int = 500, weights = nothing,
                                ntrans::Int = 1, analysis::Plans.AbstractAnalysis = Plans.LeastSquares()) where {T}
    M = length(θ)
    length(φ) == M || throw(DimensionMismatch("θ and φ must have equal length"))
    lmax >= 0 || throw(ArgumentError("lmax must be nonnegative; got $lmax"))
    ntrans >= 1 || throw(ArgumentError("ntrans must be at least 1; got $ntrans"))
    # Host arrays: the transforms visit the points one at a time.
    θT, φT = collect(T, θ), collect(T, φ)
    w = weights === nothing ? fill(one(T) / M, M) : collect(T, T.(weights) ./ sum(weights))
    return _direct_sht_plan(θT, φT, w, T(rtol), maxiter, analysis, _SHTRings(θT, φT), _SHTRecurrence(T, lmax),
                            lmax, ntrans)
end

Base.show(io::IO, p::DirectSHTSphericalPlan{T}) where {T} =
    print(io, "DirectSHTSphericalPlan{", T, "}(lmax=", p.lmax, ", M=", p.M, ", ntrans=", p.B, ")")

# The point set, the quadrature weights and the tables are read-only and shared; a task gets its own
# operator scratch and solver workspace, which analysis writes its result through.
plan_points(p::DirectSHTSphericalPlan) = _direct_plan_points(p, p.sampling)
_direct_plan_points(p, ::_SHTRings) = (p.theta, p.phi)
plan_weights(p::DirectSHTSphericalPlan) = p.weights
plan_solver(p::DirectSHTSphericalPlan) = (rtol = p.rtol, maxiter = p.maxiter, analysis = p.analysis)
Plans.spectral_backend(::DirectSHTSphericalPlan) = SB.DirectSumSpectralBackend()
Plans.batch_width(p::DirectSHTSphericalPlan) = p.B

Plans.task_local_plan(p::DirectSHTSphericalPlan) =
    _direct_sht_plan(p.theta, p.phi, p.weights, p.rtol, p.maxiter, p.analysis, p.sampling, p.tab, p.lmax, p.B)

supports_batch(::DirectSHTSphericalPlan) = true
batch_plan(p::DirectSHTSphericalPlan, n::Integer) =
    n == p.B ? p : _direct_sht_plan(p.theta, p.phi, p.weights, p.rtol, p.maxiter, p.analysis, p.sampling, p.tab,
                                  p.lmax, Int(n))

# Analysis of `field`, one column per field of a `(M, B)` stack. A field may be a length-`M` vector
# (scattered) or the `(Nθ, Nφ)` grid matrix (structured), read in the order the plan's points were
# built. The fit's LSMR stops on `‖A†r‖ ≤ rtol ‖A†b‖` and returns its best iterate at `maxiter`.
sphere_coeffs(plan::DirectSHTSphericalPlan, field::AbstractArray) =
    sphere_coeffs!(sphere_coeffs_buffer(plan), plan, field)

function sphere_coeffs!(a::AbstractArray, plan::DirectSHTSphericalPlan, field::AbstractArray)
    return _direct_analysis!(a, plan, field, plan.analysis)
end

function _direct_analysis!(a, plan::DirectSHTSphericalPlan{T}, field, ::Plans.LeastSquares) where {T}
    FTB.lsmr!(a, plan.fit, field, plan.ls;
              rtol = plan.rtol, maxiter = plan.maxiter, conlim = inv(eps(T)))
    return a
end

function _direct_analysis!(a, plan::DirectSHTSphericalPlan{T}, field, ::Plans.WeightedAdjoint) where {T}
    # The adjoint of `w ⊙ f`, through the fit's range buffer, which the adjoint does not use; a point of
    # zero weight contributes nothing whatever `f` holds there.
    t, M, B = plan.op.t, plan.M, plan.B
    @inbounds for b in 1:B, n in 1:M
        i = (b - 1) * M + n
        wn = plan.weights[n]
        t[i] = iszero(wn) ? zero(T) : wn * field[i]
    end
    return _sht_adjoint!(a, plan.sampling, plan.tab, plan.op.scr, t, B)
end

sphere_coeffs_buffer(plan::DirectSHTSphericalPlan{T}) where {T} = Matrix{T}(undef, plan.K, plan.B)
sphere_field_buffer(plan::DirectSHTSphericalPlan{T}) where {T} = Matrix{T}(undef, plan.M, plan.B)

# Synthesis: out = Σ_k h(ℓ_k)·C_k·Y_k for each column. The coefficients are laid out `(ℓ, m = -ℓ:ℓ)` in
# order, so `h` is evaluated once per degree.
function sphere_apply!(out::AbstractArray, plan::DirectSHTSphericalPlan{T}, C, h::H) where {T,H}
    _scale_direct_coeffs!(plan.ha, plan, C, h)
    _sht_synthesize!(out, plan.sampling, plan.tab, plan.op.scr, plan.ha, plan.B)
    return out
end

function _scale_direct_coeffs!(dst, plan::DirectSHTSphericalPlan{T}, C, h) where {T}
    K, B = plan.K, plan.B
    @inbounds for l in 0:plan.lmax
        hl = T(h(l))
        for b in 1:B
            o = (b - 1) * K
            for k in (o + l * l + 1):(o + (l + 1) * (l + 1))
                dst[k] = hl * C[k]
            end
        end
    end
    return dst
end

sphere_npoints(p::DirectSHTSphericalPlan) = p.M

# Spherical average under the plan's quadrature weights: uniform `1/M` by default for scattered
# points, Fejér weights on the structured grid. A plan `B > 1` wide averages each column of an
# `(M, B)` stack.
sphere_mean(plan::DirectSHTSphericalPlan, field::AbstractArray) =
    LinearAlgebra.dot(plan.weights, vec(field))

# The degree-0 harmonic is 1, so the mean is the first coefficient of each column.
coeffs_mean(plan::DirectSHTSphericalPlan, C, b::Int = 1) = C[(b - 1) * plan.K + 1]

function sphere_means!(out, plan::DirectSHTSphericalPlan, field)
    LinearAlgebra.mul!(out, transpose(reshape(field, plan.M, plan.B)), plan.weights)
    return out
end

function coeffs_means!(out, plan::DirectSHTSphericalPlan, C)
    copyto!(out, view(C, 1, :))
    return out
end

# The weighted adjoint on the Gauss–Legendre grid of band limit `L`, sharing the plan's Legendre table.
function grid_plan(p::DirectSHTSphericalPlan{T}, L::Integer) where {T}
    θ, φ, w = gauss_legendre_grid(Int(L), T)
    sampling = _SHTTensorGrid(θ[1:(Int(L) + 1)], 2Int(L) + 1)
    return _direct_sht_plan(θ, φ, w, p.rtol, p.maxiter, Plans.WeightedAdjoint(), sampling, p.tab, p.lmax, p.B)
end

# ---------------------------------------------------------------------------
# Scattered-sphere plan selection
# ---------------------------------------------------------------------------

"""
    nusht_spherical_plan(kind, θ, φ, lmax, T; rtol, maxiter, weights, ntrans, nufft, analysis)

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
    Plans.available(SB.NUFSHTSpectralBackend()) ? default_rtol(SB.NUFSHTSpectralBackend()) :
    default_rtol(SB.DirectSumSpectralBackend())

"""
    plan_spin(plan) -> (spin-0 plan, spin-1 plan) or nothing

The spin-weighted plan pair a scattered-sphere plan carries for
[`spherical_monogenic_components`](@ref ScatteringTransforms.spherical_monogenic_components), or
`nothing` for a plan built without one. Only the monogenic constructors build it. The pair is
working state, so concurrent use needs `Plans.task_local_plan`.
"""
plan_spin(::Any) = nothing

"""
    make_spherical_plan(kind, spectral, θ, φ, lmax, T; rtol, maxiter, weights, ntrans, nufft, analysis)
        -> AbstractSphericalPlan

Build the scattered-sphere plan selected by `spectral` over points `(θ, φ)` and band limit `lmax`.
`SpectralBackends.DirectSumSpectralBackend` is the dependency-free default;
`SpectralBackends.NUFSHTSpectralBackend` (or `SpectralBackends.AutoSpectralBackend` once the NUFSHT
extension is loaded) uses the NUFSHT fast path.

`ntrans = B` builds a plan that transforms `B` co-located fields per call, and `weights` are the
quadrature weights of the spherical mean. `LeastSquares()` fits coefficients and `WeightedAdjoint()`
applies the weighted adjoint. `kind = SphericalMonogenicScatteringTransform` builds NUFSHT's spin pair for
[`spherical_monogenic_components`](@ref ScatteringTransforms.spherical_monogenic_components).
The in-core plan takes the surface gradient of `(−Δ_S)^{-1/2}U⁰` and accepts only an `Auto` `nufft`.
"""
function make_spherical_plan(kind::Type, s::SB.AbstractDirectSumSpectralBackend, θ, φ, lmax, ::Type{T};
                             rtol::Real = default_rtol(s), maxiter::Int = 500, weights = nothing,
                             ntrans::Int = 1,
                             nufft::SB.AbstractSpectralBackend = SB.AutoSpectralBackend(),
                             analysis::Plans.AbstractAnalysis = Plans.LeastSquares()) where {T}
    nufft isa SB.AbstractAutoSpectralBackend || throw(ArgumentError(
        "the in-core spherical plan performs no NUFFT, so `nufft = $nufft` cannot be honoured; " *
        "pass transform = SpectralBackends.NUFSHTSpectralBackend() for a NUFFT-backed transform."))
    return DirectSHTSphericalPlan(θ, φ, lmax, T; rtol = rtol, maxiter = maxiter, weights = weights,
                                  ntrans = ntrans, analysis = analysis)
end
make_spherical_plan(kind::Type, ::SB.AbstractNUFSHTSpectralBackend, θ, φ, lmax, ::Type{T}; kwargs...) where {T} =
    nusht_spherical_plan(kind, θ, φ, lmax, T; kwargs...)
make_spherical_plan(kind::Type, ::SB.AbstractAutoSpectralBackend, θ, φ, lmax, ::Type{T}; kwargs...) where {T} =
    Plans.available(SB.NUFSHTSpectralBackend()) ?
    make_spherical_plan(kind, SB.NUFSHTSpectralBackend(), θ, φ, lmax, T; kwargs...) :
    make_spherical_plan(kind, SB.DirectSumSpectralBackend(), θ, φ, lmax, T; kwargs...)

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
    return FTB.equiangular_weights!(Vector{T}(undef, N), FTB.OpenNodes())
end

"""
    gauss_legendre_grid(lmax, T) -> (θ, φ, w)

The Gauss–Legendre grid of band limit `lmax`, flattened with colatitude fastest: the `lmax + 1` roots
of `P_{lmax+1}(cos θ)` by `2lmax + 1` equispaced longitudes, and quadrature weights `w` summing to 1.
The rule integrates the product of two harmonics of degree `≤ lmax` exactly, so its weighted adjoint
expands a degree-`lmax` field.
"""
function gauss_legendre_grid(lmax::Int, ::Type{T}) where {T}
    N, nφ = lmax + 1, 2lmax + 1
    θr, wr = _gauss_legendre_colatitudes(N, T)
    θ = vec(T[θr[j] for j in 1:N, _ in 1:nφ])
    φ = vec(T[2 * T(π) * (k - 1) / nφ for _ in 1:N, k in 1:nφ])
    w = vec(T[wr[j] / (2nφ) for j in 1:N, _ in 1:nφ])
    return θ, φ, w
end

function _gauss_legendre_colatitudes(N::Int, ::Type{T}) where {T}
    lat, w = Vector{T}(undef, N), Vector{T}(undef, N)
    FTB.gauss_legendre!(T, nothing, w, lat, N)
    return T(π) / 2 .- reverse(lat), reverse(w)
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
    return _direct_sht_plan(θf, φf, wf ./ sum(wf), T(rtol), maxiter, Plans.LeastSquares(),
                             _SHTTensorGrid(Θ, Nφ), _SHTRecurrence(T, lmax), lmax, 1)
end

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
    Plans.available(SB.FSHTSpectralBackend()) ? fsh_structured_plan(lmax, T) :
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

riesz_scratch(plan::DirectSHTSphericalPlan, field::AbstractArray) =
    _direct_riesz_scratch(plan, field, plan.sampling)

_direct_riesz_scratch(plan::DirectSHTSphericalPlan{T}, ::AbstractArray, ::_SHTRings) where {T} =
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
    rings = plan.sampling
    P, g = scr.P, scr.gc
    dc, ds, pc, ps = scr.dc, scr.ds, scr.pc, scr.ps
    s2 = sqrt(T(2))
    @inbounds for r in 1:_nrings(rings)
        _latitude_values!(P, plan.tab, rings.theta[r])
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
    return _direct_riesz_energy!(out, plan, C, h, scr, plan.sampling)
end

function _direct_riesz_energy!(out, plan, C, h, scr, ::_SHTRings)
    _scaled_coeffs!(scr, plan, C, h)
    _each_gradient((n, gθ, gφ) -> (@inbounds out[n] = gθ^2 + gφ^2), plan, scr)
    return out
end

# Pointwise monogenic decomposition on the direct SHT plan (spin-0 band-pass + spin-1 Riesz gradient).
function direct_monogenic_components(st::SphericalMonogenicScatteringTransform{<:Any, <:DirectSHTSphericalPlan},
                                     field::AbstractArray, j::Int)
    plan = st.plan
    σ²hi, σ²lo = st.sigma2[j], st.sigma2[j + 1]
    a = sphere_coeffs(plan, field)                                  # spin-0 SH coefficients
    U0 = similar(field)
    sphere_apply!(U0, plan, a, band_multiplier(σ²hi, σ²lo))         # band-pass U⁰ (real)
    # Coefficients of g = (−Δ_S)^{-1/2} U⁰, then its surface gradient `(∂_θ g, ∂_φ g / sinθ)` at each
    # point, the negation of what `_each_gradient` stores.
    uθ, uφ = similar(field), similar(field)
    _direct_riesz_components!(uθ, uφ, plan, a, riesz_potential_multiplier(σ²hi, σ²lo), plan.sampling)
    rnorm = sqrt.(uθ .^ 2 .+ uφ .^ 2)
    amplitude = sqrt.(U0 .^ 2 .+ rnorm .^ 2)
    phase = atan.(rnorm, U0)
    orientation = atan.(uφ, uθ)
    return (; bandpass = U0, riesz = (uθ, uφ), amplitude, phase, orientation)
end

function _direct_riesz_components!(uθ, uφ, plan, C, h, ::_SHTRings)
    scr = riesz_scratch(plan, uθ)
    _scaled_coeffs!(scr, plan, C, h)
    _each_gradient((n, gθ, gφ) -> (@inbounds (uθ[n] = -gθ; uφ[n] = -gφ)), plan, scr)
    return nothing
end

include("SphericalTensor.jl")

end # module SphericalCore
