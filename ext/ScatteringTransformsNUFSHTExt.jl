module ScatteringTransformsNUFSHTExt

using NUFSHT: NUFSHT
using FlowTransformBindings: FlowTransformBindings as FTB
using LinearAlgebra: LinearAlgebra
using SpectralBackends: SpectralBackends as SB
using ScatteringTransforms: ScatteringTransforms as ST

abstract type AbstractNUSHTSphericalPlan <: ST.SphericalCore.AbstractSphericalPlan end

# ---------------------------------------------------------------------------
# Scattered-sphere plan + the two SphericalCore interface methods
# ---------------------------------------------------------------------------

"""
    NUSHTSphericalPlan <: SphericalCore.AbstractSphericalPlan

Scattered-point spherical plan over a `NUFSHT.NUSHTplan` on the points `(theta, phi)`, which it
retains so a task or another process can rebuild on them. With `LeastSquares()`, analysis is a band-limited
least-squares solve to `rtol` within `maxiter` iterations on `fit`, the plan under `weights` where they are
given, which needs roughly `M ≳ (lmax+1)²` well-distributed points, and `ws` is its LSMR workspace, so an
analysis allocates none; without, it is one adjoint transform weighted by `qweights`, the mean's weights
times `4π`. `weights`, summing to 1, are the quadrature of the spherical mean and the fit, or `nothing`
for the sample mean and the unweighted fit. A plan built by the monogenic constructors also holds
`spin`, the spin-weighted plan pair `spherical_monogenic_components` synthesises with after the scalar fit.
"""
struct NUSHTSphericalPlan{P, F, V<:AbstractVector, T<:Real, NB, SP, A, W, WV, QW} <: AbstractNUSHTSphericalPlan
    plan::P
    fit::F           # `plan`, or `plan` under `weights`: the operator of the fit
    M::Int
    lmax::Int
    theta::V
    phi::V
    rtol::T
    maxiter::Int
    nufft::NB        # the *resolved* NUFFT backend driving `plan`, never an `Auto` request
    spin::SP         # (spin-0, spin-1) plan pair for `spherical_monogenic_components`, or `nothing`
    ws::W            # the fit's workspace, or `nothing` for the weighted adjoint
    weights::WV
    analysis::A
    qweights::QW     # the adjoint's weights over orthonormal harmonics, or `nothing` for the fit
end

struct TensorSphericalPlan{P, V, R, W} <: AbstractNUSHTSphericalPlan
    plan::P
    lmax::Int
    M::Int
    theta::V
    phi::R
    weights::W
    qweights::W
end

function TensorSphericalPlan(theta, phi, weights, lmax::Int, B::Int)
    T = eltype(theta)
    plan = NUFSHT.make_plan(T, theta, phi, lmax, NUFSHT.TensorProduct(); ntrans = B)
    return TensorSphericalPlan(plan, lmax, length(theta) * length(phi), theta, phi, weights,
                                (4 * T(π)) .* weights)
end

ST.SphericalCore.sphere_npoints(p::TensorSphericalPlan) = p.M
ST.SphericalCore.grid_band(p::TensorSphericalPlan) = length(p.theta) - 1
ST.SphericalCore.plan_points(::TensorSphericalPlan) = nothing
ST.SphericalCore.plan_weights(p::TensorSphericalPlan) = p.weights
ST.SphericalCore.plan_solver(::TensorSphericalPlan) = nothing
ST.SphericalCore.supports_batch(::TensorSphericalPlan) = true
ST.Plans.batch_width(p::TensorSphericalPlan) = p.plan.B
ST.Plans.close_plan!(p::TensorSphericalPlan) = (NUFSHT.close!(p.plan); nothing)
ST.Plans.spectral_backend(::TensorSphericalPlan) = SB.NUFSHTSpectralBackend()
ST.Plans.task_local_plan(p::TensorSphericalPlan) =
    TensorSphericalPlan(p.theta, p.phi, p.weights, p.lmax, p.plan.B)
ST.SphericalCore.batch_plan(p::TensorSphericalPlan, B::Integer) =
    B == p.plan.B ? p : TensorSphericalPlan(p.theta, p.phi, p.weights, p.lmax, Int(B))

function NUSHTSphericalPlan(plan, M, lmax, theta, phi, rtol, maxiter, nufft, spin = nothing;
                            weights = nothing, analysis::ST.Plans.AbstractAnalysis = ST.Plans.LeastSquares())
    return _spherical_plan(plan, M, lmax, theta, phi, rtol, maxiter, nufft, spin, weights, analysis)
end

function _spherical_plan(plan, M, lmax, theta, phi, rtol, maxiter, nufft, spin, weights, analysis)
    fit = weights === nothing ? plan : FTB.RangeWeighted(plan, weights)
    ws, qw = _analysis_work(analysis, fit, theta, weights, M)
    return NUSHTSphericalPlan{typeof(plan), typeof(fit), typeof(theta), typeof(rtol), typeof(nufft), typeof(spin),
                              typeof(analysis), typeof(ws), typeof(weights), typeof(qw)}(
        plan, fit, M, lmax, theta, phi, rtol, maxiter, nufft, spin, ws, weights, analysis, qw)
end

_analysis_work(::ST.Plans.LeastSquares, fit, θ, w, M) = (FTB.LSMRWorkspace(fit), nothing)
_analysis_work(::ST.Plans.WeightedAdjoint, fit, θ, w, M) = (nothing, _quadrature_weights(θ, w, M))

# The mean's weights, summing to 1, times the sphere's area `4π`; uniform without them.
_quadrature_weights(θ::AbstractVector, ::Nothing, M::Int) = fill!(similar(θ), 4 * eltype(θ)(π) / M)
_quadrature_weights(θ::AbstractVector, w::AbstractVector, ::Int) = (4 * eltype(θ)(π)) .* w

# `show` names the batch width and the resolved NUFFT backend, which is direct summation when no fast
# NUFFT extension is loaded.
Base.show(io::IO, p::NUSHTSphericalPlan) =
    print(io, "NUSHTSphericalPlan(lmax=", p.lmax, ", M=", p.M, ", ntrans=", p.plan.B,
          ", nufft=", nameof(typeof(p.nufft)), ", spin=", p.spin !== nothing, ")")

# Every transform writes through a `NUSHTplan`'s coefficient, field and phase buffers, so each task
# builds its own, once, from the points and band limit this wrapper retains. NUFSHT and
# FlowTransformBindings build their plans under FFTW.jl's planner lock, so a build may run from any
# task. NUFSHT transforms `ntrans` co-located fields per call, so a batched sibling is a rebuild at the
# width of the batch call.
ST.SphericalCore.supports_batch(::NUSHTSphericalPlan) = true

function ST.SphericalCore.batch_plan(p::NUSHTSphericalPlan, B::Integer)
    B == p.plan.B && return p
    plan = NUFSHT.make_plan(eltype(p.theta), p.theta, p.phi, p.lmax;
                            ntrans = Int(B), nufft = p.nufft, tol = p.plan.tol)
    # The spin pair holds the buffers its synthesis writes through, so the widened plan gets its own:
    # one task may use it while another uses `p`.
    return NUSHTSphericalPlan(plan, p.M, p.lmax, p.theta, p.phi, p.rtol, p.maxiter, p.nufft,
                              _rebuild_spin(p); weights = p.weights, analysis = p.analysis)
end

ST.SphericalCore.plan_nufft(p::NUSHTSphericalPlan) = p.nufft
ST.Plans.batch_width(p::NUSHTSphericalPlan) = p.plan.B

# `NUFSHT.close!` frees the NUFFT plans a NUSHT plan owns and is idempotent.
function ST.Plans.close_plan!(p::NUSHTSphericalPlan)
    NUFSHT.close!(p.plan)
    p.spin === nothing || (NUFSHT.close!(p.spin[1]); NUFSHT.close!(p.spin[2]))
    return nothing
end
ST.SphericalCore.plan_spin(p::NUSHTSphericalPlan) = p.spin
ST.SphericalCore.plan_points(p::NUSHTSphericalPlan) = (p.theta, p.phi)
ST.SphericalCore.plan_weights(p::NUSHTSphericalPlan) = p.weights
ST.SphericalCore.plan_solver(p::NUSHTSphericalPlan) = (rtol = p.rtol, maxiter = p.maxiter, analysis = p.analysis)
ST.Plans.spectral_backend(::NUSHTSphericalPlan) = SB.NUFSHTSpectralBackend()
ST.Plans.available(::SB.NUFSHTSpectralBackend) = true

function ST.Plans.task_local_plan(p::NUSHTSphericalPlan)
    # `ntrans` and the resolved backend are carried over, so the task's plan takes the batch the
    # cascade feeds it and runs the transform the original runs.
    plan = NUFSHT.make_plan(eltype(p.theta), p.theta, p.phi, p.lmax;
                            ntrans = p.plan.B, nufft = p.nufft, tol = p.plan.tol)
    return NUSHTSphericalPlan(plan, p.M, p.lmax, p.theta, p.phi, p.rtol, p.maxiter, p.nufft,
                              _rebuild_spin(p); weights = p.weights, analysis = p.analysis)
end

# One spin plan per spin weight over the plan's own points, on the scalar plan's NUFFT library, or
# `nothing` for a plan that carries none. The spin path drives no FastTransforms code, so it needs no
# OpenMP pin.
_spin_plans(::Type{FE}, theta, phi, lmax::Int, nufft) where {FE} =
    (NUFSHT.make_spin_plan(FE, theta, phi, lmax, 0; nufft = nufft),
     NUFSHT.make_spin_plan(FE, theta, phi, lmax, 1; nufft = nufft))

_rebuild_spin(p::NUSHTSphericalPlan) =
    p.spin === nothing ? nothing :
    _spin_plans(Complex{eltype(p.theta)}, p.theta, p.phi, p.lmax, p.nufft)

# Generic per-degree spectral multiplier h(ℓ): NUFSHT's apply_transfer! dispatches on
# `kernel_transfer(filter, ℓ)`, so any object with this method is a valid transfer.
struct _FnTransfer{F}
    f::F
end
NUFSHT.kernel_transfer(t::_FnTransfer, ℓ) = t.f(ℓ)

function ST.SphericalCore.sphere_coeffs(plan::AbstractNUSHTSphericalPlan, field::AbstractArray)
    return ST.SphericalCore.sphere_coeffs!(NUFSHT.allocate_coefficients(plan.plan), plan, field)
end

# Analysis is the band-limited least-squares solve, or the weighted adjoint; NUFSHT projects onto the
# valid `(l, m)` modes, so the coefficients come back in the degree-`<= lmax` span.
#
# `field` is an `(M,)` field or an `(M, B)` stack matching the plan's `ntrans`.
function ST.SphericalCore.sphere_coeffs!(C, plan::NUSHTSphericalPlan, field::AbstractVecOrMat)
    fill!(C, zero(eltype(C)))
    return _analyze!(C, plan, field, plan.ws)
end

function ST.SphericalCore.sphere_coeffs!(C, plan::TensorSphericalPlan, field::AbstractArray)
    NUFSHT.nusht_analyze!(C, field, plan.plan; weights = plan.qweights)
    return C
end

function _analyze!(C, plan::NUSHTSphericalPlan, field, ws::FTB.LSMRWorkspace)
    NUFSHT.nusht_solve!(C, field, plan.fit; ws = ws, rtol = plan.rtol, maxiter = plan.maxiter)
    return C
end

function _analyze!(C, plan::NUSHTSphericalPlan, field, ::Nothing)
    NUFSHT.nusht_analyze!(C, field, plan.plan; weights = plan.qweights)
    return C
end

# The caller's own coefficient array, which `sphere_coeffs!` clears and solves into.
ST.SphericalCore.sphere_coeffs_buffer(plan::AbstractNUSHTSphericalPlan) =
    NUFSHT.allocate_coefficients(plan.plan)
ST.SphericalCore.sphere_field_buffer(plan::AbstractNUSHTSphericalPlan) =
    similar(plan.theta, plan.M, plan.plan.B)

# Apply the per-degree multiplier `h(ℓ)` to a copy of the coefficients and synthesise at the points.
# `nusht_synthesize!` is the wrapped plan's own filter-then-synthesise step: it copies into scratch it
# owns and allocates on first use, so `C` survives for the next band and repeated bands allocate
# nothing.
ST.SphericalCore.sphere_apply!(out::AbstractArray, plan::AbstractNUSHTSphericalPlan, C, h) =
    NUFSHT.nusht_synthesize!(out, C, _FnTransfer(h), plan.plan)

# NUFSHT's scalar coefficients are in the real `sph_mode` layout.
ST.SphericalCore.riesz_scratch(plan::AbstractNUSHTSphericalPlan, field::AbstractArray) =
    ST.SphericalCore.sph_layout_riesz_scratch(plan, field, plan.lmax)
ST.SphericalCore.sphere_riesz_energy!(out::AbstractArray, plan::AbstractNUSHTSphericalPlan, C, h, scr) =
    ST.SphericalCore.sph_layout_riesz_energy!(out, plan, C, h, plan.lmax, scr)

# The spherical average under the plan's quadrature weights, or the sample mean over the
# (quasi-uniform) scattered points without them. An `(M, B)` stack averages each field separately, so
# a batch gets one mean per column.
ST.SphericalCore.sphere_mean(plan::NUSHTSphericalPlan, field::AbstractArray) =
    _mean(plan.weights, field, plan.M)
ST.SphericalCore.sphere_mean(plan::TensorSphericalPlan, field::AbstractArray) =
    LinearAlgebra.dot(plan.weights, vec(field))
_mean(::Nothing, field::AbstractArray, M::Int) = sum(field) / M
_mean(w::AbstractVector, field::AbstractArray, ::Int) = LinearAlgebra.dot(w, vec(field))

# `Y₀₀ = 1/√(4π)` in the orthonormal `sph_mode` layout. A sum over the one slot reads a device array
# without scalar indexing.
ST.SphericalCore.coeffs_mean(p::AbstractNUSHTSphericalPlan, C, b::Int = 1) =
    sum(view(C, 1:1, 1:1, b:b)) / sqrt(4 * eltype(p.theta)(π))

function ST.SphericalCore.coeffs_means!(out, p::AbstractNUSHTSphericalPlan, C)
    out .= reshape(view(C, 1:1, 1:1, :), :) ./ sqrt(4 * eltype(p.theta)(π))
    return out
end

function ST.SphericalCore.sphere_means!(out, p::AbstractNUSHTSphericalPlan, field)
    return _means!(out, p.weights, reshape(field, p.M, p.plan.B), p.M)
end

function _means!(out, ::Nothing, field, M)
    sum!(reshape(out, 1, :), field)
    out ./= M
    return out
end

function _means!(out, weights::AbstractVector, field, M)
    LinearAlgebra.mul!(out, transpose(field), weights)
    return out
end

# The weighted adjoint on the Gauss–Legendre grid of band limit `L`, on the plan's band limit, batch width,
# NUFFT library and tolerance, its nodes in the array type of the plan's points.
function ST.SphericalCore.grid_plan(p::NUSHTSphericalPlan, L::Integer)
    T = eltype(p.theta)
    θh, wr = ST.SphericalCore._gauss_legendre_colatitudes(Int(L) + 1, T)
    nφ = 2Int(L) + 1
    φ = range(zero(T); step = 2T(π) / nφ, length = nφ)
    θ = copyto!(similar(p.theta, length(θh)), θh)
    wh = vec(repeat(wr ./ (2nφ), 1, nφ))
    w = copyto!(similar(p.theta, length(wh)), wh)
    return TensorSphericalPlan(θ, φ, w, p.lmax, p.plan.B)
end

# ---------------------------------------------------------------------------
# The plan `spherical_scattering` / `spherical_monogenic_scattering` build when `transform` selects
# NUFSHT
# ---------------------------------------------------------------------------

"""
    nusht_spherical_plan(kind, θ, φ, lmax, T; rtol, maxiter = 500, weights = nothing, ntrans = 1,
                         nufft = AutoSpectralBackend(), analysis = LeastSquares())

`ntrans` is NUFSHT's batch size: the plan transforms `ntrans` fields sampled at the same points per
call. `nufft` names the NUFFT library, resolved before the plan is built. `weights` are the
quadrature weights of the spherical mean and of the least-squares fit, normalized to sum to 1.
`WeightedAdjoint()` applies the adjoint weighted by them.

`kind = SphericalMonogenicScatteringTransform` builds the spin-0/spin-1 pair `spherical_monogenic_components` synthesises
with (see `SphericalCore.plan_spin`). Only the monogenic constructors ask for it.
"""
function ST.SphericalCore.nusht_spherical_plan(kind::Type, pts_theta::AbstractVector, pts_phi::AbstractVector,
                                               lmax::Int, ::Type{T};
                                               rtol::Real = ST.SphericalCore.default_rtol(SB.NUFSHTSpectralBackend()),
                                               maxiter::Int = 500, weights = nothing,
                                               ntrans::Int = 1,
                                               nufft::SB.AbstractSpectralBackend = SB.AutoSpectralBackend(),
                                               analysis::ST.Plans.AbstractAnalysis = ST.Plans.LeastSquares()) where {T<:Real}
    # The plan owns copies of the points, in their own array type, so a device-resident transform
    # stays on its device.
    θ = T.(pts_theta)
    φ = T.(pts_phi)
    length(φ) == length(θ) || throw(DimensionMismatch("θ and φ must have equal length"))
    w = weights === nothing ? nothing : _point_weights(θ, T.(weights) ./ sum(weights))
    # `nufft` is resolved before building, so the plan records which NUFFT runs. It resolves against
    # the field element type `T`, the one the plan is built with.
    nb = NUFSHT._resolve_nufft(nufft, T)
    plan = NUFSHT.make_plan(T, θ, φ, lmax; ntrans = ntrans, nufft = nb)
    # NUFSHT's positional argument is the field element type: a real one selects
    # its folded real layout, and the spin field is complex.
    sp = _spin_for(kind, Complex{T}, θ, φ, lmax, nb)
    return NUSHTSphericalPlan(plan, length(θ), lmax, θ, φ, T(rtol), maxiter, nb, sp; weights = w, analysis = analysis)
end

_spin_for(::Type{ST.SphericalCore.SphericalScatteringTransform}, args...) = nothing
_spin_for(::Type{ST.SphericalCore.SphericalMonogenicScatteringTransform}, args...) = _spin_plans(args...)

# Weights `w`, one per point, on the array type of the points `θ`.
function _point_weights(θ::AbstractVector, w::AbstractVector)
    length(w) == length(θ) || throw(DimensionMismatch(
        "one weight per point: $(length(θ)) points and $(length(w)) weights"))
    return copyto!(similar(θ), w)
end

# ---------------------------------------------------------------------------
# Pointwise spherical monogenic decomposition by spin-weighted synthesis.
#
# For band-pass `U⁰_j = b_j(ℓ)·a_ℓm` (spin-0), the spin-1 Riesz field is
#   U^R_j = ð∘(−Δ_S)^{-1/2} U⁰_j,   coeffs = √(ℓ(ℓ+1))·(1/√(ℓ(ℓ+1)))·b_j(ℓ)·a_ℓm = b_j(ℓ)·a_ℓm  (ℓ≥1),
# so one coefficient array `sf = b_j(ℓ)·a` synthesised at spin 0 gives the band-pass field and at
# spin 1 the complex tangent Riesz field `U = u_θ + i u_φ`. Its amplitude √(U⁰²+|U^R|²) agrees with
# `sphere_riesz_energy!` to the SHT's accuracy.
# ---------------------------------------------------------------------------

function ST.spherical_monogenic_components(st::ST.SphericalCore.SphericalMonogenicScatteringTransform{<:Any, <:NUSHTSphericalPlan},
                                           field::AbstractVector, j::Int)
    p = st.plan
    lmax = st.lmax
    T = eltype(p.theta)
    p0, p1 = p.spin
    C = ST.SphericalCore.sphere_coeffs(p, field)
    bfn = ST.SphericalCore.band_multiplier(st.sigma2[j], st.sigma2[j + 1])
    NUFSHT.apply_transfer!(C, _FnTransfer(bfn), lmax)
    sf = NUFSHT.allocate_coefficients(p0)
    NUFSHT.scalar_to_spin0!(sf, C)

    M = p.M
    U0 = similar(p.theta, Complex{T}, M)
    UR = similar(U0)
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
