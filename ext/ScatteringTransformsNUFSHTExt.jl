module ScatteringTransformsNUFSHTExt

# Scattered-sphere plan on NUFSHT. Analysis solves `min ‖A c − x‖` over the band-limited
# coefficients: `nusht_type1!` is not the adjoint of synthesis and mis-scales degree by degree.

using NUFSHT: NUFSHT
using FlowTransformBindings: FlowTransformBindings as FTB
using LinearAlgebra: LinearAlgebra
using SpectralBackends: SpectralBackends as SB
using ScatteringTransforms: ScatteringTransforms as ST

# ---------------------------------------------------------------------------
# Scattered-sphere plan + the two SphericalCore interface methods
# ---------------------------------------------------------------------------

"""
    NUSHTSphericalPlan{P,V,T,NB,SP,W,SW,WV} <: SphericalCore.AbstractSphericalPlan

Scattered-point spherical plan over a `NUFSHT.NUSHTplan` on the points `(theta, phi)`, which it
retains so a task or another process can rebuild on them. Analysis is a band-limited least-squares
solve to `rtol` within `maxiter` iterations, which needs roughly `M ≳ (lmax+1)²` well-distributed
points; `ws` is its LSMR workspace, so an analysis allocates none. `weights`, summing to 1, are the
quadrature of the spherical mean, or `nothing` for the sample mean. A plan built by the monogenic
constructors also holds `spin`, the spin-weighted plan pair `spherical_monogenic_components`
synthesises with, and `spin_ws`, its solve's workspace.
"""
struct NUSHTSphericalPlan{P, V<:AbstractVector, T<:Real, NB, SP, W, SW, WV} <: ST.SphericalCore.AbstractSphericalPlan
    plan::P
    M::Int
    lmax::Int
    theta::V
    phi::V
    rtol::T
    maxiter::Int
    nufft::NB        # the *resolved* NUFFT backend driving `plan`, never an `Auto` request
    spin::SP         # (spin-0, spin-1) plan pair for `spherical_monogenic_components`, or `nothing`
    ws::W
    spin_ws::SW      # the spin-0 solve's workspace, or `nothing` alongside `spin`
    weights::WV
end

function NUSHTSphericalPlan(plan, M, lmax, theta, phi, rtol, maxiter, nufft, spin = nothing;
                            weights = nothing)
    ws = FTB.LSMRWorkspace(plan)
    sws = spin === nothing ? nothing : FTB.LSMRWorkspace(spin[1])
    return NUSHTSphericalPlan{typeof(plan), typeof(theta), typeof(rtol), typeof(nufft), typeof(spin),
                              typeof(ws), typeof(sws), typeof(weights)}(
        plan, M, lmax, theta, phi, rtol, maxiter, nufft, spin, ws, sws, weights)
end

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
                              _rebuild_spin(p); weights = p.weights)
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
ST.SphericalCore.plan_solver(p::NUSHTSphericalPlan) = (rtol = p.rtol, maxiter = p.maxiter)
ST.Plans.spectral_backend(::NUSHTSphericalPlan) = SB.NUFSHTSpectralBackend()

function ST.Plans.task_local_plan(p::NUSHTSphericalPlan)
    # `ntrans` and the resolved backend are carried over, so the task's plan takes the batch the
    # cascade feeds it and runs the transform the original runs.
    plan = NUFSHT.make_plan(eltype(p.theta), p.theta, p.phi, p.lmax;
                            ntrans = p.plan.B, nufft = p.nufft, tol = p.plan.tol)
    return NUSHTSphericalPlan(plan, p.M, p.lmax, p.theta, p.phi, p.rtol, p.maxiter, p.nufft,
                              _rebuild_spin(p); weights = p.weights)
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

function ST.SphericalCore.sphere_coeffs(plan::NUSHTSphericalPlan, field::AbstractVecOrMat)
    return ST.SphericalCore.sphere_coeffs!(NUFSHT.allocate_coefficients(plan.plan), plan, field)
end

# Analysis is the band-limited least-squares solve; NUFSHT projects onto the valid `(l, m)` modes
# inside its own iteration, so the coefficients come back in the degree-`<= lmax` span. A solve that
# stops at `maxiter` keeps its best iterate; one with no usable answer throws.
#
# `field` is an `(M,)` field or an `(M, B)` stack matching the plan's `ntrans`.
function ST.SphericalCore.sphere_coeffs!(C, plan::NUSHTSphericalPlan, field::AbstractVecOrMat)
    fill!(C, zero(eltype(C)))
    NUFSHT.nusht_solve!(C, field, plan.plan; ws = plan.ws, rtol = plan.rtol, maxiter = plan.maxiter)
    ST.SphericalCore.check_sphere_solve(plan.ws, plan.M, plan.lmax, plan.rtol, plan.maxiter)
    return C
end

# The caller's own coefficient array, which `sphere_coeffs!` clears and solves into.
ST.SphericalCore.sphere_coeffs_buffer(plan::NUSHTSphericalPlan) =
    NUFSHT.allocate_coefficients(plan.plan)

# Apply the per-degree multiplier `h(ℓ)` to a copy of the coefficients and synthesise at the points.
# `nusht_synthesize!` is the wrapped plan's own filter-then-synthesise step: it copies into scratch it
# owns and allocates on first use, so `C` survives for the next band and repeated bands allocate
# nothing.
ST.SphericalCore.sphere_apply!(out::AbstractVecOrMat, plan::NUSHTSphericalPlan, C, h) =
    NUFSHT.nusht_synthesize!(out, C, _FnTransfer(h), plan.plan)

# NUFSHT's scalar coefficients are in the real `sph_mode` layout.
ST.SphericalCore.riesz_scratch(plan::NUSHTSphericalPlan, field::AbstractVector) =
    ST.SphericalCore.sph_layout_riesz_scratch(plan, field, plan.lmax)
ST.SphericalCore.sphere_riesz_energy!(out::AbstractVector, plan::NUSHTSphericalPlan, C, h, scr) =
    ST.SphericalCore.sph_layout_riesz_energy!(out, plan, C, h, plan.lmax, scr)

# The spherical average under the plan's quadrature weights, or the sample mean over the
# (quasi-uniform) scattered points without them. An `(M, B)` stack averages each field separately, so
# a batch gets one mean per column.
ST.SphericalCore.sphere_mean(plan::NUSHTSphericalPlan, field::AbstractVecOrMat) =
    _mean(plan.weights, field, plan.M)
_mean(::Nothing, field::AbstractVector, M::Int) = sum(field) / M
_mean(::Nothing, field::AbstractMatrix, M::Int) = vec(sum(field; dims = 1)) ./ M
_mean(w::AbstractVector, field::AbstractVector, ::Int) = LinearAlgebra.dot(w, field)
_mean(w::AbstractVector, field::AbstractMatrix, ::Int) = transpose(field) * w

# ---------------------------------------------------------------------------
# The plan `spherical_scattering` / `spherical_monogenic_scattering` build when `transform` selects
# NUFSHT
# ---------------------------------------------------------------------------

"""
    nusht_spherical_plan(θ, φ, lmax, T; rtol, maxiter = 500, weights = nothing, ntrans = 1,
                         spin = false, nufft = AutoSpectralBackend())

`ntrans` is NUFSHT's batch size: the plan transforms `ntrans` fields sampled at the same points per
call. `nufft` names the NUFFT library, resolved before the plan is built. `weights` are the
quadrature weights of the spherical mean, normalized to sum to 1.

`spin = true` also builds the spin-0/spin-1 plan pair `spherical_monogenic_components` synthesises
with (see `SphericalCore.plan_spin`). Only the monogenic constructors ask for it.
"""
function ST.SphericalCore.nusht_spherical_plan(pts_theta::AbstractVector, pts_phi::AbstractVector,
                                               lmax::Int, ::Type{T};
                                               rtol::Real = ST.SphericalCore.default_rtol(SB.NUFSHTSpectralBackend()),
                                               maxiter::Int = 500, weights = nothing,
                                               ntrans::Int = 1, spin::Bool = false,
                                               nufft::SB.AbstractSpectralBackend = SB.AutoSpectralBackend()) where {T<:Real}
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
    sp = spin ? _spin_plans(Complex{T}, θ, φ, lmax, nb) : nothing
    return NUSHTSphericalPlan(plan, length(θ), lmax, θ, φ, T(rtol), maxiter, nb, sp; weights = w)
end

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

function ST.spherical_monogenic_components(st::ST.SphericalCore.SphericalMonogenicScattering{<:Any, <:NUSHTSphericalPlan},
                                           field::AbstractVector, j::Int)
    p = st.plan
    lmax = st.lmax
    T = eltype(p.theta)
    # The spin-weighted plans come from the plan `spherical_monogenic_scattering` built; a plan
    # without them gets a pair built here.
    p0, p1 = ST.SphericalCore.plan_spin(p) === nothing ?
             _spin_plans(Complex{T}, p.theta, p.phi, lmax, p.nufft) : ST.SphericalCore.plan_spin(p)
    ws0 = p.spin_ws === nothing ? FTB.LSMRWorkspace(p0) : p.spin_ws
    # Complex spin-0 coefficients of the field in NUFSHT's dense spin layout, by least-squares
    # inversion, so the band-pass and Riesz fields below share one set of coefficients.
    a = zeros(Complex{T}, lmax + 1, 2lmax + 1)
    NUFSHT.nusht_solve_spin!(a, Complex{T}.(field), p0; ws = ws0)

    # sf = b_j(ℓ)·a  (b_j(0)=0, so the ℓ=0 term vanishes as the Riesz operator requires).
    bfn = ST.SphericalCore.band_multiplier(st.sigma2[j + 1], st.sigma2[j])
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
