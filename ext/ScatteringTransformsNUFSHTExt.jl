module ScatteringTransformsNUFSHTExt

"""
    ScatteringTransformsNUFSHTExt — scattered-point spherical scattering on S² (NUFSHT backend)

Provides the **scattered-sphere** backend for the shared spherical scattering core
(`SphericalCore`): a scalar field sampled at arbitrary points `(θ, φ)` on S², analysed/synthesised
via NUFSHT. The DoG band-pass bank, the S0/S1/S2 cascade, and the spin-0 Bochner monogenic amplitude
all live in `SphericalCore`; this extension only supplies

  * a plan wrapper `NUSHTSphericalPlan` over `NUFSHT.NUSHTplan`;
  * `sphere_coeffs` (band-limited least-squares analysis) and `sphere_apply!` (per-degree multiply
    via `apply_transfer!` + synthesis via `nusht_type2!`);
  * `sphere_mean` (unweighted sample mean over the scattered points);

plus the constructors `spherical_scattering` / `spherical_monogenic_scattering`.

Analysis solves `min ‖A c − x‖` over the band-limited coefficients rather than applying
`nusht_type1!`, which is not the adjoint of the synthesis and mis-scales degree by degree. The
`SphericalCore` cascade analyses each field once and reuses its coefficients across all bands, so the
iterative solve runs once per field rather than once per band.
"""

using NUFSHT: NUFSHT
using SpectralBackends: SpectralBackends as SB
using ScatteringTransforms: ScatteringTransforms as ST

# ---------------------------------------------------------------------------
# Scattered-sphere plan + the two SphericalCore interface methods
# ---------------------------------------------------------------------------

"""
    NUSHTSphericalPlan{P,V,T} <: SphericalCore.AbstractSphericalPlan

Scattered-point spherical plan wrapping a `NUFSHT.NUSHTplan`, the sample count `M`, the band limit
`lmax`, the scattered points `(theta, phi)` (retained so spin-weighted plans can be built on the same
points for `spherical_monogenic_components`), and the CG-inversion tolerance/iteration cap used by
[`sphere_coeffs`](@ref). Analysis is a band-limited least-squares solve, not `nusht_type1!`:
on scattered points the adjoint mis-scales the coefficients degree-dependently, so the exact inversion
is needed for correct absolute magnitudes. Accurate inversion needs the sampling to resolve the band
limit, i.e. roughly `M ≳ (lmax+1)²` well-distributed points.
"""
struct NUSHTSphericalPlan{P, V<:AbstractVector, T<:Real, C, NB} <: ST.SphericalCore.AbstractSphericalPlan
    plan::P
    M::Int
    lmax::Int
    theta::V
    phi::V
    rtol::T
    maxiter::Int
    cbuf::C          # coefficient scratch filtered per band, so `sphere_apply!` allocates nothing
    nufft::NB        # the *resolved* NUFFT backend driving `plan`, never an `Auto` request
end

NUSHTSphericalPlan(plan, M, lmax, theta, phi, rtol, maxiter, nufft) =
    NUSHTSphericalPlan(plan, M, lmax, theta, phi, rtol, maxiter, zero(plan.C), nufft)

# The NUFFT backend and batch size are shown because they dominate this plan's speed and are
# otherwise invisible: an unresolved request falls back to direct summation when no fast NUFFT
# extension is loaded, which is asymptotically slower and looks identical from the outside.
Base.show(io::IO, p::NUSHTSphericalPlan) =
    print(io, "NUSHTSphericalPlan(lmax=", p.lmax, ", M=", p.M, ", ntrans=", p.plan.B,
          ", nufft=", nameof(typeof(p.nufft)), ")")

# A `NUSHTplan` is working state, not a lookup table: every transform writes through its coefficient,
# field and phase buffers, so tasks cannot share one. The points and band limit are retained on this
# wrapper precisely so a task can build its own.
#
# Construction goes through FFTW's planner, which FFTW documents as callable from only one thread at
# a time (of its API, only `fftw_execute` is thread safe), so the rebuild is serialised. It happens
# once per task, not once per field.
# NUFSHT transforms `ntrans` co-located fields per call, so a batched sibling is just a rebuild at a
# different width. The rebuild costs ~0.5 ms against ~175 ms for a B = 32 cascade, so it is paid
# per batch call rather than cached: caching would key on `B`, which the caller varies freely.
ST.SphericalCore.supports_batch(::NUSHTSphericalPlan) = true

# Serialised on `PLANNER_LOCK` like every other plan build here: this one is called from inside
# spawned tasks, one per batch chunk, and FFTW's planner is single-threaded.
function ST.SphericalCore.batch_plan(p::NUSHTSphericalPlan, B::Integer)
    B == p.plan.B && return p
    plan = Base.@lock PLANNER_LOCK with_serial_ft() do
        NUFSHT.make_plan(eltype(p.theta), p.theta, p.phi, p.lmax;
                         ntrans = Int(B), nufft = p.nufft, tol = p.plan.tol)
    end
    return NUSHTSphericalPlan(plan, p.M, p.lmax, p.theta, p.phi, p.rtol, p.maxiter, p.nufft)
end

ST.SphericalCore.plan_nufft(p::NUSHTSphericalPlan) = p.nufft
ST.SphericalCore.plan_points(p::NUSHTSphericalPlan) = (p.theta, p.phi)
ST.SphericalCore.plan_weights(::NUSHTSphericalPlan) = nothing   # unweighted sample mean
ST.SphericalCore.plan_solver(p::NUSHTSphericalPlan) = (rtol = p.rtol, maxiter = p.maxiter)
ST.Plans.spectral_backend(::NUSHTSphericalPlan) = SB.NUFSHTSpectralBackend()

const PLANNER_LOCK = ReentrantLock()

function ST.Plans.task_local_plan(p::NUSHTSphericalPlan)
    plan = Base.@lock PLANNER_LOCK with_serial_ft() do
        # `ntrans` and the resolved backend must be carried over: rebuilding with the defaults would
        # give the task a single-field plan on whatever `Auto` happens to pick, silently changing both
        # the batch size the cascade feeds it and the transform driving it.
        NUFSHT.make_plan(eltype(p.theta), p.theta, p.phi, p.lmax;
                         ntrans = p.plan.B, nufft = p.nufft, tol = p.plan.tol)
    end
    return NUSHTSphericalPlan(plan, p.M, p.lmax, p.theta, p.phi, p.rtol, p.maxiter, p.nufft)
end

# Generic per-degree spectral multiplier h(ℓ): NUFSHT's apply_transfer! dispatches on
# `kernel_transfer(filter, ℓ)`, so any object with this method is a valid transfer.
struct _FnTransfer{F}
    f::F
end
NUFSHT.kernel_transfer(t::_FnTransfer, ℓ) = t.f(ℓ)

function ST.SphericalCore.sphere_coeffs(plan::NUSHTSphericalPlan, field::AbstractVecOrMat)
    return ST.SphericalCore.sphere_coeffs!(zero(plan.plan.C), plan, field)
end

# Analysis is the band-limited least-squares solve; NUFSHT projects onto the valid `(l, m)` modes
# inside its own iteration, so the coefficients come back in the degree-`<= lmax` span.
#
# `field` may be a single `(M,)` field or an `(M, B)` stack, matched to the plan's `ntrans`: the
# transform costs the same setup either way, so a stack amortises it over `B` fields.
function ST.SphericalCore.sphere_coeffs!(C, plan::NUSHTSphericalPlan, field::AbstractVecOrMat)
    fill!(C, zero(eltype(C)))
    _, iters, rel = with_serial_ft() do
        NUFSHT.nusht_solve!(C, field, plan.plan; rtol = plan.rtol, maxiter = plan.maxiter)
    end
    # A solve that stops at `maxiter` has not merely fallen short of `rtol` — the iterate can have
    # diverged, so `C` is unrelated to the field. The residual is the only thing that distinguishes
    # the two, and it is discarded unless it is checked here.
    rel <= plan.rtol || throw(ST.SphericalCore.AnalysisNotConverged(Float64(rel),
                                                                    Float64(plan.rtol), iters,
                                                                    plan.maxiter, plan.plan.B))
    return C
end

ST.SphericalCore.sphere_coeffs_buffer(plan::NUSHTSphericalPlan) = zero(plan.plan.C)

# Apply the per-degree multiplier `h(ℓ)` to a copy of the coefficients and synthesise at the points.
# The copy lands in the plan's scratch so a band allocates nothing, and `C` survives for the next one.
function ST.SphericalCore.sphere_apply!(out::AbstractVecOrMat, plan::NUSHTSphericalPlan, C, h)
    C2 = plan.cbuf
    copyto!(C2, C)
    NUFSHT.apply_transfer!(C2, _FnTransfer(h), plan.lmax)
    with_serial_ft(() -> NUFSHT.nusht_type2!(out, C2, plan.plan))
    return out
end

# Unweighted sample mean over the (quasi-uniform) scattered points ≈ the spherical average. An
# `(M, B)` stack averages each field separately, so a batch gets one mean per column.
ST.SphericalCore.sphere_mean(plan::NUSHTSphericalPlan, field::AbstractVector) = sum(field) / plan.M
ST.SphericalCore.sphere_mean(plan::NUSHTSphericalPlan, field::AbstractMatrix) =
    vec(sum(field; dims = 1)) ./ plan.M

# ---------------------------------------------------------------------------
# Fast-path plan constructor (scattered-sphere seam declared in SphericalCore). The core
# `spherical_scattering` / `spherical_monogenic_scattering` build this when `spectral` selects NUFSHT.
# ---------------------------------------------------------------------------

# FastTransforms, which NUFSHT builds on, is not re-entrant from a task sharing the caller's OS
# thread: running it multithreaded from such a task silently changes what an already-built plan
# returns, permanently and with no error.
#
# `ft_set_num_threads` has no getter, but it forwards to OpenMP and `omp_get_max_threads` tracks it,
# so the count is restored afterwards rather than left mutated behind the caller's back.
#
# That symbol is reached through `libfasttransforms`, which links OpenMP and re-exports it, rather
# than through `libomp` by name: the bare name resolves on macOS but not on a stock Linux runner,
# whereas the JLL gives a real path on every platform.
function with_serial_ft(f)
    prev = ccall((:omp_get_max_threads, NUFSHT.FastTransforms.libfasttransforms), Cint, ())
    NUFSHT.FastTransforms.ft_set_num_threads(1)
    try
        return f()
    finally
        NUFSHT.FastTransforms.ft_set_num_threads(prev)
    end
end

"""
    nusht_spherical_plan(θ, φ, lmax, T; rtol, maxiter, ntrans = 1)

`ntrans` is NUFSHT's batch size: a plan transforms `ntrans` fields sampled at the same points per
call. The cascade issues many small transforms per field, and each call pays FastTransforms' parallel
region setup, so batching amortises that the way the gridded path amortises an FFT plan. `ntrans = 1`
is a single-field plan.
"""
function ST.SphericalCore.nusht_spherical_plan(pts_theta::AbstractVector, pts_phi::AbstractVector,
                                               lmax::Int, ::Type{T};
                                               rtol::Real = ST.SphericalCore.default_rtol(SB.NUFSHTSpectralBackend()),
                                               maxiter::Int = 500,
                                               ntrans::Int = 1,
                                               nufft::SB.AbstractSpectralBackend = SB.AutoSpectralBackend()) where {T<:Real}
    # Broadcast rather than `collect`: both copy (the plan must own its points, so a spec rebuilt
    # from them is independent), but `collect` materialises to a host `Array` and would strand a
    # device-resident transform on the CPU regardless of which NUFFT NUFSHT is using.
    θ = T.(pts_theta)
    φ = T.(pts_phi)
    # Resolve before building, and build against the concrete result. Handing `Auto` straight to
    # `make_plan` works, but then nothing downstream — not the plan, not `show`, not a benchmark —
    # can say whether the fast NUFFT or the O(M·K) direct-sum fallback is actually running.
    nb = NUFSHT._resolve_nufft(nufft)
    plan = with_serial_ft(() -> NUFSHT.make_plan(T, θ, φ, lmax; ntrans = ntrans, nufft = nb))
    return NUSHTSphericalPlan(plan, length(θ), lmax, θ, φ, T(rtol), maxiter, nb)
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

function ST.spherical_monogenic_components(st::ST.SphericalCore.SphericalMonogenicScattering{<:Any, <:NUSHTSphericalPlan},
                                           field::AbstractVector, j::Int)
    p = st.plan
    lmax = st.lmax
    T = eltype(p.theta)
    # Complex spin-0 coefficients of the field in NUFSHT's dense spin layout (exact CG inversion,
    # so the band-pass and Riesz fields below share one consistent set of coefficients).
    # NUFSHT's positional argument is the *field* element type, not the precision: a real one selects
    # its folded real layout. The spin field here is complex, so it must be `Complex{T}`.
    p0 = NUFSHT.make_spin_plan(Complex{T}, p.theta, p.phi, lmax, 0)
    p1 = NUFSHT.make_spin_plan(Complex{T}, p.theta, p.phi, lmax, 1)
    a = zeros(Complex{T}, lmax + 1, 2lmax + 1)
    NUFSHT.nusht_solve_spin!(a, Complex{T}.(field), p0)

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
