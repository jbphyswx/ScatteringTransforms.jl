module ScatteringTransformsNonuniformFFTsExt

"""
    ScatteringTransformsNonuniformFFTsExt — NonuniformFFTs.jl fast path for scattered planar scattering

The second fast spectral plan for the scattered-planar cascade (`ScatteredPlanar`), alongside
FINUFFT. Analysis is a Type-1 NUFFT (points → modes) or an LSMR least-squares solve;
synthesis is a Type-2 NUFFT (modes → points) scaled by `1/prod(ms)`. It implements the same
`ST.Plans.AbstractScatteringPlan` interface as the in-core `ST.Plans.DirectNUFFTPlan`, so the
cascade is identical — only the transform underneath changes.

NonuniformFFTs is pure Julia: no binary dependency, a threaded CPU path, and a
KernelAbstractions GPU path. It lays its modes out in `AbstractFFTs.fftfreq` order, which is the
lattice the wavelet bank is built on (and the order FINUFFT is asked for with `modeord = 1`), so
the two fast backends are interchangeable.

Selected by `spectral = ST.Plans.NonuniformFFTsBackend()`, or by
`SpectralBackends.NUFFTSpectralBackend()` / `AutoSpectralBackend()` when this is the loaded NUFFT.

The foreign plan is held behind the type parameter `P` and every `NonuniformFFTs` call is inside a
function body: an extension that names a runtime-only symbol in a *signature* fails to precompile,
and the tests would not catch it because the extension only loads with its trigger package.
"""

using NonuniformFFTs: NonuniformFFTs
using ScatteringTransforms: ScatteringTransforms as ST

"""
    NonuniformFFTsScatteringPlan{T,P,CV,MM,RV}

Scattered-planar spectral plan backed by a `NonuniformFFTs.PlanNUFFT` over fixed points `(x, y)`
and a uniform mode grid `ms`. `solve` selects the least-squares inversion over the plain Type-1
adjoint; the `ls_*` fields are that solver's workspace, so the solve adds nothing per call beyond the
transforms it issues.

Those transforms do allocate, which makes this the one backend where a solve allocates in steady
state: 880 B per execution and 152 880 B for a 100-iteration solve at `M = 500`, `ms = (16, 16)`.
An allocation profile attributes it to a `Threads.@threads` region in NonuniformFFTs' deconvolution
step, which builds its task scaffolding — a `Task`, a task list, a lock and a condition — on every
call, including at one thread where it parallelises nothing. The FINUFFT plan measures exactly zero
on the same case because its execution enters no Julia threaded region; its library threads are
pooled rather than created per call.
"""
struct NonuniformFFTsScatteringPlan{T, P, CV <: AbstractVector{Complex{T}},
                                    MM <: AbstractMatrix{Complex{T}},
                                    RV <: AbstractVector{T}} <: ST.Plans.AbstractScatteringPlan
    plan::P
    ms::NTuple{2, Int}
    M::Int
    invN::T                 # 1/prod(ms); makes synthesis the ifft-convention inverse
    solve::Bool
    maxiter::Int
    rtol::T
    damp::T                 # Tikhonov λ; 0 unless the mode grid is over-specified
    sx::RV                  # (M) points already scaled to the 2π-periodic domain, retained so a
    sy::RV                  #     task can build its own plan
    eps::T                  # requested relative tolerance, under the constructor's own keyword name.
                            #     Kept instead of the half-support it maps to, so a rebuild re-applies
                            #     `_half_support` rather than inverting a step function.
    nthreads::Int           # the FFT thread count (0 = whatever the process planner is set to)
    cj::CV                  # (M) nonuniform exec buffer
    ls_v::MM                # (ms) solver scratch: the four LSMR mode vectors, with `cj` and `ls_t`
    ls_w::MM                #      its two point vectors. The transforms overwrite rather than
    ls_h::MM                #      accumulate, so `A·v` and `A†·u` each need their own destination.
    ls_hbar::MM
    ls_t::CV                # (M)
end

# `PlanNUFFT` holds the scratch every execution writes through, so tasks cannot share one. The
# points are retained above precisely so a task can build its own.
#
# The rebuild carries the plan's own thread count, defaulting to one, for the reason given on the
# FINUFFT sibling: one plan exists per task, so leaving each one the whole machine oversubscribes and
# makes every execution spawn a Julia task per FFT thread.
function ST.Plans.task_local_plan(p::NonuniformFFTsScatteringPlan{T}) where {T}
    return _plan_at(p.ms, p.M, p.sx, p.sy, p.eps, T, p.solve, p.maxiter, p.rtol,
                    ST.Plans.per_task_nthreads(p.nthreads), p.damp)
end

# Where the plan lives follows the points, so there is nothing to pass: `get_backend` reads the
# KernelAbstractions backend off `sx` and every buffer is `similar` to it. Host points give a host
# plan, device points a device-resident one, through the same code — and a task's rebuilt plan lands
# on the same device, because it rebuilds from these same points.
function _plan_at(ms::NTuple{2, Int}, M::Int, sx, sy, eps::T, ::Type{T}, solve, maxiter,
                  rtol::T, nthreads::Int = 0, damp::T = zero(T)) where {T}
    halfsupport = _half_support(eps)
    # Serialised on the package-wide planner lock: a host plan's smooth-grid FFT is planned through
    # the same libfftw3 every other backend here plans through, and this builder runs inside spawned
    # tasks. `nthreads > 0` additionally pins that planner's thread count, which the FFT plan bakes in
    # — a count of 0 leaves it at whatever the process has, which is NonuniformFFTs' own default.
    function build()
        pl = NonuniformFFTs.PlanNUFFT(Complex{T}, ms;
                                      m = NonuniformFFTs.HalfSupport(halfsupport),
                                      backend = NonuniformFFTs.KA.get_backend(sx))
        NonuniformFFTs.set_points!(pl, (sx, sy))
        return pl
    end
    plan = Base.@lock ST.Plans.PLANNER_LOCK begin
        nthreads > 0 ? ST.Plans.with_fft_nthreads(build, nthreads) : build()
    end
    pts() = similar(sx, Complex{T}, M)
    modes() = similar(sx, Complex{T}, ms)
    return NonuniformFFTsScatteringPlan(
        plan, ms, M, one(T) / prod(ms), solve, maxiter, rtol, damp, sx, sy, eps, nthreads,
        pts(), modes(), modes(), modes(), modes(), pts())
end

# The plan holds device/threading state and scratch, so it prints as one line rather than dumping
# its internals, and a concurrent task takes its own.
Base.show(io::IO, p::NonuniformFFTsScatteringPlan{T}) where {T} =
    print(io, "NonuniformFFTsScatteringPlan{", T, "}(ms=", p.ms, ", M=", p.M,
          ", solve=", p.solve, ")")
Base.show(io::IO, ::MIME"text/plain", p::NonuniformFFTsScatteringPlan) = show(io, p)

ST.Plans.spectral_backend(::NonuniformFFTsScatteringPlan) = ST.Plans.NonuniformFFTsBackend()
ST.Plans.plan_points(p::NonuniformFFTsScatteringPlan) = (p.sx, p.sy)
ST.Plans.plan_analysis(p::NonuniformFFTsScatteringPlan) =
    (solve = p.solve, maxiter = p.maxiter, rtol = p.rtol, damp = p.damp, eps = p.eps,
     nufft_nthreads = p.nthreads)

# NonuniformFFTs expresses accuracy as the convolution kernel's half-support, not as a tolerance
# (its documented default, `m = 4` with `σ = 2.0`, gives ~1e-7 relative for `Float64`). The `eps`
# the scattered-plan interface takes — a FINUFFT-style tolerance — is therefore mapped to the
# smallest half-support that meets it, so the two fast backends honour the same request.
_half_support(tol::Real) = tol >= 1.0e-4 ? 2 : tol >= 1.0e-7 ? 4 : tol >= 1.0e-10 ? 6 : 8

function ST.Plans.nonuniformffts_scattered_plan(x, y, ms::NTuple{2, Int}, ::Type{T};
                                                period = nothing, solve::Bool = false,
                                                maxiter::Int = 100, eps = nothing,
                                                rtol::Real = ST.Plans.default_solver_rtol(T, ST.Plans.NonuniformFFTsBackend(), eps),
                                                damp::Real = 0, ntrans::Int = 1,
                                                nufft_nthreads::Int = 0) where {T}
    # Accepted so the caller need not know which backend it will get, and deliberately not acted on:
    # this plan reports `batch_width == 1`, so the cascade keeps its per-field loop. NonuniformFFTs
    # does expose `ntransforms`, but measured against the same transforms issued one at a time it is
    # 1.16x at `B = 8` and 0.70x at `B = 32` on this package's mode-grid sizes — batching it would
    # cost throughput at the batch sizes that matter.
    M = length(x)
    length(y) == M || throw(DimensionMismatch("x and y must have equal length"))
    ST.Plans.warn_underdetermined(M, ms, solve, damp)
    xmin, ymin = T(minimum(x)), T(minimum(y))
    # Same default period as the in-core plan: a uniform 0:m-1 grid maps to the exact DFT nodes.
    px = period === nothing ? (T(maximum(x)) - xmin) * ms[1] / (ms[1] - 1) : T(period[1])
    py = period === nothing ? (T(maximum(y)) - ymin) * ms[2] / (ms[2] - 1) : T(period[2])
    sx = T(2π) .* (T.(x) .- xmin) ./ px
    sy = T(2π) .* (T.(y) .- ymin) ./ py

    tol = eps === nothing ? ST.Plans.default_nufft_eps(T) : eps
    return _plan_at(ms, M, sx, sy, T(tol), T, solve, maxiter, T(rtol), nufft_nthreads, T(damp))
end

# Synthesis: modes → points (Type-2), scaled by 1/prod(ms) so it is the ifft-convention inverse.
function ST.Plans.inverse_transform!(out_pts::AbstractVector, plan::NonuniformFFTsScatteringPlan,
                                     Xmodes::AbstractMatrix)
    NonuniformFFTs.exec_type2!(plan.cj, plan.plan, Xmodes)
    @. out_pts = plan.cj * plan.invN
    return out_pts
end

# Analysis: points → modes. Type-1 adjoint (fft-equivalent on a uniform grid) unless `solve`.
function ST.Plans.forward_transform!(Xmodes::AbstractMatrix, plan::NonuniformFFTsScatteringPlan,
                                     x_pts::AbstractVector)
    if plan.solve
        _lsmr_solve!(Xmodes, plan, x_pts)
    else
        copyto!(plan.cj, x_pts)
        NonuniformFFTs.exec_type1!(Xmodes, plan.plan, plan.cj)
    end
    return Xmodes
end

# Least-squares inversion, sharing the core solver with the in-core and FINUFFT paths so the three
# agree to solver tolerance. `A f̃ = x` is solved and scaled by `N` afterwards rather than inflating
# the right-hand side by `prod(ms)` — see the FINUFFT sibling.
function _lsmr_solve!(f::AbstractMatrix, plan::NonuniformFFTsScatteringPlan{T},
                      x_pts::AbstractVector) where {T}
    info = ST.Plans.lsmr_solve!(f,
        (dst, src) -> NonuniformFFTs.exec_type2!(dst, plan.plan, src),   # A : modes → points
        (dst, src) -> NonuniformFFTs.exec_type1!(dst, plan.plan, src),   # A†: points → modes
        x_pts, plan.cj, plan.ls_t, plan.ls_v, plan.ls_w, plan.ls_h, plan.ls_hbar;
        damp = plan.damp, atol = plan.rtol, btol = plan.rtol,
        conlim = inv(Base.eps(T)), maxiter = plan.maxiter)
    ST.Plans._check_solve(info, plan.M, plan.ms, plan.rtol, plan.maxiter)
    f .*= one(T) / plan.invN
    return f
end

end # module ScatteringTransformsNonuniformFFTsExt