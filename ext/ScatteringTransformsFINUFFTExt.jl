module ScatteringTransformsFINUFFTExt

"""
    ScatteringTransformsFINUFFTExt — FINUFFT fast path for scattered / nonuniform planar scattering

Supplies the **fast** spectral plan for the scattered-planar cascade (`ScatteredPlanar`): a reusable
FINUFFT guru plan over fixed scattered points `(x, y)` and a uniform Fourier mode grid `ms`. Analysis
is a Type-1 NUFFT (points → modes) or a conjugate-gradient least-squares solve; synthesis is a Type-2
NUFFT (modes → points) scaled by `1/prod(ms)`. It implements the same `ST.Plans.AbstractScatteringPlan`
interface as the in-core `ST.Plans.DirectNUFFTPlan`, so the cascade is identical — this only replaces the
`O(M·prod(ms))` direct summation with FINUFFT's `O((M + prod(ms))·log)` transform.

Plans use FFT mode ordering (`modeord=1`) so the mode grid matches the filter bank's `fftfreq` lattice;
on a uniform `0:m-1` grid Type-1/Type-2 reduce to `fft`/`ifft`. The core `scattered_planar_scattering`
selects this plan when `spectral` is `SB.NUFFTSpectralBackend()` (or `SB.AutoSpectralBackend()` with
this extension loaded).

The Type-1 adjoint (`solve=false`) equals the true DFT on a uniform grid and is
accurate for adequately-sampled band-limited fields, but on gappy/irregular sampling it is only the
adjoint, not the inverse. `solve=true` runs a conjugate-gradient least-squares inversion for the true
band-limited coefficients — slower, but the principled choice for irregular data. The caller picks per
their sampling.
"""

using FINUFFT: FINUFFT
using LinearAlgebra: LinearAlgebra
using ScatteringTransforms: ScatteringTransforms as ST

# ---------------------------------------------------------------------------
# NUFFT spectral plan: analysis (points → modes, Type-1 or CG-solve) + synthesis (modes → points).
# Implements the `ST.Plans.AbstractScatteringPlan` interface so the cascade reuses `wavelet_convolve!`.
# ---------------------------------------------------------------------------

# The guru plans are themselves mutable and own their C plans, so the finalizer that frees one goes
# there (see `nufft_guru_make`) and this wrapper stays immutable.
#
# The buffers carry no fixed rank: a guru plan's `ntrans` is chosen at build time and cannot vary per
# execution, so a `B = 1` plan holds `(M)` / `(ms)` buffers and a batched one `(M, B)` / `(ms…, B)`.
# A plan is therefore either single-field or batched, never both.
struct NUFFTScatteringPlan{T, G, CV<:AbstractArray{Complex{T}},
                           MM<:AbstractArray{Complex{T}},
                           RV<:AbstractVector{T}} <: ST.Plans.AbstractScatteringPlan
    # A type parameter rather than `finufft_plan{T}`, so the same wrapper holds a host guru plan or a
    # cuFINUFFT one. Concrete either way — it is fixed per instantiation.
    guru1::G                      # Type-1 (points → modes), iflag −1, FFT mode order
    guru2::G                      # Type-2 (modes → points), iflag +1, FFT mode order
    ms::NTuple{2,Int}
    M::Int
    B::Int                        # fields transformed per execution (FINUFFT `ntrans`)
    invN::T                       # 1/prod(ms); makes synthesis the ifft-convention inverse
    solve::Bool
    maxiter::Int
    rtol::T
    sx::RV                        # (M) points already scaled to FINUFFT's 2π-periodic domain,
    sy::RV                        #     retained so a task can build its own guru plans
    eps::T                        # the tolerance those plans were made with
    nthreads::Int                 # and the thread count they were made with (0 = FINUFFT's default)
    cj::CV                        # (M[, B]) nonuniform exec buffer (shared by Type-1/Type-2)
    r::MM                         # (ms[, B]) CG residual / rhs
    p::MM                         # (ms[, B]) CG search direction
    Ap::MM                        # (ms[, B]) CG A†A·p
    tmp_pts::CV                   # (M[, B]) CG scratch
end

function _make_plan(x, y, ms::NTuple{2,Int}, ::Type{T}, period, eps, solve, maxiter, rtol,
                    B::Int = 1, nthreads::Int = 0) where {T}
    M = length(x)
    length(y) == M || throw(DimensionMismatch("x and y must have equal length"))
    xmin, ymin = T(minimum(x)), T(minimum(y))
    # Default period so a uniform 0:m-1 grid (span m-1) maps to the exact DFT nodes 2π·(0:m-1)/m.
    px = period === nothing ? (T(maximum(x)) - xmin) * ms[1] / (ms[1] - 1) : T(period[1])
    py = period === nothing ? (T(maximum(y)) - ymin) * ms[2] / (ms[2] - 1) : T(period[2])
    sx = T(2π) .* (T.(x) .- xmin) ./ px
    sy = T(2π) .* (T.(y) .- ymin) ./ py
    return _plan_at(ms, M, sx, sy, T(eps), T, solve, maxiter, T(rtol), B, nthreads)
end

# Host seam. `finufft_plan` is mutable and owns the C plan, so the finalizer that frees it goes there
# and this package's wrapper stays immutable; `finufft_destroy!` is idempotent, so an explicit destroy
# before collection is harmless.
function ST.Plans.nufft_guru_make(::AbstractArray, type::Integer, ms::NTuple{2, Int},
                                  iflag::Integer, ntrans::Integer, eps::Real, ::Type{T};
                                  nthreads::Integer = 0) where {T}
    g = FINUFFT.finufft_makeplan(type, collect(ms), iflag, ntrans, eps;
                                 dtype = T, modeord = 1, nthreads = nthreads)
    finalizer(FINUFFT.finufft_destroy!, g)
    return g
end

ST.Plans.nufft_guru_setpts!(g::FINUFFT.finufft_plan, x, y) = (FINUFFT.finufft_setpts!(g, x, y); g)
ST.Plans.nufft_guru_exec!(g::FINUFFT.finufft_plan, input, output) =
    (FINUFFT.finufft_exec!(g, input, output); output)

function _plan_at(ms::NTuple{2,Int}, M::Int, sx, sy, eps::T, ::Type{T}, solve, maxiter,
                  rtol::T, B::Int = 1, nthreads::Int = 0) where {T}
    # FINUFFT plans through the same libfftw3 as every other backend here, and that planner takes one
    # thread at a time, so construction is serialised on the package-wide lock. It happens once per
    # task, never per transform.
    guru1, guru2 = Base.@lock ST.Plans.PLANNER_LOCK begin
        # Built through the seam, so device points get a cuFINUFFT plan and host points a host one.
        g1 = ST.Plans.nufft_guru_make(sx, 1, ms, -1, B, eps, T; nthreads = nthreads)
        g2 = ST.Plans.nufft_guru_make(sx, 2, ms, +1, B, eps, T; nthreads = nthreads)
        ST.Plans.nufft_guru_setpts!(g1, sx, sy)
        ST.Plans.nufft_guru_setpts!(g2, sx, sy)
        (g1, g2)
    end
    # `similar` to the points, so a device point set gives device-resident buffers to match the
    # cuFINUFFT plan the seam returned for it.
    pts(n) = B == 1 ? similar(sx, Complex{T}, n) : similar(sx, Complex{T}, n, B)
    modes() = B == 1 ? similar(sx, Complex{T}, ms) : similar(sx, Complex{T}, (ms..., B))
    return NUFFTScatteringPlan(
        guru1, guru2, ms, M, B, one(T) / prod(ms), solve, maxiter, rtol, sx, sy, eps, nthreads,
        pts(M), modes(), modes(), modes(), pts(M))
end

# The guru plans hold C pointers and reference-keeping arrays; the default `show` would walk all of
# it. One line instead.
Base.show(io::IO, p::NUFFTScatteringPlan{T}) where {T} =
    print(io, "NUFFTScatteringPlan{", T, "}(ms=", p.ms, ", M=", p.M, ", ntrans=", p.B,
          ", solve=", p.solve, ")")
Base.show(io::IO, ::MIME"text/plain", p::NUFFTScatteringPlan) = show(io, p)

ST.Plans.spectral_backend(::NUFFTScatteringPlan) = ST.Plans.FINUFFTBackend()
ST.Plans.plan_points(p::NUFFTScatteringPlan) = (p.sx, p.sy)
ST.Plans.plan_analysis(p::NUFFTScatteringPlan) =
    (solve = p.solve, maxiter = p.maxiter, rtol = p.rtol, eps = p.eps,
     nufft_nthreads = p.nthreads)

# A guru plan carries the working buffers each execution writes through, so tasks cannot share one —
# doing so silently corrupts every concurrent transform. The scaled points are retained on the
# wrapper precisely so a task can build its own.
#
# The rebuild divides the machine among the concurrent tasks rather than inheriting FINUFFT's default
# of "all cores": one of these plans exists per task, so the default would put `nthreads()` tasks ×
# `CPU_THREADS` library threads on `CPU_THREADS` cores, and each execution would additionally spawn a
# Julia task per library thread through FFTW.jl's thread callback. An even split keeps the cores
# filled exactly once — and collapses to a single thread, i.e. no nesting at all, when Julia already
# has a task per core.
ST.Plans.task_local_plan(p::NUFFTScatteringPlan{T}) where {T} =
    _plan_at(p.ms, p.M, p.sx, p.sy, p.eps, T, p.solve, p.maxiter, p.rtol, p.B,
             max(1, cld(Sys.CPU_THREADS, Threads.nthreads())))

ST.Plans.batch_width(p::NUFFTScatteringPlan) = p.B

# Fast-path plan constructor filled into the core `ST.Plans.finufft_scattered_plan` declaration; the core
# `scattered_planar_scattering` cascade builds it when `spectral` selects the FINUFFT backend.
function ST.Plans.finufft_scattered_plan(x, y, ms::NTuple{2,Int}, ::Type{T}; period = nothing,
                                      solve::Bool = false, maxiter::Int = 100, rtol::Real = 1.0e-8,
                                      eps = nothing, ntrans::Int = 1,
                                      nufft_nthreads::Int = 0) where {T}
    ε = eps === nothing ? (T === Float32 ? 1.0e-6 : 1.0e-9) : eps
    return _make_plan(x, y, ms, T, period, ε, solve, maxiter, rtol, ntrans, nufft_nthreads)
end

# Synthesis: modes → points, scaled by 1/prod(ms) (ifft convention). For a Type-2 plan
# `finufft_exec!(plan, input, output)` takes input=modes, output=points.
function ST.Plans.inverse_transform!(out_pts::AbstractVector, plan::NUFFTScatteringPlan, Xmodes::AbstractMatrix)
    ST.Plans.nufft_guru_exec!(plan.guru2, Xmodes, plan.cj)
    @. out_pts = plan.cj * plan.invN
    return out_pts
end

# Batched forms. A `B`-wide guru plan executes exactly `B` transforms per call, so these are the only
# valid shapes for it, just as the shapes above are the only valid ones for a `B = 1` plan.
function ST.Plans.inverse_transform!(out_pts::AbstractMatrix, plan::NUFFTScatteringPlan,
                                     Xmodes::AbstractArray{<:Any,3})
    ST.Plans.nufft_guru_exec!(plan.guru2, Xmodes, plan.cj)
    @. out_pts = plan.cj * plan.invN
    return out_pts
end

function ST.Plans.forward_transform!(Xmodes::AbstractArray{<:Any,3}, plan::NUFFTScatteringPlan,
                                     x_pts::AbstractMatrix)
    plan.solve && throw(ArgumentError(
        "the CG least-squares analysis (`solve = true`) has no batched form; build the plan with " *
        "`ntrans = 1` or use `solve = false`."))
    copyto!(plan.cj, x_pts)
    ST.Plans.nufft_guru_exec!(plan.guru1, plan.cj, Xmodes)
    return Xmodes
end

# Analysis: points → modes. Type-1 adjoint (fft-equivalent on a uniform grid) unless `solve`.
function ST.Plans.forward_transform!(Xmodes::AbstractMatrix, plan::NUFFTScatteringPlan, x_pts::AbstractVector)
    if plan.solve
        _cg_solve!(Xmodes, plan, x_pts)
    else
        copyto!(plan.cj, x_pts)
        ST.Plans.nufft_guru_exec!(plan.guru1, plan.cj, Xmodes)
    end
    return Xmodes
end

# CG least-squares inversion: find modes `f` with Type2(f) ≈ prod(ms)·x (so synthesis, = Type2/N,
# recovers x). Solves the normal equations (A†A) f = A† (N·x) with A = Type-2, A† = Type-1.
function _cg_solve!(f::AbstractMatrix, plan::NUFFTScatteringPlan{T}, x_pts::AbstractVector) where {T}
    N = one(T) / plan.invN
    copyto!(plan.cj, x_pts)
    ST.Plans.nufft_guru_exec!(plan.guru1, plan.cj, plan.r)         # r = A†x  (modes)
    plan.r .*= N                                               # r = A†(N·x) = rhs
    fill!(f, zero(Complex{T}))
    copyto!(plan.p, plan.r)
    rsold = real(LinearAlgebra.dot(vec(plan.r), vec(plan.r)))
    rs0 = rsold
    rs0 == 0 && return f
    @inbounds for _ in 1:plan.maxiter
        ST.Plans.nufft_guru_exec!(plan.guru2, plan.p, plan.tmp_pts)    # tmp = A·p (Type-2: modes→points)
        ST.Plans.nufft_guru_exec!(plan.guru1, plan.tmp_pts, plan.Ap)   # Ap  = A†A·p (Type-1: points→modes)
        α = rsold / real(LinearAlgebra.dot(vec(plan.p), vec(plan.Ap)))
        f .+= α .* plan.p
        plan.r .-= α .* plan.Ap
        rsnew = real(LinearAlgebra.dot(vec(plan.r), vec(plan.r)))
        sqrt(rsnew) <= plan.rtol * sqrt(rs0) && break
        plan.p .= plan.r .+ (rsnew / rsold) .* plan.p
        rsold = rsnew
    end
    return f
end

end # module ScatteringTransformsFINUFFTExt
