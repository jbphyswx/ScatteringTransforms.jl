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
# `guru1`/`guru2` are opaque FINUFFT C-plan handles (kept untyped); the array buffers are type
# parameters so the struct stays generic and concretely typed.
# ---------------------------------------------------------------------------

mutable struct NUFFTScatteringPlan{T, CV<:AbstractVector{Complex{T}},
                                   MM<:AbstractMatrix{Complex{T}}} <: ST.Plans.AbstractScatteringPlan
    guru1                         # Type-1 (points → modes), iflag −1, FFT mode order
    guru2                         # Type-2 (modes → points), iflag +1, FFT mode order
    ms::NTuple{2,Int}
    M::Int
    invN::T                       # 1/prod(ms); makes synthesis the ifft-convention inverse
    solve::Bool
    maxiter::Int
    rtol::T
    cj::CV                        # (M) nonuniform exec buffer (shared by Type-1/Type-2)
    r::MM                         # (ms) CG residual / rhs
    p::MM                         # (ms) CG search direction
    Ap::MM                        # (ms) CG A†A·p
    tmp_pts::CV                   # (M) CG scratch
end

function _make_plan(x, y, ms::NTuple{2,Int}, ::Type{T}, period, eps, solve, maxiter, rtol) where {T}
    M = length(x)
    length(y) == M || throw(DimensionMismatch("x and y must have equal length"))
    xmin, ymin = T(minimum(x)), T(minimum(y))
    # Default period so a uniform 0:m-1 grid (span m-1) maps to the exact DFT nodes 2π·(0:m-1)/m.
    px = period === nothing ? (T(maximum(x)) - xmin) * ms[1] / (ms[1] - 1) : T(period[1])
    py = period === nothing ? (T(maximum(y)) - ymin) * ms[2] / (ms[2] - 1) : T(period[2])
    sx = T(2π) .* (T.(x) .- xmin) ./ px
    sy = T(2π) .* (T.(y) .- ymin) ./ py
    guru1 = FINUFFT.finufft_makeplan(1, collect(ms), -1, 1, T(eps); dtype = T, modeord = 1)
    guru2 = FINUFFT.finufft_makeplan(2, collect(ms), +1, 1, T(eps); dtype = T, modeord = 1)
    FINUFFT.finufft_setpts!(guru1, sx, sy)
    FINUFFT.finufft_setpts!(guru2, sx, sy)
    plan = NUFFTScatteringPlan(
        guru1, guru2, ms, M, one(T) / prod(ms), solve, maxiter, T(rtol),
        Vector{Complex{T}}(undef, M),
        Matrix{Complex{T}}(undef, ms), Matrix{Complex{T}}(undef, ms), Matrix{Complex{T}}(undef, ms),
        Vector{Complex{T}}(undef, M))
    finalizer(pl -> (FINUFFT.finufft_destroy!(pl.guru1); FINUFFT.finufft_destroy!(pl.guru2)), plan)
    return plan
end

# `guru1`/`guru2` are opaque C handles; the default `show` would walk them. One line instead.
Base.show(io::IO, p::NUFFTScatteringPlan{T}) where {T} =
    print(io, "NUFFTScatteringPlan{", T, "}(ms=", p.ms, ", M=", p.M, ", solve=", p.solve, ")")
Base.show(io::IO, ::MIME"text/plain", p::NUFFTScatteringPlan) = show(io, p)

ST.Plans.spectral_backend(::NUFFTScatteringPlan) = ST.Plans.FINUFFTBackend()

# The guru plans hold internal state that FINUFFT does not document as re-entrant, so a task takes
# its own plan rather than sharing one.
ST.Plans.task_local_plan(p::NUFFTScatteringPlan) = p

# Fast-path plan constructor filled into the core `ST.Plans.finufft_scattered_plan` declaration; the core
# `scattered_planar_scattering` cascade builds it when `spectral` selects the FINUFFT backend.
function ST.Plans.finufft_scattered_plan(x, y, ms::NTuple{2,Int}, ::Type{T}; period = nothing,
                                      solve::Bool = false, maxiter::Int = 100, rtol::Real = 1.0e-8,
                                      eps = nothing) where {T}
    ε = eps === nothing ? (T === Float32 ? 1.0e-6 : 1.0e-9) : eps
    return _make_plan(x, y, ms, T, period, ε, solve, maxiter, rtol)
end

# Synthesis: modes → points, scaled by 1/prod(ms) (ifft convention). For a Type-2 plan
# `finufft_exec!(plan, input, output)` takes input=modes, output=points.
function ST.Plans.inverse_transform!(out_pts::AbstractVector, plan::NUFFTScatteringPlan, Xmodes::AbstractMatrix)
    FINUFFT.finufft_exec!(plan.guru2, Xmodes, plan.cj)
    @. out_pts = plan.cj * plan.invN
    return out_pts
end

# Analysis: points → modes. Type-1 adjoint (fft-equivalent on a uniform grid) unless `solve`.
function ST.Plans.forward_transform!(Xmodes::AbstractMatrix, plan::NUFFTScatteringPlan, x_pts::AbstractVector)
    if plan.solve
        _cg_solve!(Xmodes, plan, x_pts)
    else
        copyto!(plan.cj, x_pts)
        FINUFFT.finufft_exec!(plan.guru1, plan.cj, Xmodes)
    end
    return Xmodes
end

# CG least-squares inversion: find modes `f` with Type2(f) ≈ prod(ms)·x (so synthesis, = Type2/N,
# recovers x). Solves the normal equations (A†A) f = A† (N·x) with A = Type-2, A† = Type-1.
function _cg_solve!(f::AbstractMatrix, plan::NUFFTScatteringPlan{T}, x_pts::AbstractVector) where {T}
    N = one(T) / plan.invN
    copyto!(plan.cj, x_pts)
    FINUFFT.finufft_exec!(plan.guru1, plan.cj, plan.r)         # r = A†x  (modes)
    plan.r .*= N                                               # r = A†(N·x) = rhs
    fill!(f, zero(Complex{T}))
    copyto!(plan.p, plan.r)
    rsold = real(LinearAlgebra.dot(vec(plan.r), vec(plan.r)))
    rs0 = rsold
    rs0 == 0 && return f
    @inbounds for _ in 1:plan.maxiter
        FINUFFT.finufft_exec!(plan.guru2, plan.p, plan.tmp_pts)    # tmp = A·p (Type-2: modes→points)
        FINUFFT.finufft_exec!(plan.guru1, plan.tmp_pts, plan.Ap)   # Ap  = A†A·p (Type-1: points→modes)
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
