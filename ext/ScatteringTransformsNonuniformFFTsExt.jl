module ScatteringTransformsNonuniformFFTsExt

"""
    ScatteringTransformsNonuniformFFTsExt — NonuniformFFTs.jl fast path for scattered planar scattering

The second fast spectral plan for the scattered-planar cascade (`ScatteredPlanar`), alongside
FINUFFT. Analysis is a Type-1 NUFFT (points → modes) or a conjugate-gradient least-squares solve;
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
using LinearAlgebra: LinearAlgebra
using ScatteringTransforms: ScatteringTransforms as ST

"""
    NonuniformFFTsScatteringPlan{T,P,CV,MM}

Scattered-planar spectral plan backed by a `NonuniformFFTs.PlanNUFFT` over fixed points `(x, y)`
and a uniform mode grid `ms`. `solve` selects the conjugate-gradient least-squares inversion over
the plain Type-1 adjoint; `r`/`p`/`Ap`/`tmp_pts` are that solver's workspace, so a solve allocates
nothing per call.
"""
struct NonuniformFFTsScatteringPlan{T, P, CV <: AbstractVector{Complex{T}},
                                    MM <: AbstractMatrix{Complex{T}}} <: ST.Plans.AbstractScatteringPlan
    plan::P
    ms::NTuple{2, Int}
    M::Int
    invN::T                 # 1/prod(ms); makes synthesis the ifft-convention inverse
    solve::Bool
    maxiter::Int
    rtol::T
    cj::CV                  # (M) nonuniform exec buffer
    r::MM                   # (ms) CG residual / rhs
    p::MM                   # (ms) CG search direction
    Ap::MM                  # (ms) CG A†A·p
    tmp_pts::CV             # (M) CG scratch
end

# The plan holds device/threading state and scratch, so it prints as one line rather than dumping
# its internals, and a concurrent task takes its own.
Base.show(io::IO, p::NonuniformFFTsScatteringPlan{T}) where {T} =
    print(io, "NonuniformFFTsScatteringPlan{", T, "}(ms=", p.ms, ", M=", p.M,
          ", solve=", p.solve, ")")
Base.show(io::IO, ::MIME"text/plain", p::NonuniformFFTsScatteringPlan) = show(io, p)

ST.Plans.spectral_backend(::NonuniformFFTsScatteringPlan) = ST.Plans.NonuniformFFTsBackend()

# NonuniformFFTs expresses accuracy as the convolution kernel's half-support, not as a tolerance
# (its documented default, `m = 4` with `σ = 2.0`, gives ~1e-7 relative for `Float64`). The `eps`
# the scattered-plan interface takes — a FINUFFT-style tolerance — is therefore mapped to the
# smallest half-support that meets it, so the two fast backends honour the same request.
_half_support(tol::Real) = tol >= 1.0e-4 ? 2 : tol >= 1.0e-7 ? 4 : tol >= 1.0e-10 ? 6 : 8

function ST.Plans.nonuniformffts_scattered_plan(x, y, ms::NTuple{2, Int}, ::Type{T};
                                                period = nothing, solve::Bool = false,
                                                maxiter::Int = 100, rtol::Real = 1.0e-8,
                                                eps = nothing) where {T}
    M = length(x)
    length(y) == M || throw(DimensionMismatch("x and y must have equal length"))
    xmin, ymin = T(minimum(x)), T(minimum(y))
    # Same default period as the in-core plan: a uniform 0:m-1 grid maps to the exact DFT nodes.
    px = period === nothing ? (T(maximum(x)) - xmin) * ms[1] / (ms[1] - 1) : T(period[1])
    py = period === nothing ? (T(maximum(y)) - ymin) * ms[2] / (ms[2] - 1) : T(period[2])
    sx = T(2π) .* (T.(x) .- xmin) ./ px
    sy = T(2π) .* (T.(y) .- ymin) ./ py

    tol = eps === nothing ? (T === Float32 ? 1.0e-6 : 1.0e-9) : Float64(eps)
    plan = NonuniformFFTs.PlanNUFFT(Complex{T}, ms;
                                    m = NonuniformFFTs.HalfSupport(_half_support(tol)))
    NonuniformFFTs.set_points!(plan, (sx, sy))

    return NonuniformFFTsScatteringPlan{T, typeof(plan), Vector{Complex{T}}, Matrix{Complex{T}}}(
        plan, ms, M, one(T) / prod(ms), solve, maxiter, T(rtol),
        Vector{Complex{T}}(undef, M),
        Matrix{Complex{T}}(undef, ms), Matrix{Complex{T}}(undef, ms), Matrix{Complex{T}}(undef, ms),
        Vector{Complex{T}}(undef, M))
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
        _cg_solve!(Xmodes, plan, x_pts)
    else
        copyto!(plan.cj, x_pts)
        NonuniformFFTs.exec_type1!(Xmodes, plan.plan, plan.cj)
    end
    return Xmodes
end

# CG least-squares inversion of the normal equations (A†A)f = A†(N·x), A = Type-2, A† = Type-1 — so
# synthesis (Type-2/N) of the recovered modes reproduces the sampled values. Mirrors the in-core and
# FINUFFT paths exactly, including the stopping rule, so the three agree to solver tolerance.
function _cg_solve!(f::AbstractMatrix, plan::NonuniformFFTsScatteringPlan{T},
                    x_pts::AbstractVector) where {T}
    N = one(T) / plan.invN
    copyto!(plan.cj, x_pts)
    NonuniformFFTs.exec_type1!(plan.r, plan.plan, plan.cj)                 # r = A†x  (modes)
    plan.r .*= N                                               # r = A†(N·x) = rhs
    fill!(f, zero(Complex{T}))
    copyto!(plan.p, plan.r)
    rsold = real(LinearAlgebra.dot(vec(plan.r), vec(plan.r)))
    rs0 = rsold
    rs0 == 0 && return f
    @inbounds for _ in 1:plan.maxiter
        NonuniformFFTs.exec_type2!(plan.tmp_pts, plan.plan, plan.p)        # tmp = A·p    (points)
        NonuniformFFTs.exec_type1!(plan.Ap, plan.plan, plan.tmp_pts)       # Ap  = A†A·p  (modes)
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

end # module ScatteringTransformsNonuniformFFTsExt