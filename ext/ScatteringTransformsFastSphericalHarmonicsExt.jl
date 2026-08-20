module ScatteringTransformsFastSphericalHarmonicsExt

"""
    ScatteringTransformsFastSphericalHarmonicsExt — structured-grid spherical scattering (fast SHT)

The **structured (uniform)** spherical backend for the shared spherical scattering core
(`SphericalCore`), alongside the scattered-point NUFSHT backend. A scalar field is sampled on the
Clenshaw–Curtis grid of `FastSphericalHarmonics`
(`Nθ = lmax+1`, `Nφ = 2lmax+1`) and analysed/synthesised with the fast `sph_transform!` /
`sph_evaluate!`. The DoG band-pass bank, S0/S1/S2 cascade, and the spin-0 Bochner monogenic amplitude
are shared with `SphericalCore`; this extension only supplies the plan and the two interface methods.

The spherical average (`sphere_mean`) is the exact quadrature integral, evaluated directly as a
weighted sum over the grid. Reading it from the degree-0 harmonic coefficient instead would be
equally exact but would cost a full forward SHT — and the cascade takes `J + J(J−1)/2 + 1` means
per transform.
"""

using FastSphericalHarmonics: FastSphericalHarmonics as FSH
using LinearAlgebra: LinearAlgebra
using SpectralBackends: SpectralBackends as SB
using ScatteringTransforms: ScatteringTransforms as ST

"""
    SHTSphericalPlan{T,V,M} <: SphericalCore.AbstractSphericalPlan

Structured spherical plan on a Clenshaw–Curtis grid (`N = lmax+1` colatitudes, `2N-1` longitudes),
backed by `FastSphericalHarmonics`. The forward SHT is exact for band-limited fields, so this backend
needs no iterative solve. `FastSphericalHarmonics` supports `Float64` only, so `T == Float64`.

`θweights` are the colatitude quadrature weights (Fejér first rule, summing to 2 as `∫sinθ dθ` does);
`cbuf` is the coefficient scratch `sphere_apply!` filters in, so a band costs no allocation.
"""
struct SHTSphericalPlan{T, V <: AbstractVector{T}, M <: AbstractMatrix{T}, C} <: ST.SphericalCore.AbstractSphericalPlan
    lmax::Int
    inv_sqrt4pi::T         # Y₀₀ = 1/√(4π)
    θweights::V            # (Nθ) quadrature weights over colatitude
    invnφ::T               # 1/Nφ — the longitude average, which is uniform
    cbuf::M                # (Nθ, Nφ) coefficient scratch
    cache::C               # FastSphericalHarmonics transform plans, built once (see below)
end

# `sph_transform!`/`sph_evaluate!` take the plan cache as a keyword argument whose default
# constructs a *fresh, empty* cache — so calling them without one rebuilds both FastTransforms
# plans on every single call, and the cascade calls them once per field plus once per band. Owning
# the cache on our plan builds them once instead.
#
# Their construction goes through FFTW's planner, which is documented as callable from only one
# thread at a time (only `fftw_execute` is thread safe), so every cache is populated under this
# lock — including the per-task copies below, which are built inside spawned tasks.
const PLANNER_LOCK = ReentrantLock()

# FastTransforms is not re-entrant from a task sharing the caller's OS thread: running it
# multithreaded from such a task silently changes what an already-built plan returns, permanently and
# with no error.
#
# `ft_set_num_threads` has no getter, but it forwards to OpenMP and `omp_get_max_threads` tracks it,
# so the count is restored afterwards rather than left mutated behind the caller's back.
function with_serial_ft(f)
    prev = ccall((:omp_get_max_threads, "libomp"), Cint, ())
    FSH.FastTransforms.ft_set_num_threads(1)
    try
        return f()
    finally
        FSH.FastTransforms.ft_set_num_threads(prev)
    end
end

function _warmed_cache(nθ::Int, nφ::Int)
    cache = FSH.SphPlanCache{Float64}()
    scratch = zeros(Float64, nθ, nφ)
    Base.@lock PLANNER_LOCK begin
        with_serial_ft() do
            FSH.sph_transform!(scratch; cache = cache)
            FSH.sph_evaluate!(scratch; cache = cache)
        end
    end
    return cache
end

function SHTSphericalPlan(lmax::Int)
    nθ, nφ = lmax + 1, 2lmax + 1
    return SHTSphericalPlan(lmax, 1 / sqrt(4π), ST.SphericalCore._fejer1_weights(nθ, Float64),
                            1 / nφ, Matrix{Float64}(undef, nθ, nφ), _warmed_cache(nθ, nφ))
end

Base.show(io::IO, p::SHTSphericalPlan{T}) where {T} =
    print(io, "SHTSphericalPlan{", T, "}(lmax=", p.lmax, ")")

# `cbuf` is mutated per band and the cached plans hold their own scratch, so a task needs both.
# A structured plan is defined by its band limit alone, so it carries no points to rebuild from and
# its quadrature follows from the grid. Analysis is an exact transform, not an iterative solve.
ST.SphericalCore.plan_points(::SHTSphericalPlan) = nothing
ST.SphericalCore.plan_weights(::SHTSphericalPlan) = nothing
ST.SphericalCore.plan_solver(::SHTSphericalPlan) = nothing
ST.Plans.spectral_backend(::SHTSphericalPlan) = SB.FSHTSpectralBackend()

ST.Plans.task_local_plan(p::SHTSphericalPlan) =
    SHTSphericalPlan(p.lmax, p.inv_sqrt4pi, p.θweights, p.invnφ, similar(p.cbuf),
                     _warmed_cache(size(p.cbuf)...))

# Multiply each spherical-harmonic coefficient of degree ℓ by h(ℓ), in the FastSphericalHarmonics
# triangular `sph_mode` layout (in place).
function _apply_degree_multiplier!(C::AbstractMatrix, h, lmax::Int)
    @inbounds for l in 0:lmax
        hl = h(l)
        for m in -l:l
            C[FSH.sph_mode(l, m)] *= hl
        end
    end
    return C
end

# Analysis: exact forward SHT (grid → SH coefficients).
function ST.SphericalCore.sphere_coeffs(plan::SHTSphericalPlan, field::AbstractMatrix)
    return ST.SphericalCore.sphere_coeffs!(Matrix{Float64}(undef, size(field)), plan, field)
end

function ST.SphericalCore.sphere_coeffs!(C::AbstractMatrix, plan::SHTSphericalPlan,
                                         field::AbstractMatrix)
    copyto!(C, field)
    with_serial_ft(() -> FSH.sph_transform!(C; cache = plan.cache))
    return C
end

ST.SphericalCore.sphere_coeffs_buffer(plan::SHTSphericalPlan) =
    Matrix{Float64}(undef, plan.lmax + 1, 2plan.lmax + 1)

# Apply the per-degree multiplier h(ℓ) to a copy of the coefficients and synthesise back to the grid.
# The copy goes into the plan's scratch, so a band allocates nothing; `C` itself is left intact for
# the next band.
function ST.SphericalCore.sphere_apply!(out::AbstractMatrix, plan::SHTSphericalPlan, C, h)
    C2 = plan.cbuf
    copyto!(C2, C)
    _apply_degree_multiplier!(C2, h, plan.lmax)
    with_serial_ft(() -> FSH.sph_evaluate!(C2; cache = plan.cache))
    copyto!(out, C2)
    return out
end

# ⟨f⟩ = (1/4π)∬ f sinθ dθ dφ. Longitude is uniform, so its integral is the plain row mean; colatitude
# uses the Fejér weights, which integrate ∫₀^π g sinθ dθ exactly for band-limited `g` and sum to 2 —
# hence the trailing ½. Equal to the degree-0 coefficient route, without its forward SHT.
function ST.SphericalCore.sphere_mean(plan::SHTSphericalPlan{T}, field::AbstractMatrix) where {T}
    nθ, nφ = size(field)
    acc = zero(T)
    @inbounds for k in 1:nφ, j in 1:nθ
        acc += plan.θweights[j] * field[j, k]
    end
    return acc * plan.invnφ / 2
end

# ---------------------------------------------------------------------------
# Fast-path structured plan constructor (structured-sphere seam declared in SphericalCore). The core
# `structured_spherical_scattering` / `structured_spherical_monogenic_scattering` build this when
# `spectral` selects the fast SHT. The grid (`structured_sphere_points`) is computed in core and
# matches `FSH.sph_points`, so a field sampled there is valid input to this plan.
# ---------------------------------------------------------------------------

function ST.SphericalCore.fsh_structured_plan(lmax::Int, ::Type{T}) where {T}
    T === Float64 || throw(ArgumentError("FastSphericalHarmonics supports Float64 only; got $T."))
    return SHTSphericalPlan(lmax)
end

end # module ScatteringTransformsFastSphericalHarmonicsExt
