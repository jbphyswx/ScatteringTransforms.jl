module ScatteringTransformsFastSphericalHarmonicsExt

# Structured-grid spherical plan on FastSphericalHarmonics: a field sampled on its `(lmax+1, 2lmax+1)`
# midpoint grid is analysed and synthesised with `sph_transform!` / `sph_evaluate!`. `sphere_mean` is
# the exact quadrature sum over the grid, which needs no forward SHT.

using FastSphericalHarmonics: FastSphericalHarmonics as FSH
using FlowTransformBindings: FlowTransformBindings as FTB
using LinearAlgebra: LinearAlgebra
using SpectralBackends: SpectralBackends as SB
using ScatteringTransforms: ScatteringTransforms as ST

"""
    SHTSphericalPlan{T,V,M,C} <: SphericalCore.AbstractSphericalPlan

Structured spherical plan on the equiangular midpoint grid (`N = lmax+1` colatitudes
`θ_j = π(j-½)/N`, `2N-1` longitudes), backed by `FastSphericalHarmonics`. The forward SHT is exact
for band-limited fields, so this backend needs no iterative solve. `FastSphericalHarmonics`
supports `Float64` only, so `T == Float64`.

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

# `sph_transform!`/`sph_evaluate!` build a fresh plan cache on every call unless given one, so the
# plan owns a cache. FastTransforms plans on FFTW.jl's process-global libfftw3 planner, so every
# cache, the per-task copies built inside spawned tasks included, is populated under FFTW.jl's
# planner lock at one planner thread (`Plans.with_fft_nthreads`).
#
# Every FastTransforms call goes through `FTB.with_fasttransforms_threads`, which sets the OpenMP count
# on the OS thread making the call and restores it after.
function _warmed_cache(nθ::Int, nφ::Int)
    cache = FSH.SphPlanCache{Float64}()
    scratch = zeros(Float64, nθ, nφ)
    ST.Plans.with_fft_nthreads(1) do
        FTB.with_fasttransforms_threads() do
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

# A structured plan is defined by its band limit: it has no points, its quadrature follows from the
# grid, and its analysis is an exact transform with no solver settings.
ST.SphericalCore.plan_points(::SHTSphericalPlan) = nothing
ST.SphericalCore.plan_weights(::SHTSphericalPlan) = nothing
ST.SphericalCore.plan_solver(::SHTSphericalPlan) = nothing
ST.Plans.spectral_backend(::SHTSphericalPlan) = SB.FSHTSpectralBackend()

# `cbuf` is mutated per band and the cached plans hold their own scratch, so a task needs both.
ST.Plans.task_local_plan(p::SHTSphericalPlan) =
    SHTSphericalPlan(p.lmax, p.inv_sqrt4pi, p.θweights, p.invnφ, similar(p.cbuf),
                     _warmed_cache(size(p.cbuf)...))

# Multiply every coefficient of degree ℓ by h(ℓ), in place. Column `c` carries `|m| = c ÷ 2` and row
# `r` carries `l - |m|`; the `(nθ, nφ)` matrix also holds degrees above `nθ-1` at high `|m|`, and
# those are multiplied too.
function _apply_degree_multiplier!(C::AbstractMatrix, h)
    nθ, nφ = size(C)
    @inbounds for c in 1:nφ
        am = c >> 1
        for r in 1:nθ
            C[r, c] *= h(r - 1 + am)
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
    FTB.with_fasttransforms_threads(() -> FSH.sph_transform!(C; cache = plan.cache))
    return C
end

ST.SphericalCore.sphere_coeffs_buffer(plan::SHTSphericalPlan) =
    Matrix{Float64}(undef, plan.lmax + 1, 2plan.lmax + 1)

# A structured plan is defined by its band limit alone, so narrowing is just building a smaller one.
ST.SphericalCore.sphere_plan_at(::SHTSphericalPlan, lmax::Integer) = SHTSphericalPlan(Int(lmax))
ST.SphericalCore.sphere_lmax(p::SHTSphericalPlan) = p.lmax

# A coefficient's position encodes `(l, m)` the same way at every grid size (column `m`, row
# `l - |m|`), so the coefficients a narrower grid represents are its leading sub-block, degrees above
# `nθ-1` at high `|m|` included.
function ST.SphericalCore.sphere_restrict!(Cd::AbstractMatrix, ::SHTSphericalPlan,
                                           Cs::AbstractMatrix, ::SHTSphericalPlan)
    nθ, nφ = size(Cd)
    @inbounds copyto!(Cd, view(Cs, 1:nθ, 1:nφ))
    return Cd
end

# Apply the per-degree multiplier h(ℓ) to a copy of the coefficients and synthesise back to the grid.
# The copy goes into the plan's scratch, so a band allocates nothing; `C` itself is left intact for
# the next band.
function ST.SphericalCore.sphere_apply!(out::AbstractMatrix, plan::SHTSphericalPlan, C, h)
    C2 = plan.cbuf
    copyto!(C2, C)
    _apply_degree_multiplier!(C2, h)
    FTB.with_fasttransforms_threads(() -> FSH.sph_evaluate!(C2; cache = plan.cache))
    copyto!(out, C2)
    return out
end

# The coefficients are in the real `sph_mode` layout.
ST.SphericalCore.riesz_scratch(plan::SHTSphericalPlan, field::AbstractMatrix) =
    ST.SphericalCore.sph_layout_riesz_scratch(plan, field, plan.lmax)
ST.SphericalCore.sphere_riesz_energy!(out::AbstractMatrix, plan::SHTSphericalPlan, C, h, scr) =
    ST.SphericalCore.sph_layout_riesz_energy!(out, plan, C, h, plan.lmax, scr)

# ⟨f⟩ = (1/4π)∬ f sinθ dθ dφ. Longitude is uniform, so its integral is the plain row mean; colatitude
# uses the Fejér weights, which integrate ∫₀^π g sinθ dθ exactly for band-limited `g` and sum to 2,
# hence the trailing ½.
function ST.SphericalCore.sphere_mean(plan::SHTSphericalPlan{T}, field::AbstractMatrix) where {T}
    nθ, nφ = size(field)
    acc = zero(T)
    @inbounds for k in 1:nφ, j in 1:nθ
        acc += plan.θweights[j] * field[j, k]
    end
    return acc * plan.invnφ / 2
end

# ---------------------------------------------------------------------------
# The structured plan `SphericalCore.make_structured_plan` builds for `FSHTSpectralBackend`. Fields
# sampled at `structured_sphere_points` are valid input.
# ---------------------------------------------------------------------------

function ST.SphericalCore.fsh_structured_plan(lmax::Int, ::Type{T}) where {T}
    T === Float64 || throw(ArgumentError("FastSphericalHarmonics supports Float64 only; got $T."))
    return SHTSphericalPlan(lmax)
end

end # module ScatteringTransformsFastSphericalHarmonicsExt
