module ScatteringTransformsFastSphericalHarmonicsExt

"""
    ScatteringTransformsFastSphericalHarmonicsExt — structured-grid spherical scattering (fast SHT)

The **structured (uniform)** spherical backend for the shared spherical scattering core
(`SphericalCore`), alongside the scattered-point NUFSHT backend. A scalar field is sampled on the
Clenshaw–Curtis grid of `FastSphericalHarmonics`
(`Nθ = lmax+1`, `Nφ = 2lmax+1`) and analysed/synthesised with the fast `sph_transform!` /
`sph_evaluate!`. The DoG band-pass bank, S0/S1/S2 cascade, and the spin-0 Bochner monogenic amplitude
are shared with `SphericalCore`; this extension only supplies the plan and the two interface methods.

The spherical average (`sphere_mean`) is the exact Clenshaw–Curtis quadrature integral, read directly
from the degree-0 harmonic coefficient (`c₀₀ · Y₀₀`, `Y₀₀ = 1/√(4π)`).
"""

using FastSphericalHarmonics: FastSphericalHarmonics
using ScatteringTransforms: ScatteringTransforms

const ST = ScatteringTransforms
const SC = ST.SphericalCore
const FSH = FastSphericalHarmonics

"""
    SHTSphericalPlan{T} <: SphericalCore.AbstractSphericalPlan

Structured spherical plan on a Clenshaw–Curtis grid (`N = lmax+1` colatitudes, `2N-1` longitudes),
backed by `FastSphericalHarmonics`. The forward SHT is exact for band-limited fields, so this backend
needs no iterative solve. `FastSphericalHarmonics` supports `Float64` only, so `T == Float64`.
"""
struct SHTSphericalPlan{T} <: SC.AbstractSphericalPlan
    lmax::Int
    inv_sqrt4pi::T         # Y₀₀ = 1/√(4π): turns the ℓ=0 coefficient into the spherical mean
end

SHTSphericalPlan(lmax::Int) = SHTSphericalPlan{Float64}(lmax, 1 / sqrt(4π))

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
function SC.sphere_coeffs(plan::SHTSphericalPlan, field::AbstractMatrix)
    C = Matrix{Float64}(undef, size(field))
    copyto!(C, field)
    FSH.sph_transform!(C)
    return C
end

# Apply the per-degree multiplier h(ℓ) to a copy of the coefficients and synthesise back to the grid.
function SC.sphere_apply!(out::AbstractMatrix, plan::SHTSphericalPlan, C, h)
    C2 = copy(C)
    _apply_degree_multiplier!(C2, h, plan.lmax)
    FSH.sph_evaluate!(C2)
    copyto!(out, C2)
    return out
end

# Exact spherical average via the degree-0 coefficient: ⟨f⟩ = c₀₀ · Y₀₀ (Clenshaw–Curtis quadrature).
function SC.sphere_mean(plan::SHTSphericalPlan, field::AbstractMatrix)
    C = Matrix{Float64}(undef, size(field))
    copyto!(C, field)
    FSH.sph_transform!(C)
    return C[FSH.sph_mode(0, 0)] * plan.inv_sqrt4pi
end

# ---------------------------------------------------------------------------
# Constructors + grid helper (structured-sphere entry points declared in core)
# ---------------------------------------------------------------------------

ST.structured_sphere_points(lmax::Int) = FSH.sph_points(lmax + 1)

function ST.structured_spherical_scattering(lmax::Int, J::Int; max_order::Int = 2)
    plan = SHTSphericalPlan(lmax)
    return SC.SphericalScattering{Float64, typeof(plan)}(lmax, J, max_order, plan,
                                                         SC.dog_sigma2(lmax, J, Float64))
end

function ST.structured_spherical_monogenic_scattering(lmax::Int, J::Int; max_order::Int = 2)
    plan = SHTSphericalPlan(lmax)
    return SC.SphericalMonogenicScattering{Float64, typeof(plan)}(lmax, J, max_order, plan,
                                                                  SC.dog_sigma2(lmax, J, Float64))
end

end # module ScatteringTransformsFastSphericalHarmonicsExt
