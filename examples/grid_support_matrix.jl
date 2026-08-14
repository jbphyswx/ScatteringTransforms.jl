"""
    grid_support_matrix.jl

The grid-support matrix: planar (Cartesian) and spherical (S²) scattering, on uniform/structured and
nonuniform/scattered sampling. Shows the NUFFT scattered-planar path (reproducing the gridded FFT
transform on a uniform grid), the fast-SHT structured-sphere path, the scattered-sphere NUFSHT path,
and pointwise spherical monogenic orientation/phase (spin-1 Riesz vector).
Run with: `julia --project=. grid_support_matrix.jl`
"""

using ScatteringTransforms: ScatteringTransforms as ST
using SpectralBackends: SpectralBackends as SB
using FFTW: FFTW
using FINUFFT: FINUFFT                        # scattered / nonuniform planar (Cartesian) path
using FastSphericalHarmonics: FastSphericalHarmonics   # structured (uniform) sphere path
using NUFSHT: NUFSHT                          # scattered sphere path
using Test: Test

println("="^64)
println("ScatteringTransforms.jl — grid-support matrix")
println("="^64)

# ---------------------------------------------------------------------------
# Cartesian, nonuniform / scattered (NUFFT). On a uniform grid it reduces to the FFT transform.
# ---------------------------------------------------------------------------
println("\n1. Scattered-planar scattering (NUFFT)")
Ny, Nx, J, L = 24, 24, 3, 4
f = randn(Ny, Nx)
grid = ST.Scattering2D.ScatteringTransform2D((Ny, Nx), J; L=L, max_order=2, spectral=SB.FFTSpectralBackend())
n1 = vec([Float64(i) for i in 0:Ny-1, j in 0:Nx-1])
n2 = vec([Float64(j) for i in 0:Ny-1, j in 0:Nx-1])
sca = ST.scattered_planar_scattering(n1, n2, (Ny, Nx), J; L=L, max_order=2, period=(Ny, Nx))
relerr = maximum(abs.(ST.Coefficients.flatten2d(sca(vec(f))) .- ST.Coefficients.flatten2d(grid(f)))) /
         maximum(abs.(ST.Coefficients.flatten2d(grid(f))))
println("   uniform-grid parity vs gridded FFT: rel-err = ", relerr)
println("   (irregular data: pass solve=true for the exact band-limited CG inversion)")

# ---------------------------------------------------------------------------
# Sphere: structured (fast SHT) and scattered (NUFSHT).
# ---------------------------------------------------------------------------
println("\n2. Structured spherical scattering (fast SHT on a Clenshaw–Curtis grid)")
lmax, Js = 16, 3
Θ, Φ = ST.structured_sphere_points(lmax)
gfield = [cos(θ)^2 - 1/3 + 0.4sin(θ)*cos(φ) for θ in Θ, φ in Φ]
sst = ST.structured_spherical_scattering(lmax, Js)
rs = sst(gfield)
println("   S0 = ", round(rs.S0, digits=4), " | S1 = ", round.(rs.S1, digits=4))

println("\n3. Scattered spherical scattering (NUFSHT)")
M = 800; gr = (sqrt(5)-1)/2
θ = [acos(1 - 2*(k-0.5)/M) for k in 1:M]; φ = [2π*mod(k*gr, 1) for k in 1:M]
scst = ST.spherical_scattering(θ, φ, lmax, Js)
rsc = scst(randn(M))
println("   S1 = ", round.(rsc.S1, digits=4))

# ---------------------------------------------------------------------------
# Pointwise spherical monogenic orientation/phase (spin-1 Riesz vector).
# ---------------------------------------------------------------------------
println("\n4. Spherical monogenic components (spin-1 orientation/phase)")
mst = ST.spherical_monogenic_scattering(θ, φ, lmax, Js)
field = randn(M)
comp = ST.spherical_monogenic_components(mst, field, 2)
println("   fields: ", propertynames(comp))
println("   amplitude ∈ [", round(minimum(comp.amplitude), digits=3), ", ",
        round(maximum(comp.amplitude), digits=3), "], orientation range = ",
        round(maximum(comp.orientation) - minimum(comp.orientation), digits=3), " rad")

Test.@testset "grid_support_matrix smoke" begin
    Test.@test relerr < 1e-6                                  # NUFFT reproduces gridded FFT on a grid
    Test.@test all(rs.S1 .>= 0) && all(isfinite, rs.S1)
    Test.@test all(rsc.S1 .>= 0)
    Test.@test comp.amplitude ≈ sqrt.(comp.bandpass.^2 .+ comp.riesz[1].^2 .+ comp.riesz[2].^2)
end

println("\n", "="^64, "\nDone.\n", "="^64)
