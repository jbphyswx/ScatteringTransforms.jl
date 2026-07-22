# Structured (uniform) spherical scattering via the fast SHT path, on the
# FastSphericalHarmonics Clenshaw–Curtis grid. Shares SphericalCore's DoG bank + cascade with the
# scattered NUFSHT backend; only the spectral transform (exact fast SHT) differs.
using FastSphericalHarmonics: FastSphericalHarmonics
using NUFSHT: NUFSHT   # scattered backend for the cross-backend agreement test + closed-form sYlm

Test.@testset "Structured spherical scattering (fast SHT on a Clenshaw–Curtis grid)" begin
    lmax, J = 24, 3
    Θ, Φ = ScatteringTransforms.structured_sphere_points(lmax)
    Test.@test length(Θ) == lmax + 1
    Test.@test length(Φ) == 2lmax + 1

    st = ScatteringTransforms.structured_spherical_scattering(lmax, J)

    # sphere_mean is the exact CC quadrature: a constant field integrates to itself; a pure
    # harmonic of degree ℓ>0 integrates to ~0.
    Test.@test st(fill(2.5, lmax + 1, 2lmax + 1)).S0 ≈ 2.5
    C = zeros(lmax + 1, 2lmax + 1)
    C[FastSphericalHarmonics.sph_mode(2, 0)] = 1.0
    Y20 = FastSphericalHarmonics.sph_evaluate(C)                 # pure ₂Y₀ field on the grid
    Test.@test abs(st(Y20).S0) < 1e-10

    # Smooth band-limited field.
    f = [cos(θ)^2 - 1/3 + 0.5 * sin(θ) * cos(φ) for θ in Θ, φ in Φ]
    res = st(f)
    Test.@test length(res.S1) == J
    Test.@test all(res.S1 .>= 0)
    Test.@test all(isfinite, res.S1)
    Test.@test size(res.S2) == (J, J)
    for j1 in 1:J, j2 in j1:J                                    # strictly coarser j2 < j1 only
        Test.@test res.S2[j1, j2] == 0
    end

    # The fast SHT is exact, so z-rotation covariance is tight (much tighter than the scattered path).
    α = 0.7
    fα = [cos(θ)^2 - 1/3 + 0.5 * sin(θ) * cos(φ + α) for θ in Θ, φ in Φ]
    rα = st(fα)
    Test.@test maximum(abs.(res.S1 .- rα.S1)) / maximum(abs.(res.S1)) < 1e-3

    # Monogenic amplitude ≥ analytic modulus pointwise ⇒ ⟨A_j⟩ ≥ ⟨|U⁰_j|⟩ for every scale.
    stm = ScatteringTransforms.structured_spherical_monogenic_scattering(lmax, J)
    rm = stm(f)
    Test.@test all(isfinite, rm.S1)
    Test.@test all(rm.S1 .>= res.S1 .- 1e-8)
end

# Cross-backend agreement: the two spherical backends compute the SAME transform, so on a smooth
# band-limited field (adequately sampled for the scattered CG inversion) their S0/S1/S2 must agree.
# This is the test that pins the *absolute* scale — the scattered path's exact analysis
# (`nusht_solve!`) makes it match the exact structured SHT; the earlier adjoint analysis was 6–9× off
# and only passed the scale-invariant checks. First order is compared tightly; second order is looser
# because its input is a modulus (broadband), which the two backends truncate at the shared band limit
# slightly differently — so the tolerance there guards the ~7× regression, not machine agreement.
Test.@testset "Structured (SHT) and scattered (NUFSHT) spherical scattering agree" begin
    lmax, J = 16, 3
    Θ, Φ = ScatteringTransforms.structured_sphere_points(lmax)
    M = 3000                                   # ≫ (lmax+1)² = 289 for an accurate scattered inversion
    gr = (sqrt(5) - 1) / 2
    θ = [acos(1 - 2 * (k - 0.5) / M) for k in 1:M]
    φ = [2π * mod(k * gr, 1) for k in 1:M]
    sst = ScatteringTransforms.structured_spherical_scattering(lmax, J)
    scst = ScatteringTransforms.spherical_scattering(θ, φ, lmax, J)
    for gfun in ((θ, φ) -> cos(θ)^2 - 1/3 + 0.5sin(θ) * cos(φ),   # smooth, content well below lmax
                 (θ, φ) -> cos(4θ) + 0.5sin(3θ) * cos(2φ))
        rstr = sst([gfun(θ, φ) for θ in Θ, φ in Φ])
        rsca = scst([gfun(θ[k], φ[k]) for k in 1:M])
        Test.@test rsca.S0 ≈ rstr.S0 atol = 1e-3
        Test.@test maximum(abs.(rsca.S1 .- rstr.S1)) / maximum(abs.(rstr.S1)) < 0.02
        Test.@test maximum(abs.(rsca.S2 .- rstr.S2)) / (maximum(abs.(rstr.S2)) + eps()) < 0.10
    end
end

# The in-core dependency-free direct-summation SH backend and the NUFSHT fast path solve the same
# scattered least-squares SH inversion, so on a band-limited, adequately-sampled field (M ≫ (lmax+1)²)
# their scattering coefficients agree closely — S1 essentially to solver tolerance, S2 looser (its
# input is a broadband modulus truncated slightly differently). This verifies `spherical_scattering`
# works with no external library (`spectral = DirectSHTBackend()`).
Test.@testset "Dependency-free direct SH backend agrees with NUFSHT (scattered S²)" begin
    lmax, J, M = 12, 3, 2500
    gr = (sqrt(5) - 1) / 2
    θ = [acos(1 - 2 * (k - 0.5) / M) for k in 1:M]
    φ = [2π * mod(k * gr, 1) for k in 1:M]
    SC = ScatteringTransforms.SphericalCore
    for gfun in ((θ, φ) -> cos(θ)^2 - 1/3 + 0.5sin(θ) * cos(φ),
                 (θ, φ) -> cos(2θ) + 0.4sin(θ) * cos(2φ))
        f = [gfun(θ[k], φ[k]) for k in 1:M]
        sdir = ScatteringTransforms.spherical_scattering(θ, φ, lmax, J; spectral = SC.DirectSHTBackend())
        snu  = ScatteringTransforms.spherical_scattering(θ, φ, lmax, J; spectral = SC.NUSHTBackend())
        Test.@test sdir.plan isa SC.DirectSHTSphericalPlan
        rdir, rnu = sdir(f), snu(f)
        Test.@test rdir.S0 ≈ rnu.S0 atol = 1e-6
        Test.@test maximum(abs.(rdir.S1 .- rnu.S1)) / maximum(abs.(rnu.S1)) < 0.01
        Test.@test maximum(abs.(rdir.S2 .- rnu.S2)) / (maximum(abs.(rnu.S2)) + eps()) < 0.05
    end
end

# The in-core dependency-free direct SHT on the structured (equiangular) grid vs the
# FastSphericalHarmonics fast exact SHT: on a band-limited field the least-squares SH fit is exact, so
# S1 matches to solver tolerance and the exact Fejér-quadrature mean matches. This verifies
# `structured_spherical_scattering` works with no external library (`spectral = DirectSHTBackend()`).
Test.@testset "Dependency-free structured SHT agrees with FastSphericalHarmonics" begin
    lmax, J = 16, 3
    Θ, Φ = ScatteringTransforms.structured_sphere_points(lmax)
    SC = ScatteringTransforms.SphericalCore
    for gfun in ((θ, φ) -> cos(θ)^2 - 1/3 + 0.5sin(θ) * cos(φ),      # band-limited to ℓ ≤ 2
                 (θ, φ) -> sin(θ)^2 * cos(2φ) + 0.6cos(θ))
        f = [gfun(θ, φ) for θ in Θ, φ in Φ]
        sdir = ScatteringTransforms.structured_spherical_scattering(lmax, J; spectral = SC.DirectSHTBackend())
        sfsh = ScatteringTransforms.structured_spherical_scattering(lmax, J; spectral = SC.SHTBackend())
        Test.@test sdir.plan isa SC.DirectSHTSphericalPlan
        rdir, rfsh = sdir(f), sfsh(f)
        Test.@test rdir.S0 ≈ rfsh.S0 atol = 1e-8
        Test.@test maximum(abs.(rdir.S1 .- rfsh.S1)) / maximum(abs.(rfsh.S1)) < 0.01
        Test.@test maximum(abs.(rdir.S2 .- rfsh.S2)) / (maximum(abs.(rfsh.S2)) + eps()) < 0.05
    end
end
