# Pointwise spherical monogenic orientation/phase on S², via spin-1 scattered synthesis
# (NUFSHT). Validated against the closed-form spin-weighted harmonic ₁Y_ℓm: for a field that is a
# single real harmonic Y_{ℓ0,0}, the band-pass and Riesz-vector fields have exact analytic forms.
using NUFSHT: NUFSHT

Test.@testset "Spherical monogenic components (spin-1 orientation/phase on S²)" begin
    lmax, J, M = 16, 3, 900
    gr = (sqrt(5) - 1) / 2                     # deterministic quasi-uniform Fibonacci-sphere points
    θ = [acos(1 - 2 * (k - 0.5) / M) for k in 1:M]
    φ = [2π * mod(k * gr, 1) for k in 1:M]
    st = ScatteringTransforms.spherical_monogenic_scattering(θ, φ, lmax, J)

    # Choose a degree ℓ0 and the scale j whose band-pass weight b_j(ℓ0) is largest.
    ℓ0 = 5
    sig = ScatteringTransforms.SphericalCore.dog_sigma2(lmax, J, Float64)
    bvals = [ScatteringTransforms.SphericalCore.band_multiplier(sig[j + 1], sig[j])(ℓ0) for j in 1:J]
    j = argmax(abs.(bvals))
    bl = bvals[j]

    field = Float64[real(NUFSHT.sYlm(0, ℓ0, 0, θ[k], φ[k])) for k in 1:M]     # real harmonic Y_{ℓ0,0}
    comp = ScatteringTransforms.spherical_monogenic_components(st, field, j)

    # Closed forms: band-pass = b_j(ℓ0)·Y_{ℓ0,0}; Riesz vector = b_j(ℓ0)·₁Y_{ℓ0,0}.
    Test.@test comp.bandpass ≈ bl .* field rtol = 1e-6
    URc = comp.riesz[1] .+ im .* comp.riesz[2]
    ref = Complex{Float64}[bl * NUFSHT.sYlm(1, ℓ0, 0, θ[k], φ[k]) for k in 1:M]
    Test.@test maximum(abs.(URc .- ref)) / maximum(abs.(ref)) < 1e-6

    Test.@testset "the spin plans are built with the transform, not per call" begin
        SC = ScatteringTransforms.SphericalCore
        # Only the monogenic constructor builds them.
        Test.@test SC.plan_spin(st.plan) !== nothing
        Test.@test SC.plan_spin(ScatteringTransforms.spherical_scattering(θ, φ, lmax, J).plan) ===
                   nothing

        # Bytes against the same decomposition on a plan that carries no spin pair: the difference
        # is at least one spin-plan build.
        nospin = SC.nusht_spherical_plan(θ, φ, lmax, Float64; spin = false)
        Test.@test SC.plan_spin(nospin) === nothing
        st_nospin = SC.SphericalMonogenicScattering(lmax, J, st.max_order, nospin, st.sigma2)
        components(s, f, jj) = ScatteringTransforms.spherical_monogenic_components(s, f, jj)
        components(st, field, j)                                              # warm up both paths
        components(st_nospin, field, j)
        cached = @allocated components(st, field, j)
        uncached = @allocated components(st_nospin, field, j)
        NUFSHT.make_spin_plan(ComplexF64, θ, φ, lmax, 0)                      # warm up
        build_bytes = @allocated NUFSHT.make_spin_plan(ComplexF64, θ, φ, lmax, 0)
        Test.@test cached < uncached
        Test.@test uncached - cached > build_bytes
        Test.@test components(st_nospin, field, j).amplitude ≈ components(st, field, j).amplitude

        # A task's own copy carries its own pair: the spin plans hold the buffers their synthesis
        # writes through.
        tl = ScatteringTransforms.Plans.task_local_plan(st.plan)
        Test.@test SC.plan_spin(tl) !== nothing
        Test.@test SC.plan_spin(tl) !== SC.plan_spin(st.plan)

        # The spin pair runs on the scalar plan's NUFFT library, in a task's copy and a widened plan too.
        for lib in (FTB.FINUFFTBackend(), FTB.NonuniformFFTsBackend())
            sl = ScatteringTransforms.spherical_monogenic_scattering(θ, φ, lmax, J; nufft = lib)
            copies = (ScatteringTransforms.Plans.task_local_plan(sl.plan), SC.batch_plan(sl.plan, 2))
            for p in (sl.plan, copies...)
                Test.@test p.nufft === lib
                Test.@test FTB._backend(NUFSHT._nufft2(p.plan).plan) === lib
                Test.@test all(q -> FTB._backend(NUFSHT._nufft2(q).plan) === lib, SC.plan_spin(p))
            end
            foreach(ScatteringTransforms.Plans.close_plan!, copies)
            ScatteringTransforms.close_transform!(sl)
        end
    end
end

# The same closed forms on the direct SH backend, where the spin-1 Riesz field is the surface gradient
# of g; `sYlm` is the analytic reference.
Test.@testset "Spherical monogenic components: dependency-free direct SH backend" begin
    lmax, J, M = 16, 3, 1200
    gr = (sqrt(5) - 1) / 2
    θ = [acos(1 - 2 * (k - 0.5) / M) for k in 1:M]
    φ = [2π * mod(k * gr, 1) for k in 1:M]
    SC = ScatteringTransforms.SphericalCore
    st = ScatteringTransforms.spherical_monogenic_scattering(θ, φ, lmax, J; transform = SpectralBackends.DirectSumSpectralBackend())
    Test.@test st.plan isa SC.DirectSHTSphericalPlan
    ℓ0 = 5
    sig = SC.dog_sigma2(lmax, J, Float64)
    bvals = [SC.band_multiplier(sig[j + 1], sig[j])(ℓ0) for j in 1:J]
    j = argmax(abs.(bvals)); bl = bvals[j]
    field = Float64[real(NUFSHT.sYlm(0, ℓ0, 0, θ[k], φ[k])) for k in 1:M]     # band-limited (ℓ = ℓ0)
    comp = ScatteringTransforms.spherical_monogenic_components(st, field, j)
    Test.@test comp.bandpass ≈ bl .* field rtol = 1e-5
    URc = comp.riesz[1] .+ im .* comp.riesz[2]
    ref = Complex{Float64}[bl * NUFSHT.sYlm(1, ℓ0, 0, θ[k], φ[k]) for k in 1:M]
    Test.@test maximum(abs.(URc .- ref)) / maximum(abs.(ref)) < 1e-5
end

Test.@testset "Riesz energy |∇_S g|² equals the finite-difference gradient on every backend" begin
    # `g` has content at the band limit `lmax`, and `g²` at twice it.
    SC = ScatteringTransforms.SphericalCore
    SBk = SpectralBackends
    lmax = 10
    modes = ((10, 3, 0.7 - 0.2im), (7, -2, 0.4 + 0.5im), (4, 0, 0.9 + 0im), (1, 1, 0.3 - 0.6im))
    g(t, f) = sum(real(c * NUFSHT.sYlm(0, ℓ, m, t, f)) for (ℓ, m, c) in modes)
    hd = 1.0e-6
    grad2(t, f) = ((g(t + hd, f) - g(t - hd, f)) / (2hd))^2 +
                  ((g(t, f + hd) - g(t, f - hd)) / (2hd) / sin(t))^2

    M = 700
    θs = [acos(1 - 2 * (k - 0.5) / M) for k in 1:M]
    φs = [2π * mod(k * (sqrt(5) - 1) / 2, 1) for k in 1:M]
    Θ, Φ = ScatteringTransforms.structured_sphere_points(lmax)
    θg, φg = [t for t in Θ, _ in Φ], [f for _ in Θ, f in Φ]
    cases = (("direct, scattered", SC.make_spherical_plan(SBk.DirectSumSpectralBackend(), θs, φs, lmax,
                                                          Float64; rtol = 1e-12), θs, φs),
             ("NUFSHT, scattered", SC.make_spherical_plan(SBk.NUFSHTSpectralBackend(), θs, φs, lmax,
                                                          Float64; rtol = 1e-9), θs, φs),
             ("direct, structured", SC.make_structured_plan(SBk.DirectSumSpectralBackend(), lmax,
                                                            Float64; rtol = 1e-12), θg, φg),
             ("FastSphericalHarmonics", SC.make_structured_plan(SBk.FSHTSpectralBackend(), lmax,
                                                                Float64), θg, φg))
    for (name, plan, θp, φp) in cases
        field = g.(θp, φp)
        ref = grad2.(θp, φp)
        out = similar(field)
        SC.sphere_riesz_energy!(out, plan, SC.sphere_coeffs(plan, field), Returns(1.0),
                                SC.riesz_scratch(plan, field))
        Test.@test maximum(abs, out .- ref) < 1e-7 * maximum(ref)
        ScatteringTransforms.Plans.close_plan!(plan)
    end

    # The cascade's first order is the mean of the monogenic amplitude under its grid's quadrature, which
    # the spin-1 synthesis of `spherical_monogenic_components` at the grid's nodes computes independently.
    # `g` is band limited, so the fit at the points and the adjoint on the grid recover the same field.
    J = 3
    st = ScatteringTransforms.spherical_monogenic_scattering(θs, φs, lmax, J; rtol = 1e-9)
    S1 = st(g.(θs, φs)).S1
    θq, φq, wq = SC.gauss_legendre_grid(SC.grid_band(st.grid), Float64)
    stq = ScatteringTransforms.spherical_monogenic_scattering(θq, φq, lmax, J; weights = wq, solve = false)
    for j in 1:J
        amp = ScatteringTransforms.spherical_monogenic_components(stq, g.(θq, φq), j).amplitude
        Test.@test S1[j] ≈ sum(wq .* amp) rtol = 1e-6
    end
    foreach(ScatteringTransforms.close_transform!, (st, stq))
end

Test.@testset "Riesz components equal the surface gradient (finite-difference ground truth)" begin
    # Each component separately, against central differences of the analytic harmonic. The Riesz field
    # is the surface gradient of `g = (−Δ_S)^{-1/2} U⁰`: `u_θ = ∂_θ g` and `u_φ = (1/sinθ) ∂_φ g`.
    # `∂_φ g` vanishes at `m = 0`, so the cases include `m ≠ 0`.
    SC = ScatteringTransforms.SphericalCore
    lmax, J, M = 12, 3, 400
    gr = (sqrt(5) - 1) / 2
    θ = [acos(1 - 2 * (k - 0.5) / M) for k in 1:M]
    φ = [2π * mod(k * gr, 1) for k in 1:M]

    # A real spherical harmonic in the plan's own convention, from the undifferentiated Legendre
    # recurrence.
    tab = SC._LegendreTable(Float64, lmax)
    function realY(ℓ, m, t, f)
        P = Matrix{Float64}(undef, lmax + 1, lmax + 1)
        SC._assoc_legendre!(P, tab, cos(t), sin(t))
        am = abs(m)
        Pv = P[ℓ + 1, am + 1]
        return m == 0 ? Pv : (m > 0 ? sqrt(2) * Pv * cos(m * f) : sqrt(2) * Pv * sin(am * f))
    end

    st = ScatteringTransforms.spherical_monogenic_scattering(θ, φ, lmax, J;
             transform = SpectralBackends.DirectSumSpectralBackend())
    sig = SC.dog_sigma2(lmax, J, Float64)
    h = 1.0e-6
    for (ℓ0, m) in ((5, 0), (5, 2), (6, -3))
        bvals = [SC.band_multiplier(sig[j + 1], sig[j])(ℓ0) for j in 1:J]
        j = argmax(abs.(bvals))
        bl = bvals[j]
        s = sqrt(ℓ0 * (ℓ0 + 1))
        field = [realY(ℓ0, m, θ[k], φ[k]) for k in 1:M]
        comp = ScatteringTransforms.spherical_monogenic_components(st, field, j)
        gθ = [bl * (realY(ℓ0, m, θ[k] + h, φ[k]) - realY(ℓ0, m, θ[k] - h, φ[k])) / (2h) / s
              for k in 1:M]
        gφ = [bl * (realY(ℓ0, m, θ[k], φ[k] + h) - realY(ℓ0, m, θ[k], φ[k] - h)) /
              (2h) / s / sin(θ[k]) for k in 1:M]
        # Absolute scale: `∂_φ g` is identically zero at `m = 0`.
        scale = maximum(abs, gθ)
        Test.@test maximum(abs, comp.riesz[1] .- gθ) < 1e-6 * scale
        Test.@test maximum(abs, comp.riesz[2] .- gφ) < 1e-6 * scale
    end
end

Test.@testset "The Riesz potential multiplier is b_j(ℓ)/√(ℓ(ℓ+1)) in the bank's precision" begin
    # `b_j(ℓ) = exp(-σ²_hi ℓ(ℓ+1)/2) − exp(-σ²_lo ℓ(ℓ+1)/2)`, and `0` at `ℓ = 0`.
    SC = ScatteringTransforms.SphericalCore
    for T in (Float32, Float64)
        hi, lo = T(0.02), T(0.08)
        h = SC.riesz_potential_multiplier(hi, lo)
        Test.@test h(0) === zero(T)
        for ℓ in (1, 4, 9)
            n = ℓ * (ℓ + 1)
            e1, e2 = exp(-Float64(hi) * n / 2), exp(-Float64(lo) * n / 2)
            want = (e1 - e2) / sqrt(n)
            # A few ulps in each exponential, magnified by the cancellation in their difference.
            tol = 4 * eps(T) * (e1 + e2) / (e1 - e2) + 4 * eps(T)
            Test.@test h(ℓ) isa T
            Test.@test h(ℓ) ≈ want rtol = tol
        end
    end
end
