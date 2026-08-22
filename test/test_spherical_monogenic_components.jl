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

    # API: five named fields, each length M; riesz is the (u_θ, u_φ) tangent vector.
    Test.@test propertynames(comp) == (:bandpass, :riesz, :amplitude, :phase, :orientation)
    Test.@test length(comp.bandpass) == M && length(comp.riesz) == 2
    Test.@test length(comp.riesz[1]) == M && length(comp.riesz[2]) == M

    # Internal consistency of amplitude / phase / orientation.
    rnorm = sqrt.(comp.riesz[1] .^ 2 .+ comp.riesz[2] .^ 2)
    Test.@test comp.amplitude ≈ sqrt.(comp.bandpass .^ 2 .+ rnorm .^ 2)
    Test.@test comp.phase ≈ atan.(rnorm, comp.bandpass)
    Test.@test comp.orientation ≈ atan.(comp.riesz[2], comp.riesz[1])
    Test.@test all(comp.amplitude .>= abs.(comp.bandpass) .- 1e-10)      # monogenic amp ≥ |bandpass|

    # Closed forms: band-pass = b_j(ℓ0)·Y_{ℓ0,0}; Riesz vector = b_j(ℓ0)·₁Y_{ℓ0,0}.
    Test.@test comp.bandpass ≈ bl .* field rtol = 1e-6
    URc = comp.riesz[1] .+ im .* comp.riesz[2]
    ref = Complex{Float64}[bl * NUFSHT.sYlm(1, ℓ0, 0, θ[k], φ[k]) for k in 1:M]
    Test.@test maximum(abs.(URc .- ref)) / maximum(abs.(ref)) < 1e-6

    # z-rotation covariance of the amplitude field (SH transform is exact ⇒ tight).
    α = 0.6
    fα = Float64[real(NUFSHT.sYlm(0, ℓ0, 0, θ[k], φ[k])) for k in 1:M]       # m=0 ⇒ φ-invariant
    Test.@test comp.amplitude ≈ ScatteringTransforms.spherical_monogenic_components(st, fα, j).amplitude

    Test.@testset "the spin plans are built with the transform, not per call" begin
        SC = ScatteringTransforms.SphericalCore
        # Only the monogenic constructor asks for them, so a plain spherical transform on the same
        # points pays nothing for a decomposition it does not perform.
        Test.@test SC.plan_spin(st.plan) !== nothing
        Test.@test SC.plan_spin(ScatteringTransforms.spherical_scattering(θ, φ, lmax, J).plan) ===
                   nothing

        # Gated by counting bytes against the same decomposition on a plan that carries no pair, so
        # the constant is measured here rather than assumed: the difference between the two is what
        # the caching removes, and it must be at least one plan build (it is two plus their Wigner-d
        # tables). A single call allocates more than one plan build on its own — the CG solve and
        # seven length-M outputs — so the per-call figure is only meaningful against its own
        # uncached twin.
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
        # And the two agree: taking the pair off the plan changes only where it came from.
        Test.@test components(st_nospin, field, j).amplitude ≈ components(st, field, j).amplitude

        # A task's own copy carries its own pair: the spin plans hold the buffers their synthesis
        # writes through, so sharing one across tasks would corrupt both.
        tl = ScatteringTransforms.Plans.task_local_plan(st.plan)
        Test.@test SC.plan_spin(tl) !== nothing
        Test.@test SC.plan_spin(tl) !== SC.plan_spin(st.plan)
    end
end

# Same validation for the in-core dependency-free direct SH backend (no NUFSHT for the transform — the
# spin-1 Riesz field is the surface gradient of g). Band-pass = b_j(ℓ0)·Y_{ℓ0,0} and Riesz vector =
# b_j(ℓ0)·₁Y_{ℓ0,0}, matching the closed-form spin-weighted harmonic (`sYlm` is used only as the
# analytic reference here).
Test.@testset "Spherical monogenic components: dependency-free direct SH backend" begin
    lmax, J, M = 16, 3, 1200
    gr = (sqrt(5) - 1) / 2
    θ = [acos(1 - 2 * (k - 0.5) / M) for k in 1:M]
    φ = [2π * mod(k * gr, 1) for k in 1:M]
    SC = ScatteringTransforms.SphericalCore
    st = ScatteringTransforms.spherical_monogenic_scattering(θ, φ, lmax, J; spectral = SpectralBackends.DirectSumSpectralBackend())
    Test.@test st.plan isa SC.DirectSHTSphericalPlan
    ℓ0 = 5
    sig = SC.dog_sigma2(lmax, J, Float64)
    bvals = [SC.band_multiplier(sig[j + 1], sig[j])(ℓ0) for j in 1:J]
    j = argmax(abs.(bvals)); bl = bvals[j]
    field = Float64[real(NUFSHT.sYlm(0, ℓ0, 0, θ[k], φ[k])) for k in 1:M]     # band-limited (ℓ = ℓ0)
    comp = ScatteringTransforms.spherical_monogenic_components(st, field, j)
    Test.@test comp.amplitude ≈ sqrt.(comp.bandpass .^ 2 .+ comp.riesz[1] .^ 2 .+ comp.riesz[2] .^ 2)
    Test.@test all(comp.amplitude .>= abs.(comp.bandpass) .- 1e-10)
    Test.@test comp.bandpass ≈ bl .* field rtol = 1e-5
    URc = comp.riesz[1] .+ im .* comp.riesz[2]
    ref = Complex{Float64}[bl * NUFSHT.sYlm(1, ℓ0, 0, θ[k], φ[k]) for k in 1:M]
    Test.@test maximum(abs.(URc .- ref)) / maximum(abs.(ref)) < 1e-5
end

Test.@testset "Riesz components equal the surface gradient (finite-difference ground truth)" begin
    # The comparison above is backend-against-backend, so it cannot pin a sign: it checks the complex
    # combination `u_θ + i·u_φ`, and a flip in both components against a flipped reference passes.
    # That is not hypothetical — it is how a sign error in both survived here.
    #
    # This gates each component separately against something neither backend supplies. The Riesz field
    # is the surface gradient of `g = (−Δ_S)^{-1/2} U⁰`: `u_θ = ∂_θ g` and `u_φ = (1/sinθ) ∂_φ g`, so
    # central differences of the analytic harmonic are the ground truth. `m = 0` alone is not enough —
    # `∂_φ g` vanishes there, which is exactly what hid the `u_φ` sign.
    SC = ScatteringTransforms.SphericalCore
    lmax, J, M = 12, 3, 400
    gr = (sqrt(5) - 1) / 2
    θ = [acos(1 - 2 * (k - 0.5) / M) for k in 1:M]
    φ = [2π * mod(k * gr, 1) for k in 1:M]

    # A real spherical harmonic in the plan's own convention, from the *undifferentiated* Legendre
    # recurrence, so the θ-derivative under test is not used to build its own reference.
    function realY(ℓ, m, t, f)
        P = Matrix{Float64}(undef, lmax + 1, lmax + 1)
        SC._assoc_legendre!(P, cos(t), lmax)
        am = abs(m)
        Pv = P[ℓ + 1, am + 1]
        return m == 0 ? Pv : (m > 0 ? sqrt(2) * Pv * cos(m * f) : sqrt(2) * Pv * sin(am * f))
    end

    st = ScatteringTransforms.spherical_monogenic_scattering(θ, φ, lmax, J;
             spectral = SpectralBackends.DirectSumSpectralBackend())
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
        # Absolute scale, not relative: `∂_φ g` is identically zero at `m = 0`, so a relative
        # tolerance there would divide by zero and report nonsense either way.
        scale = maximum(abs, gθ)
        Test.@test maximum(abs, comp.riesz[1] .- gθ) < 1e-6 * scale
        Test.@test maximum(abs, comp.riesz[2] .- gφ) < 1e-6 * scale
    end
end
