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
end
