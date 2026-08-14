"""
    monogenic.jl

Monogenic (Riesz) wavelet scattering — the rotation-covariant alternative to oriented-modulus
scattering. Instead of fixing discrete orientations and taking an analytic modulus, the monogenic
transform pairs an *isotropic* band-pass with its Riesz components and reports a **monogenic
amplitude** (a smooth envelope) plus a **continuous local orientation/phase**.

Demonstrates: 1D/2D/3D `MonogenicScattering`, the Riesz-multiplier partition of unity, the
amplitude/orientation decomposition (`monogenic_components`), and rotation behaviour.

Run with: `julia --project=. monogenic.jl`
"""

using ScatteringTransforms: ScatteringTransforms as ST
using SpectralBackends: SpectralBackends as SB
using FFTW: FFTW
using Statistics: Statistics
using Test: Test

println("="^64)
println("Monogenic (Riesz) scattering")
println("="^64)

# ── Riesz multipliers tile unity off the DC bin ───────────────────────────────
R = ST.Monogenic.riesz_multipliers((32, 32), Float64)
partition = sum(abs2.(Rd) for Rd in R)
println("\nRiesz partition Σ_d|R_d|² = 1 off-DC: ",
        maximum(abs.([partition[i] for i in CartesianIndices(partition) if i != CartesianIndex(1, 1)] .- 1)) < 1e-12)

# ── 1D / 2D / 3D monogenic scattering ─────────────────────────────────────────
let
    c1 = ST.Monogenic.MonogenicScattering(256, 5; Q=1, max_order=2)(randn(256))
    c2 = ST.Monogenic.MonogenicScattering((48, 48), 3; Q=1, max_order=2)(randn(48, 48))
    c3 = ST.Monogenic.MonogenicScattering((16, 16, 16), 2; max_order=2)(randn(16, 16, 16))
    println("1D/2D/3D monogenic S1 lengths: ",
            length(ST.Coefficients.first_order(c1)), " / ", length(ST.Coefficients.first_order(c2)), " / ", length(ST.Coefficients.first_order(c3)))
end

# ── amplitude envelope + continuous orientation on an oriented field ──────────
# Concentric rings (constant wavelength): local orientation rotates with azimuth.
M = 128
cx = cy = (M + 1) / 2
rr = [sqrt((i - cx)^2 + (j - cy)^2) for i in 1:M, j in 1:M]
field = cos.(2π .* rr ./ (M / 16))
st = ST.Monogenic.MonogenicScattering((M, M), 5; Q=1, max_order=1, spectral=SB.FFTSpectralBackend())
best = argmax([sum(abs2, ST.Monogenic.monogenic_components(st, field, j).bandpass) for j in 1:5])
comp = ST.Monogenic.monogenic_components(st, field, best)

# the monogenic amplitude is a smooth envelope: far less oscillatory than the band-pass field
env_osc = Statistics.std(diff(vec(comp.amplitude)))
bp_osc = Statistics.std(diff(vec(comp.bandpass)))
println("band-pass oscillation ", round(bp_osc; sigdigits=3),
        "  ≫  amplitude oscillation ", round(env_osc; sigdigits=3))

# orientation recovered continuously; for these rings it is the radial angle
amp = comp.amplitude
mask = amp .> 0.3 * maximum(amp)
ang_err = Float64[]
for k in CartesianIndices(field)
    mask[k] || continue
    radial = mod(atan(k[2] - cy, k[1] - cx), π)
    est = mod(atan(comp.riesz[2][k], comp.riesz[1][k]), π)
    push!(ang_err, min(abs(est - radial), π - abs(est - radial)))
end
println("median orientation error vs radial direction: ", round(Statistics.median(ang_err); digits=3), " rad")

# ── spherical monogenic scattering (needs `using NUFSHT`) ─────────────────────
println("\nspherical monogenic scattering: load `using NUFSHT` then ",
        "`spherical_monogenic_scattering(θ, φ, lmax, J)` — amplitude via a spin-0 identity.")

Test.@testset "monogenic" begin
    Test.@test maximum(abs.([partition[i] for i in CartesianIndices(partition) if i != CartesianIndex(1, 1)] .- 1)) < 1e-12
    Test.@test env_osc < bp_osc / 3                       # amplitude is a smooth envelope
    Test.@test Statistics.median(ang_err) < 0.1           # orientation tracks the radial direction
end

println("\n", "="^64, "\nDone.\n", "="^64)
