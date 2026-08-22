"""
    synthesis_and_inverse.jl

Reconstruction from the scattering representation, the three levels the literature uses:

  1. **Exact linear wavelet-frame inverse** — `iwavelet ∘ wavelet_transform` recovers the field
     to machine precision (the bank is a tight frame; no information is lost *before* the
     modulus).
  2. **Phase retrieval** — recover a field from the first-order moduli `|x ⋆ ψ_λ|` alone
     (Gerchberg–Saxton alternating projections); determined up to a global sign.
  3. **Gradient-descent synthesis** — from noise, descend `‖S(x̂) − S(x)‖²` so the *scattering
     coefficients* match (Bruna–Mallat microcanonical models). Yields a new sample with the same
     multiscale statistics, not the original field. Uses DifferentiationInterface + an AD backend.

Run with: `julia --project=. synthesis_and_inverse.jl`
"""

using ScatteringTransforms: ScatteringTransforms as ST
using SpectralBackends: SpectralBackends as SB
using FFTW: FFTW                                       # O(N log N) spectral fast path
using DifferentiationInterface: DifferentiationInterface as DI
using ADTypes: ADTypes
# using Mooncake: Mooncake                                        # the AD backend implementation
using Enzyme: Enzyme
using Statistics: Statistics
using Test: Test

println("="^64)
println("Reconstruction from the scattering transform")
println("="^64)

# A structured 1/f-like signal.
function spectral_signal(N, α)
    k = [0; 1:(N ÷ 2 - 1); (N ÷ 2); (N ÷ 2 - 1):-1:1]
    fhat = ((k .+ 1.0) .^ (α / 2)) .* exp.(im .* 2π .* rand(N)); fhat[1] = 0
    s = real(FFTW.ifft(fhat)); return s ./ Statistics.std(s)
end

N, J = 256, 6
sig = spectral_signal(N, -5/3)
# FFTW fast path for the (AD-free) exact inverse and phase retrieval.
st = ST.Scattering1D.ScatteringTransform1D(N, J; Q=1, max_order=2, spectral=SB.FFTSpectralBackend())

# ── 1. exact linear inverse ───────────────────────────────────────────────────
wt = ST.Inverse.wavelet_transform(st, sig)            # complex (pre-modulus) wavelet + low-pass fields
xr = ST.Inverse.iwavelet(st, wt)
inv_err = maximum(abs.(xr .- sig)) / maximum(abs.(sig))
println("\n1. exact linear inverse:   rel. err = ", inv_err, "  (machine precision)")

# ── 2. phase retrieval from first-order moduli ────────────────────────────────
moduli = [abs.(w) for w in wt.wavelet]
xhat = ST.Inverse.reconstruct_phase(st, moduli; iters=400, init=randn(N))
mod_hat = [abs.(w) for w in ST.Inverse.wavelet_transform(st, xhat).wavelet]
mod_relerr = sqrt(sum(sum(abs2, mh .- m) for (mh, m) in zip(mod_hat, moduli))) /
             sqrt(sum(sum(abs2, m) for m in moduli))
println("2. phase retrieval:        modulus rel. err = ", round(mod_relerr; sigdigits=3),
        "  (recovered up to a global sign)")

# ── 3. gradient-descent synthesis from scattering coefficients ────────────────
# Use the in-core direct-sum forward: it is differentiable by *every* AD backend with no special
# rules (FFTW-fast-path reverse-mode AD needs the backend's FFT rules — see the docs).
st_ad = ST.Scattering1D.ScatteringTransform1D(N, J; Q=1, max_order=2, spectral=SB.DirectSumSpectralBackend())
# Runtime activity is required, not optional: the cascade maps a closure over the filter bank, so
# every closure holds constant filters next to the active input and Enzyme's static activity analysis
# cannot tell them apart. Without it, `AutoEnzyme()` raises `EnzymeRuntimeActivityError`.
ad = ADTypes.AutoEnzyme(; mode=Enzyme.set_runtime_activity(Enzyme.Reverse))
res = ST.synthesize(st_ad, sig; backend=ad, init=randn(N), iters=300, lr=0.05)
cT, cS = ST.ScatteringCore.scattering(st_ad, sig), ST.ScatteringCore.scattering(st_ad, res.field)
s1_relerr = sqrt(sum(abs2, ST.Coefficients.first_order(cS) .- ST.Coefficients.first_order(cT))) /
            sqrt(sum(abs2, ST.Coefficients.first_order(cT)))
println("3. coefficient synthesis:  loss ", round(res.losses[1]; sigdigits=3), " → ",
        round(res.losses[end]; sigdigits=3), ";  S₁ rel. err = ", round(s1_relerr; sigdigits=3))
println("   (a matching sample, NOT the original field — the modulus discards local phase)")

Test.@testset "synthesis_and_inverse" begin
    Test.@test inv_err < 1e-8                       # exact frame inverse
    Test.@test mod_relerr < 0.2                     # phase retrieval converges
    Test.@test res.losses[end] < res.losses[1] / 5  # synthesis objective drops
    Test.@test s1_relerr < 0.1                       # coefficients match
end

println("\n", "="^64, "\nDone.\n", "="^64)
