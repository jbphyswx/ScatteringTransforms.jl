module ScatteringTransformsTests

using Test: Test
using Aqua: Aqua
using ExplicitImports: ExplicitImports as EI

using ScatteringTransforms: ScatteringTransforms
using ComputationalBackends: ComputationalBackends
using SpectralBackends: SpectralBackends
using FFTW: FFTW
using OhMyThreads: OhMyThreads
using Distributed: Distributed
using MPI: MPI
using NUFSHT: NUFSHT
# Both fast NUFFT backends load before any test file, so what an `Auto` spectral backend resolves
# to does not depend on include order.
using FINUFFT: FINUFFT
using NonuniformFFTs: NonuniformFFTs
using FlowTransformBindings: FlowTransformBindings as FTB
using FlowGeometries: FlowGeometries
using Statistics: Statistics
using Random: Random
using DifferentiationInterface: DifferentiationInterface as DI  # loaded so its extension is present for the explicit-imports audit below

Test.@testset "Aqua.jl quality tests" begin
    Aqua.test_all(ScatteringTransforms)
end

Test.@testset "Explicit imports (no implicit / no stale)" begin
    # The core module and every loaded extension. Each check throws on a violation.
    Test.@test EI.check_no_implicit_imports(ScatteringTransforms) === nothing
    # The core imports FlowTransformBindings only so its `__init__` runs before FastTransforms loads.
    Test.@test EI.check_no_stale_explicit_imports(ScatteringTransforms;
                                                  ignore = (:FlowTransformBindings,)) === nothing
    for extname in (
        :ScatteringTransformsCairoMakieExt,
        :ScatteringTransformsDifferentiationInterfaceExt,
        :ScatteringTransformsFFTWExt,
        :ScatteringTransformsFlowGeometriesExt,
        :ScatteringTransformsKernelAbstractionsExt,
        :ScatteringTransformsNUFSHTExt,
        :ScatteringTransformsOhMyThreadsExt,
        :ScatteringTransformsDistributedExt,
        :ScatteringTransformsMPIExt,
    )
        ext = Base.get_extension(ScatteringTransforms, extname)
        ext === nothing && continue
        Test.@test EI.check_no_implicit_imports(ext) === nothing
        Test.@test EI.check_no_stale_explicit_imports(ext) === nothing
    end
end

# JET optimization/abstract-interpretation audit of the hot paths (skipped on pre-release Julia,
# where JET refuses to run).
include("test_jet.jl")

include("test_enzyme.jl")
# Mooncake is out of the test set for now; Enzyme covers reverse-mode autodiff.
# include("test_mooncake.jl")

# Structural gates: spectral-execution counts, workspace scaling, backend honesty.
include("test_structure.jl")

Test.@testset "1D Morlet Wavelet Mathematical Properties" begin
    N = 512
    ψ = ScatteringTransforms.Filters.frequency_response(ScatteringTransforms.Filters.Morlet1D(N, 2; Q=1))
    # Analytic: exactly zero on the negative frequencies, bins N/2+2:N (bin N/2+1 is ±½).
    Test.@test all(iszero, ψ[(N ÷ 2 + 2):N])
end

Test.@testset "Littlewood–Paley sum of every bank: A(0) = 1, 0 < A ≤ 1" begin
    # For a real field `‖Wx‖² = (1/N) Σ_k A(k)|x̂(k)|²` with
    # `A = |φ̂|² + ½ Σ_λ (|ψ̂_λ(k)|² + |ψ̂_λ(−k)|²)` (Andén & Mallat 2014): `A ≤ 1` bounds the transform
    # by the identity, and `min A > 0` makes it invertible.
    FBk = ScatteringTransforms.FilterBanks
    Mono = ScatteringTransforms.Monogenic
    for T in (Float64, Float32), c in (FBk.StoredFilters(), FBk.ComputedFilters())
        banks = (FBk.build_filter_bank1d(T, 256, 5, c; Q = 1),
                 FBk.build_filter_bank1d(T, 255, 4, c; Q = 2),
                 FBk.build_filter_bank2d(T, (64, 64), 3, c; L = 8),
                 FBk.build_filter_bank2d(T, (63, 64), 3, c; L = 4),
                 FBk.build_filter_bank3d(T, (16, 16, 16), 2, c; n_orient = 6),
                 Mono.build_monogenic_bank(T, (32, 32), 3, c; Q = 1),
                 Mono.build_monogenic_bank(T, (33,), 4, c; Q = 2))
        for fb in banks
            A = FBk.littlewood_paley(fb)
            Test.@test abs(A[1] - 1) <= 8 * eps(T)
            Test.@test maximum(A) <= 1 + 8 * eps(T)
            Test.@test minimum(A) > 0
        end
    end
end

Test.@testset "1D Littlewood–Paley sum from the Morlet formula, Nyquist bin included" begin
    # Each bin carries the continuous sum at its frequency. On an even grid the Nyquist bin is
    # `ω = ±½` at once, and the analytic wavelets reach it from `+½`.
    FBk = ScatteringTransforms.FilterBanks
    for N in (256, 255), Q in (1, 2)
        J = 4
        fb = FBk.build_filter_bank1d(N, J; Q = Q)
        c = FBk.lp_scale_1d(Q)
        function Acont(ω)
            s = exp(-(2π * ω * 0.8 * 2.0^J)^2)                 # |φ̂|², σ = 0.8·2^J
            for n in 0:(J * Q - 1)
                ξ, f = 0.5 / 2.0^(n / Q), 2.0^(-1 / Q)
                σ = ξ * (1 - f) / (1 + f) / sqrt(log(2.0))    # crossing at r = √½
                g(w) = w < 0 ? 0.0 :
                       exp(-((w - ξ) / σ)^2 / 2) - exp(-(ξ / σ)^2 / 2) * exp(-(w / σ)^2 / 2)
                s += c^2 * (g(ω)^2 + g(-ω)^2) / 2
            end
            return s
        end
        fr = [k < (N + 1) ÷ 2 ? k / N : (k - N) / N for k in 0:(N - 1)]
        Test.@test maximum(abs, FBk.littlewood_paley(fb) .- Acont.(fr)) < 1e-14
    end
end

Test.@testset "A is the energy per frequency of the wavelet transform of a real field" begin
    # Parseval: `Σ_λ ‖x⋆ψ_λ‖² + ‖x⋆φ‖² = (1/N) Σ_k A(k)|x̂(k)|²`; for the monogenic bank the band-pass
    # and each Riesz component count separately.
    FBk = ScatteringTransforms.FilterBanks
    Mono = ScatteringTransforms.Monogenic
    Random.seed!(7)
    for (st, x) in ((ScatteringTransforms.Scattering1D.ScatteringTransform1D(128, 4; Q = 2), randn(128)),
                    (ScatteringTransforms.Scattering2D.ScatteringTransform2D((32, 31), 3; L = 4),
                     randn(32, 31)),
                    (ScatteringTransforms.Scattering3D.ScatteringTransform3D((12, 12, 12), 2),
                     randn(12, 12, 12)))
        wt = ScatteringTransforms.Inverse.wavelet_transform(st, x)
        E = sum(w -> sum(abs2, w), wt.wavelet) + sum(abs2, wt.lowpass)
        A = FBk.littlewood_paley(st.filter_bank)
        Test.@test E ≈ sum(A .* abs2.(FFTW.fft(x))) / length(x) rtol = 1e-12
    end
    for (dims, J) in (((64,), 4), ((32, 32), 3), ((12, 12, 12), 2))
        st = Mono.MonogenicScattering(dims, J; Q = 1, max_order = 1)
        x = randn(dims...)
        E = sum(abs2, real.(FFTW.ifft(FFTW.fft(x) .* st.filter_bank.averaging)))
        for j in 1:J
            m = Mono.monogenic_components(st, x, j)
            E += sum(abs2, m.bandpass) + sum(r -> sum(abs2, r), m.riesz)
        end
        A = FBk.littlewood_paley(st.filter_bank)
        Test.@test E ≈ sum(A .* abs2.(FFTW.fft(x))) / length(x) rtol = 1e-12
    end
end

Test.@testset "The Littlewood–Paley constant makes A ≤ 1 tight inside the band" begin
    # Between its finest and coarsest wavelets a deep bank reaches the maximum of the bank continued
    # over every scale, which the constant sets to 1. Tolerances are the grid's sampling of that
    # maximum: ≥ 250 bins per octave in 1D, ≥ 10 per orientation period in 2D.
    FBk = ScatteringTransforms.FilterBanks
    for (fb, tol) in ((FBk.build_filter_bank1d(4096, 9; Q = 1), 1e-3),
                      (FBk.build_filter_bank1d(4096, 8; Q = 2), 1e-3),
                      (FBk.build_filter_bank2d((256, 256), 5; L = 8), 1e-2),
                      (FBk.build_filter_bank2d((256, 256), 5; L = 4), 1e-2))
        A = FBk.littlewood_paley(fb)
        Test.@test 1 - tol < maximum(A[2:end]) <= 1
    end
end

Test.@testset "S1 of a scale does not depend on how many scales the bank has" begin
    # The constant depends on the wavelet design only, so wavelet `j` is the same function in a
    # bank of any depth; undecimated, its coefficient is the same arithmetic.
    Random.seed!(11)
    C = ScatteringTransforms.Coefficients
    x = randn(256)
    S1(J) = C.first_order(ScatteringTransforms.Scattering1D.ScatteringTransform1D(256, J; Q = 1,
                                                                                max_order = 1)(x))
    Test.@test S1(4) == S1(6)[1:4]
    x2 = randn(64, 64)
    S12(J) = C.first_order(ScatteringTransforms.Scattering2D.ScatteringTransform2D((64, 64), J; L = 4,
                                                                                 max_order = 1)(x2))
    Test.@test S12(2) == S12(4)[1:8]
end

Test.@testset "1D Filter Bank with Q > 1 Fractional Center Frequencies" begin
    N = 256
    J = 3
    Q = 4
    bank = ScatteringTransforms.FilterBanks.build_filter_bank1d(N, J; Q=Q)
    Test.@test length(bank.wavelets) == J * Q

    center_freqs = [m.center_freq for m in bank.meta]
    Test.@test all(center_freqs[i] > center_freqs[i+1] for i in 1:length(center_freqs)-1)

    # center_freq = 0.5 * 2^(-j_eff)
    for (i, m) in enumerate(bank.meta)
        expected_freq = 0.5 / (2.0^m.j_eff)
        Test.@test isapprox(m.center_freq, expected_freq, rtol=1e-10)
    end
end

Test.@testset "Path graph: 2D second order is scale-increasing" begin
    # A second-order path pairs a wavelet with every wavelet of a strictly coarser scale, at every
    # orientation, and `S2` is zero off those pairs.
    PG = ScatteringTransforms.PathGraph
    Ny, Nx = 64, 64
    J = 3
    L = 4
    st = ScatteringTransforms.Scattering2D.ScatteringTransform2D((Ny, Nx), J; L=L, max_order=2)
    meta = st.filter_bank.meta
    tree = st.tree

    n_paths = 0
    for p in PG.order_range(tree, 2)
        idx = PG.path_indices(tree, p)
        i1, i2 = idx[1], idx[2]
        Test.@test meta[i2].scale > meta[i1].scale   # strictly coarser only
        n_paths += 1
    end
    # binom(J,2) scale pairs, each with L*L orientation pairs.
    Test.@test n_paths == (J * (J - 1) ÷ 2) * L * L

    image = randn(Ny, Nx)
    coeffs = st(image)
    S2 = ScatteringTransforms.Coefficients.second_order(coeffs)
    nw = J * L
    for i1 in 1:nw, i2 in 1:nw
        if meta[i2].scale <= meta[i1].scale
            Test.@test S2[i1, i2] == 0
        end
    end
    Test.@test any(S2[i1, i2] > 0 for i1 in 1:nw for i2 in 1:nw if meta[i2].scale > meta[i1].scale)
end


Test.@testset "1D Scattering Transform Tests" begin
    N = 256
    J = 4

    st = ScatteringTransforms.Scattering1D.ScatteringTransform1D(N, J; Q=1, max_order=2)
    signal = randn(N)

    coeffs = st(signal)
    Test.@test isapprox(ScatteringTransforms.Coefficients.zeroth_order(coeffs), Statistics.mean(signal), atol=1e-10)

    S1 = ScatteringTransforms.Coefficients.first_order(coeffs)
    Test.@test length(S1) == J
    Test.@test all(S1 .>= 0)

    S2 = ScatteringTransforms.Coefficients.second_order(coeffs)
    Test.@test size(S2) == (J, J)

    # In place, into a container whose S0 is a 1-element array.
    nw = length(st.filter_bank.wavelets)
    cm = ScatteringTransforms.Coefficients.ScatteringCoefficients1D(Vector{Float64}(undef, nw), zeros(nw, nw); S0=[0.0])
    ScatteringTransforms.Scattering1D.scattering_transform!(cm, st, signal)
    Test.@test ScatteringTransforms.Coefficients.zeroth_order(cm) ≈ Statistics.mean(signal)
end

Test.@testset "2D Filter Tests" begin
    # Analytic on the half-plane: zero where `k∥ = kx cos θ + ky sin θ < 0`.
    Ny, Nx = 64, 64
    resp = ScatteringTransforms.Filters.frequency_response(
        ScatteringTransforms.Filters.Morlet2D((Ny, Nx), 2, π/4; L=8))
    freqs_x = 2π .* collect(FFTW.fftfreq(Nx))
    freqs_y = 2π .* collect(FFTW.fftfreq(Ny))
    ct, st = cos(π/4), sin(π/4)
    any_negative_halfplane = false
    for ix in 1:Nx, iy in 1:Ny
        kx = freqs_x[ix]
        ky = freqs_y[iy]
        kxr = kx * ct + ky * st
        if kxr < -1e-5 && abs(resp[iy, ix]) > 1e-10
            any_negative_halfplane = true
        end
    end
    Test.@test !any_negative_halfplane
end

Test.@testset "2D Morlet equals the FFT of its spatial form" begin
    # Built in space: a Gabor with widths σ∥ along θ and σ⊥ = σ∥·L/4 across it, periodized over
    # neighbouring images, minus a Gaussian scaled to zero mean, divided by the Gaussian's integral
    # 2πσ∥σ⊥. Its DFT is the analytic envelope sampled on the grid; at j = 2 the frequency aliases
    # sit e⁻⁵⁰ below it. Compared on k∥ ≥ 0, where the wavelet here is defined as analytic.
    Ny, Nx = 64, 64
    j = 2
    for L in (4, 8), l in (0, 1, 3)
        θ = π * l / L
        m = ScatteringTransforms.Filters.Morlet2D((Ny, Nx), j, θ; L = L)
        σpar, σperp, k0 = 0.8 * 2.0^j, 0.8 * 2.0^j * L / 4, 3π / (4 * 2.0^j)
        g(k) = [sum(exp(-(((x + ex * Nx) * cos(θ) + (y + ey * Ny) * sin(θ))^2 / (2σpar^2)) -
                        ((-(x + ex * Nx) * sin(θ) + (y + ey * Ny) * cos(θ))^2 / (2σperp^2)) +
                        im * k * ((x + ex * Nx) * cos(θ) + (y + ey * Ny) * sin(θ)))
                    for ex in -2:2, ey in -2:2) for y in 0:(Ny - 1), x in 0:(Nx - 1)]
        gξ, g0 = g(k0), g(0.0)
        ref = FFTW.fft((gξ .- (sum(gξ) / sum(g0)) .* g0) ./ (2π * σpar * σperp))
        ψ = ScatteringTransforms.Filters.frequency_response(m)
        fx, fy = FFTW.fftfreq(Nx), FFTW.fftfreq(Ny)
        kpar = [2π * (fx[ix] * cos(θ) + fy[iy] * sin(θ)) for iy in 1:Ny, ix in 1:Nx]
        half = kpar .>= 0
        Test.@test maximum(abs, ψ[half] .- ref[half]) < 1e-12
    end
end

Test.@testset "2D wavelets resolve orientation (S1 of a sinusoid)" begin
    # For a sinusoid on a grid wavevector, |x ⋆ ψ| is the constant |ψ̂(k)|·A/2, so S1 is |ψ̂(k)|
    # up to the bank's scale. At j = 2 the wavelet at θ = 0 against the one at θ = π/4 gives
    # ψ̂₀(k)/ψ̂₂(k) ≈ 43, the factor exp(((k₀ sin π/4)·σ⊥)²/2) doing most of it.
    N, J, L = 64, 3, 8
    st = ScatteringTransforms.Scattering2D.ScatteringTransform2D((N, N), J; L = L, max_order = 1)
    x = [cos(2π * 6 * (ix - 1) / N) for iy in 1:N, ix in 1:N]     # |k| = 6·2π/64 = k₀ at j = 2
    S1 = reshape(ScatteringTransforms.Coefficients.first_order(st(x)), L, J)
    Test.@test S1[1, 3] / S1[3, 3] > 10
end

Test.@testset "2D Wavelet Orientation and Scale Selectivity" begin
    Ny, Nx = 64, 64
    J = 3
    L = 8
    st = ScatteringTransforms.Scattering2D.ScatteringTransform2D((Ny, Nx), J; L=L, max_order=1)
    
    # k0 for j=1 is 3π / (4 * 2^1) = 3π/8 ≈ 1.178.
    k0 = 3π / 8
    theta_wave = π/4 # orient index 2 (1-based: index 3 since orient=0, 1, 2)
    
    # Generate grid
    X = reshape(range(0, 2π, length=Nx+1)[1:Nx], 1, Nx)
    Y = reshape(range(0, 2π, length=Ny+1)[1:Ny], Ny, 1)
    k_wave = round(k0 * Nx / 2π)
    
    plane_wave = cos.(k_wave .* (X .* cos(theta_wave) .+ Y .* sin(theta_wave)))
    
    coeffs = st(plane_wave)
    S1 = ScatteringTransforms.Coefficients.first_order(coeffs)
    S1_matrix = reshape(S1, L, J) # L orientations x J scales
    
    # We expect the peak to be at:
    # Scale index 2 (j=1)
    # Orientation index 3 (θ = 2*π/8 = π/4)
    peak_idx = argmax(S1_matrix)
    orient_p, scale_p = peak_idx[1], peak_idx[2]
    
    Test.@test scale_p == 2
    Test.@test orient_p == 3
end

Test.@testset "3D volumetric scattering transform" begin
    N = (16, 16, 16)
    J = 2
    n_orient = 6
    st = ScatteringTransforms.Scattering3D.ScatteringTransform3D(N, J; n_orient=n_orient, max_order=2)
    vol = randn(N...)
    coeffs = st(vol)

    Test.@test isapprox(ScatteringTransforms.Coefficients.zeroth_order(coeffs), Statistics.mean(vol); atol=1e-10)
    S1 = ScatteringTransforms.Coefficients.first_order(coeffs)
    Test.@test length(S1) == J * n_orient
    Test.@test all(S1 .>= 0)
    S2 = ScatteringTransforms.Coefficients.second_order(coeffs)
    Test.@test size(S2) == (J * n_orient, J * n_orient)
    Test.@test Statistics.maximum(S2) > 0

    # second-order paths are scale-increasing only
    meta = st.filter_bank.meta
    PG = ScatteringTransforms.PathGraph
    for p in PG.order_range(st.tree, 2)
        i1, i2 = PG.path_indices(st.tree, p)
        Test.@test meta[i2].scale > meta[i1].scale
    end

    # in-core direct sum matches the FFTW fast path in 3D
    st_d = ScatteringTransforms.Scattering3D.ScatteringTransform3D(N, J; n_orient=n_orient, max_order=2, transform=SpectralBackends.DirectSumSpectralBackend())
    st_f = ScatteringTransforms.Scattering3D.ScatteringTransform3D(N, J; n_orient=n_orient, max_order=2, transform=SpectralBackends.FFTSpectralBackend())
    Test.@test isapprox(ScatteringTransforms.Coefficients.first_order(st_d(vol)),
                        ScatteringTransforms.Coefficients.first_order(st_f(vol)); rtol=1e-6)
    Test.@test isapprox(ScatteringTransforms.Coefficients.second_order(st_d(vol)),
                        ScatteringTransforms.Coefficients.second_order(st_f(vol)); rtol=1e-6)
end

Test.@testset "Spherical scattering (NUFSHT, smooth difference-of-Gaussians bands)" begin
    lmax = 12
    J = 3
    M = 1000                      # ≳ (lmax+1)² = 169 so the least-squares analysis is well-determined
    gr = (sqrt(5) - 1) / 2        # quasi-uniform Fibonacci-sphere points
    θ = [acos(1 - 2 * (k - 0.5) / M) for k in 1:M]
    φ = [2π * mod(k * gr, 1) for k in 1:M]
    st = ScatteringTransforms.spherical_scattering(θ, φ, lmax, J)
    field = [cos(4 * θ[k]) + 0.6 * cos(3 * φ[k]) * sin(2 * θ[k]) - 0.4 * cos(2 * θ[k]) for k in 1:M]  # band-limited
    res = st(field)

    Test.@test res.S0 ≈ Statistics.mean(field)
    Test.@test length(res.S1) == J
    Test.@test all(res.S1 .>= 0)
    Test.@test size(res.S2) == (J, J)
    # second order only for strictly coarser j2 < j1 ⇒ diagonal + upper triangle are exactly zero
    for j1 in 1:J, j2 in j1:J
        Test.@test res.S2[j1, j2] == 0
    end
    Test.@test any(res.S2[j1, j2] > 0 for j1 in 1:J for j2 in 1:(j1 - 1))
end

Test.@testset "Spherical monogenic scattering (Riesz amplitude)" begin
    lmax = 12
    J = 3
    M = 1200                      # ≳ (lmax+1)² = 169 so the least-squares analysis is well-determined
    # Deterministic, well-distributed Fibonacci-sphere points.
    gr = (sqrt(5) - 1) / 2
    θ = [acos(1 - 2 * (k - 0.5) / M) for k in 1:M]
    φ = [2π * mod(k * gr, 1) for k in 1:M]
    st = ScatteringTransforms.spherical_monogenic_scattering(θ, φ, lmax, J)
    res = st([cos(4 * θ[k]) + 0.5 * sin(3 * θ[k]) * cos(2 * φ[k]) for k in 1:M])  # band-limited
    Test.@test length(res.S1) == J
    Test.@test all(res.S1 .>= 0)              # monogenic amplitude is non-negative
    Test.@test size(res.S2) == (J, J)
    for j1 in 1:J, j2 in j1:J                 # second order only for strictly coarser j2 < j1
        Test.@test res.S2[j1, j2] == 0
    end
    Test.@test any(res.S2[j1, j2] > 0 for j1 in 1:J for j2 in 1:(j1 - 1))

    # Rotation covariance about z: a smooth field and its φ-rotation give matching coefficients.
    α = 0.7
    f = cos.(θ) .^ 2 .- 1/3 .+ 0.5 .* sin.(θ) .* cos.(φ)
    fα = cos.(θ) .^ 2 .- 1/3 .+ 0.5 .* sin.(θ) .* cos.(φ .+ α)
    r0 = st(f); rα = st(fα)
    Test.@test maximum(abs.(r0.S1 .- rα.S1)) / maximum(abs.(r0.S1)) < 0.05

    # Pointwise orientation/phase on S² (spin-1 Riesz vector) is validated in
    # test_spherical_monogenic_components.jl.
end

Test.@testset "Translation invariance (approximate)" begin
    N = 256
    J = 4

    st = ScatteringTransforms.Scattering1D.ScatteringTransform1D(N, J; Q=1, max_order=1)

    x = range(0, 2π, length=N+1)[1:N]
    signal = sin.(3x)
    signal_shifted = circshift(signal, 10)

    # `|x ⋆ ψ|` of a grid sinusoid is constant, so a circular shift leaves S1 unchanged.
    S1_1 = ScatteringTransforms.Coefficients.first_order(st(signal))
    S1_2 = ScatteringTransforms.Coefficients.first_order(st(signal_shifted))
    Test.@test S1_2 ≈ S1_1 rtol = 1e-12
end

Test.@testset "Batched transforms reuse the plan and match per-signal results" begin
    st2 = ScatteringTransforms.Scattering2D.ScatteringTransform2D((32, 32), 3; L=4, max_order=2)
    X2 = randn(32, 32, 4)
    B2 = ScatteringTransforms.scattering_batch(st2, X2)
    for b in 1:4
        Test.@test isapprox(B2[:, b], ScatteringTransforms.Coefficients.flatten2d(st2(view(X2, :, :, b))); rtol=1e-10)
    end
end

Test.@testset "Periodized cascade: bit-exact at oversampling ≥ J, lossy below" begin
    N = 256
    J = 5
    signal = randn(N)
    build(ov) = ScatteringTransforms.Scattering1D.ScatteringTransform1D(N, J; Q=1, max_order=2,
                                                                        oversampling=ov)
    exact = build(J)(signal)
    C = ScatteringTransforms.Coefficients

    # Nothing decimated: every alias block is the whole spectrum, so the products are the same
    # floating-point operations in the same order.
    c_off = build(J + 3)(signal)
    Test.@test C.first_order(c_off) == C.first_order(exact)
    Test.@test C.second_order(c_off) == C.second_order(exact)
    Test.@test C.zeroth_order(c_off) == C.zeroth_order(exact)

    # S0 is the plain field mean and never involves a wavelet.
    Test.@test C.zeroth_order(build(0)(signal)) ≈ C.zeroth_order(exact)

    # A modulus has no compact spectrum, so decimating it always aliases by a signal-dependent
    # amount. Asserted: zero loss at `oversampling = J`, and a nonzero loss at 0.
    S1e, S2e = C.first_order(exact), C.second_order(exact)
    err(ov, get, ref) = let c = build(ov)(signal)
        sum(abs2, get(c) .- ref) / sum(abs2, ref)
    end
    Test.@test err(J, C.second_order, S2e) == 0
    Test.@test err(0, C.second_order, S2e) > 0

    # Order 1 is decimated too: its coefficient is the mean of |U₁| over the grid its band needs.
    Test.@test err(J, C.first_order, S1e) == 0
    Test.@test 0 < err(0, C.first_order, S1e) < 1e-2
end

Test.@testset "Distributed batch (single process) matches serial" begin
    # DistributedBackend distributes batch columns across workers; each worker rebuilds the
    # transform from a spec (plans aren't serializable) and runs the inner backend. With no
    # added workers, pmap runs locally and must equal the serial batch.
    N = 96
    J = 4
    st = ScatteringTransforms.Scattering1D.ScatteringTransform1D(N, J; Q=1, max_order=2)
    X = randn(N, 6)
    Test.@test ScatteringTransforms.scattering_batch(st, X; execution = ComputationalBackends.DistributedBackend()) ≈
               ScatteringTransforms.scattering_batch(st, X)

    st2 = ScatteringTransforms.Scattering2D.ScatteringTransform2D((24, 24), 3; L=4, max_order=2)
    X2 = randn(24, 24, 5)
    Test.@test ScatteringTransforms.scattering_batch(st2, X2; execution = ComputationalBackends.DistributedBackend()) ≈
               ScatteringTransforms.scattering_batch(st2, X2)
end

Test.@testset "MPI batch (single rank) matches serial" begin
    # SPMD: each rank computes its column block and Allgatherv combines them. With a single rank
    # (no mpiexec needed) the result must equal the serial batch.
    MPI.Initialized() || MPI.Init()
    N = 96
    J = 4
    st = ScatteringTransforms.Scattering1D.ScatteringTransform1D(N, J; Q=1, max_order=2)
    X = randn(N, 6)
    Test.@test ScatteringTransforms.scattering_batch(st, X; execution = ComputationalBackends.MPIBackend()) ≈
               ScatteringTransforms.scattering_batch(st, X)
    st2 = ScatteringTransforms.Scattering2D.ScatteringTransform2D((24, 24), 3; L=4, max_order=2)
    X2 = randn(24, 24, 5)
    Test.@test ScatteringTransforms.scattering_batch(st2, X2; execution = ComputationalBackends.MPIBackend()) ≈
               ScatteringTransforms.scattering_batch(st2, X2)
end

Test.@testset "Spectral plans: in-core direct sum matches FFTW fast path" begin
    # Direct summation and FFTW compute the same DFT.
    N = 128
    J = 4
    signal = randn(N)
    st_d = ScatteringTransforms.Scattering1D.ScatteringTransform1D(N, J; Q=1, max_order=2, transform=SpectralBackends.DirectSumSpectralBackend())
    st_f = ScatteringTransforms.Scattering1D.ScatteringTransform1D(N, J; Q=1, max_order=2, transform=SpectralBackends.FFTSpectralBackend())
    cd, cf = st_d(signal), st_f(signal)
    Test.@test isapprox(ScatteringTransforms.Coefficients.zeroth_order(cd), ScatteringTransforms.Coefficients.zeroth_order(cf); rtol=1e-6)
    Test.@test isapprox(ScatteringTransforms.Coefficients.first_order(cd), ScatteringTransforms.Coefficients.first_order(cf); rtol=1e-6)
    Test.@test isapprox(ScatteringTransforms.Coefficients.second_order(cd), ScatteringTransforms.Coefficients.second_order(cf); rtol=1e-6)

    img = randn(32, 32)
    s2d = ScatteringTransforms.Scattering2D.ScatteringTransform2D((32, 32), 3; L=4, max_order=2, transform=SpectralBackends.DirectSumSpectralBackend())
    s2f = ScatteringTransforms.Scattering2D.ScatteringTransform2D((32, 32), 3; L=4, max_order=2, transform=SpectralBackends.FFTSpectralBackend())
    Test.@test isapprox(ScatteringTransforms.Coefficients.first_order(s2d(img)),
                        ScatteringTransforms.Coefficients.first_order(s2f(img)); rtol=1e-6)
    Test.@test isapprox(ScatteringTransforms.Coefficients.second_order(s2d(img)),
                        ScatteringTransforms.Coefficients.second_order(s2f(img)); rtol=1e-6)
end

Test.@testset "1D localized field: mean equals averaged coefficient" begin
    # The localized (Mallat) field S_p x = (|U_p x| ⋆ φ_J) ↓ s. With s = 1 (no decimation) and
    # φ̂(0) = 1, the spatial mean of each path's field equals that path's globally-averaged
    # coefficient.
    PG = ScatteringTransforms.PathGraph
    N = 256
    J = 4
    st = ScatteringTransforms.Scattering1D.ScatteringTransform1D(N, J; Q=1, max_order=2)
    signal = randn(N)
    coeffs = st(signal)
    sf = ScatteringTransforms.ScatteringFields.scattering_field(st, signal; subsample=1)
    tree = st.tree

    root = first(PG.order_range(tree, 0))
    Test.@test isapprox(Statistics.mean(ScatteringTransforms.ScatteringFields.path_field(sf, root)),
                        ScatteringTransforms.Coefficients.zeroth_order(coeffs); atol=1e-10)

    S1 = ScatteringTransforms.Coefficients.first_order(coeffs)
    for p in PG.order_range(tree, 1)
        j = PG.path_indices(tree, p)[1]
        Test.@test isapprox(Statistics.mean(ScatteringTransforms.ScatteringFields.path_field(sf, p)), S1[j]; atol=1e-8)
    end

    S2 = ScatteringTransforms.Coefficients.second_order(coeffs)
    for p in PG.order_range(tree, 2)
        idx = PG.path_indices(tree, p)
        j1, j2 = idx[1], idx[2]
        Test.@test isapprox(Statistics.mean(ScatteringTransforms.ScatteringFields.path_field(sf, p)), S2[j1, j2]; atol=1e-8)
    end

    # Decimated, the mean is still the coefficient: φ_J vanishes on the subsampling lattice.
    sf8 = ScatteringTransforms.ScatteringFields.scattering_field(st, signal; subsample=8)
    Test.@test size(sf8.data, 1) == N ÷ 8
    for p in PG.order_range(tree, 1)
        j = PG.path_indices(tree, p)[1]
        Test.@test isapprox(Statistics.mean(ScatteringTransforms.ScatteringFields.path_field(sf8, p)),
                            S1[j]; rtol=1e-12)
    end
    for p in PG.order_range(tree, 2)
        idx = PG.path_indices(tree, p)
        Test.@test isapprox(Statistics.mean(ScatteringTransforms.ScatteringFields.path_field(sf8, p)),
                            S2[idx[1], idx[2]]; rtol=1e-12)
    end
end

Test.@testset "2D localized field: mean equals averaged coefficient" begin
    PG = ScatteringTransforms.PathGraph
    Ny, Nx = 64, 64
    J = 3
    L = 4
    st = ScatteringTransforms.Scattering2D.ScatteringTransform2D((Ny, Nx), J; L=L, max_order=2)
    image = randn(Ny, Nx)
    coeffs = st(image)
    sf = ScatteringTransforms.ScatteringFields.scattering_field(st, image; subsample=1)
    tree = st.tree

    root = first(PG.order_range(tree, 0))
    Test.@test isapprox(Statistics.mean(ScatteringTransforms.ScatteringFields.path_field(sf, root)),
                        ScatteringTransforms.Coefficients.zeroth_order(coeffs); atol=1e-10)
    S1 = ScatteringTransforms.Coefficients.first_order(coeffs)
    for p in PG.order_range(tree, 1)
        j = PG.path_indices(tree, p)[1]
        Test.@test isapprox(Statistics.mean(ScatteringTransforms.ScatteringFields.path_field(sf, p)), S1[j]; atol=1e-8)
    end
    S2 = ScatteringTransforms.Coefficients.second_order(coeffs)
    for p in PG.order_range(tree, 2)
        idx = PG.path_indices(tree, p)
        j1, j2 = idx[1], idx[2]
        Test.@test isapprox(Statistics.mean(ScatteringTransforms.ScatteringFields.path_field(sf, p)), S2[j1, j2]; atol=1e-8)
    end

    sf2 = ScatteringTransforms.ScatteringFields.scattering_field(st, image; subsample=2)
    Test.@test size(sf2.data) == (Ny ÷ 2, Nx ÷ 2, PG.npaths(tree))
    for p in PG.order_range(tree, 1)
        j = PG.path_indices(tree, p)[1]
        Test.@test isapprox(Statistics.mean(ScatteringTransforms.ScatteringFields.path_field(sf2, p)),
                            S1[j]; rtol=1e-12)
    end
    for p in PG.order_range(tree, 2)
        idx = PG.path_indices(tree, p)
        Test.@test isapprox(Statistics.mean(ScatteringTransforms.ScatteringFields.path_field(sf2, p)),
                            S2[idx[1], idx[2]]; rtol=1e-12)
    end
end

Test.@testset "Computed filter banks match stored ones everywhere" begin
    # A computed bank evaluates each wavelet on demand into one shared scratch array, so each task
    # needs its own scratch.
    Random.seed!(41)
    FB = SpectralBackends.FFTSpectralBackend()
    CBk = ComputationalBackends
    FBk = ScatteringTransforms.FilterBanks
    flat(c) = vcat(c.S0, vec(c.S1), vec(c.S2))

    for (mk, x, run!, mkc) in (
            ((α, fs) -> ScatteringTransforms.Scattering1D.ScatteringTransform1D(128, 4; Q=1,
                            max_order=2, oversampling=α, transform=FB, filters=fs),
             randn(128), ScatteringTransforms.Scattering1D.scattering_transform!,
             st -> ScatteringTransforms.batch_coeffs(st, Float64)),
            ((α, fs) -> ScatteringTransforms.Scattering2D.ScatteringTransform2D((32, 32), 3; L=4,
                            max_order=2, oversampling=α, transform=FB, filters=fs),
             randn(32, 32), ScatteringTransforms.Scattering2D.scattering_transform2d!,
             st -> ScatteringTransforms.batch_coeffs(st, Float64)),
            ((α, fs) -> ScatteringTransforms.Scattering3D.ScatteringTransform3D((16, 16, 16), 2;
                            n_orient=4, max_order=2, oversampling=α, transform=FB, filters=fs),
             randn(16, 16, 16), ScatteringTransforms.Scattering3D.scattering_transform3d!,
             st -> ScatteringTransforms.batch_coeffs(st, Float64)))
        for α in (4, 1, 0)
            stored, computed = mk(α, FBk.StoredFilters()), mk(α, FBk.ComputedFilters())
            Test.@test ScatteringTransforms.FilterBanks.iscomputed(computed.filter_bank)
            Test.@test !ScatteringTransforms.FilterBanks.iscomputed(stored.filter_bank)
            # Same arithmetic in the same order.
            Test.@test flat(computed(x)) == flat(stored(x))

            # Each task gets its own bank, so its filter list must point at *that* bank's scratch.
            ser, thr = mkc(computed), mkc(computed)
            run!(ser, computed, x; execution = CBk.SerialBackend())
            for _ in 1:6
                run!(thr, computed, x; execution = CBk.ThreadedBackend())
                Test.@test flat(thr) == flat(ser)
            end
        end
    end

    # A worker rebuilding from a spec gets a computed bank too.
    st = ScatteringTransforms.Scattering2D.ScatteringTransform2D((32, 32), 3; L=4, max_order=2,
             oversampling=1, transform=FB, filters=FBk.ComputedFilters())
    rebuilt = ScatteringTransforms.rebuild_transform(ScatteringTransforms.transform_spec(st))
    Test.@test ScatteringTransforms.FilterBanks.iscomputed(rebuilt.filter_bank)
    x2 = randn(32, 32)
    Test.@test flat(rebuilt(x2)) == flat(st(x2))

    # And the batched cascade, whose filters carry a trailing singleton over the shared scratch.
    Xb = randn(32, 32, 3)
    Bb = ScatteringTransforms.scattering_batch(st, Xb)
    for b in 1:3
        Test.@test view(Bb, :, b) == ScatteringTransforms.Coefficients.flatten2d(st(view(Xb, :, :, b)))
    end
end

Test.@testset "Complex input on every gridded surface" begin
    # A modulus is real, so `S1`/`S2` of a complex field are real and equal the real field's
    # exactly when the imaginary part is zero. Only `S0 = ⟨x⟩` is complex, and the flat layout and
    # the order-0 localized field carry it.
    Random.seed!(31)
    C = ScatteringTransforms.Coefficients
    SF = ScatteringTransforms.ScatteringFields

    for (st, x, flat) in (
            (ScatteringTransforms.Scattering1D.ScatteringTransform1D(64, 3; Q=1, max_order=2),
             randn(64), C.flatten1d),
            (ScatteringTransforms.Scattering2D.ScatteringTransform2D((32, 32), 3; L=4, max_order=2),
             randn(32, 32), C.flatten2d),
            (ScatteringTransforms.Scattering3D.ScatteringTransform3D((8, 8, 8), 2; n_orient=4,
                                                                     max_order=2),
             randn(8, 8, 8), C.flatten2d))
        cr, cc = st(x), st(complex(x))
        Test.@test C.first_order(cc) == C.first_order(cr)
        Test.@test C.second_order(cc) == C.second_order(cr)
        Test.@test eltype(C.first_order(cc)) <: Real
        Test.@test C.zeroth_order(cc) isa Complex

        # A real field stays real.
        Test.@test eltype(flat(cr)) <: Real
        fc = flat(cc)
        Test.@test eltype(fc) <: Complex
        Test.@test fc[C.flat_row_s0()] ≈ C.zeroth_order(cc)
        Test.@test real(fc) ≈ flat(cr)

        # A genuinely complex field: the AD forward must agree with the in-place one.
        z = complex.(x, randn(size(x)...))
        Test.@test flat(ScatteringTransforms.ScatteringCore.scattering(st, z)) ≈ flat(st(z))
    end

    # Batched columns must match the per-slice transform, complex `S0` included.
    st2 = ScatteringTransforms.Scattering2D.ScatteringTransform2D((32, 32), 3; L=4, max_order=2)
    Z = randn(ComplexF64, 32, 32, 3)
    B = ScatteringTransforms.scattering_batch(st2, Z)
    for b in 1:3
        Test.@test view(B, :, b) ≈ C.flatten2d(st2(view(Z, :, :, b)))
    end

    # The localized field carries the complex order-0 field, and its mean is still `S0`.
    z2 = randn(ComplexF64, 32, 32)
    sf = SF.scattering_field(st2, z2; subsample=4)
    root = first(ScatteringTransforms.PathGraph.order_range(st2.tree, 0))
    Test.@test Statistics.mean(SF.path_field(sf, root)) ≈ ScatteringTransforms.Coefficients.zeroth_order(st2(z2))
    Test.@test eltype(SF.scattering_field(st2, real(z2); subsample=4).data) <: Real
end

Test.@testset "Every bank's averaging filter is the Gaussian φ_J" begin
    FBk = ScatteringTransforms.FilterBanks
    N, J = 256, 4
    φ = ScatteringTransforms.Filters.gaussian_lowpass(Float64, (N,), J)
    for c in (FBk.StoredFilters(), FBk.ComputedFilters())
        Test.@test FBk.build_filter_bank1d(Float64, N, J, c; Q = 2).averaging == φ
        Test.@test FBk.build_filter_bank2d(Float64, (32, 48), 3, c; L = 4).averaging ==
                   ScatteringTransforms.Filters.gaussian_lowpass(Float64, (32, 48), 3)
        Test.@test FBk.build_filter_bank3d(Float64, (8, 8, 8), 2, c).averaging ==
                   ScatteringTransforms.Filters.gaussian_lowpass(Float64, (8, 8, 8), 2)
        Test.@test ScatteringTransforms.Monogenic.build_monogenic_bank(Float64, (16, 16), 3, c).averaging ==
                   ScatteringTransforms.Filters.gaussian_lowpass(Float64, (16, 16), 3)
    end
    # φ̂(k) = exp(-k²σ²/2) at angular frequency k = 2πm/N, σ = 0.8·2^J.
    Test.@test φ ≈ [exp(-(2π * (m < N ÷ 2 ? m : m - N) / N * 0.8 * 2^J)^2 / 2) for m in 0:(N - 1)] rtol = 1e-14
    Test.@test φ[1] == 1.0
    # Every frequency that folds onto DC under the default 2^(J-1) subsampling is negligible.
    s = 1 << (J - 1)
    Test.@test maximum(abs, φ[(N ÷ s) .* (1:(s - 1)) .+ 1]) < 1e-20
end

Test.@testset "Reduced descriptors: sparsity, shape (anisotropy), normalize, log" begin
    Ny, Nx = 64, 64
    J = 3
    L = 4
    st = ScatteringTransforms.Scattering2D.ScatteringTransform2D((Ny, Nx), J; L=L, max_order=2)

    # Anisotropic field (oriented stripes) -> nonzero shape; isotropic noise -> ~0 shape.
    xs = range(0, 8π, length=Nx)'
    aniso = repeat(sin.(xs), Ny, 1) .+ 0.01 .* randn(Ny, Nx)
    ca = st(aniso)
    ra = ScatteringTransforms.Scattering2D.compute_shape_sparsity(ScatteringTransforms.Coefficients.first_order(ca),
            ScatteringTransforms.Coefficients.second_order(ca), st.filter_bank.meta)
    Test.@test any(ra.sparsity .> 0)
    Test.@test maximum(abs, ra.shape) > 0.05      # clear anisotropy signal

    iso = randn(Ny, Nx)
    ci = st(iso)
    ri = ScatteringTransforms.Scattering2D.compute_shape_sparsity(ScatteringTransforms.Coefficients.first_order(ci),
            ScatteringTransforms.Coefficients.second_order(ci), st.filter_bank.meta)
    # Isotropic noise: anisotropy averages down well below the oriented case.
    Test.@test maximum(abs, ri.shape) < maximum(abs, ra.shape)

    # Normalized + log reductions.
    nc = ScatteringTransforms.Reductions.normalized_coefficients(ca)
    Test.@test nc.s1 ≈ ScatteringTransforms.Coefficients.first_order(ca) ./ ScatteringTransforms.Coefficients.zeroth_order(ca)
    S1a = ScatteringTransforms.Coefficients.first_order(ca)
    S2a = ScatteringTransforms.Coefficients.second_order(ca)
    for j1 in axes(S2a, 1), j2 in axes(S2a, 2)
        if S1a[j1] > 0
            Test.@test nc.s2[j1, j2] ≈ S2a[j1, j2] / S1a[j1]
        end
    end
    # The pad keeps the structural zeros of `S2` finite under the log.
    lc = ScatteringTransforms.Reductions.log_coefficients(ca)
    Test.@test any(iszero, S2a)
    Test.@test all(isfinite, lc.logS2)
end

Test.@testset "Non-mutating scattering(st,x) matches the in-place st(x) (1D/2D/3D)" begin
    # The autodiff-friendly forward must reproduce the production (mutating) transform exactly.
    let
        N, J = 64, 4
        x = randn(N)
        for spec in (SpectralBackends.DirectSumSpectralBackend(), SpectralBackends.FFTSpectralBackend())
            st = ScatteringTransforms.Scattering1D.ScatteringTransform1D(N, J; Q=2, max_order=2, transform=spec)
            cm = st(x); cs = ScatteringTransforms.ScatteringCore.scattering(st, x)
            Test.@test ScatteringTransforms.Coefficients.zeroth_order(cm) ≈ ScatteringTransforms.Coefficients.zeroth_order(cs)
            Test.@test ScatteringTransforms.Coefficients.first_order(cm) ≈ ScatteringTransforms.Coefficients.first_order(cs)
            Test.@test ScatteringTransforms.Coefficients.second_order(cm) ≈ ScatteringTransforms.Coefficients.second_order(cs)
        end
    end
    let
        st = ScatteringTransforms.Scattering2D.ScatteringTransform2D((16, 16), 2; L=4, max_order=2)
        x = randn(16, 16)
        cm = st(x); cs = ScatteringTransforms.ScatteringCore.scattering(st, x)
        Test.@test ScatteringTransforms.Coefficients.first_order(cm) ≈ ScatteringTransforms.Coefficients.first_order(cs)
        Test.@test ScatteringTransforms.Coefficients.second_order(cm) ≈ ScatteringTransforms.Coefficients.second_order(cs)
    end
    let
        st = ScatteringTransforms.Scattering3D.ScatteringTransform3D((8, 8, 8), 2; n_orient=6, max_order=2)
        x = randn(8, 8, 8)
        cm = st(x); cs = ScatteringTransforms.ScatteringCore.scattering(st, x)
        Test.@test ScatteringTransforms.Coefficients.first_order(cm) ≈ ScatteringTransforms.Coefficients.first_order(cs)
        Test.@test ScatteringTransforms.Coefficients.second_order(cm) ≈ ScatteringTransforms.Coefficients.second_order(cs)
    end
    # Float32 in, Float32 out, on both paths.
    let
        st = ScatteringTransforms.Scattering1D.ScatteringTransform1D(Float32, 32, 3; Q=1, max_order=2)
        x = randn(Float32, 32)
        cm = st(x); cs = ScatteringTransforms.ScatteringCore.scattering(st, x)
        Test.@test eltype(ScatteringTransforms.Coefficients.first_order(cm)) == Float32
        Test.@test ScatteringTransforms.Coefficients.zeroth_order(cm) isa Float32
        Test.@test eltype(ScatteringTransforms.Coefficients.first_order(cs)) == Float32
    end
end

Test.@testset "Exact linear wavelet-frame inverse: iwavelet ∘ wavelet_transform ≈ id (1D/2D/3D)" begin
    let
        N, J = 128, 5
        x = randn(N)
        st = ScatteringTransforms.Scattering1D.ScatteringTransform1D(N, J; Q=1, max_order=1)
        xr = ScatteringTransforms.Inverse.iwavelet(st, ScatteringTransforms.Inverse.wavelet_transform(st, x))
        Test.@test xr ≈ x  rtol=1e-9
    end
    let
        N, J = (32, 32), 3
        x = randn(N)
        st = ScatteringTransforms.Scattering2D.ScatteringTransform2D(N, J; L=4, max_order=1)
        xr = ScatteringTransforms.Inverse.iwavelet(st, ScatteringTransforms.Inverse.wavelet_transform(st, x))
        Test.@test xr ≈ x  rtol=1e-9
    end
    let
        N, J = (16, 16, 16), 2
        x = randn(N)
        st = ScatteringTransforms.Scattering3D.ScatteringTransform3D(N, J; n_orient=6, max_order=1)
        xr = ScatteringTransforms.Inverse.iwavelet(st, ScatteringTransforms.Inverse.wavelet_transform(st, x))
        Test.@test xr ≈ x  rtol=1e-9
    end
end

Test.@testset "Phase retrieval (Gerchberg–Saxton): error reduction" begin
    # Each round projects onto the fields with the target moduli and low-pass, then back onto the
    # range of the wavelet layer by its least-squares inverse, so the distance `d` between the two
    # sets never increases (Fienup 1982) and the true field is a fixed point.
    Random.seed!(123)
    Inv = ScatteringTransforms.Inverse
    N, J = 128, 6
    x = randn(N)
    st = ScatteringTransforms.Scattering1D.ScatteringTransform1D(N, J; Q = 2, max_order = 1)
    wt = Inv.wavelet_transform(st, x)
    moduli = [abs.(w) for w in wt.wavelet]
    lp = copy(wt.lowpass)
    function d(xh)
        w = Inv.wavelet_transform(st, xh)
        return sqrt(sum(sum(abs2, abs.(a) .- m) for (a, m) in zip(w.wavelet, moduli)) +
                    sum(abs2, w.lowpass .- lp))
    end
    Test.@test Inv.reconstruct_phase(st, moduli; iters = 5, init = x, seed_lowpass = lp) ≈ x rtol = 1e-12
    x0 = randn(N)
    ds = [d(k == 0 ? x0 : Inv.reconstruct_phase(st, moduli; iters = k, init = x0, seed_lowpass = lp))
          for k in (0, 25, 100, 400)]
    Test.@test all(ds[i + 1] <= ds[i] * (1 + 1e-12) for i in 1:3)
    Test.@test ds[end] < ds[1]
end

Test.@testset "Monogenic (Riesz) scattering: multipliers, transforms" begin
    # `R_d(k) = -i k_d/|k|` is purely imaginary, `Σ_d |R_d(k)|² = 1` off the DC bin, and `R_d = 0`
    # on axis d's Nyquist bin, where the odd multiplier's two aliases `±i·½/|k|` average to zero.
    for dims in ((32,), (16, 16), (8, 8, 8), (15, 16), (9, 9, 9))
        R = ScatteringTransforms.Monogenic.riesz_multipliers(dims, Float64)
        Test.@test all(Rd -> all(iszero ∘ real, Rd), R)
        s = sum(abs2.(Rd) for Rd in R)
        nyq(I, d) = iseven(dims[d]) && I[d] - 1 == dims[d] ÷ 2
        freq(I) = [(i - 1 < (n + 1) ÷ 2 ? i - 1 : i - 1 - n) / n for (i, n) in zip(Tuple(I), dims)]
        ref = map(CartesianIndices(dims)) do I
            k = freq(I)
            iszero(k) ? 0.0 : sum(k[d]^2 for d in eachindex(k) if !nyq(I, d); init = 0.0) / sum(abs2, k)
        end
        Test.@test s[1] == 0                              # DC
        Test.@test maximum(abs, s .- ref) < 1e-14
        Test.@test all(R[d][I] == 0 for d in eachindex(dims) for I in CartesianIndices(dims) if nyq(I, d))
    end

    # Float32 in, Float32 out.
    let
        stf = ScatteringTransforms.Monogenic.MonogenicScattering(Float32, (32, 32), 2; Q=1, max_order=2)
        cf = stf(randn(Float32, 32, 32))
        Test.@test eltype(ScatteringTransforms.Coefficients.first_order(cf)) == Float32
    end

    # A complex field `a + ib` has amplitude `√(A(a)² + A(b)²)`, from the decompositions of its real
    # and imaginary parts, and its own complex mean as `S0`. Undecimated, so `S1[j]` is the mean of
    # that amplitude over the full grid.
    let
        Mono = ScatteringTransforms.Monogenic
        dims, J = (32, 32), 3
        st = Mono.MonogenicScattering(dims, J; Q = 1, max_order = 1)
        a, b = randn(dims), randn(dims)
        c = st(complex.(a, b))
        Aa = [Mono.monogenic_components(st, a, j).amplitude for j in 1:J]
        Ab = [Mono.monogenic_components(st, b, j).amplitude for j in 1:J]
        Test.@test ScatteringTransforms.Coefficients.first_order(c) ≈
                   [sum(sqrt.(Aa[j] .^ 2 .+ Ab[j] .^ 2)) / prod(dims) for j in 1:J] rtol = 1e-12
        Test.@test ScatteringTransforms.Coefficients.zeroth_order(c) ≈ complex(sum(a), sum(b)) / prod(dims)
        # The local phase and orientation are a real field's.
        Test.@test_throws ArgumentError Mono.monogenic_components(st, complex.(a, b), 1)
    end
end

Test.@testset "Monogenic periodized cascade converges to the undecimated one" begin
    # The Riesz-weighted band-pass `R_d ψ_j` is periodized as one filter; the periodization of a
    # product differs from the product of periodizations.
    Random.seed!(53)
    FB = SpectralBackends.FFTSpectralBackend()
    flat(c) = vcat(c.S0, vec(c.S1), vec(c.S2))

    for (dims, J) in (((128,), 4), ((32, 32), 3), ((16, 16, 16), 2))
        x = randn(dims...)
        mk(α; filters=ScatteringTransforms.FilterBanks.StoredFilters()) =
            ScatteringTransforms.Monogenic.MonogenicScattering(
                Float64, dims, J; Q=1, max_order=2, transform=FB, oversampling=α, filters=filters)
        cref = mk(J)(x)
        # Nothing decimated: the same operations in the same order, so equality.
        Test.@test flat(mk(J + 2)(x)) == flat(cref)

        e(α) = let c = mk(α)(x)
            (maximum(abs, c.S1 .- cref.S1) / maximum(abs, cref.S1),
             maximum(abs, c.S2 .- cref.S2) / maximum(abs, cref.S2))
        end
        e1, e0 = e(1), e(0)
        Test.@test e1[1] <= e0[1] + 1e-12          # error falls monotonically with oversampling
        Test.@test e1[2] <= e0[2] + 1e-12
        Test.@test e0[2] > 0                        # and is nonzero below it
        # `S0` is the plain field mean, so no decimation can move it.
        Test.@test mk(0)(x).S0 ≈ cref.S0

        # A computed bank evaluates into shared scratch.
        Test.@test flat(mk(1; filters=ScatteringTransforms.FilterBanks.ComputedFilters())(x)) == flat(mk(1)(x))
    end
end

Test.@testset "Monogenic: rotation invariance + continuous orientation recovery" begin
    # Averaged monogenic coefficients are rotation-invariant (90° rotation is exact on a grid).
    let
        M, J = 64, 3
        f = [sin(2π * 3 * i / M) + 0.5 * cos(2π * 5 * j / M) for i in 0:M-1, j in 0:M-1] .+
            0.1 .* randn(M, M)
        st = ScatteringTransforms.Monogenic.MonogenicScattering((M, M), J; Q=1, max_order=2)
        c0 = st(f); c90 = st(rotr90(f))
        rel = maximum(abs.(ScatteringTransforms.Coefficients.first_order(c0) .- ScatteringTransforms.Coefficients.first_order(c90))) /
              maximum(abs.(ScatteringTransforms.Coefficients.first_order(c0)))
        Test.@test rel < 1e-6
    end
    # The Riesz vector recovers a plane wave's orientation.
    let
        M, θ, n = 64, 0.6, 8
        kx, ky = cos(θ), sin(θ)
        f = [cos(2π * n * (kx * i + ky * j) / M) for i in 0:M-1, j in 0:M-1]
        st = ScatteringTransforms.Monogenic.MonogenicScattering((M, M), 5; Q=1, max_order=1)
        best = argmax([sum(abs2, ScatteringTransforms.Monogenic.monogenic_components(st, f, jj).bandpass) for jj in 1:5])
        comp = ScatteringTransforms.Monogenic.monogenic_components(st, f, best)
        r1, r2, amp = comp.riesz[1], comp.riesz[2], comp.amplitude
        mask = amp .> 0.5 * maximum(amp)
        c2 = sum((r1[k]^2 - r2[k]^2) for k in CartesianIndices(f) if mask[k])
        s2 = sum((2 * r1[k] * r2[k]) for k in CartesianIndices(f) if mask[k])
        est = mod(atan(s2, c2) / 2, π)                    # orientation is defined mod π
        Test.@test min(abs(est - θ), π - abs(est - θ)) < 0.05
    end
end

# Structured-grid spherical scattering (fast SHT).
include("test_spherical_sht.jl")

# Policy around the least-squares solve.
include("test_lsmr.jl")

# Scattered / nonuniform planar scattering (NUFFT).
include("test_scattered_planar.jl")

# Pointwise spherical monogenic orientation/phase (spin-1 synthesis).
include("test_spherical_monogenic_components.jl")

# Plan construction from concurrent tasks — the guards over the process-global FFTW planner and
# FastTransforms thread count both backends share.
include("test_concurrency.jl")

# Allocation discipline: hot paths zero-alloc; allocating paths minimal + data-size-independent.
include("test_allocs.jl")

# Vendor-neutral GPU path on the KernelAbstractions CPU backend (no GPU hardware needed).
include("test_gpu.jl")

end # module
