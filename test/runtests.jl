module ScatteringTransformsTests

using Test: Test
using Aqua: Aqua
using ExplicitImports: ExplicitImports as EI

# Use the required import style: using X: X
using ScatteringTransforms: ScatteringTransforms
using ComputationalBackends: ComputationalBackends
using SpectralBackends: SpectralBackends
using FFTW: FFTW
using OhMyThreads: OhMyThreads
using Distributed: Distributed
using MPI: MPI
using NUFSHT: NUFSHT
# Both fast NUFFT backends are loaded here rather than inside a single test file, so which one
# an `Auto` spectral backend resolves to does not depend on include order. Without this the
# earlier test files silently exercise the in-core reference instead of a fast path.
using FINUFFT: FINUFFT
using NonuniformFFTs: NonuniformFFTs
using Statistics: Statistics
using Random: Random
using DifferentiationInterface: DifferentiationInterface as DI  # loaded so its extension is present for the explicit-imports audit below
# Mooncake (and ADTypes.AutoMooncake) are loaded inside the gated test_mooncake.jl: Mooncake can't
# precompile on pre-release Julia, so its load and autodiff tests are skipped there — mirroring JET.

# Run Aqua quality tests first
Test.@testset "Aqua.jl quality tests" begin
    Aqua.test_all(ScatteringTransforms)
end

Test.@testset "Explicit imports (no implicit / no stale)" begin
    # Enforce the package style: no reliance on bare `using` re-exports, no dead imports.
    # Checks the core module and every loaded backend extension (skipped if the weakdep
    # isn't loaded in the test environment).
    Test.@test (EI.check_no_implicit_imports(ScatteringTransforms); true)
    Test.@test (EI.check_no_stale_explicit_imports(ScatteringTransforms); true)
    for extname in (
        :ScatteringTransformsCairoMakieExt,
        :ScatteringTransformsDifferentiationInterfaceExt,
        :ScatteringTransformsFFTWExt,
        :ScatteringTransformsKernelAbstractionsExt,
        :ScatteringTransformsNUFSHTExt,
        :ScatteringTransformsOhMyThreadsExt,
        :ScatteringTransformsDistributedExt,
        :ScatteringTransformsMPIExt,
        :ScatteringTransformsNUFSHTExt,
    )
        ext = Base.get_extension(ScatteringTransforms, extname)
        ext === nothing && continue
        Test.@test (EI.check_no_implicit_imports(ext); true)
        Test.@test (EI.check_no_stale_explicit_imports(ext); true)
    end
end

# JET optimization/abstract-interpretation audit of the hot paths (skipped on pre-release Julia,
# where JET refuses to run — see test_jet.jl).
include("test_jet.jl")

# Reverse-mode autodiff coverage via DifferentiationInterface. Enzyme is the backend under test;
# the equivalent Mooncake suite is kept but off by default, because Mooncake currently conflicts
# with the CUDA version FINUFFT is pinned to. Run it with `ST_TEST_MOONCAKE=1`.
include("test_enzyme.jl")
get(ENV, "ST_TEST_MOONCAKE", "0") == "1" && include("test_mooncake.jl")

# Structural gates: spectral-execution counts, workspace scaling, backend honesty — the properties
# that a performance regression would break first, asserted by counting rather than by timing.
include("test_structure.jl")

Test.@testset "Type stability (concrete struct fields + inferred transforms)" begin
    # Every field of the transform struct must be concretely typed — in particular the FFT plan
    # fields (an untyped `Any` there forces dynamic dispatch on every `mul!`) and the 2D filter-bank
    # field (its matrix container must stay a type parameter, not be dropped to an abstract type).
    N = 256
    J = 4
    st = ScatteringTransforms.Scattering1D.ScatteringTransform1D(N, J; Q=1, max_order=2)
    for ft in fieldtypes(typeof(st))
        Test.@test isconcretetype(ft)
    end
    signal = randn(N)
    Test.@test (Test.@inferred st(signal); true)

    st2 = ScatteringTransforms.Scattering2D.ScatteringTransform2D((64, 64), 3; L=4, max_order=2)
    for ft in fieldtypes(typeof(st2))
        Test.@test isconcretetype(ft)
    end
    image = randn(64, 64)
    Test.@test (Test.@inferred st2(image); true)

    # Element type is preserved end-to-end (Float32 in -> Float32 out).
    stf = ScatteringTransforms.Scattering1D.ScatteringTransform1D(Float32, N, J; Q=1, max_order=2)
    cf = stf(Float32.(signal))
    Test.@test eltype(ScatteringTransforms.Coefficients.first_order(cf)) == Float32
    Test.@test ScatteringTransforms.Coefficients.zeroth_order(cf) isa Float32
end

Test.@testset "1D Morlet Wavelet Mathematical Properties" begin
    N = 512
    j = 2
    Q = 1
    r = sqrt(0.5)
    
    morlet = ScatteringTransforms.Filters.Morlet1D(N, j; Q=Q)
    ψ = ScatteringTransforms.Filters.frequency_response(morlet)
    freqs = FFTW.fftfreq(N)
    
    # Test 1: Center frequency matches expected formula
    # xi = 0.5 * 2^(-j/Q)
    expected_center = 0.5 / (2.0^(j/Q))
    Test.@test isapprox(morlet.center_freq, expected_center, rtol=1e-10)
    
    # Test 2: Bandwidth matches kymatio formula
    # sigma = xi * (1-2^(-1/Q))/(1+2^(-1/Q)) / sqrt(2*log(1/r))
    factor = 1.0 / (2.0^(1.0/Q))
    term1 = (1.0 - factor) / (1.0 + factor)
    term2 = 1.0 / sqrt(2 * log(1.0/r))
    expected_sigma = expected_center * term1 * term2
    Test.@test isapprox(morlet.bandwidth, expected_sigma, rtol=1e-10)
    
    # Test 3: Peak is at center frequency (within 1 FFT bin)
    pos_idx = findall(freqs .>= 0)
    peak_idx = argmax(abs.(ψ[pos_idx]))
    peak_freq = freqs[pos_idx][peak_idx]
    freq_spacing = 1.0 / N
    Test.@test abs(peak_freq - morlet.center_freq) < 2 * freq_spacing
    
    # Test 4: Zero mean (admissibility) - DC component should be ~0
    dc_idx = argmin(abs.(freqs))
    Test.@test abs(ψ[dc_idx]) < 0.01
    
    # Test 5: Analytic - negative frequencies should be exactly zero
    neg_idx = findall(freqs .< 0)
    Test.@test all(abs.(ψ[neg_idx]) .< 1e-10)
    
    # Test 6: Verify Gaussian shape at specific points
    # At center frequency, |ψ| should be close to 1 (before any normalization)
    ψ_abs = abs.(ψ)
    center_idx = argmin(abs.(freqs .- morlet.center_freq))
    
    # At center ± sigma, |ψ| should be exp(-0.5) ≈ 0.6065
    sigma = morlet.bandwidth
    idx_plus = argmin(abs.(freqs .- (morlet.center_freq + sigma)))
    idx_minus = argmin(abs.(freqs .- (morlet.center_freq - sigma)))
    
    # Only check if indices are in positive frequency region
    if freqs[idx_plus] >= 0
        val_plus = ψ_abs[idx_plus] / ψ_abs[center_idx]  # Normalize
        Test.@test isapprox(val_plus, exp(-0.5), rtol=0.2)
    end
    if freqs[idx_minus] >= 0
        val_minus = ψ_abs[idx_minus] / ψ_abs[center_idx]
        Test.@test isapprox(val_minus, exp(-0.5), rtol=0.2)
    end
end

Test.@testset "Filter bank is a tight frame (Littlewood-Paley ≡ 1)" begin
    # |φ(ω)|² + Σⱼ|ψⱼ(ω)|² ≡ 1: non-expansive, no frequency amplified.
    #
    # The identity is exact by construction — `φ = √(1 − Σⱼ|ψⱼ|²)`, so the sum is that expression
    # squared again — and it holds over the whole band, not just the positive half. Measured
    # deviation is one ulp (2.220e-16 in Float64, 1.192e-07 in Float32, i.e. `eps(T)` in each), so
    # the tolerance is a small multiple of `eps` rather than a round number that would let a bank
    # drift a long way before failing.
    for (N, J, Q) in ((1024, 6, 1), (512, 4, 2))
        fb = ScatteringTransforms.FilterBanks.build_filter_bank1d(N, J; Q=Q)
        lp = abs2.(fb.averaging) .+ sum(abs2.(ψ) for ψ in fb.wavelets)
        Test.@test maximum(abs, lp .- 1) < 4 * eps(Float64)
    end
    fb2 = ScatteringTransforms.FilterBanks.build_filter_bank2d((64, 64), 3; L=8)
    lp2 = abs2.(fb2.averaging) .+ sum(abs2.(ψ) for ψ in fb2.wavelets)
    Test.@test maximum(abs, lp2 .- 1) < 4 * eps(Float64)
end

Test.@testset "Filter banks are real, and stored as such" begin
    # A Morlet's Fourier response is real — analyticity is the half-plane support, not a complex
    # value — so the bank is the same numbers in half the memory, and a wavelet multiply is
    # complex×real. The bank is the largest single item in a transform, so this is not a detail.
    #
    # The narrower container is asserted alongside the tight-frame sum it has to keep satisfying,
    # so a bank that reached a real element type by discarding information would fail here rather
    # than pass on the type check alone.
    for T in (Float64, Float32)
        banks = (ScatteringTransforms.FilterBanks.build_filter_bank1d(T, 128, 4; Q=2),
                 ScatteringTransforms.FilterBanks.build_filter_bank2d(T, (32, 32), 3; L=4),
                 ScatteringTransforms.FilterBanks.build_filter_bank3d(T, (16, 16, 16), 2;
                                                                     n_orient=4),
                 ScatteringTransforms.Monogenic.build_monogenic_bank(T, (16, 16), 3))
        for fb in banks
            Test.@test eltype(fb.averaging) == T
            Test.@test all(ψ -> eltype(ψ) == T, fb.wavelets)
            lp = abs2.(fb.averaging) .+ sum(abs2.(ψ) for ψ in fb.wavelets)
            Test.@test maximum(abs, lp .- 1) < 4 * eps(T)
        end
        # The Riesz multipliers are genuinely complex (`R_d(k) = -i k_d/|k|`) and stay so — the
        # saving comes from the wavelets being real, not from narrowing everything in sight.
        mb = banks[end]
        Test.@test all(R -> eltype(R) == Complex{T}, mb.riesz)
        Test.@test maximum(R -> maximum(abs ∘ real, R), mb.riesz) == 0
        Test.@test maximum(R -> maximum(abs ∘ imag, R), mb.riesz) > 0
    end
end

Test.@testset "1D Filter Bank Tests" begin
    N = 256
    J = 4
    bank = ScatteringTransforms.FilterBanks.build_filter_bank1d(N, J; Q=1)
    
    Test.@test bank.J == J
    Test.@test bank.Q == 1
    Test.@test length(bank.wavelets) == J  # One wavelet per octave for Q=1
    Test.@test length(bank.averaging) == N
end

Test.@testset "1D Filter Bank with Q > 1 Fractional Center Frequencies" begin
    N = 256
    J = 3
    Q = 4
    bank = ScatteringTransforms.FilterBanks.build_filter_bank1d(N, J; Q=Q)
    
    Test.@test bank.J == J
    Test.@test bank.Q == Q
    Test.@test length(bank.wavelets) == J * Q
    
    # Verify that center frequencies are strictly unique and decreasing
    center_freqs = [m.center_freq for m in bank.meta]
    Test.@test all(center_freqs[i] > center_freqs[i+1] for i in 1:length(center_freqs)-1)
    
    # Verify that fractional frequency spacing matches math formula:
    # center_freq = 0.5 * 2^(-j_eff)
    for (i, m) in enumerate(bank.meta)
        expected_freq = 0.5 / (2.0^m.j_eff)
        Test.@test isapprox(m.center_freq, expected_freq, rtol=1e-10)
    end
end

Test.@testset "2D S2 Coefficients Zero-Initialization Verification" begin
    # Checks the 2D second-order storage contract: S2 is a dense J×J matrix whose inadmissible
    # (non-frequency-decreasing) entries are zeroed.
    Ny, Nx = 64, 64
    J = 3
    L = 4
    st = ScatteringTransforms.Scattering2D.ScatteringTransform2D((Ny, Nx), J; L=L, max_order=2)
    image = randn(Ny, Nx)
    coeffs = st(image)
    S2 = ScatteringTransforms.Coefficients.second_order(coeffs)

    # Verify diagonal and lower triangle are exactly zero (due to zeros allocation fix)
    n = size(S2, 1)
    for j1 in 1:n
        for j2 in 1:j1
            Test.@test S2[j1, j2] == 0.0
        end
    end
    
    # Verify upper triangle has positive values (meaningful computed second-order scattering)
    upper_vals = [S2[j1, j2] for j1 in 1:n for j2 in (j1+1):n]
    Test.@test all(upper_vals .>= 0.0)
    Test.@test any(upper_vals .> 0.0)
end

Test.@testset "Path graph: 2D second order is scale-increasing (correctness fix)" begin
    # The 2D second-order constraint must be SCALE strictly increasing over ALL orientation
    # pairs — not a flat-index upper triangle (which wrongly mixed scale and orientation and
    # included same-scale, different-orientation pairs).
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
    # Same-scale (incl. different-orientation) second-order entries are exactly zero.
    for i1 in 1:nw, i2 in 1:nw
        if meta[i1].scale == meta[i2].scale
            Test.@test S2[i1, i2] == 0
        end
    end
    # Genuine cross-scale second-order energy exists.
    Test.@test any(S2[i1, i2] > 0 for i1 in 1:nw for i2 in 1:nw if meta[i2].scale > meta[i1].scale)
end


Test.@testset "1D Scattering Transform Tests" begin
    N = 256
    J = 4
    
    st = ScatteringTransforms.Scattering1D.ScatteringTransform1D(N, J; Q=1, max_order=2)
    signal = randn(N)
    
    coeffs = st(signal)
    
    # Test S0 (average) - using accessor functions
    Test.@test isapprox(ScatteringTransforms.Coefficients.zeroth_order(coeffs), Statistics.mean(signal), atol=1e-10)
    
    # Test S1 (first order)
    S1 = ScatteringTransforms.Coefficients.first_order(coeffs)
    Test.@test length(S1) == J  # One coefficient per scale
    Test.@test all(S1 .>= 0)  # Modulus makes them non-negative
    
    # Test S2 (second order)
    S2 = ScatteringTransforms.Coefficients.second_order(coeffs)
    Test.@test size(S2) == (J, J)
    
    # Test S2 has meaningful values (not all near-zero due to filter bug)
    # With proper wavelet formulas, S2 should have measurable energy
    Test.@test Statistics.maximum(S2) > 1e-8
end

Test.@testset "Wavelet Center Frequencies" begin
    N = 512
    J = 6
    
    for j in 0:(J-1)
        morlet = ScatteringTransforms.Filters.Morlet1D(N, j; Q=1)
        expected_center = 0.5 / (2.0^j)  # xi = 0.5 * 2^(-j)
        
        # Center frequency should match expected value within tolerance
        Test.@test isapprox(morlet.center_freq, expected_center, rtol=1e-10)
    end
end

Test.@testset "Wavelet Constant-Q Property" begin
    N = 512
    J = 6
    
    Q_values = Float64[]
    for j in 0:(J-1)
        morlet = ScatteringTransforms.Filters.Morlet1D(N, j; Q=1)
        Q = morlet.center_freq / morlet.bandwidth
        Base.push!(Q_values, Q)
    end
    
    # Q should be approximately constant across all scales
    Q_mean = Statistics.mean(Q_values)
    for (j, Q) in enumerate(Q_values)
        Test.@test isapprox(Q, Q_mean, rtol=0.1)
    end
end

Test.@testset "Wavelet Frequency Response Peak Location" begin
    N = 512
    
    for j in 0:3
        morlet = ScatteringTransforms.Filters.Morlet1D(N, j; Q=1)
        ψ = ScatteringTransforms.Filters.frequency_response(morlet)
        ψ_abs = abs.(ψ)
        
        freqs = FFTW.fftfreq(N)
        peak_idx = argmax(ψ_abs)
        peak_freq = freqs[peak_idx]
        
        # Peak should be near center frequency (within 1 frequency bin)
        freq_spacing = 1.0 / N
        Test.@test abs(peak_freq - morlet.center_freq) < freq_spacing * 2
    end
end

Test.@testset "Filter Bank Wavelet Energy" begin
    N = 256
    J = 4
    bank = ScatteringTransforms.FilterBanks.build_filter_bank1d(N, J; Q=1)
    
    # Each wavelet must have non-negligible in-band energy (guards against a bandwidth error that
    # collapses the filters to near-zero magnitude).
    for (j, ψ) in enumerate(bank.wavelets)
        energy = Statistics.maximum(Base.abs.(ψ))
        # With correct formulas, all wavelets should have ~0.01-0.1 energy
        Test.@test energy > 0.001
    end
end

Test.@testset "Wavelet Shape is Gaussian" begin
    N = 512
    
    for j in 0:3
        morlet = ScatteringTransforms.Filters.Morlet1D(N, j; Q=1)
        ψ = ScatteringTransforms.Filters.frequency_response(morlet)
        ψ_abs = abs.(ψ)
        freqs = FFTW.fftfreq(N)
        
        # Find peak among positive frequencies only
        last_pos = findlast(freqs .>= 0)
        peak_idx = argmax(ψ_abs[1:last_pos])
        
        # Check monotonicity before peak (always within positive freqs)
        left_side = ψ_abs[peak_idx-10:peak_idx-1]
        is_increasing = all(diff(left_side) .> 0)
        Test.@test is_increasing
        
        # Check monotonicity after peak, but only if there's room
        # For j=0, peak is at Nyquist boundary - no room to check after
        if peak_idx + 10 <= last_pos
            right_side = ψ_abs[peak_idx+1:peak_idx+10]
            is_decreasing = all(diff(right_side) .< 0)
            Test.@test is_decreasing
        end
    end
end

Test.@testset "2D Filter Tests" begin
    Ny, Nx = 64, 64
    morlet = ScatteringTransforms.Filters.Morlet2D((Ny, Nx), 2, π/4; L=8)
    
    resp = ScatteringTransforms.Filters.frequency_response(morlet)
    Test.@test size(resp) == (Ny, Nx)
    # A Morlet's *Fourier* response is real: its analyticity is the half-plane support (asserted
    # below), not a complex value. Storing it as such is what halves the filter bank.
    Test.@test eltype(resp) == Float64
    Test.@test eltype(ScatteringTransforms.Filters.frequency_response(
        ScatteringTransforms.Filters.Morlet2D{Float32}((Ny, Nx), 2, π/4; L=8))) == Float32

    # 1. Peak location
    # Since theta = π/4 and j = 2, the center frequency is k0 = 3π / (4 * 2^2) = 3π / 16 ≈ 0.589.
    # Frequency grid: kx = _fftfreq(Nx, ix-1) * 2π, ky = _fftfreq(Ny, iy-1) * 2π.
    freqs_x = [ScatteringTransforms.Filters._fftfreq(Nx, ix-1) * 2π for ix in 1:Nx]
    freqs_y = [ScatteringTransforms.Filters._fftfreq(Ny, iy-1) * 2π for iy in 1:Ny]
    
    # Find peak index
    peak_idx = argmax(abs.(resp))
    iy_p, ix_p = peak_idx[1], peak_idx[2]
    kx_p, ky_p = freqs_x[ix_p], freqs_y[iy_p]
    
    # Check that rotated kxr is close to k0
    ct, st = cos(π/4), sin(π/4)
    kxr_p = kx_p * ct + ky_p * st
    kyr_p = -kx_p * st + ky_p * ct
    
    Test.@test isapprox(kxr_p, morlet.center_freq, atol=0.2)
    Test.@test isapprox(kyr_p, 0.0, atol=0.2)
    
    # 2. Analytic property (half-plane): zero when kxr < 0
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
    
    # 3. DC Response (zero mean)
    Test.@test abs(resp[1, 1]) < 1e-10
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

Test.@testset "2D Filter Bank Tests" begin
    N = (64, 64)
    J = 3
    L = 4
    
    bank = ScatteringTransforms.FilterBanks.build_filter_bank2d(N, J; L=L)
    
    Test.@test bank.J == J
    Test.@test bank.L == L
    Test.@test length(bank.wavelets) == J * L
    Test.@test size(bank.averaging) == N
end

Test.@testset "2D Scattering Transform Tests" begin
    Ny, Nx = 64, 64
    J = 3
    L = 4
    
    st = ScatteringTransforms.Scattering2D.ScatteringTransform2D((Ny, Nx), J; L=L, max_order=2)
    image = randn(Ny, Nx)
    
    coeffs = st(image)
    
    # Test S0 exists
    S0 = ScatteringTransforms.Coefficients.zeroth_order(coeffs)
    Test.@test isa(S0, Float64)
    
    # Test S1
    S1 = ScatteringTransforms.Coefficients.first_order(coeffs)
    Test.@test length(S1) == J * L
    
    # Test S2
    S2 = ScatteringTransforms.Coefficients.second_order(coeffs)
    Test.@test size(S2) == (J * L, J * L)
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
    st_d = ScatteringTransforms.Scattering3D.ScatteringTransform3D(N, J; n_orient=n_orient, max_order=2, spectral=SpectralBackends.DirectSumSpectralBackend())
    st_f = ScatteringTransforms.Scattering3D.ScatteringTransform3D(N, J; n_orient=n_orient, max_order=2, spectral=SpectralBackends.FFTSpectralBackend())
    Test.@test isapprox(ScatteringTransforms.Coefficients.first_order(st_d(vol)),
                        ScatteringTransforms.Coefficients.first_order(st_f(vol)); rtol=1e-6)
    Test.@test isapprox(ScatteringTransforms.Coefficients.second_order(st_d(vol)),
                        ScatteringTransforms.Coefficients.second_order(st_f(vol)); rtol=1e-6)
end

Test.@testset "Spherical scattering (NUFSHT, smooth difference-of-Gaussians bands)" begin
    lmax = 12
    J = 3
    M = 1000                      # ≳ (lmax+1)² = 169 so the exact CG analysis is well-determined
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
    Test.@test all(isfinite, res.S1)
end

Test.@testset "Spherical MONOGENIC scattering (Riesz amplitude via spin-0 Bochner identity)" begin
    lmax = 12
    J = 3
    M = 1200                      # ≳ (lmax+1)² = 169 so the exact CG analysis is well-determined
    # Deterministic, well-distributed Fibonacci-sphere points.
    gr = (sqrt(5) - 1) / 2
    θ = [acos(1 - 2 * (k - 0.5) / M) for k in 1:M]
    φ = [2π * mod(k * gr, 1) for k in 1:M]
    st = ScatteringTransforms.spherical_monogenic_scattering(θ, φ, lmax, J)
    res = st([cos(4 * θ[k]) + 0.5 * sin(3 * θ[k]) * cos(2 * φ[k]) for k in 1:M])  # band-limited
    Test.@test length(res.S1) == J
    Test.@test all(res.S1 .>= 0)              # monogenic amplitude is non-negative
    Test.@test all(isfinite, res.S1)
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

Test.@testset "3D Morlet wavelet: analytic + zero-mean" begin
    N = (16, 16, 16)
    dirs = ScatteringTransforms.Filters.fibonacci_directions(6, Float64)
    m = ScatteringTransforms.Filters.Morlet3D(N, 1, dirs[1])
    ψ = ScatteringTransforms.Filters.frequency_response(m)
    Test.@test size(ψ) == N
    Test.@test abs(ψ[1, 1, 1]) < 1e-10            # zero mean (DC)
    Test.@test all(isfinite, ψ)
end

Test.@testset "Translation invariance (approximate)" begin
    N = 256
    J = 4
    
    st = ScatteringTransforms.Scattering1D.ScatteringTransform1D(N, J; Q=1, max_order=1)
    
    # Create a periodic signal
    x = range(0, 2π, length=N+1)[1:N]
    signal = sin.(3x)
    
    # Shift by small amount
    shift = 10
    signal_shifted = circshift(signal, shift)
    
    coeffs1 = st(signal)
    coeffs2 = st(signal_shifted)
    
    # S1 should be approximately translation invariant
    S1_1 = ScatteringTransforms.Coefficients.first_order(coeffs1)
    S1_2 = ScatteringTransforms.Coefficients.first_order(coeffs2)
    rel_diff = abs.(S1_1 .- S1_2) ./ (S1_1 .+ 1e-10)
    Test.@test all(rel_diff .< 0.1)  # Within 10% due to edge effects
end

Test.@testset "Mutable S0 container → in-place update, truly zero-alloc streaming" begin
    # A mutable S0 container makes scattering_transform! mutate in place (same object) and
    # allocate nothing in steady state — the dispatch that powers zero-allocation streaming.
    N = 256
    J = 4
    st = ScatteringTransforms.Scattering1D.ScatteringTransform1D(N, J; Q=1, max_order=2)
    nw = length(st.filter_bank.wavelets)
    coeffs = ScatteringTransforms.Coefficients.ScatteringCoefficients1D(Vector{Float64}(undef, nw), zeros(nw, nw); S0=[0.0])
    sig = randn(N)
    r = ScatteringTransforms.Scattering1D.scattering_transform!(coeffs, st, sig)
    Test.@test r === coeffs                                                  # mutated in place
    Test.@test ScatteringTransforms.Coefficients.zeroth_order(coeffs) ≈ Statistics.mean(sig)
    ScatteringTransforms.Scattering1D.scattering_transform!(coeffs, st, sig)              # warm up
    Test.@test (@allocated ScatteringTransforms.Scattering1D.scattering_transform!(coeffs, st, sig)) == 0
end

Test.@testset "Batched transforms reuse the plan and match per-signal results" begin
    N = 128
    J = 4
    st = ScatteringTransforms.Scattering1D.ScatteringTransform1D(N, J; Q=1, max_order=2)
    X = randn(N, 5)
    B = ScatteringTransforms.scattering_batch(st, X)
    Test.@test size(B, 2) == 5
    for b in 1:5
        Test.@test isapprox(B[:, b], ScatteringTransforms.Coefficients.flatten1d(st(view(X, :, b))); rtol=1e-10)
    end
    # steady-state batch call allocates only the output + per-column scalar-S0 wrappers (no
    # per-signal workspace), i.e. far less than B independent transforms would.
    Test.@test (ScatteringTransforms.scattering_batch(st, X); true)

    # 2D
    st2 = ScatteringTransforms.Scattering2D.ScatteringTransform2D((32, 32), 3; L=4, max_order=2)
    X2 = randn(32, 32, 4)
    B2 = ScatteringTransforms.scattering_batch(st2, X2)
    Test.@test size(B2, 2) == 4
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

    # Nothing decimated ⇒ every alias block is the whole spectrum ⇒ the products are the same
    # floating-point operations in the same order, so this is equality, not approximate equality.
    c_off = build(J + 3)(signal)
    Test.@test C.first_order(c_off) == C.first_order(exact)
    Test.@test C.second_order(c_off) == C.second_order(exact)
    Test.@test C.zeroth_order(c_off) == C.zeroth_order(exact)

    # S0 is the plain field mean and never involves a wavelet, so no decimation can move it.
    Test.@test C.zeroth_order(build(0)(signal)) ≈ C.zeroth_order(exact)

    # What is asserted is where the loss goes to zero, not how big it is below that. A modulus is
    # nonlinear, so the envelope it produces has no compact spectrum and decimating it always aliases
    # something; how much depends on the signal and has no per-draw bound. What holds for every draw
    # is that retaining enough band is exactly lossless, and that below it the loss is real rather
    # than the cascade quietly declining to decimate.
    S1e, S2e = C.first_order(exact), C.second_order(exact)
    err(ov, get, ref) = let c = build(ov)(signal)
        sum(abs2, get(c) .- ref) / sum(abs2, ref)
    end
    Test.@test err(J, C.second_order, S2e) == 0
    Test.@test err(0, C.second_order, S2e) > 0

    # Order 1 is decimated too — the coefficient is the mean of |U₁| over the grid that wavelet's
    # band needs, not over the full grid. Exact where nothing is decimated, and bounded below that.
    Test.@test err(J, C.first_order, S1e) == 0
    Test.@test 0 < err(0, C.first_order, S1e) < 1e-2
end

Test.@testset "Threaded batch (OhMyThreads) matches serial batch" begin
    # ThreadedBackend parallelizes scattering_batch over the batch; each task uses its own
    # workspace, so results must be identical to the serial batch.
    N = 96
    J = 4
    st = ScatteringTransforms.Scattering1D.ScatteringTransform1D(N, J; Q=1, max_order=2)
    X = randn(N, 6)
    serial = ScatteringTransforms.scattering_batch(st, X)
    threaded = ScatteringTransforms.scattering_batch(ComputationalBackends.ThreadedBackend(), st, X)
    Test.@test threaded ≈ serial

    st2 = ScatteringTransforms.Scattering2D.ScatteringTransform2D((24, 24), 3; L=4, max_order=2)
    X2 = randn(24, 24, 5)
    Test.@test ScatteringTransforms.scattering_batch(ComputationalBackends.ThreadedBackend(), st2, X2) ≈
               ScatteringTransforms.scattering_batch(st2, X2)
end

Test.@testset "Distributed batch (single process) matches serial" begin
    # DistributedBackend distributes batch columns across workers; each worker rebuilds the
    # transform from a spec (plans aren't serializable) and runs the inner backend. With no
    # added workers, pmap runs locally and must equal the serial batch.
    N = 96
    J = 4
    st = ScatteringTransforms.Scattering1D.ScatteringTransform1D(N, J; Q=1, max_order=2)
    X = randn(N, 6)
    Test.@test ScatteringTransforms.scattering_batch(ComputationalBackends.DistributedBackend(), st, X) ≈
               ScatteringTransforms.scattering_batch(st, X)

    st2 = ScatteringTransforms.Scattering2D.ScatteringTransform2D((24, 24), 3; L=4, max_order=2)
    X2 = randn(24, 24, 5)
    Test.@test ScatteringTransforms.scattering_batch(ComputationalBackends.DistributedBackend(), st2, X2) ≈
               ScatteringTransforms.scattering_batch(st2, X2)
end

Test.@testset "MPI batch (single rank) matches serial" begin
    # SPMD: each rank computes its column block and Allgatherv combines them. With a single rank
    # (no mpiexec needed) the result must equal the serial batch. Multi-rank verification is for
    # `mpiexec -n k` on the user's cluster.
    MPI.Initialized() || MPI.Init()
    N = 96
    J = 4
    st = ScatteringTransforms.Scattering1D.ScatteringTransform1D(N, J; Q=1, max_order=2)
    X = randn(N, 6)
    Test.@test ScatteringTransforms.scattering_batch(ComputationalBackends.MPIBackend(), st, X) ≈
               ScatteringTransforms.scattering_batch(st, X)
    st2 = ScatteringTransforms.Scattering2D.ScatteringTransform2D((24, 24), 3; L=4, max_order=2)
    X2 = randn(24, 24, 5)
    Test.@test ScatteringTransforms.scattering_batch(ComputationalBackends.MPIBackend(), st2, X2) ≈
               ScatteringTransforms.scattering_batch(st2, X2)
end

Test.@testset "Spectral plans: in-core direct sum matches FFTW fast path" begin
    # The slow in-core DirectSumPlan default and the FFTW extension fast path must agree.
    N = 128
    J = 4
    signal = randn(N)
    st_d = ScatteringTransforms.Scattering1D.ScatteringTransform1D(N, J; Q=1, max_order=2, spectral=SpectralBackends.DirectSumSpectralBackend())
    st_f = ScatteringTransforms.Scattering1D.ScatteringTransform1D(N, J; Q=1, max_order=2, spectral=SpectralBackends.FFTSpectralBackend())
    cd, cf = st_d(signal), st_f(signal)
    Test.@test isapprox(ScatteringTransforms.Coefficients.zeroth_order(cd), ScatteringTransforms.Coefficients.zeroth_order(cf); rtol=1e-6)
    Test.@test isapprox(ScatteringTransforms.Coefficients.first_order(cd), ScatteringTransforms.Coefficients.first_order(cf); rtol=1e-6)
    Test.@test isapprox(ScatteringTransforms.Coefficients.second_order(cd), ScatteringTransforms.Coefficients.second_order(cf); rtol=1e-6)

    img = randn(32, 32)
    s2d = ScatteringTransforms.Scattering2D.ScatteringTransform2D((32, 32), 3; L=4, max_order=2, spectral=SpectralBackends.DirectSumSpectralBackend())
    s2f = ScatteringTransforms.Scattering2D.ScatteringTransform2D((32, 32), 3; L=4, max_order=2, spectral=SpectralBackends.FFTSpectralBackend())
    Test.@test isapprox(ScatteringTransforms.Coefficients.first_order(s2d(img)),
                        ScatteringTransforms.Coefficients.first_order(s2f(img)); rtol=1e-6)
    Test.@test isapprox(ScatteringTransforms.Coefficients.second_order(s2d(img)),
                        ScatteringTransforms.Coefficients.second_order(s2f(img)); rtol=1e-6)
end

Test.@testset "1D localized field: mean equals averaged coefficient" begin
    # The localized (Mallat) field S_p x = (|U_p x| ⋆ φ_J) ↓ s. With s = 1 (no decimation) and
    # φ̂(0) = 1, the spatial mean of each path's field must equal that path's globally-averaged
    # coefficient — the two outputs are consistent by construction.
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

    # Decimation must not weaken that: φ_J is a genuine Gaussian low-pass, so the subsampled field
    # carries the same DC and its mean is still the coefficient. Smoothing with the bank's
    # `averaging` instead — an all-pass on the analytic complement — folds full-band energy onto DC
    # and puts this 8% out at s=8.
    sf8 = ScatteringTransforms.ScatteringFields.scattering_field(st, signal; subsample=8)
    Test.@test size(sf8.data, 1) == N ÷ 8
    Test.@test all(isfinite, sf8.data)
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
    Test.@test all(isfinite, sf2.data)
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

Test.@testset "Computed filter banks (cache=false) match stored ones everywhere" begin
    # A computed bank evaluates each wavelet on demand into one shared scratch array instead of
    # storing `J·L` of them. That has to be invisible in the result and safe under threading — the
    # sharing is exactly what makes it a race if a task is handed someone else's scratch.
    Random.seed!(41)
    FB = SpectralBackends.FFTSpectralBackend()
    CBk = ComputationalBackends
    flat(c) = vcat(c.S0, vec(c.S1), vec(c.S2))

    for (mk, x, run!, mkc) in (
            ((α, ca) -> ScatteringTransforms.Scattering1D.ScatteringTransform1D(128, 4; Q=1,
                            max_order=2, oversampling=α, spectral=FB, cache=ca),
             randn(128), ScatteringTransforms.Scattering1D.scattering_transform!,
             st -> ScatteringTransforms.batch_coeffs(st, Float64)),
            ((α, ca) -> ScatteringTransforms.Scattering2D.ScatteringTransform2D((32, 32), 3; L=4,
                            max_order=2, oversampling=α, spectral=FB, cache=ca),
             randn(32, 32), ScatteringTransforms.Scattering2D.scattering_transform2d!,
             st -> ScatteringTransforms.batch_coeffs(st, Float64)),
            ((α, ca) -> ScatteringTransforms.Scattering3D.ScatteringTransform3D((16, 16, 16), 2;
                            n_orient=4, max_order=2, oversampling=α, spectral=FB, cache=ca),
             randn(16, 16, 16), ScatteringTransforms.Scattering3D.scattering_transform3d!,
             st -> ScatteringTransforms.batch_coeffs(st, Float64)))
        for α in (4, 1, 0)
            stored, computed = mk(α, true), mk(α, false)
            Test.@test ScatteringTransforms.FilterBanks.iscomputed(computed.filter_bank)
            Test.@test !ScatteringTransforms.FilterBanks.iscomputed(stored.filter_bank)
            # Same arithmetic in the same order, so equality rather than approximate equality.
            Test.@test flat(computed(x)) == flat(stored(x))

            # Each task gets its own bank, so its filter list must point at *that* bank's scratch.
            ser, thr = mkc(computed), mkc(computed)
            run!(ser, CBk.SerialBackend(), computed, x)
            for _ in 1:6
                run!(thr, CBk.ThreadedBackend(), computed, x)
                Test.@test flat(thr) == flat(ser)
            end
        end
    end

    # A worker rebuilding from a spec must get a computed bank too, not silently a stored one.
    st = ScatteringTransforms.Scattering2D.ScatteringTransform2D((32, 32), 3; L=4, max_order=2,
             oversampling=1, spectral=FB, cache=false)
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
    # A modulus is real, so `S1`/`S2` of a complex field are real and must equal the real field's
    # exactly when the imaginary part is zero. Only `S0 = ⟨x⟩` is complex, and the flat layout and
    # the order-0 localized field have to be wide enough to carry it rather than throwing or
    # silently dropping the imaginary part.
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

        # A real field must not be widened just because complex input is supported.
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
    Test.@test eltype(B) <: Complex
    for b in 1:3
        Test.@test view(B, :, b) ≈ C.flatten2d(st2(view(Z, :, :, b)))
    end

    # The localized field carries the complex order-0 field, and its mean is still `S0`.
    z2 = randn(ComplexF64, 32, 32)
    sf = SF.scattering_field(st2, z2; subsample=4)
    Test.@test eltype(sf.data) <: Complex
    root = first(ScatteringTransforms.PathGraph.order_range(st2.tree, 0))
    Test.@test Statistics.mean(SF.path_field(sf, root)) ≈ ScatteringTransforms.Coefficients.zeroth_order(st2(z2))
    Test.@test eltype(SF.scattering_field(st2, real(z2); subsample=4).data) <: Real
end

Test.@testset "Localized-field low-pass is a low-pass, not the tight-frame complement" begin
    # The bank's `averaging` is `√(max(0, 1-Σ|ψ|²))`, and every ψ̂ is analytic, so it is identically
    # 1 across the analytic complement. Subsampling a field smoothed with it aliases full-band
    # energy onto DC, which is why the localized transform uses its own Gaussian φ_J.
    N, J = 256, 4
    fb = ScatteringTransforms.FilterBanks.build_filter_bank1d(N, J; Q=1)
    neg = (N ÷ 2 + 2):N
    Test.@test all(≈(1.0), fb.averaging[neg])

    φ = ScatteringTransforms.Filters.gaussian_lowpass(Float64, (N,), J)
    Test.@test φ[1] == 1.0
    # Every frequency that folds onto DC under the default 2^(J-1) subsampling is negligible.
    s = 1 << (J - 1)
    Test.@test maximum(abs, φ[(N ÷ s) .* (1:(s - 1)) .+ 1]) < 1e-20
    Test.@test all(<=(0), diff(φ[1:(N ÷ 2)]))             # monotone decreasing on positives

    φ2 = ScatteringTransforms.Filters.gaussian_lowpass(Float32, (32, 32), 2)
    Test.@test eltype(φ2) == Float32
    Test.@test φ2[1, 1] == 1.0f0
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
    Test.@test any(ra.shape .!= 0)                # shape is implemented (was all-zero before)
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
    lc = ScatteringTransforms.Reductions.log_coefficients(ca)
    Test.@test all(isfinite, lc.logS1)
    Test.@test length(lc.logS1) == length(S1a)
end

Test.@testset "Non-mutating scattering(st,x) matches the in-place st(x) (1D/2D/3D)" begin
    # The autodiff-friendly forward must reproduce the production (mutating) transform exactly.
    let
        N, J = 64, 4
        x = randn(N)
        for spec in (SpectralBackends.DirectSumSpectralBackend(), SpectralBackends.FFTSpectralBackend())
            st = ScatteringTransforms.Scattering1D.ScatteringTransform1D(N, J; Q=2, max_order=2, spectral=spec)
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
    # Element-type genericity: Float32 in -> Float32 out through the non-mutating path.
    let
        st = ScatteringTransforms.Scattering1D.ScatteringTransform1D(Float32, 32, 3; Q=1, max_order=2)
        cs = ScatteringTransforms.ScatteringCore.scattering(st, randn(Float32, 32))
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

Test.@testset "Phase retrieval (Gerchberg–Saxton): reconstructed moduli match target" begin
    # The first-order band-pass moduli |x⋆ψ_λ| carry no information about the low-pass (smooth)
    # component, so reconstructing from the moduli alone leaves it an unconstrained null space — a
    # genuine under-determinacy that capped accuracy at ~12% error and flaked against a 0.15 bar.
    # Seeding the low-pass channel with the target's (a documented `reconstruct_phase` option) closes
    # that null space, and GS then recovers the field to machine precision for ANY init (rel ≈ 1e-4
    # across every seed tried, identically on Julia 1.11 and 1.12) — so the tight bar below holds for
    # all seeds and the RNG seed is only for reproducible output, not to pass a threshold.
    Random.seed!(123)
    N, J = 128, 6
    x = randn(N)
    st = ScatteringTransforms.Scattering1D.ScatteringTransform1D(N, J; Q=2, max_order=1)
    wt = ScatteringTransforms.Inverse.wavelet_transform(st, x)
    moduli = [abs.(w) for w in wt.wavelet]
    xhat = ScatteringTransforms.Inverse.reconstruct_phase(st, moduli; iters=400, init=randn(N),
                                                          seed_lowpass=wt.lowpass)
    wt2 = ScatteringTransforms.Inverse.wavelet_transform(st, xhat)
    num = sqrt(sum(sum(abs2, abs.(w2) .- m) for (w2, m) in zip(wt2.wavelet, moduli)))
    den = sqrt(sum(sum(abs2, m) for m in moduli))
    Test.@test num / den < 0.01      # near-exact recovery once the low-pass null space is removed
end

Test.@testset "Monogenic (Riesz) scattering: partition, tight frame, transforms" begin
    # Riesz multipliers partition unity off the DC bin: Σ_d |R_d(k)|² = 1, and vanish at DC.
    for dims in ((32,), (16, 16), (8, 8, 8))
        R = ScatteringTransforms.Monogenic.riesz_multipliers(dims, Float64)
        s = sum(abs2.(Rd) for Rd in R)
        Test.@test s[1] == 0                              # DC
        offdc = [s[i] for i in CartesianIndices(dims) if i != first(CartesianIndices(dims))]
        Test.@test maximum(abs.(offdc .- 1)) < 1e-12
    end

    # Isotropic bank is a tight frame: Σ_j |ψ̂_j|² + |φ̂|² ≡ 1.
    let
        fb = ScatteringTransforms.Monogenic.build_monogenic_bank((32, 32), 3; Q=1)
        s = abs2.(fb.averaging)
        for ψ in fb.wavelets
            s = s .+ abs2.(ψ)
        end
        Test.@test maximum(abs.(s .- 1)) < 1e-12
    end

    # 1D/2D/3D transforms run, finite, correct coefficient counts; Float32 preserved.
    for (dims, J, n) in (((128,), 5, 5), ((32, 32), 3, 3), ((16, 16, 16), 2, 2))
        st = ScatteringTransforms.Monogenic.MonogenicScattering(dims, J; Q=1, max_order=2)
        c = st(randn(dims...))
        Test.@test length(ScatteringTransforms.Coefficients.first_order(c)) == n
        Test.@test all(isfinite, ScatteringTransforms.Coefficients.first_order(c))
        Test.@test all(isfinite, ScatteringTransforms.Coefficients.second_order(c))
    end
    let
        stf = ScatteringTransforms.Monogenic.MonogenicScattering(Float32, (32, 32), 2; Q=1, max_order=2)
        cf = stf(randn(Float32, 32, 32))
        Test.@test eltype(ScatteringTransforms.Coefficients.first_order(cf)) == Float32
    end
end

Test.@testset "Monogenic periodized cascade converges to the undecimated one" begin
    # The monogenic amplitude band-passes by `D+1` filters per wavelet, so it is the most
    # transform-heavy cascade here and gains the most from producing each on its own grid. The
    # Riesz-weighted band-pass `R_d ψ_j` has to be periodized as one filter: periodizing `R_d` and
    # `ψ_j` separately and multiplying is a different function, and would show up as an error floor
    # that never reaches zero.
    Random.seed!(53)
    FB = SpectralBackends.FFTSpectralBackend()
    flat(c) = vcat(c.S0, vec(c.S1), vec(c.S2))

    for (dims, J) in (((128,), 4), ((32, 32), 3), ((16, 16, 16), 2))
        x = randn(dims...)
        mk(α; cache=true) = ScatteringTransforms.Monogenic.MonogenicScattering(
            Float64, dims, J; Q=1, max_order=2, spectral=FB, oversampling=α, cache=cache)
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
        Test.@test e0[2] > 0                        # and is real below it, not a silent no-op
        # `S0` is the plain field mean, so no decimation can move it.
        Test.@test mk(0)(x).S0 ≈ cref.S0

        # A computed bank evaluates into shared scratch; the Riesz products must still be right.
        Test.@test flat(mk(1; cache=false)(x)) == flat(mk(1)(x))
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
    # The Riesz vector recovers a plane wave's orientation (continuously, not quantized).
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

# Structured-grid spherical scattering (fast SHT) — completes the grid-support matrix.
include("test_spherical_sht.jl")

# The least-squares solver against LAPACK on dense operators, before any transform is involved.
include("test_lsmr.jl")

# Scattered / nonuniform planar scattering (NUFFT) — completes the Cartesian side of the matrix.
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
