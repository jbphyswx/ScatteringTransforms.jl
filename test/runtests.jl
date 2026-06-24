module ScatteringTransformsTests

using Test: Test
using Aqua: Aqua
using ExplicitImports: ExplicitImports as EI

# Use the required import style: using X: X
using ScatteringTransforms: ScatteringTransforms
using FFTW: FFTW
using Statistics: Statistics

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
        :ScatteringTransformsCUDAExt,
        :ScatteringTransformsCairoMakieExt,
        :ScatteringTransformsKernelAbstractionsExt,
        :ScatteringTransformsNUFSHTExt,
    )
        ext = Base.get_extension(ScatteringTransforms, extname)
        ext === nothing && continue
        Test.@test (EI.check_no_implicit_imports(ext); true)
        Test.@test (EI.check_no_stale_explicit_imports(ext); true)
    end
end

Test.@testset "Type stability (concrete struct fields + inferred transforms)" begin
    # Every field of the transform struct must be concretely typed — in particular the FFT
    # plan fields (previously untyped `Any`, causing dynamic dispatch on every `mul!`) and the
    # 2D filter-bank field (previously `FilterBank2D{T}` with the matrix param dropped).
    N = 256
    J = 4
    st = ScatteringTransforms.ScatteringTransform1D(N, J; Q=1, max_order=2)
    for ft in fieldtypes(typeof(st))
        Test.@test isconcretetype(ft)
    end
    signal = randn(N)
    Test.@test (Test.@inferred st(signal); true)

    st2 = ScatteringTransforms.ScatteringTransform2D((64, 64), 3; L=4, max_order=2)
    for ft in fieldtypes(typeof(st2))
        Test.@test isconcretetype(ft)
    end
    image = randn(64, 64)
    Test.@test (Test.@inferred st2(image); true)

    # Element type is preserved end-to-end (Float32 in -> Float32 out).
    stf = ScatteringTransforms.ScatteringTransform1D(N, J; Q=1, max_order=2, T=Float32)
    cf = stf(Float32.(signal))
    Test.@test eltype(ScatteringTransforms.first_order(cf)) == Float32
    Test.@test ScatteringTransforms.zeroth_order(cf) isa Float32
end

Test.@testset "1D Morlet Wavelet Mathematical Properties" begin
    N = 512
    j = 2
    Q = 1
    r = sqrt(0.5)
    
    morlet = ScatteringTransforms.Morlet1D(N, j; Q=Q)
    ψ = ScatteringTransforms.frequency_response(morlet)
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

Test.@testset "1D Filter Bank Tests" begin
    N = 256
    J = 4
    bank = ScatteringTransforms.build_filter_bank1d(N, J; Q=1)
    
    Test.@test bank.J == J
    Test.@test bank.Q == 1
    Test.@test length(bank.wavelets) == J  # One wavelet per octave for Q=1
    Test.@test length(bank.averaging) == N
end

Test.@testset "1D Filter Bank with Q > 1 Fractional Center Frequencies" begin
    N = 256
    J = 3
    Q = 4
    bank = ScatteringTransforms.build_filter_bank1d(N, J; Q=Q)
    
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
    # TODO(Phase 2): this asserts the PROVISIONAL dense-matrix-with-zero-triangle storage
    # contract. The dense S2 matrix is a placeholder for the true scattering path structure;
    # it will be replaced by a path-indexed container and these assertions become
    # path-structure checks (counts per order, frequency-decreasing constraint). Until then
    # keep verifying the current contract so the suite stays green.
    Ny, Nx = 64, 64
    J = 3
    L = 4
    st = ScatteringTransforms.ScatteringTransform2D((Ny, Nx), J; L=L, max_order=2)
    image = randn(Ny, Nx)
    coeffs = st(image)
    S2 = ScatteringTransforms.second_order(coeffs)

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
    st = ScatteringTransforms.ScatteringTransform2D((Ny, Nx), J; L=L, max_order=2)
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
    S2 = ScatteringTransforms.second_order(coeffs)
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
    
    st = ScatteringTransforms.ScatteringTransform1D(N, J; Q=1, max_order=2)
    signal = randn(N)
    
    coeffs = st(signal)
    
    # Test S0 (average) - using accessor functions
    Test.@test isapprox(ScatteringTransforms.zeroth_order(coeffs), Statistics.mean(signal), atol=1e-10)
    
    # Test S1 (first order)
    S1 = ScatteringTransforms.first_order(coeffs)
    Test.@test length(S1) == J  # One coefficient per scale
    Test.@test all(S1 .>= 0)  # Modulus makes them non-negative
    
    # Test S2 (second order)
    S2 = ScatteringTransforms.second_order(coeffs)
    Test.@test size(S2) == (J, J)
    
    # Test S2 has meaningful values (not all near-zero due to filter bug)
    # With proper wavelet formulas, S2 should have measurable energy
    Test.@test Statistics.maximum(S2) > 1e-8
end

Test.@testset "Wavelet Center Frequencies" begin
    N = 512
    J = 6
    
    for j in 0:(J-1)
        morlet = ScatteringTransforms.Morlet1D(N, j; Q=1)
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
        morlet = ScatteringTransforms.Morlet1D(N, j; Q=1)
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
        morlet = ScatteringTransforms.Morlet1D(N, j; Q=1)
        ψ = ScatteringTransforms.frequency_response(morlet)
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
    bank = ScatteringTransforms.build_filter_bank1d(N, J; Q=1)
    
    # Each wavelet should have non-negligible energy
    # This is a regression test - previously bandwidth was wrong causing near-zero energy
    for (j, ψ) in enumerate(bank.wavelets)
        energy = Statistics.maximum(Base.abs.(ψ))
        # With correct formulas, all wavelets should have ~0.01-0.1 energy
        Test.@test energy > 0.001
    end
end

Test.@testset "Wavelet Shape is Gaussian" begin
    N = 512
    
    for j in 0:3
        morlet = ScatteringTransforms.Morlet1D(N, j; Q=1)
        ψ = ScatteringTransforms.frequency_response(morlet)
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
    morlet = ScatteringTransforms.Morlet2D((Ny, Nx), 2, π/4; L=8)
    
    resp = ScatteringTransforms.frequency_response(morlet)
    Test.@test size(resp) == (Ny, Nx)
    Test.@test eltype(resp) == ComplexF64
    
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
    st = ScatteringTransforms.ScatteringTransform2D((Ny, Nx), J; L=L, max_order=1)
    
    # k0 for j=1 is 3π / (4 * 2^1) = 3π/8 ≈ 1.178.
    k0 = 3π / 8
    theta_wave = π/4 # orient index 2 (1-based: index 3 since orient=0, 1, 2)
    
    # Generate grid
    X = reshape(range(0, 2π, length=Nx+1)[1:Nx], 1, Nx)
    Y = reshape(range(0, 2π, length=Ny+1)[1:Ny], Ny, 1)
    k_wave = round(k0 * Nx / 2π)
    
    plane_wave = cos.(k_wave .* (X .* cos(theta_wave) .+ Y .* sin(theta_wave)))
    
    coeffs = st(plane_wave)
    S1 = ScatteringTransforms.first_order(coeffs)
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
    
    bank = ScatteringTransforms.build_filter_bank2d(N, J; L=L)
    
    Test.@test bank.J == J
    Test.@test bank.L == L
    Test.@test length(bank.wavelets) == J * L
    Test.@test size(bank.averaging) == N
end

Test.@testset "2D Scattering Transform Tests" begin
    Ny, Nx = 64, 64
    J = 3
    L = 4
    
    st = ScatteringTransforms.ScatteringTransform2D((Ny, Nx), J; L=L, max_order=2)
    image = randn(Ny, Nx)
    
    coeffs = st(image)
    
    # Test S0 exists
    S0 = ScatteringTransforms.zeroth_order(coeffs)
    Test.@test isa(S0, Float64)
    
    # Test S1
    S1 = ScatteringTransforms.first_order(coeffs)
    Test.@test length(S1) == J * L
    
    # Test S2
    S2 = ScatteringTransforms.second_order(coeffs)
    Test.@test size(S2) == (J * L, J * L)
end

Test.@testset "Translation invariance (approximate)" begin
    N = 256
    J = 4
    
    st = ScatteringTransforms.ScatteringTransform1D(N, J; Q=1, max_order=1)
    
    # Create a periodic signal
    x = range(0, 2π, length=N+1)[1:N]
    signal = sin.(3x)
    
    # Shift by small amount
    shift = 10
    signal_shifted = circshift(signal, shift)
    
    coeffs1 = st(signal)
    coeffs2 = st(signal_shifted)
    
    # S1 should be approximately translation invariant
    S1_1 = ScatteringTransforms.first_order(coeffs1)
    S1_2 = ScatteringTransforms.first_order(coeffs2)
    rel_diff = abs.(S1_1 .- S1_2) ./ (S1_1 .+ 1e-10)
    Test.@test all(rel_diff .< 0.1)  # Within 10% due to edge effects
end

Test.@testset "1D localized field: mean equals averaged coefficient" begin
    # The localized (Mallat) field S_p x = (|U_p x| ⋆ φ_J) ↓ s. With s = 1 (no decimation) and
    # φ̂(0) = 1, the spatial mean of each path's field must equal that path's globally-averaged
    # coefficient — the two outputs are consistent by construction.
    PG = ScatteringTransforms.PathGraph
    N = 256
    J = 4
    st = ScatteringTransforms.ScatteringTransform1D(N, J; Q=1, max_order=2)
    signal = randn(N)
    coeffs = st(signal)
    sf = ScatteringTransforms.scattering_field(st, signal; subsample=1)
    tree = st.tree

    root = first(PG.order_range(tree, 0))
    Test.@test isapprox(Statistics.mean(ScatteringTransforms.path_field(sf, root)),
                        ScatteringTransforms.zeroth_order(coeffs); atol=1e-10)

    S1 = ScatteringTransforms.first_order(coeffs)
    for p in PG.order_range(tree, 1)
        j = PG.path_indices(tree, p)[1]
        Test.@test isapprox(Statistics.mean(ScatteringTransforms.path_field(sf, p)), S1[j]; atol=1e-8)
    end

    S2 = ScatteringTransforms.second_order(coeffs)
    for p in PG.order_range(tree, 2)
        idx = PG.path_indices(tree, p)
        j1, j2 = idx[1], idx[2]
        Test.@test isapprox(Statistics.mean(ScatteringTransforms.path_field(sf, p)), S2[j1, j2]; atol=1e-8)
    end

    # Decimation: subsample=8 -> field length N/8, all finite. (The decimated mean is a
    # finite-sample estimate of the full mean, not exact, so we don't assert equality here;
    # the subsample=1 case above is the exact consistency check.)
    sf8 = ScatteringTransforms.scattering_field(st, signal; subsample=8)
    Test.@test size(sf8.data, 1) == N ÷ 8
    Test.@test all(isfinite, sf8.data)
end

Test.@testset "2D localized field: mean equals averaged coefficient" begin
    PG = ScatteringTransforms.PathGraph
    Ny, Nx = 64, 64
    J = 3
    L = 4
    st = ScatteringTransforms.ScatteringTransform2D((Ny, Nx), J; L=L, max_order=2)
    image = randn(Ny, Nx)
    coeffs = st(image)
    sf = ScatteringTransforms.scattering_field(st, image; subsample=1)
    tree = st.tree

    root = first(PG.order_range(tree, 0))
    Test.@test isapprox(Statistics.mean(ScatteringTransforms.path_field(sf, root)),
                        ScatteringTransforms.zeroth_order(coeffs); atol=1e-10)
    S1 = ScatteringTransforms.first_order(coeffs)
    for p in PG.order_range(tree, 1)
        j = PG.path_indices(tree, p)[1]
        Test.@test isapprox(Statistics.mean(ScatteringTransforms.path_field(sf, p)), S1[j]; atol=1e-8)
    end
    S2 = ScatteringTransforms.second_order(coeffs)
    for p in PG.order_range(tree, 2)
        idx = PG.path_indices(tree, p)
        j1, j2 = idx[1], idx[2]
        Test.@test isapprox(Statistics.mean(ScatteringTransforms.path_field(sf, p)), S2[j1, j2]; atol=1e-8)
    end

    sf2 = ScatteringTransforms.scattering_field(st, image; subsample=2)
    Test.@test size(sf2.data) == (Ny ÷ 2, Nx ÷ 2, PG.npaths(tree))
    Test.@test all(isfinite, sf2.data)
end

Test.@testset "compute_shape_sparsity: shape reduction (TODO Phase 6)" begin
    # `compute_shape_sparsity` currently returns `shape` as all zeros — the anisotropy/shape
    # reduction (RWST / Cheng-Ménard s22) is not yet implemented. Recorded as broken so the
    # intent to implement it in Phase 6 is tracked and flips green when done.
    Ny, Nx = 64, 64
    J = 3
    L = 4
    st = ScatteringTransforms.ScatteringTransform2D((Ny, Nx), J; L=L, max_order=2)
    image = randn(Ny, Nx)
    coeffs = st(image)
    S1 = ScatteringTransforms.first_order(coeffs)
    S2 = ScatteringTransforms.second_order(coeffs)
    res = ScatteringTransforms.Scattering2D.compute_shape_sparsity(S1, S2, st.filter_bank.meta)
    Test.@test_broken any(res.shape .!= 0)
end

end # module
