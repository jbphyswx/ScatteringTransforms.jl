module ScatteringTransformsTests

using Test: Test

# Use the required import style: using X: X
using ScatteringTransforms: ScatteringTransforms
using FFTW: FFTW
using Statistics: Statistics

# Run Aqua quality tests first
Test.@testset "Aqua.jl quality tests" begin
    using Aqua: Aqua
    Aqua.test_all(ScatteringTransforms)
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
    using FFTW: FFTW
    
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
    using FFTW: FFTW
    
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

end # module
