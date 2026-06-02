"""
    generate_figures.jl

Generate static figures for documentation.
Run from docs/ directory: julia --project=.. generate_assets/generate_figures.jl
"""

using ScatteringTransforms: ScatteringTransforms
using Plots: Plots
using Statistics: Statistics
using FFTW: FFTW

Base.println("Generating documentation assets...")

# Professional plot defaults
Plots.default(fontfamily="sans-serif", framestyle=:box, grid=false)

# Create output directory
assets_dir = Base.joinpath(Base.@__DIR__, "..", "src", "assets")
Base.mkpath(assets_dir)

# ============================================================================
# Figure 1: 1D Signal and Scattering Coefficients
# ============================================================================
Base.println("Generating Figure 1: 1D Scattering Example...")

N = 2048
t = range(0, 4π, length=N)

# Create a signal with actual multi-scale structure
# Pink noise (1/f spectrum) has structure at all scales
function pink_noise(n)
    white = Base.randn(n)
    fft_white = FFTW.fft(white)
    # Frequency bins for real FFT: [0, 1, 2, ..., n/2, n/2-1, ..., 1]
    freqs = [0; 1:(n÷2-1); (n÷2); (n÷2-1):-1:1]
    pink_fft = fft_white ./ Base.sqrt.(freqs .+ 0.1)  # Add 0.1 to avoid div by zero at DC
    return Base.real(FFTW.ifft(pink_fft))
end

signal = pink_noise(N) .+ 0.5 .* Base.sin.(5*t)  # Pink noise + low freq oscillation

st = ScatteringTransforms.ScatteringTransform1D(N, 6; Q=1, max_order=2)
coeffs = st(signal)

# Check we have meaningful values
S1_max = Statistics.maximum(coeffs.S1)
S2_max = Statistics.maximum(coeffs.S2)
Base.println("  S1 max: $(S1_max)")
Base.println("  S2 max: $(S2_max)")

# Panel 1: Signal
p1 = Plots.plot(t, signal, title="(a) Input Signal: Pink Noise + Oscillation",
          xlabel="Time", ylabel="Amplitude", lw=1.0, color=:black,
          legend=false,
          titlefontsize=11, guidefontsize=10, tickfontsize=9,
          left_margin=8Plots.mm, bottom_margin=6Plots.mm,
          right_margin=2Plots.mm, top_margin=4Plots.mm)

# Panel 2: S1 - First order (log scale for better visualization)
S1_display = coeffs.S1 .+ 1e-10  # Add small offset for log scale
p2 = Plots.bar(1:Base.length(coeffs.S1), S1_display, 
         title="(b) S₁: First-Order (Log Scale)",
         xlabel="Scale Index (j)", ylabel="Energy",
         color=Plots.cgrad(:viridis, Base.length(coeffs.S1), categorical=true),
         yscale=:log10,
         legend=false,
         titlefontsize=11, guidefontsize=10, tickfontsize=9,
         left_margin=8Plots.mm, bottom_margin=8Plots.mm,
         right_margin=2Plots.mm, top_margin=4Plots.mm)

# Panel 3: S2 - Second-order (lower triangle set to NaN to preserve dynamic range and avoid -10 floor)
S2_display = [j2 > j1 ? coeffs.S2[j1, j2] : NaN for j1 in 1:6, j2 in 1:6]
S2_log = [isnan(x) || isinf(x) || x <= 0 ? NaN : log10(x) for x in S2_display]
p3 = Plots.heatmap(S2_log, 
             title="(c) S₂: Second-Order (Log10)",
             xlabel="Scale j2", ylabel="Scale j1",
             color=:viridis, aspect_ratio=1,
             xticks=1:6, yticks=1:6,
             titlefontsize=11, guidefontsize=10, tickfontsize=9,
             left_margin=6Plots.mm, bottom_margin=8Plots.mm,
             right_margin=8Plots.mm, top_margin=4Plots.mm)

# Condensed, highly professional layout putting signal wide on top, and bar/heatmap side-by-side below
l = Plots.@layout [
    a{0.4h}
    [b c]
]
p_combined = Plots.plot(p1, p2, p3, layout=l, size=(900, 600), margin=2Plots.mm, dpi=300)
Plots.savefig(p_combined, Base.joinpath(assets_dir, "1d_scattering_example.png"))
Base.println("  ✓ 1d_scattering_example.png")

# ============================================================================
# Figure 2: Filter Bank Visualization - INFORMATIVE VERSION
# ============================================================================
Base.println("Generating Figure 2: Filter Bank...")

filters = st.filter_bank
wavelet_responses = [Base.abs.(ψ) for ψ in filters.wavelets]
Nfreq = length(wavelet_responses[1])
Nhalf = Nfreq ÷ 2
freq_axis = range(0.0, 0.5, length=Nhalf)

# Plot frequency responses with meaningful annotations
p_fb = Plots.plot(title="Morlet Wavelet Filter Bank: Frequency Tiling",
         xlabel="Normalized Frequency (cycles/sample)", ylabel="Filter Magnitude",
         size=(900, 320), dpi=300,
         legend=:outerright,
         titlefontsize=12, guidefontsize=11, tickfontsize=10,
         legendfontsize=9,
         left_margin=8Plots.mm,
         bottom_margin=8Plots.mm,
         right_margin=2Plots.mm,
         top_margin=4Plots.mm)

# 1. Plot the scaling function (low-pass filter)
ϕ_abs = Base.abs.(filters.averaging[1:Nhalf])
Plots.plot!(p_fb, freq_axis, ϕ_abs, label="Lowpass ϕ", lw=3.0, color=:black, linestyle=:dash)

# 2. Use premium, high-contrast colors for scales (no yellow/light-green)
colors = [:darkblue, :royalblue, :teal, :forestgreen, :darkorange, :crimson]
for (i, response) in enumerate(wavelet_responses)
    label = "Scale j=$(i-1)"
    response_half = response[1:Nhalf]
    Plots.plot!(p_fb, freq_axis, response_half, label=label, lw=2.5, color=colors[i], alpha=0.8)
end

Plots.savefig(p_fb, Base.joinpath(assets_dir, "filter_bank.png"))
Base.println("  ✓ filter_bank.png")

# ============================================================================
# Figure 3: 2D Scattering Example - Better Test Image
# ============================================================================
Base.println("Generating Figure 3: 2D Scattering Example...")

M = 128
x = range(0, 8π, length=M)
y = range(0, 8π, length=M)

# Create 2D fractal-like texture using multiple sine waves at different scales
image = Base.zeros(M, M)
for i in 1:M, j in 1:M
    xi, yi = x[i], y[j]
    # Multiple octaves of noise-like structure
    image[i,j] += Base.sin(xi) * Base.cos(yi)  # Base frequency
    image[i,j] += 0.5 * Base.sin(2*xi) * Base.cos(2*yi)  # Octave 2
    image[i,j] += 0.25 * Base.sin(4*xi) * Base.cos(4*yi)  # Octave 3
    image[i,j] += 0.125 * Base.sin(8*xi) * Base.cos(8*yi)  # Octave 4
end
# Normalize
image = image ./ Statistics.maximum(Base.abs.(image))
# Add pink noise
image .+= 0.3 .* Base.randn(M, M)

st2d = ScatteringTransforms.ScatteringTransform2D((M, M), 3; L=6, max_order=2)
coeffs_2d = st2d(image)

Base.println("  2D S1 range: [$(Statistics.minimum(coeffs_2d.S1)), $(Statistics.maximum(coeffs_2d.S1))]")
Base.println("  2D S2 max: $(Statistics.maximum(coeffs_2d.S2))")

# Normalize S2 for display
S2_2d = coeffs_2d.S2
if Statistics.maximum(S2_2d) > 0
    S2_2d = S2_2d ./ Statistics.maximum(S2_2d)
end

p1 = Plots.heatmap(image, title="(a) Input Texture", 
             color=:viridis, aspect_ratio=1,
             titlefontsize=11, guidefontsize=10, tickfontsize=9,
             left_margin=5Plots.mm, bottom_margin=6Plots.mm,
             right_margin=8Plots.mm, top_margin=4Plots.mm)

# Group S1 by scale for better visualization
J, L = st2d.filter_bank.J, st2d.filter_bank.L
S1_matrix = Base.reshape(coeffs_2d.S1, L, J)'

p2 = Plots.heatmap(S1_matrix, title="(b) S₁: Scale-Orientation",
             xlabel="Orientation (θ)", ylabel="Scale (j)",
             color=:viridis, aspect_ratio=1,
             xticks=1:L, yticks=1:J,
             titlefontsize=11, guidefontsize=10, tickfontsize=9,
             left_margin=5Plots.mm, bottom_margin=6Plots.mm,
             right_margin=8Plots.mm, top_margin=4Plots.mm)

# S2 with log scale, set lower triangle/diagonal to NaN to preserve dynamic range
n_w = size(S2_2d, 1)
S2_2d_display = [j2 > j1 ? S2_2d[j1, j2] : NaN for j1 in 1:n_w, j2 in 1:n_w]
S2_2d_log = [isnan(x) || isinf(x) || x <= 0 ? NaN : log10(x) for x in S2_2d_display]

p3 = Plots.heatmap(S2_2d_log, title="(c) S₂: Second-Order (Log10)",
             xlabel="Wavelet Index (j₂, θ₂)", ylabel="Wavelet Index (j₁, θ₁)",
             color=:viridis, aspect_ratio=1,
             xticks=[1, 7, 13, 18], yticks=[1, 7, 13, 18],
             titlefontsize=11, guidefontsize=10, tickfontsize=9,
             left_margin=5Plots.mm, bottom_margin=6Plots.mm,
             right_margin=8Plots.mm, top_margin=4Plots.mm)

p_2d = Plots.plot(p1, p2, p3, layout=(1,3), size=(1050, 275), margin=2Plots.mm, dpi=300)
Plots.savefig(p_2d, Base.joinpath(assets_dir, "2d_scattering_example.png"))
Base.println("  ✓ 2d_scattering_example.png")

# ============================================================================
# Figure 4: Zero-Allocation Performance Comparison - Real Benchmark
# ============================================================================
Base.println("Generating Figure 4: Performance Comparison...")

# Warm up compilation to get accurate, stable benchmark results
warmup_st = ScatteringTransforms.ScatteringTransform1D(128, 4; Q=1, max_order=2)
warmup_sig = Base.randn(128)
warmup_st(warmup_sig)
warmup_coeffs = ScatteringTransforms.ScatteringCoefficients1D(Base.length(warmup_st.filter_bank.wavelets), Float64; compute_S2=true)
ScatteringTransforms.scattering_transform!(warmup_coeffs, warmup_st, warmup_sig)

sizes = [512, 1024, 2048, 4096, 8192]
naive_times = Float64[]
zero_alloc_times = Float64[]

for N in sizes
    local st, signal, coeffs
    st = ScatteringTransforms.ScatteringTransform1D(N, 6; Q=1, max_order=2)
    signal = Base.randn(N)
    
    # Garbage collect to ensure identical starting conditions
    GC.gc()
    
    # Naive: allocate every time (run 100 iterations)
    t1 = Base.@elapsed for _ in 1:100 st(signal) end
    Base.push!(naive_times, t1 * 10)  # Convert to per-call ms: (t1 / 100) * 1000
    
    # Garbage collect
    GC.gc()
    
    # Zero-allocation: pre-allocated (run 100 iterations)
    coeffs = ScatteringTransforms.ScatteringCoefficients1D(Base.length(st.filter_bank.wavelets), Float64; compute_S2=true)
    t2 = Base.@elapsed for _ in 1:100 ScatteringTransforms.scattering_transform!(coeffs, st, signal) end
    Base.push!(zero_alloc_times, t2 * 10)  # Convert to per-call ms: (t2 / 100) * 1000
end

p_perf = Plots.plot(sizes, naive_times, label="Naive (Allocates)", marker=:circle, lw=2, markersize=8,
         title="Performance: Zero-Allocation vs Naive",
         xlabel="Signal Size (N)", ylabel="Time per Call (ms)",
         xscale=:log2, yscale=:log10,
         legend=:topleft, framestyle=:box,
         titlefontsize=12, guidefontsize=11, tickfontsize=10,
         legendfontsize=10, dpi=300,
         size=(800, 480),
         left_margin=8Plots.mm, bottom_margin=8Plots.mm,
         right_margin=3Plots.mm, top_margin=4Plots.mm)

# Use dynamic speedup in legend label to avoid overlapping text annotation
speedup = naive_times[end] / zero_alloc_times[end]
speedup_str = "$(Base.round(speedup, digits=1))x Speedup"
Plots.plot!(p_perf, sizes, zero_alloc_times, label="Zero-Allocation ($(speedup_str))", marker=:square, lw=2, markersize=8)

Plots.savefig(p_perf, Base.joinpath(assets_dir, "performance_comparison.png"))
Base.println("  ✓ performance_comparison.png")

Base.println("\nAll assets generated in: $(assets_dir)")
