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
Plots.default(fontfamily="Computer Modern", framestyle=:box, grid=false)

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

st = ScatteringTransform1D(N, 6; Q=1, max_order=2)
coeffs = st(signal)

# Check we have meaningful values
S1_max = Statistics.maximum(coeffs.S1)
S2_max = Statistics.maximum(coeffs.S2)
Base.println("  S1 max: $(S1_max)")
Base.println("  S2 max: $(S2_max)")

# Panel 1: Signal
p1 = Plots.plot(t, signal, title="(a) Input Signal: Pink Noise (1/f Spectrum)",
          xlabel="Time", ylabel="Amplitude", lw=1.0, color=:black,
          titlefontsize=11, guidefontsize=10, tickfontsize=9)

# Panel 2: S1 - First order (log scale for better visualization)
S1_display = coeffs.S1 .+ 1e-10  # Add small offset for log scale
p2 = Plots.bar(1:Base.length(coeffs.S1), S1_display, 
         title="(b) S₁: First-Order Scattering (Log Scale)",
         xlabel="Scale Index (j)", ylabel="Energy (log scale)",
         color=Base.range(0.3, 0.9, length=Base.length(coeffs.S1)),
         yscale=:log10,
         titlefontsize=11, guidefontsize=10, tickfontsize=9)

# Panel 3: S2 - Second order (log scale)
S2_display = coeffs.S2 .+ 1e-10
p3 = Plots.heatmap(Base.log10.(S2_display), 
             title="(c) S₂: Second-Order Scattering (Log₁₀ Scale)",
             xlabel="Scale j₂", ylabel="Scale j₁",
             color=:viridis, aspect_ratio=1,
             colorbar_title="log₁₀(Energy)",
             titlefontsize=11, guidefontsize=10, tickfontsize=9)

Plots.plot(p1, p2, p3, layout=(3,1), size=(800, 900), margin=8Plots.mm, dpi=150)
Plots.savefig(Base.joinpath(assets_dir, "1d_scattering_example.png"))
Base.println("  ✓ 1d_scattering_example.png")

# ============================================================================
# Figure 2: Filter Bank Visualization - INFORMATIVE VERSION
# ============================================================================
Base.println("Generating Figure 2: Filter Bank...")

filters = st.filter_bank
wavelet_responses = [Base.abs.(ψ) for ψ in filters.wavelets]
Nfreq = length(wavelet_responses[1])
freq_axis = 0:(Nfreq-1)

# Plot frequency responses with meaningful annotations
p = Plots.plot(title="(a) Morlet Wavelet Filter Bank: Frequency Tiling",
         xlabel="Frequency Index", ylabel="Magnitude (|ψ̂(ω)|)",
         size=(900, 500), dpi=150,
         titlefontsize=12, guidefontsize=11, tickfontsize=10,
         legendfontsize=9)

# Use color gradient to show scale progression
colors = Plots.cgrad(:viridis, Base.length(wavelet_responses), categorical=true)
for (i, response) in enumerate(wavelet_responses)
    # Only label every other to avoid clutter
    label = Base.isodd(i) ? "Scale j=$(i)" : ""
    Plots.plot!(freq_axis, response, label=label, lw=2.5, color=colors[i], alpha=0.8)
end

# Add annotation explaining what user should see
Plots.annotate!(0.5, 0.95, Plots.text("Key insight: Each wavelet covers a specific frequency band.\nLower j = higher frequency. Filter bank provides complete coverage.",
                          9, :left, :top))

Plots.savefig(Base.joinpath(assets_dir, "filter_bank.png"))
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

st2d = ScatteringTransform2D((M, M), 3; L=6, max_order=2)
coeffs_2d = st2d(image)

Base.println("  2D S1 range: [$(Statistics.minimum(coeffs_2d.S1)), $(Statistics.maximum(coeffs_2d.S1))]")
Base.println("  2D S2 max: $(Statistics.maximum(coeffs_2d.S2))")

# Normalize S2 for display
S2_2d = coeffs_2d.S2
if Statistics.maximum(S2_2d) > 0
    S2_2d = S2_2d ./ Statistics.maximum(S2_2d)
end

p1 = Plots.heatmap(image, title="(a) Input: Multi-Scale Fractal Texture", 
             color=:viridis, aspect_ratio=1, colorbar_title="Intensity",
             titlefontsize=11, guidefontsize=10, tickfontsize=9)

# Group S1 by scale for better visualization
J, L = st2d.filter_bank.J, st2d.filter_bank.L
S1_matrix = Base.reshape(coeffs_2d.S1, L, J)'

p2 = Plots.heatmap(S1_matrix, title="(b) S₁: Scale-Orientation Decomposition",
             xlabel="Orientation (θ)", ylabel="Scale (j)",
             color=:viridis, aspect_ratio=1, colorbar_title="Energy",
             xticks=1:L, yticks=1:J,
             titlefontsize=11, guidefontsize=10, tickfontsize=9)

# S2 with log scale for better visualization
S2_display = S2_2d .+ 1e-10
p3 = Plots.heatmap(Base.log10.(S2_display), title="(c) S₂: Second-Order (Log₁₀ Scale)",
             xlabel="Wavelet Index (j₂,θ₂)", ylabel="Wavelet Index (j₁,θ₁)",
             color=:viridis, aspect_ratio=1, colorbar_title="log₁₀(Energy)",
             titlefontsize=11, guidefontsize=10, tickfontsize=9)

Plots.plot(p1, p2, p3, layout=(1,3), size=(1400, 450), margin=10Plots.mm, dpi=150)
Plots.savefig(Base.joinpath(assets_dir, "2d_scattering_example.png"))
Base.println("  ✓ 2d_scattering_example.png")

# ============================================================================
# Figure 4: Zero-Allocation Performance Comparison - Real Benchmark
# ============================================================================
Base.println("Generating Figure 4: Performance Comparison...")

# Actually benchmark to get real numbers
sizes = [512, 1024, 2048, 4096, 8192]
naive_times = Float64[]
zero_alloc_times = Float64[]

for N in sizes
    st = ScatteringTransform1D(N, 6; Q=1, max_order=2)
    signal = Base.randn(N)
    
    # Naive: allocate every time
    t1 = Base.@elapsed for _ in 1:10 st(signal) end
    Base.push!(naive_times, t1 * 100)  # Convert to per-call ms
    
    # Zero-allocation: pre-allocated
    coeffs = ScatteringCoefficients1D(Base.length(st.filter_bank.wavelets), Float64; compute_S2=true)
    t2 = Base.@elapsed for _ in 1:10 scattering_transform!(coeffs, st, signal) end
    Base.push!(zero_alloc_times, t2 * 100)
end

p = Plots.plot(sizes, naive_times, label="Naive (Allocates)", marker=:circle, lw=2, markersize=8,
         title="(d) Performance: Zero-Allocation vs Naive",
         xlabel="Signal Size (N)", ylabel="Time per Call (ms)",
         xscale=:log2, yscale=:log10,
         legend=:topleft, framestyle=:box,
         titlefontsize=12, guidefontsize=11, tickfontsize=10,
         legendfontsize=10, dpi=150)
Plots.plot!(sizes, zero_alloc_times, label="Zero-Allocation", marker=:square, lw=2, markersize=8)

# Add speedup annotation
speedup = naive_times[end] / zero_alloc_times[end]
Plots.annotate!(0.7, 0.3, Plots.text("$(Base.round(speedup, digits=1))× faster at N=$(sizes[end])", 10, :left))

Plots.savefig(Base.joinpath(assets_dir, "performance_comparison.png"))
Base.println("  ✓ performance_comparison.png")

Base.println("\nAll assets generated in: $(assets_dir)")
