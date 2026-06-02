"""
    generate_figures_simple.jl

Generate documentation figures using proper import policy.
"""

using ScatteringTransforms: ScatteringTransforms, ScatteringTransform1D, ScatteringTransform2D
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
t = Base.range(0, 4π, length=N)

# Create pink noise (1/f spectrum) for multi-scale structure
function pink_noise(n)
    white = Base.randn(n)
    fft_white = FFTW.fft(white)
    freqs = [0; 1:(n÷2-1); (n÷2); (n÷2-1):-1:1]
    pink_fft = fft_white ./ Base.sqrt.(freqs .+ 0.1)
    return Base.real(FFTW.ifft(pink_fft))
end

signal = pink_noise(N) .+ 0.5 .* Base.sin.(5*t)

st = ScatteringTransform1D(N, 6; Q=1, max_order=2)
coeffs = st(signal)

S1_max = Statistics.maximum(coeffs.S1)
S2_max = Statistics.maximum(coeffs.S2)
Base.println("  S1 max: $(S1_max)")
Base.println("  S2 max: $(S2_max)")

# Panel 1: Signal
p1 = Plots.plot(t, signal, title="(a) Input Signal: Pink Noise",
          xlabel="Time", ylabel="Amplitude", lw=1.0, color=:black,
          titlefontsize=11, guidefontsize=10, tickfontsize=9)

# Panel 2: S1 - First order with log scale
S1_display = coeffs.S1 .+ 1e-10
p2 = Plots.bar(1:Base.length(coeffs.S1), S1_display, 
         title="(b) S₁: First-Order Scattering",
         xlabel="Scale Index (j)", ylabel="Energy",
         color=:steelblue,
         yscale=:log10,
         titlefontsize=11, guidefontsize=10, tickfontsize=9)

# Panel 3: S2 heatmap
S2_display = coeffs.S2 .+ 1e-10
p3 = Plots.heatmap(Base.log10.(S2_display), 
             title="(c) S₂: Second-Order (Log Scale)",
             xlabel="Scale j₂", ylabel="Scale j₁",
             color=:viridis, aspect_ratio=1,
             colorbar_title="log₁₀(Energy)",
             titlefontsize=11, guidefontsize=10, tickfontsize=9)

Plots.plot(p1, p2, p3, layout=(3,1), size=(800, 900), margin=8Plots.mm, dpi=150)
Plots.savefig(Base.joinpath(assets_dir, "1d_scattering_example.png"))
Base.println("  ✓ 1d_scattering_example.png")

# ============================================================================
# Figure 2: 2D Scattering
# ============================================================================
Base.println("Generating Figure 2: 2D Scattering Example...")

M = 128
x = Base.range(0, 8π, length=M)
y = Base.range(0, 8π, length=M)

image = Base.zeros(M, M)
for i in 1:M, j in 1:M
    xi, yi = x[i], y[j]
    image[i,j] += Base.sin(xi) * Base.cos(yi)
    image[i,j] += 0.5 * Base.sin(2*xi) * Base.cos(2*yi)
    image[i,j] += 0.25 * Base.sin(4*xi) * Base.cos(4*yi)
end
image = image ./ Statistics.maximum(Base.abs.(image))
image .+= 0.3 .* Base.randn(M, M)

st2d = ScatteringTransform2D((M, M), 3; L=6, max_order=1)
coeffs_2d = st2d(image)

Base.println("  2D S1 range: [$(Statistics.minimum(coeffs_2d.S1)), $(Statistics.maximum(coeffs_2d.S1))]")

p1 = Plots.heatmap(image, title="(a) Input: Multi-Scale Texture", 
             color=:viridis, aspect_ratio=1,
             titlefontsize=11, guidefontsize=10, tickfontsize=9)

J, L = st2d.filter_bank.J, st2d.filter_bank.L
S1_matrix = Base.reshape(coeffs_2d.S1, L, J)'

p2 = Plots.heatmap(S1_matrix, title="(b) S₁: Scale-Orientation",
             xlabel="Orientation", ylabel="Scale",
             color=:viridis, aspect_ratio=1,
             xticks=1:L, yticks=1:J,
             titlefontsize=11, guidefontsize=10, tickfontsize=9)

# Filter bank visualization for 2D
filters_2d = st2d.filter_bank
sample_filter = Base.abs.(filters_2d.wavelets[1])
p3 = Plots.heatmap(sample_filter, title="(c) Sample Wavelet (j=1, θ=1)",
             color=:viridis, aspect_ratio=1,
             titlefontsize=11, guidefontsize=10, tickfontsize=9)

Plots.plot(p1, p2, p3, layout=(1,3), size=(1200, 400), margin=10Plots.mm, dpi=150)
Plots.savefig(Base.joinpath(assets_dir, "2d_scattering_example.png"))
Base.println("  ✓ 2d_scattering_example.png")

# ============================================================================
# Figure 3: Informative Filter Bank Visualization - HIGH RES
# ============================================================================
Base.println("Generating Figure 3: Filter Bank...")

N_viz = 1024  # Higher resolution for smoother curves
st_viz = ScatteringTransform1D(N_viz, 6; Q=1, max_order=1)
filters_viz = st_viz.filter_bank

# Get positive frequencies only for cleaner visualization
freqs = FFTW.fftfreq(N_viz, 1.0)[1:N_viz÷2+1]  # Normalized frequency [0, 0.5]

p_fb = Plots.plot(title="Morlet Wavelet Filter Bank (Q=1, J=6)",
         xlabel="Normalized Frequency (cycles/sample)", 
         ylabel="Magnitude (normalized)",
         xlims=(0, 0.55),
         ylims=(0, 1.05),
         size=(900, 550),
         dpi=150,
         titlefontsize=13, guidefontsize=11, tickfontsize=10,
         legend=:outerright, legendfontsize=9,
         grid=true, gridalpha=0.3,
         rightmargin=15Plots.mm)

# Plot each wavelet with distinct colors
colors_fb = [:darkblue, :blue, :teal, :green, :orange, :red]
for (j, ψ) in enumerate(filters_viz.wavelets)
    ψ_abs = Base.abs.(ψ[1:N_viz÷2+1])
    ψ_norm = ψ_abs / Base.maximum(ψ_abs .+ 1e-10)
    center_freq = 0.5 / 2^(j-1)
    Plots.plot!(freqs, ψ_norm, label="j=$j (f₀=$(round(center_freq, digits=4)))", 
                lw=3, color=colors_fb[j], fill=true, fillalpha=0.15)
end

Plots.savefig(Base.joinpath(assets_dir, "filter_bank.png"))
Base.println("  ✓ filter_bank.png")

Base.println("\nAll assets generated!")
