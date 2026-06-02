using ScatteringTransforms: ScatteringTransforms, ScatteringTransform1D
using Plots: Plots
using FFTW: FFTW

assets_dir = "/home/jbenjami/Code/jbphyswx/ScatteringTransforms.jl/docs/src/assets"

println("Generating filter bank figure...")

N = 512
st = ScatteringTransform1D(N, 6; Q=1, max_order=1)
filters = st.filter_bank

# Get positive frequency axis (excluding negative Nyquist at end)
freqs = FFTW.fftfreq(N, 1.0)[1:N÷2]  # Changed from N÷2+1 to N÷2

# Create the plot with minimal styling
Plots.gr()

p = Plots.plot(
    title = "Wavelet Filter Bank (Q=1, J=6)",
    xlabel = "Normalized Frequency (cycles/sample)",
    ylabel = "Magnitude",
    xlim = (0.005, 0.52),  # Avoid showing cut-off at Nyquist for j=1
    ylim = (0, 1.05),
    size = (900, 500),
    dpi = 150,
    titlefont = 13,
    guidefont = 11,
    tickfont = 10,
    legend = :topright,
    legendfont = 9,
    framestyle = :box,
    grid = true
)


# Plot each wavelet as clean line without fill
colors = [:purple, :blue, :teal, :green, :orange, :red]
for (j, ψ) in enumerate(filters.wavelets)
    ψ_abs = abs.(ψ[1:N÷2])
    ψ_norm = ψ_abs / maximum(ψ_abs .+ 1e-10)
    freq_center = 0.5 / 2^(j-1)
    
    Plots.plot!(freqs, ψ_norm, 
        label = "j=$j (fc=$(round(freq_center, digits=3)))",
        color = colors[j],
        linewidth = 2
    )
end

Plots.savefig("$assets_dir/filter_bank.png")
println("Done! Saved to $assets_dir/filter_bank.png")
