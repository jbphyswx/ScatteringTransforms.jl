module ScatteringTransformsCairoMakieExt

using CairoMakie: CairoMakie
using ScatteringTransforms: ScatteringTransforms

# Implement plot_filter_bank for 1D Filter Banks
function ScatteringTransforms.plot_filter_bank(fb::ScatteringTransforms.FilterBank1D)
    N = length(fb.averaging)
    Nhalf = N ÷ 2
    freq_axis = collect(range(0.0, 0.5, length=Nhalf))
    
    fig = CairoMakie.Figure(size=(1000, 380))
    
    # Panel (a): Filter frequency tiling
    ax_a = CairoMakie.Axis(fig[1, 1],
        title="(a) Morlet filter bank: frequency tiling",
        xlabel="Normalized frequency (cycles/sample)",
        ylabel="Filter magnitude",
        limits=((0.0, 0.52), (0.0, 1.05))
    )
    
    # Plot low-pass filter
    CairoMakie.lines!(ax_a, freq_axis, abs.(fb.averaging[1:Nhalf]),
        label="Low-pass ϕ", linewidth=2.5, linestyle=:dash, color=:black)
        
    # Plot wavelets
    colors = CairoMakie.distinguishable_colors(length(fb.wavelets) + 1)
    for (i, ψ) in enumerate(fb.wavelets)
        # Exclude the first color if it is white/black for better visibility
        col = colors[i+1]
        CairoMakie.lines!(ax_a, freq_axis, abs.(ψ[1:Nhalf]),
            label="j=$(fb.meta[i].scale), q=$(fb.meta[i].q)", linewidth=2.0, color=col)
    end
    CairoMakie.axislegend(ax_a, position=:rt, labelsize=9)
    
    # Panel (b): Littlewood-Paley sum
    lp_sum = abs2.(fb.averaging[1:Nhalf]) .+ sum(abs2.(ψ[1:Nhalf]) for ψ in fb.wavelets)
    ax_b = CairoMakie.Axis(fig[1, 2],
        title="(b) Littlewood-Paley tiling: ‖ϕ‖² + Σ‖ψⱼ‖² ≈ 1",
        xlabel="Normalized frequency",
        ylabel="Energy sum",
        limits=((0.0, 0.52), (0.0, 1.3))
    )
    
    CairoMakie.lines!(ax_b, freq_axis, lp_sum, color=:black, linewidth=2.0)
    CairoMakie.hlines!(ax_b, [1.0], color=:red, linestyle=:dash, linewidth=1.5)
    
    return fig
end

# Implement plot_filter_bank for 2D Filter Banks
function ScatteringTransforms.plot_filter_bank(fb::ScatteringTransforms.FilterBank2D)
    Ny, Nx = size(fb.averaging)
    
    # Littlewood-Paley sum matrix
    lp_sum = abs2.(fb.averaging) .+ sum(abs2.(ψ) for ψ in fb.wavelets)
    
    # Pick a sample wavelet response (e.g. j=1, θ=π/4)
    # Wavelets are in fb.wavelets, let's pick one around the middle scale and orientation
    sample_idx = min(3, length(fb.wavelets))
    ψ_sample = fb.wavelets[sample_idx]
    
    fig = CairoMakie.Figure(size=(1000, 420))
    
    # Panel (a): 2D energy sum
    ax_a = CairoMakie.Axis(fig[1, 1],
        title="(a) 2D Littlewood-Paley Energy Sum",
        xlabel="kx index",
        ylabel="ky index",
        aspect=CairoMakie.AxisAspect(1)
    )
    hm_a = CairoMakie.heatmap!(ax_a, lp_sum, colormap=:viridis)
    CairoMakie.Colorbar(fig[1, 2], hm_a)
    
    # Panel (b): Sample Wavelet
    ax_b = CairoMakie.Axis(fig[1, 3],
        title="(b) Sample Wavelet magnitude (j=$(fb.meta[sample_idx].scale), θ=$(round(fb.meta[sample_idx].theta, digits=2)))",
        xlabel="kx index",
        ylabel="ky index",
        aspect=CairoMakie.AxisAspect(1)
    )
    hm_b = CairoMakie.heatmap!(ax_b, abs.(ψ_sample), colormap=:plasma)
    CairoMakie.Colorbar(fig[1, 4], hm_b)
    
    return fig
end

# Implement plot_coefficients for 1D Coefficients
function ScatteringTransforms.plot_coefficients(c::ScatteringTransforms.ScatteringCoefficients1D; signal=nothing)
    has_sig = !isnothing(signal)
    
    J_eff = c.n_wavelets
    scale_labels = ["2^$(j-1)" for j in 1:J_eff]
    
    if has_sig
        fig = CairoMakie.Figure(size=(900, 850))
        
        # Panel (a): Input Signal
        ax_a = CairoMakie.Axis(fig[1, 1],
            title="(a) Input Signal",
            xlabel="Sample Index",
            ylabel="Amplitude"
        )
        CairoMakie.lines!(ax_a, signal, color=:black, linewidth=1.0)
        
        # Panel (b): S1 coefficients
        ax_b = CairoMakie.Axis(fig[2, 1],
            title="(b) S₁: First-Order Scattering Coefficients",
            xlabel="Scale Index (j)",
            ylabel="Energy (log₁₀ scale)",
            xticks=(1:J_eff, scale_labels),
            yscale=log10
        )
        CairoMakie.barplot!(ax_b, 1:J_eff, c.S1 .+ 1e-10, color=:steelblue)
        
        # Panel (c): S2 heatmap
        ax_c = CairoMakie.Axis(fig[3, 1],
            title="(c) S₂: Second-Order Cross-Scale Coupling",
            xlabel="Finer scale j₂",
            ylabel="Coarser scale j₁",
            xticks=(1:J_eff, scale_labels),
            yticks=(1:J_eff, scale_labels),
            aspect=CairoMakie.AxisAspect(1)
        )
        # Zero out diagonal / lower triangle for plotting
        S2_disp = copy(c.S2)
        S2_disp[S2_disp .== 0] .= NaN  # blank invalid/empty scattering paths
        for i in 1:J_eff, j in 1:J_eff
            if j <= i
                S2_disp[i, j] = NaN
            end
        end
        hm = CairoMakie.heatmap!(ax_c, S2_disp, colormap=:plasma, nan_color=:transparent)
        CairoMakie.Colorbar(fig[3, 2], hm, label="Energy")
    else
        fig = CairoMakie.Figure(size=(900, 420))
        
        # Panel (a): S1 coefficients
        ax_a = CairoMakie.Axis(fig[1, 1],
            title="(a) S₁: First-Order Scattering Coefficients",
            xlabel="Scale Index (j)",
            ylabel="Energy (log₁₀ scale)",
            xticks=(1:J_eff, scale_labels),
            yscale=log10
        )
        CairoMakie.barplot!(ax_a, 1:J_eff, c.S1 .+ 1e-10, color=:steelblue)
        
        # Panel (b): S2 heatmap
        ax_b = CairoMakie.Axis(fig[1, 2],
            title="(b) S₂: Second-Order Cross-Scale Coupling",
            xlabel="Finer scale j₂",
            ylabel="Coarser scale j₁",
            xticks=(1:J_eff, scale_labels),
            yticks=(1:J_eff, scale_labels),
            aspect=CairoMakie.AxisAspect(1)
        )
        S2_disp = copy(c.S2)
        S2_disp[S2_disp .== 0] .= NaN  # blank invalid/empty scattering paths
        for i in 1:J_eff, j in 1:J_eff
            if j <= i
                S2_disp[i, j] = NaN
            end
        end
        hm = CairoMakie.heatmap!(ax_b, S2_disp, colormap=:plasma, nan_color=:transparent)
        CairoMakie.Colorbar(fig[1, 3], hm, label="Energy")
    end
    
    return fig
end

# Implement plot_coefficients for 2D Coefficients
function ScatteringTransforms.plot_coefficients(c::ScatteringTransforms.ScatteringCoefficients2D; image=nothing)
    has_img = !isnothing(image)
    J = c.n_scales
    L = c.n_orientations
    
    if has_img
        fig = CairoMakie.Figure(size=(1200, 420))
        
        # Panel (a): Input Image
        ax_a = CairoMakie.Axis(fig[1, 1],
            title="(a) Input Image/Texture",
            xlabel="x",
            ylabel="y",
            aspect=CairoMakie.AxisAspect(1)
        )
        CairoMakie.heatmap!(ax_a, image, colormap=:viridis)
        
        # Panel (b): S1 scale-orientation heatmap
        ax_b = CairoMakie.Axis(fig[1, 2],
            title="(b) S₁: Scale-Orientation Energy",
            xlabel="Orientation (θ)",
            ylabel="Scale (j)",
            xticks=(1:L, ["$(round(Int, 180*l/L))°" for l in 0:L-1]),
            yticks=(1:J, ["2^$(j-1)" for j in 1:J]),
            aspect=CairoMakie.AxisAspect(1)
        )
        S1_matrix = reshape(c.S1, L, J)' # J × L
        hm_b = CairoMakie.heatmap!(ax_b, 1:L, 1:J, S1_matrix', colormap=:viridis)
        CairoMakie.Colorbar(fig[1, 3], hm_b)
        
        # Panel (c): S2 cross-scale/orientation coupling
        ax_c = CairoMakie.Axis(fig[1, 4],
            title="(c) S₂: Cross-scale coupling",
            xlabel="Wavelet Index 2",
            ylabel="Wavelet Index 1",
            aspect=CairoMakie.AxisAspect(1)
        )
        S2_disp = copy(c.S2)
        S2_disp[S2_disp .== 0] .= NaN  # blank invalid/empty scattering paths
        nw = J * L
        for i in 1:nw, j in 1:nw
            if j <= i
                S2_disp[i, j] = NaN
            end
        end
        hm_c = CairoMakie.heatmap!(ax_c, S2_disp, colormap=:plasma, nan_color=:transparent)
        CairoMakie.Colorbar(fig[1, 5], hm_c)
    else
        fig = CairoMakie.Figure(size=(900, 420))
        
        # Panel (a): S1 scale-orientation heatmap
        ax_a = CairoMakie.Axis(fig[1, 1],
            title="(a) S₁: Scale-Orientation Energy",
            xlabel="Orientation (θ)",
            ylabel="Scale (j)",
            xticks=(1:L, ["$(round(Int, 180*l/L))°" for l in 0:L-1]),
            yticks=(1:J, ["2^$(j-1)" for j in 1:J]),
            aspect=CairoMakie.AxisAspect(1)
        )
        S1_matrix = reshape(c.S1, L, J)' # J × L
        hm_a = CairoMakie.heatmap!(ax_a, 1:L, 1:J, S1_matrix', colormap=:viridis)
        CairoMakie.Colorbar(fig[1, 2], hm_a)
        
        # Panel (b): S2 heatmap
        ax_b = CairoMakie.Axis(fig[1, 3],
            title="(b) S₂: Cross-scale coupling",
            xlabel="Wavelet Index 2",
            ylabel="Wavelet Index 1",
            aspect=CairoMakie.AxisAspect(1)
        )
        S2_disp = copy(c.S2)
        S2_disp[S2_disp .== 0] .= NaN  # blank invalid/empty scattering paths
        nw = J * L
        for i in 1:nw, j in 1:nw
            if j <= i
                S2_disp[i, j] = NaN
            end
        end
        hm_b = CairoMakie.heatmap!(ax_b, S2_disp, colormap=:plasma, nan_color=:transparent)
        CairoMakie.Colorbar(fig[1, 4], hm_b)
    end
    
    return fig
end

end # module
