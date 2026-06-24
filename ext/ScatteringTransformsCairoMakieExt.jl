module ScatteringTransformsCairoMakieExt

using CairoMakie: CairoMakie
using ScatteringTransforms: ScatteringTransforms

# Sequential colormap for non-negative energy/coefficient heatmaps: light (≈ white) at zero so
# it reads cleanly on a white page (avoids the dark-purple-on-white look). Invalid/empty
# scattering paths are shaded a light transparent gray rather than left blank.
const _ENERGY_CMAP = :dense
const _NAN_COLOR = (:gray, 0.18)

# Implement plot_filter_bank for 1D Filter Banks
function ScatteringTransforms.plot_filter_bank(fb::ScatteringTransforms.FilterBank1D)
    MK = CairoMakie
    N = length(fb.averaging)
    Nhalf = N ÷ 2
    freq = collect(range(0.0, 0.5, length=Nhalf))
    nwav = length(fb.wavelets)
    lp = abs.(fb.averaging[1:Nhalf])

    # Saturated, white-visible, frequency-ordered colors (turbo, trimmed extremes).
    cols = [MK.cgrad(:turbo)[t] for t in range(0.06, 0.94, length=nwav)]

    fig = MK.Figure(size=(1150, 410))

    # Panel (a): frequency tiling — wider; low-pass shaded, wavelets coloured by scale.
    ax_a = MK.Axis(fig[1, 1];
        title="(a) Morlet filter bank: frequency tiling",
        xlabel="normalized frequency (cycles/sample)",
        ylabel="filter magnitude  |ψ̂|, |φ̂|",
        limits=((0.0, 0.5), (0.0, 1.08)))
    MK.band!(ax_a, freq, zeros(Nhalf), lp; color=(:gray, 0.20))
    MK.lines!(ax_a, freq, lp; color=:black, linewidth=2.6, linestyle=:dash, label="low-pass φ")
    for (i, ψ) in enumerate(fb.wavelets)
        lbl = fb.Q > 1 ? "j=$(fb.meta[i].scale), q=$(fb.meta[i].q)" : "j=$(fb.meta[i].scale)"
        MK.lines!(ax_a, freq, abs.(ψ[1:Nhalf]); color=cols[i], linewidth=2.4, label=lbl)
    end
    MK.axislegend(ax_a; position=:rt, labelsize=9, nbanks=(fb.Q > 1 || nwav > 6) ? 2 : 1,
                  framevisible=true)

    # Panel (b): the tight frame — individual squared responses tile up to ≈ 1.
    lp_sum = abs2.(fb.averaging[1:Nhalf]) .+ sum(abs2.(ψ[1:Nhalf]) for ψ in fb.wavelets)
    ax_b = MK.Axis(fig[1, 2];
        title="(b) Littlewood–Paley sum ≈ 1",
        xlabel="normalized frequency",
        ylabel="‖φ̂‖² + Σⱼ‖ψ̂ⱼ‖²",
        limits=((0.0, 0.5), (0.0, 1.22)))
    MK.lines!(ax_b, freq, abs2.(fb.averaging[1:Nhalf]); color=(:black, 0.35), linewidth=1.0)
    for (i, ψ) in enumerate(fb.wavelets)
        MK.lines!(ax_b, freq, abs2.(ψ[1:Nhalf]); color=(cols[i], 0.55), linewidth=1.0)
    end
    MK.lines!(ax_b, freq, lp_sum; color=:black, linewidth=2.6, label="total")
    MK.hlines!(ax_b, [1.0]; color=:red, linestyle=:dash, linewidth=1.5)
    MK.axislegend(ax_b; position=:rb, labelsize=9)

    MK.colsize!(fig.layout, 1, MK.Relative(0.62))
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
    
    # Blank the empty/invalid second-order paths (lower triangle + diagonal) for display.
    S2_display(S2) = begin
        D = copy(S2); D[D .== 0] .= NaN
        for i in axes(D, 1), j in axes(D, 2)
            j <= i && (D[i, j] = NaN)
        end
        D
    end

    if has_sig
        MK = CairoMakie
        # Left column: signal (a) over S₁ bars (b); right column: the S₂ coupling matrix (c).
        fig = MK.Figure(size=(1080, 470))

        ax_a = MK.Axis(fig[1, 1]; title="(a) input signal", xlabel="sample index",
                       ylabel="amplitude")
        MK.lines!(ax_a, signal, color=:black, linewidth=1.0)

        ax_b = MK.Axis(fig[2, 1]; title="(b) S₁: first-order coefficients",
                       xlabel="scale  j", ylabel="energy (log₁₀)",
                       xticks=(1:J_eff, scale_labels), yscale=log10)
        MK.barplot!(ax_b, 1:J_eff, c.S1 .+ 1e-10, color=:steelblue)

        ax_c = MK.Axis(fig[1:2, 2]; title="(c) S₂: cross-scale coupling  ⟨‖x⋆ψⱼ₁‖⋆ψⱼ₂‖⟩",
                       xlabel="finer scale  j₂", ylabel="coarser scale  j₁",
                       xticks=(1:J_eff, scale_labels), yticks=(1:J_eff, scale_labels),
                       aspect=MK.DataAspect())
        hm = MK.heatmap!(ax_c, S2_display(c.S2), colormap=_ENERGY_CMAP, nan_color=_NAN_COLOR)
        MK.Colorbar(fig[1:2, 3], hm, label="energy")
        MK.colsize!(fig.layout, 1, MK.Relative(0.54))
        return fig
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
        hm = CairoMakie.heatmap!(ax_b, S2_disp, colormap=_ENERGY_CMAP, nan_color=_NAN_COLOR)
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
        hm_b = CairoMakie.heatmap!(ax_b, 1:L, 1:J, S1_matrix', colormap=_ENERGY_CMAP)
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
        hm_c = CairoMakie.heatmap!(ax_c, S2_disp, colormap=_ENERGY_CMAP, nan_color=_NAN_COLOR)
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
        hm_a = CairoMakie.heatmap!(ax_a, 1:L, 1:J, S1_matrix', colormap=_ENERGY_CMAP)
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
        hm_b = CairoMakie.heatmap!(ax_b, S2_disp, colormap=_ENERGY_CMAP, nan_color=_NAN_COLOR)
        CairoMakie.Colorbar(fig[1, 4], hm_b)
    end
    
    return fig
end

end # module
