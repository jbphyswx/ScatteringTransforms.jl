"""
    generate_figures.jl

Generate static figures for documentation using CairoMakie.
Run from repo root: julia --project=docs docs/generate_assets/generate_figures.jl
"""

using ScatteringTransforms: ScatteringTransforms
using CairoMakie: CairoMakie
using Statistics: Statistics
using FFTW: FFTW
using Random: Random

Random.seed!(42)

assets_dir = Base.joinpath(Base.@__DIR__, "..", "src", "assets")
Base.mkpath(assets_dir)

# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers for synthesizing signals
# ─────────────────────────────────────────────────────────────────────────────

function spectral_signal(N::Int, α::Real)
    k    = [0; 1:(N÷2-1); (N÷2); (N÷2-1):-1:1]
    amp  = (k .+ 1.0) .^ (α / 2)
    ϕ    = 2π .* Base.rand(N)
    fhat = amp .* exp.(im .* ϕ)
    fhat[1] = 0.0
    return Base.real(FFTW.ifft(fhat))
end

function spectral_field_2d(M::Int, α::Real)
    Ny, Nx = M, M
    ky_vec = [k < (Ny+1)÷2 ? k : k-Ny for k in 0:Ny-1]
    kx_vec = [k < (Nx+1)÷2 ? k : k-Nx for k in 0:Nx-1]
    fhat   = zeros(Complex{Float64}, Ny, Nx)
    for ix in 1:Nx, iy in 1:Ny
        kr = Base.sqrt(Float64(kx_vec[ix])^2 + Float64(ky_vec[iy])^2)
        amp = (kr + 1.0)^(α / 2)
        fhat[iy, ix] = amp * exp(2π * im * Base.rand())
    end
    fhat[1, 1] = 0.0
    field = Base.real(FFTW.ifft(fhat))
    return field ./ Statistics.std(field)
end

# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: Filter bank tiling and Littlewood-Paley sum
# ─────────────────────────────────────────────────────────────────────────────
Base.println("Figure 1: Filter bank...")
N_fb = 1024
J_fb = 6
st_fb = ScatteringTransforms.ScatteringTransform1D(N_fb, J_fb; Q=1, max_order=1)
fig_fb = ScatteringTransforms.plot_filter_bank(st_fb.filter_bank)
CairoMakie.save(Base.joinpath(assets_dir, "filter_bank.png"), fig_fb)
Base.println("  ✓ filter_bank.png")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: 1D Kolmogorov turbulence — signal, S1, S2
# ─────────────────────────────────────────────────────────────────────────────
Base.println("Figure 2: 1D scattering example...")
N = 2048
J = 7
signal = spectral_signal(N, -5/3)
signal = signal ./ Statistics.std(signal)
st1d = ScatteringTransforms.ScatteringTransform1D(N, J; Q=1, max_order=2)
coeffs_1d = st1d(signal)
fig_1d = ScatteringTransforms.plot_coefficients(coeffs_1d; signal=signal)
CairoMakie.save(Base.joinpath(assets_dir, "1d_scattering_example.png"), fig_1d)
Base.println("  ✓ 1d_scattering_example.png")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: 2D turbulent field — field, S1, S2
# ─────────────────────────────────────────────────────────────────────────────
Base.println("Figure 3: 2D scattering example...")
M = 128
J2 = 3
L2 = 8
field = spectral_field_2d(M, -3.0)
st2d = ScatteringTransforms.ScatteringTransform2D((M, M), J2; L=L2, max_order=2)
coeffs_2d = st2d(field)
fig_2d = ScatteringTransforms.plot_coefficients(coeffs_2d; image=field)
CairoMakie.save(Base.joinpath(assets_dir, "2d_scattering_example.png"), fig_2d)
Base.println("  ✓ 2d_scattering_example.png")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 4: Discriminability Demo (Kolmogorov vs Gaussian phase-randomized)
# ─────────────────────────────────────────────────────────────────────────────
Base.println("Figure 4: Discriminability demo...")
N_d = 2048
J_d = 6

sigA = zeros(Float64, N_d)
# Place 12 sparse impulses
for idx in [150, 320, 510, 720, 930, 1100, 1280, 1420, 1610, 1750, 1920]
    sigA[idx] = randn()
end

# Smooth slightly to create localized features
freqs = FFTW.fftfreq(N_d)
smooth_filter = exp.(-freqs.^2 ./ (2 * 0.02^2))
sigA = Base.real(FFTW.ifft(FFTW.fft(sigA) .* smooth_filter))
sigA = sigA ./ Statistics.std(sigA)

fA   = FFTW.fft(sigA)
ampA = Base.abs.(fA)
ϕB   = 2π .* Base.rand(N_d)
fB   = ampA .* exp.(im .* ϕB)
fB[1] = 0.0
sigB  = Base.real(FFTW.ifft(fB))
sigB  = sigB ./ Statistics.std(sigB)

st_d   = ScatteringTransforms.ScatteringTransform1D(N_d, J_d; Q=1, max_order=2)
cA     = st_d(sigA)
cB     = st_d(sigB)

# Set up the custom discriminability figure using CairoMakie
fig_disc = CairoMakie.Figure(size=(950, 1000))

zi = 1:1000
t_d = collect(0:N_d-1) ./ N_d

# Panel (a): Signal A
ax_a = CairoMakie.Axis(fig_disc[1, 1],
    title="(a) Signal A: Intermittent process (correlated phases)",
    xlabel="x", ylabel="u(x)"
)
CairoMakie.lines!(ax_a, t_d[zi], sigA[zi], color=:steelblue, linewidth=1.5)

# Panel (b): Signal B
ax_b = CairoMakie.Axis(fig_disc[1, 2],
    title="(b) Signal B: Randomized phases (Gaussian surrogate)",
    xlabel="x", ylabel="u(x)"
)
CairoMakie.lines!(ax_b, t_d[zi], sigB[zi], color=:firebrick, linewidth=1.5)

# Panel (c): Power spectrum
pA = Base.abs.(FFTW.fft(sigA)) .^ 2 ./ N_d
pB = Base.abs.(FFTW.fft(sigB)) .^ 2 ./ N_d
kk = collect(1:N_d÷2)
ax_c = CairoMakie.Axis(fig_disc[2, 1:2],
    title="(c) Power spectra: identical by construction",
    xlabel="Wavenumber k", ylabel="E(k)",
    xscale=log10, yscale=log10
)
CairoMakie.lines!(ax_c, kk, pA[2:N_d÷2+1], color=:steelblue, linewidth=2.5, label="A")
CairoMakie.lines!(ax_c, kk, pB[2:N_d÷2+1], color=:firebrick, linewidth=1.5, linestyle=:dash, label="B")
CairoMakie.axislegend(ax_c, position=:rt)

# Panel (d): S1 coefficients
S1A = cA.S1
S1B = cB.S1
scale_labels_d = ["2^$(j-1)" for j in 1:J_d]
ax_d = CairoMakie.Axis(fig_disc[3, 1:2],
    title="(d) S₁: nearly identical (first order is phase-blind)",
    xlabel="Scale j", ylabel="S₁(j)",
    xticks=(1:J_d, scale_labels_d)
)
CairoMakie.scatterlines!(ax_d, 1:J_d, S1A, color=:steelblue, marker=:circle, markersize=10, linewidth=2.0, label="A")
CairoMakie.scatterlines!(ax_d, 1:J_d, S1B, color=:firebrick, marker=:rect, markersize=10, linewidth=1.5, linestyle=:dash, label="B")
CairoMakie.axislegend(ax_d, position=:rt)

# Panels (e) & (f): S2 Heatmaps
# Find clim
S2A_disp = copy(cA.S2)
S2B_disp = copy(cB.S2)
for i in 1:J_d, j in 1:J_d
    if j <= i
        S2A_disp[i, j] = NaN
        S2B_disp[i, j] = NaN
    end
end
min_val = min(minimum(filter(!isnan, S2A_disp)), minimum(filter(!isnan, S2B_disp)))
max_val = max(maximum(filter(!isnan, S2A_disp)), maximum(filter(!isnan, S2B_disp)))
crange = (min_val, max_val)

ax_e = CairoMakie.Axis(fig_disc[4, 1],
    title="(e) S₂ for A  [turbulent: cross-scale coupling]",
    xlabel="j₂", ylabel="j₁",
    xticks=(1:J_d, scale_labels_d), yticks=(1:J_d, scale_labels_d),
    aspect=CairoMakie.AxisAspect(1)
)
hm_e = CairoMakie.heatmap!(ax_e, S2A_disp, colormap=:plasma, colorrange=crange, nan_color=:transparent)

ax_f = CairoMakie.Axis(fig_disc[4, 2],
    title="(f) S₂ for B  [Gaussian: weak coupling]",
    xlabel="j₂", ylabel="j₁",
    xticks=(1:J_d, scale_labels_d), yticks=(1:J_d, scale_labels_d),
    aspect=CairoMakie.AxisAspect(1)
)
hm_f = CairoMakie.heatmap!(ax_f, S2B_disp, colormap=:plasma, colorrange=crange, nan_color=:transparent)
CairoMakie.Colorbar(fig_disc[4, 3], hm_e)

CairoMakie.save(Base.joinpath(assets_dir, "discriminability.png"), fig_disc)
Base.println("  ✓ discriminability.png")

Base.println("\nAll assets generated successfully in: $(assets_dir)")
