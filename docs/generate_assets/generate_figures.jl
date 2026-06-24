"""
    generate_figures.jl

Generate the documentation/README figures with CairoMakie against the CURRENT API.
Run from repo root: `julia --project=docs docs/generate_assets/generate_figures.jl`
"""

using ScatteringTransforms: ScatteringTransforms as ST
using CairoMakie: CairoMakie as MK
using Statistics: Statistics
using FFTW: FFTW
using Random: Random

Random.seed!(42)

const ASSETS = Base.joinpath(Base.@__DIR__, "..", "src", "assets")
Base.mkpath(ASSETS)
save(name, fig) = (MK.save(Base.joinpath(ASSETS, name), fig); Base.println("  ✓ ", name))

# ── synthetic signals ───────────────────────────────────────────────────────
function spectral_signal(N::Int, α::Real)
    k = [0; 1:(N ÷ 2 - 1); (N ÷ 2); (N ÷ 2 - 1):-1:1]
    fhat = ((k .+ 1.0) .^ (α / 2)) .* exp.(im .* 2π .* Base.rand(N))
    fhat[1] = 0.0
    s = Base.real(FFTW.ifft(fhat))
    return s ./ Statistics.std(s)
end

function spectral_field_2d(M::Int, α::Real)
    kv = [k < (M + 1) ÷ 2 ? k : k - M for k in 0:M-1]
    fhat = [(Base.sqrt(kv[ix]^2 + kv[iy]^2) + 1.0)^(α / 2) * exp(2π * im * Base.rand()) for iy in 1:M, ix in 1:M]
    fhat[1, 1] = 0.0
    f = Base.real(FFTW.ifft(fhat))
    return f ./ Statistics.std(f)
end

# ── 1. filter bank tiling + Littlewood–Paley (now a tight frame ≡ 1) ──────────
Base.println("Figure 1: filter bank...")
save("filter_bank.png",
     ST.plot_filter_bank(ST.ScatteringTransform1D(1024, 6; Q=1, max_order=1).filter_bank))

# ── 2. 1D scattering of a 1/f (turbulent) signal ──────────────────────────────
Base.println("Figure 2: 1D scattering...")
let N = 2048, J = 7
    sig = spectral_signal(N, -5/3)
    st = ST.ScatteringTransform1D(N, J; Q=1, max_order=2)
    save("1d_scattering_example.png", ST.plot_coefficients(st(sig); signal=sig))
end

# ── 3. 2D scattering of a turbulent field ─────────────────────────────────────
Base.println("Figure 3: 2D scattering...")
let M = 128, J = 3, L = 8
    field = spectral_field_2d(M, -3.0)
    st = ST.ScatteringTransform2D((M, M), J; L=L, max_order=2)
    save("2d_scattering_example.png", ST.plot_coefficients(st(field); image=field))
end

# ── 4. discriminability: identical power spectra, very different scattering ────
Base.println("Figure 4: discriminability...")
let N = 2048, J = 6
    # Signal A: strongly intermittent — a few sharp, well-separated localized bursts at
    # different scales (sparse, non-Gaussian).
    sigA = zeros(N)
    for (c, w) in ((250, 4), (600, 9), (980, 18), (1350, 6), (1750, 24))
        for n in (c - 3w):(c + 3w)
            (1 <= n <= N) && (sigA[n] += exp(-((n - c) / w)^2) * sin((2π / w) * (n - c)))
        end
    end
    sigA ./= Statistics.std(sigA)
    # Signal B: phase-randomized surrogate — identical power spectrum, but Gaussian.
    fB = Base.abs.(FFTW.fft(sigA)) .* exp.(im .* 2π .* Base.rand(N)); fB[1] = 0
    sigB = Base.real(FFTW.ifft(fB)); sigB ./= Statistics.std(sigB)

    st = ST.ScatteringTransform1D(N, J; Q=1, max_order=2)
    cA, cB = st(sigA), st(sigB)
    s2A = copy(ST.normalized_coefficients(cA).s2)   # S₂/S₁ : cross-scale coupling
    s2B = copy(ST.normalized_coefficients(cB).s2)
    s2A[s2A .== 0] .= NaN; s2B[s2B .== 0] .= NaN
    ratio = Base.round(Base.sum(filter(!isnan, s2A)) / Base.sum(filter(!isnan, s2B)); digits=1)
    cr = (0.0, Base.maximum(filter(!isnan, s2A)))
    lab = ["2^$(j-1)" for j in 1:J]

    fig = MK.Figure(size=(1000, 880))
    MK.Label(fig[0, 1:3], "Same power spectrum, different scattering — A couples across scales $(ratio)× more than B";
             fontsize=17, font=:bold)
    axa = MK.Axis(fig[1, 1]; title="(a) A: intermittent (sparse bursts)", xlabel="x"); MK.lines!(axa, sigA[1:1100]; color=:steelblue)
    axb = MK.Axis(fig[1, 2]; title="(b) B: phase-randomized surrogate", xlabel="x"); MK.lines!(axb, sigB[1:1100]; color=:firebrick)
    pA = Base.abs.(FFTW.fft(sigA)) .^ 2 ./ N; pB = Base.abs.(FFTW.fft(sigB)) .^ 2 ./ N
    axc = MK.Axis(fig[2, 1:3]; title="(c) power spectra: identical by construction",
                  xlabel="wavenumber k", ylabel="E(k)", xscale=log10, yscale=log10)
    MK.lines!(axc, 1:N÷2, pA[2:N÷2+1]; color=:steelblue, linewidth=3, label="A")
    MK.lines!(axc, 1:N÷2, pB[2:N÷2+1]; color=:firebrick, linewidth=1.5, linestyle=:dash, label="B")
    MK.axislegend(axc; position=:lb)
    axd = MK.Axis(fig[3, 1]; title="(d) S₂/S₁ for A — strong coupling", xlabel="j₂", ylabel="j₁",
                  xticks=(1:J, lab), yticks=(1:J, lab), aspect=MK.DataAspect())
    MK.heatmap!(axd, s2A; colormap=:plasma, colorrange=cr, nan_color=:transparent)
    axe = MK.Axis(fig[3, 2]; title="(e) S₂/S₁ for B — weak coupling", xlabel="j₂", ylabel="j₁",
                  xticks=(1:J, lab), yticks=(1:J, lab), aspect=MK.DataAspect())
    he = MK.heatmap!(axe, s2B; colormap=:plasma, colorrange=cr, nan_color=:transparent)
    MK.Colorbar(fig[3, 3], he; label="S₂/S₁")
    MK.rowsize!(fig.layout, 3, MK.Relative(0.4))
    save("discriminability.png", fig)
end

# ── 5. backends: in-core direct sum vs FFTW fast path ─────────────────────────
Base.println("Figure 5: backend performance...")
let J = 6
    Ns = (256, 512, 1024, 2048, 4096)
    td = Float64[]; tf = Float64[]
    for N in Ns
        x = spectral_signal(N, -5/3)
        sd = ST.ScatteringTransform1D(N, J; Q=1, max_order=2, spectral=:direct)
        sf = ST.ScatteringTransform1D(N, J; Q=1, max_order=2, spectral=:fftw)
        sd(x); sf(x)  # warmup
        push!(td, Base.minimum(Base.@elapsed(sd(x)) for _ in 1:3) * 1e3)
        push!(tf, Base.minimum(Base.@elapsed(sf(x)) for _ in 1:3) * 1e3)
    end
    fig = MK.Figure(size=(760, 480))
    ax = MK.Axis(fig[1, 1]; title="Spectral backend: in-core direct sum vs FFTW fast path",
                 xlabel="signal length N", ylabel="time per transform (ms)",
                 xscale=log10, yscale=log10, xticks=(collect(Ns), string.(collect(Ns))))
    MK.scatterlines!(ax, collect(Ns), td; color=:firebrick, marker=:rect, label="direct sum (in-core, O(N²))")
    MK.scatterlines!(ax, collect(Ns), tf; color=:steelblue, marker=:circle, label="FFTW fast path (O(N log N))")
    MK.axislegend(ax; position=:lt)
    save("backend_performance.png", fig)
end

# ── 6. localized (Mallat) scattering field ────────────────────────────────────
Base.println("Figure 6: localized field...")
let N = 1024, J = 6
    # two localized bursts at different scales
    sig = spectral_signal(N, -1.0)
    sig[200:260] .+= 2.0 .* sin.(range(0, 8π, length=61))
    sig[700:740] .+= 2.0 .* sin.(range(0, 24π, length=41))
    st = ST.ScatteringTransform1D(N, J; Q=1, max_order=2)
    sf = ST.scattering_field(st, sig; subsample=4)
    o1 = ST.PathGraph.order_range(st.tree, 1)              # first-order paths
    M = ST.path_field(sf, first(o1)) |> length
    field = reduce(hcat, [ST.path_field(sf, p) for p in o1])'  # (n_paths1, M)
    fig = MK.Figure(size=(820, 560))
    axs = MK.Axis(fig[1, 1]; title="(a) signal with bursts at two scales", xlabel="x", ylabel="u")
    MK.lines!(axs, sig; color=:black)
    axf = MK.Axis(fig[2, 1]; title="(b) localized first-order field S₁(x): energy localizes in scale & space",
                  xlabel="position (subsampled)", ylabel="scale j")
    hm = MK.heatmap!(axf, 1:M, 1:length(o1), field'; colormap=:viridis)
    MK.Colorbar(fig[2, 2], hm)
    MK.rowsize!(fig.layout, 1, MK.Relative(0.32))
    save("localized_field.png", fig)
end

# ── 7. reduced descriptors: anisotropy distinguishes oriented vs isotropic ─────
Base.println("Figure 7: reductions...")
let M = 128, J = 3, L = 8
    xs = range(0, 8π, length=M)'
    oriented = repeat(Base.sin.(xs), M, 1) .+ 0.05 .* Base.randn(M, M)
    iso = spectral_field_2d(M, -3.0)
    st = ST.ScatteringTransform2D((M, M), J; L=L, max_order=2)
    ro = ST.compute_shape_sparsity(ST.first_order(st(oriented)), ST.second_order(st(oriented)), st.filter_bank.meta)
    ri = ST.compute_shape_sparsity(ST.first_order(st(iso)), ST.second_order(st(iso)), st.filter_bank.meta)
    fig = MK.Figure(size=(900, 360))
    a1 = MK.Axis(fig[1, 1]; title="(a) oriented texture", aspect=MK.AxisAspect(1)); MK.heatmap!(a1, oriented; colormap=:viridis)
    a2 = MK.Axis(fig[1, 2]; title="(b) isotropic texture", aspect=MK.AxisAspect(1)); MK.heatmap!(a2, iso; colormap=:viridis)
    a3 = MK.Axis(fig[1, 3]; title="(c) anisotropy s₂₂ (peak)", xlabel="texture", ylabel="max |s₂₂|",
                 xticks=(1:2, ["oriented", "isotropic"]))
    MK.barplot!(a3, 1:2, [maximum(abs, ro.shape), maximum(abs, ri.shape)]; color=[:firebrick, :steelblue])
    save("reductions.png", fig)
end

Base.println("\nAll figures written to ", ASSETS)
