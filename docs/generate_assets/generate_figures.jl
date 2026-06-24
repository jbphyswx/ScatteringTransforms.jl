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
using DifferentiationInterface: DifferentiationInterface as DI
using ADTypes: AutoMooncake
import Mooncake
using NUFSHT: NUFSHT          # enables spherical (monogenic) scattering

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

# ── 3. 2D scattering: scale–orientation selectivity across three textures ─────
Base.println("Figure 3: 2D scattering (multi-texture)...")
let M = 128, J = 3, L = 8
    st = ST.ScatteringTransform2D((M, M), J; L=L, max_order=2)
    iso = spectral_field_2d(M, -3.0)                                  # isotropic 1/f
    θ0 = 0.6                                                          # anisotropic: oriented stripes
    aniso = [Base.sin(2π * 6 * (Base.cos(θ0) * i + Base.sin(θ0) * j) / M) for i in 0:M-1, j in 0:M-1] .+
            0.05 .* Base.randn(M, M)
    Random.seed!(3)                                                   # eddy field: signed Gaussian vortices
    eddy = Base.zeros(M, M)
    for _ in 1:14
        cx, cy, s, σ = Base.rand() * M, Base.rand() * M, Base.rand([-1, 1]), 6 + 8 * Base.rand()
        for i in 1:M, j in 1:M
            eddy[i, j] += s * Base.exp(-((i - cx)^2 + (j - cy)^2) / (2σ^2))
        end
    end
    eddy ./= Statistics.std(eddy)

    fields = [("isotropic (1/f²)", iso), ("anisotropic (oriented)", aniso), ("eddy field (vortices)", eddy)]
    orient_ticks = (1:L, ["$(Base.round(Int, 180 * l / L))°" for l in 0:L-1])
    scale_ticks = (1:J, ["2^$(j-1)" for j in 1:J])
    letters = 'a':'z'

    fig = MK.Figure(size=(900, 1060))
    MK.Label(fig[0, 1:3], "2D scattering: first-order energy S₁ resolves scale × orientation";
             fontsize=16, font=:bold)
    for (row, (name, fld)) in enumerate(fields)
        c = st(fld)
        S1m = reshape(c.S1, L, J)'                # J×L (scale × orientation)
        cmax = Base.maximum(Base.abs.(fld))
        af = MK.Axis(fig[row, 1]; title="($(letters[2row-1])) $name", aspect=MK.DataAspect())
        MK.heatmap!(af, fld; colormap=:balance, colorrange=(-cmax, cmax))   # signed field: 0 = white
        MK.hidedecorations!(af)
        as = MK.Axis(fig[row, 2]; title="($(letters[2row])) S₁(scale, orientation)",
                     xlabel="orientation θ", ylabel="scale j",
                     xticks=orient_ticks, yticks=scale_ticks)
        hm = MK.heatmap!(as, 1:L, 1:J, permutedims(S1m); colormap=:dense)   # energy: light = 0
        MK.Colorbar(fig[row, 3], hm; label="S₁")
    end
    MK.colsize!(fig.layout, 1, MK.Relative(0.46))
    MK.colsize!(fig.layout, 3, MK.Relative(0.04))
    save("2d_scattering_example.png", fig)
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
    MK.heatmap!(axd, s2A; colormap=:dense, colorrange=cr, nan_color=(:gray, 0.18))
    axe = MK.Axis(fig[3, 2]; title="(e) S₂/S₁ for B — weak coupling", xlabel="j₂", ylabel="j₁",
                  xticks=(1:J, lab), yticks=(1:J, lab), aspect=MK.DataAspect())
    he = MK.heatmap!(axe, s2B; colormap=:dense, colorrange=cr, nan_color=(:gray, 0.18))
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
        sd = ST.ScatteringTransform1D(N, J; Q=1, max_order=2, spectral=ST.DirectSumBackend())
        sf = ST.ScatteringTransform1D(N, J; Q=1, max_order=2, spectral=ST.FFTBackend())
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
    fig = MK.Figure(size=(1120, 420))
    MK.Label(fig[0, 1:3], "Reduced descriptor s₂₂ (anisotropy) separates oriented from isotropic texture";
             fontsize=15, font=:bold)
    a1 = MK.Axis(fig[1, 1]; title="(a) oriented texture", xlabel="x", ylabel="y", aspect=MK.DataAspect())
    MK.heatmap!(a1, oriented; colormap=:viridis)
    a2 = MK.Axis(fig[1, 2]; title="(b) isotropic texture", xlabel="x", ylabel="y", aspect=MK.DataAspect())
    MK.heatmap!(a2, iso; colormap=:viridis)
    a3 = MK.Axis(fig[1, 3]; title="(c) anisotropy s₂₂ (peak)", ylabel="max |s₂₂|",
                 xticks=(1:2, ["oriented", "isotropic"]))
    MK.barplot!(a3, 1:2, [maximum(abs, ro.shape), maximum(abs, ri.shape)];
                color=[:firebrick, :steelblue])
    MK.colsize!(fig.layout, 3, MK.Relative(0.26))
    save("reductions.png", fig)
end

# ── 8. reconstruction: exact linear inverse + scattering-coefficient synthesis ─
Base.println("Figure 8: reconstruction & synthesis...")
let N = 384, J = 6
    sig = spectral_signal(N, -5/3)
    st = ST.ScatteringTransform1D(N, J; Q=1, max_order=2, spectral=ST.FFTBackend())

    # (1) exact linear wavelet-frame inverse — recovers the field to machine precision.
    xr = ST.iwavelet(st, ST.wavelet_transform(st, sig))
    inv_err = Base.maximum(Base.abs.(xr .- sig)) / Base.maximum(Base.abs.(sig))

    # (2) gradient-descent synthesis from the scattering coefficients (Bruna–Mallat): from noise,
    #     match S(x̂) to S(sig). Recovers a *sample* with the same multiscale statistics. Uses the
    #     in-core direct-sum forward (portable across AD backends; FFTW reverse-mode AD needs the
    #     backend's FFT rules).
    Random.seed!(7)
    st_ad = ST.ScatteringTransform1D(N, J; Q=1, max_order=2, spectral=ST.DirectSumBackend())
    res = ST.synthesize(st_ad, sig; backend=AutoMooncake(), init=Base.randn(N), iters=300, lr=0.05)
    syn = res.field

    cT = ST.scattering(st_ad, sig); cS = ST.scattering(st_ad, syn)
    s1T = ST.first_order(cT); s1S = ST.first_order(cS)

    fig = MK.Figure(size=(1000, 760))
    MK.Label(fig[0, 1:2], "Reconstruction from the scattering representation"; fontsize=17, font=:bold)
    a1 = MK.Axis(fig[1, 1]; title="(a) exact linear inverse  (rel. err $(Base.round(inv_err; sigdigits=2)))",
                 xlabel="x", ylabel="u")
    MK.lines!(a1, sig; color=:black, label="original")
    MK.lines!(a1, xr; color=:orange, linestyle=:dash, label="iwavelet")
    MK.axislegend(a1; position=:rt)
    a2 = MK.Axis(fig[1, 2]; title="(b) scattering-synthesis loss (Adam)", xlabel="iteration",
                 ylabel="‖S(x̂)−S(x)‖² (norm.)", yscale=log10)
    MK.lines!(a2, res.losses; color=:purple)
    a3 = MK.Axis(fig[2, 1]; title="(c) original vs synthesized sample (same statistics, not same field)",
                 xlabel="x", ylabel="u")
    MK.lines!(a3, sig; color=:black, label="original")
    MK.lines!(a3, syn; color=:seagreen, label="synthesized")
    MK.axislegend(a3; position=:rt)
    a4 = MK.Axis(fig[2, 2]; title="(d) first-order coefficients match", xlabel="S₁ path", ylabel="S₁")
    MK.scatterlines!(a4, s1T; color=:black, label="original")
    MK.scatterlines!(a4, s1S; color=:seagreen, marker=:rect, label="synthesized")
    MK.axislegend(a4; position=:rt)
    save("reconstruction_synthesis.png", fig)
end

# ── 9. monogenic (Riesz) scattering: amplitude envelope + continuous orientation ─
Base.println("Figure 9: monogenic...")
let M = 128
    # Constant-wavelength concentric rings: band-limited (no aliasing), and the local orientation
    # rotates with azimuth — a clean test for the monogenic amplitude (a smooth envelope) and the
    # continuously-recovered orientation (a radial pinwheel).
    cx = cy = (M + 1) / 2
    rr = [Base.sqrt((i - cx)^2 + (j - cy)^2) for i in 1:M, j in 1:M]
    field = Base.cos.(2π .* rr ./ (M / 16))
    st = ST.MonogenicScattering((M, M), 5; Q=1, max_order=1, spectral=ST.FFTBackend())
    best = Base.argmax([Base.sum(abs2, ST.monogenic_components(st, field, j).bandpass) for j in 1:5])
    comp = ST.monogenic_components(st, field, best)
    amp = comp.amplitude
    θ = [mod(Base.atan(comp.riesz[2][k], comp.riesz[1][k]), Base.π) for k in CartesianIndices(field)]
    θmask = [amp[k] > 0.15 * Base.maximum(amp) ? θ[k] : NaN for k in CartesianIndices(field)]

    fig = MK.Figure(size=(980, 940))
    MK.Label(fig[0, 1:2],
        "Monogenic (Riesz) wavelet analysis — amplitude envelope & continuous orientation";
        fontsize=16, font=:bold)
    a1 = MK.Axis(fig[1, 1]; title="(a) field: concentric rings", aspect=MK.DataAspect())
    MK.heatmap!(a1, field; colormap=:balance)
    a2 = MK.Axis(fig[1, 2]; title="(b) band-pass U⁰ = x ⋆ ψⱼ  (oscillatory)", aspect=MK.DataAspect())
    MK.heatmap!(a2, comp.bandpass; colormap=:balance)
    g3 = fig[2, 1] = MK.GridLayout()
    a3 = MK.Axis(g3[1, 1]; title="(c) monogenic amplitude √(U⁰²+|U^R|²)", aspect=MK.DataAspect())
    hm3 = MK.heatmap!(a3, amp; colormap=:viridis)
    MK.Colorbar(g3[1, 2], hm3)
    g4 = fig[2, 2] = MK.GridLayout()
    a4 = MK.Axis(g4[1, 1]; title="(d) local orientation atan(U^R)  (radial pinwheel)",
                 aspect=MK.DataAspect())
    hm4 = MK.heatmap!(a4, θmask; colormap=:twilight, colorrange=(0, Base.π), nan_color=:white)
    MK.Colorbar(g4[1, 2], hm4; ticks=([0, Base.π/2, Base.π], ["0", "π/2", "π"]))
    save("monogenic.png", fig)
end

# ── 10. 2D reconstruction: exact linear inverse + scattering-coefficient synthesis ─
Base.println("Figure 10: 2D reconstruction & synthesis (Mooncake; this one is slow)...")
let M = 40, J = 3, L = 6
    # A clear, *stationary* texture (oriented stripes + mild noise): global scattering coefficients
    # are translation-invariant texture statistics, so synthesis reproduces the texture (same
    # scale/orientation content), not the exact field — most legible on a real texture.
    θ0 = 0.6
    target = [Base.sin(2π * 4 * (Base.cos(θ0) * i + Base.sin(θ0) * j) / M) for i in 0:M-1, j in 0:M-1] .+
             0.15 .* Base.randn(M, M)
    target ./= Statistics.std(target)
    # (1) exact linear inverse — machine precision, via the FFTW fast path.
    stf = ST.ScatteringTransform2D((M, M), J; L=L, max_order=2, spectral=ST.FFTBackend())
    xr = ST.iwavelet(stf, ST.wavelet_transform(stf, target))
    inv_err = Base.maximum(Base.abs.(xr .- target)) / Base.maximum(Base.abs.(target))
    # (2) coefficient synthesis from noise (direct-sum forward, portable under reverse-mode AD).
    Random.seed!(11)
    st_ad = ST.ScatteringTransform2D((M, M), J; L=L, max_order=2, spectral=ST.DirectSumBackend())
    res = ST.synthesize(st_ad, target; backend=AutoMooncake(), init=Base.randn(M, M),
                        iters=160, lr=0.06)
    syn = res.field

    cr = (Base.minimum(target), Base.maximum(target))
    fig = MK.Figure(size=(1040, 760))
    MK.Label(fig[0, 1:2], "2D reconstruction from the scattering representation"; fontsize=16, font=:bold)
    a1 = MK.Axis(fig[1, 1]; title="(a) target texture (oriented)", aspect=MK.DataAspect())
    MK.heatmap!(a1, target; colormap=:viridis, colorrange=cr)
    a2 = MK.Axis(fig[1, 2]; title="(b) exact linear inverse  (rel. err $(Base.round(inv_err; sigdigits=2)))",
                 aspect=MK.DataAspect())
    MK.heatmap!(a2, xr; colormap=:viridis, colorrange=cr)
    a3 = MK.Axis(fig[2, 1]; title="(c) synthesized from noise — same texture, not same field",
                 aspect=MK.DataAspect())
    MK.heatmap!(a3, syn; colormap=:viridis)
    a4 = MK.Axis(fig[2, 2]; title="(d) synthesis loss (Adam)", xlabel="iteration",
                 ylabel="‖S(x̂)−S(x)‖² (norm.)", yscale=log10)
    MK.lines!(a4, res.losses; color=:purple, linewidth=2)
    save("reconstruction_2d.png", fig)
end

# ── 11. spherical scattering on S² (scattered points via NUFSHT) ──────────────
Base.println("Figure 11: spherical scattering...")
let Msph = 4500, lmax = 28, J = 4
    # quasi-uniform Fibonacci-sphere points + a smooth multi-scale band-limited field
    gr = (Base.sqrt(5.0) - 1) / 2
    θ = [Base.acos(1 - 2 * (k - 0.5) / Msph) for k in 1:Msph]
    φ = [2π * mod(k * gr, 1) for k in 1:Msph]
    field = Base.cos.(6 .* θ) .+ 0.6 .* Base.cos.(4 .* φ) .* Base.sin.(3 .* θ) .+
            0.4 .* Base.sin.(8 .* θ) .* Base.cos.(2 .* φ)
    x = Base.sin.(θ) .* Base.cos.(φ); y = Base.sin.(θ) .* Base.sin.(φ); z = Base.cos.(θ)

    sa = ST.spherical_scattering(θ, φ, lmax, J)(field)            # analytic modulus
    sm = ST.spherical_monogenic_scattering(θ, φ, lmax, J)(field)  # monogenic amplitude

    # Orthographic projection of the visible hemisphere onto the viewing plane: unlike Axis3 (which
    # inscribes the sphere in a cubic bbox and wastes most of the panel), the disk fills the square.
    av, ev = 0.6, 0.5
    u = (-Base.sin(av)) .* x .+ Base.cos(av) .* y
    v = (-Base.cos(av) * Base.sin(ev)) .* x .+ (-Base.sin(av) * Base.sin(ev)) .* y .+ Base.cos(ev) .* z
    depth = (Base.cos(av) * Base.cos(ev)) .* x .+ (Base.sin(av) * Base.cos(ev)) .* y .+ Base.sin(ev) .* z
    front = depth .>= -0.02                # the visible cap (silhouette = the unit circle)
    ord = sortperm(depth[front])           # draw far → near

    fig = MK.Figure(size=(1180, 560))
    MK.Label(fig[0, 1:2], "Spherical scattering on S² (scattered points, NUFSHT)"; fontsize=16, font=:bold)
    ax = MK.Axis(fig[1, 1]; title="(a) band-limited field on S² (orthographic)",
                 aspect=MK.DataAspect(), limits=((-1.02, 1.02), (-1.02, 1.02)))
    MK.scatter!(ax, u[front][ord], v[front][ord]; color=field[front][ord],
                colormap=:balance, markersize=13)
    MK.hidedecorations!(ax); MK.hidespines!(ax)

    gb = fig[1, 2] = MK.GridLayout()
    axb = MK.Axis(gb[1, 1]; title="(b) first-order coefficients S₁ by scale",
                  xlabel="scale band j", ylabel="S₁", xticks=1:J)
    js = repeat(1:J, 2)
    vals = vcat(sa.S1, sm.S1)
    grp = vcat(fill(1, J), fill(2, J))
    MK.barplot!(axb, js, vals; dodge=grp, color=[g == 1 ? :steelblue : :seagreen for g in grp])
    MK.axislegend(axb,
        [MK.PolyElement(color=:steelblue), MK.PolyElement(color=:seagreen)],
        ["analytic |·|", "monogenic"]; position=:lt, framevisible=false)
    MK.colsize!(fig.layout, 1, MK.Relative(0.46))   # ≈ square panel so the disk fills it
    save("spherical_scattering.png", fig)
end

Base.println("\nAll figures written to ", ASSETS)
