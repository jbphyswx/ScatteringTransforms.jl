"""
    suite.jl — the performance gate

Three scale-free criteria, because an absolute millisecond target says nothing without the machine
it was measured on:

 1. **Roofline ratio** — achieved time against the cost of the same cascade's steps done back to
    back with nothing else. A step is a spectral multiply, an inverse transform, and a fused
    modulus+mean; the cascade issues `(1 + nparents) + (nw + npaths)` of them. A ratio near `1`
    means everything except the transform itself is free.

    The floor is deliberately a *full step*, not just the FFT. Measured here, the FFT is only
    ~a quarter of a step: the multiply and the modulus each move `O(N)` words against the
    transform's `O(N log N)` of compute, so the cascade is memory-bandwidth bound and an FFT-only
    roofline is optimistic by ~4x.

 2. **Parallel efficiency** — `t_serial / t_threaded`, against the *physical* core count. Threads
    beyond that share one memory controller, so on a bandwidth-bound kernel they do not scale.

 3. **Allocation** — zero, in steady state, on every `!` path.

Run:  julia --project=benchmark -t<threads> benchmark/suite.jl [quick|full]
"""

using FFTW: FFTW
using OhMyThreads: OhMyThreads
using FastSphericalHarmonics: FastSphericalHarmonics
# Loaded for their extensions, not called directly: without them the nonuniform surfaces resolve to
# the in-core reference and the suite times that instead of the transform.
using FINUFFT: FINUFFT
using NUFSHT: NUFSHT
using ComputationalBackends: ComputationalBackends as CB
using SpectralBackends: SpectralBackends as SB
using ScatteringTransforms: ScatteringTransforms as ST
using LinearAlgebra: LinearAlgebra
using Printf: Printf

const TIER = isempty(ARGS) ? "quick" : ARGS[1]
const FB = SB.FFTSpectralBackend()

bench(f, r = 3) = (f(); minimum(@elapsed(f()) for _ in 1:r))

# For the micro-kernels the floor is built from. A single 20 us kernel timed on its own is mostly
# timer and call overhead — enough to put the reported floor an order of magnitude over the truth,
# which shows up as a transform "beating" its own floor. Repeat inside the timed region until the
# measurement is well clear of that.
function bench_kernel(f, target = 5.0e-3)
    f()
    t1 = minimum(@elapsed(f()) for _ in 1:3)
    k = max(1, ceil(Int, target / max(t1, 1.0e-9)))
    best = Inf
    for _ in 1:5
        t = @elapsed for _ in 1:k
            f()
        end
        best = min(best, t / k)
    end
    return best
end

# Spectral-plan executions one cascade must issue, read off the tree rather than assumed.
function transform_count(st)
    nw = length(st.filter_bank.wavelets)
    npaths = length(ST.PathGraph.order_range(st.tree, 2))
    nparents = count(g -> !isempty(g[2]), st.groups)
    return (1 + nparents) + (nw + npaths)
end

# The floor's plans must be built the way the package builds its own — pinned to one FFT thread.
# FFTW's count is process-global and FastSphericalHarmonics raises it to 4 simply by being loaded,
# and a plan built for 4 threads spawns 4 tasks on *every* execution. On a small grid that overhead
# exceeds the transform, which inflates the measured floor until the cascade appears to beat it.
plan_pinned(f) = ST.Plans.with_fft_nthreads(f, 1)

# One complete cascade step at a given grid size: multiply, inverse transform, modulus+mean.
function step_time(::Type{T}, spatial) where {T}
    xf, psi = randn(Complex{T}, spatial), randn(Complex{T}, spatial)
    c1, c2 = similar(xf), similar(xf)
    p = plan_pinned(() -> FFTW.plan_ifft(xf; flags = FFTW.MEASURE))
    return bench_kernel() do
        @. c1 = xf * psi
        LinearAlgebra.mul!(c2, p, c1)
        s = zero(T)
        @inbounds @simd for i in eachindex(c2)
            s += abs(c2[i])
        end
        s
    end
end

function fft_time(::Type{T}, spatial) where {T}
    a = zeros(Complex{T}, spatial)
    b = similar(a)
    p = plan_pinned(() -> FFTW.plan_fft(a; flags = FFTW.MEASURE))
    return bench_kernel(() -> LinearAlgebra.mul!(b, p, a))
end

header(s) = println("\n", s, "\n", "-"^length(s))

# ---------------------------------------------------------------------------
# 1 + 2. Gridded surfaces: roofline and parallel efficiency, single field and batch
# ---------------------------------------------------------------------------

function gridded(label, st, x, X, spatial)
    T = real(eltype(st.filter_bank.averaging))
    ntr = transform_count(st)
    tstep, tfft = step_time(T, spatial), fft_time(T, spatial)
    B = size(X)[end]

    tser = bench(() -> ST.scattering_batch(st, X))
    tthr = bench(() -> ST.scattering_batch(CB.ThreadedBackend(), st, X))
    floor_serial = ntr * tstep * B
    Printf.@printf("%-24s B=%-4d steps/slice=%-4d  step=%6.1f us (FFT is %2.0f%% of it)\n",
                   label, B, ntr, tstep * 1e6, 100 * tfft / tstep)
    Printf.@printf("    serial   %9.1f ms   %5.2fx floor\n", tser * 1e3, tser / floor_serial)
    Printf.@printf("    threaded %9.1f ms   %5.2fx floor   %.2fx serial on %d threads\n",
                   tthr * 1e3, tthr / (floor_serial / Threads.nthreads()), tser / tthr,
                   Threads.nthreads())

    # A single field has no batch to spread over, so threading takes the wavelet axis instead.
    t1s = bench(() -> st(x))
    Printf.@printf("    one field %8.1f ms   %5.2fx floor\n", t1s * 1e3, t1s / (ntr * tstep))
    return nothing
end

# ---------------------------------------------------------------------------
# 3. Multi-resolution second order — speedup against the error it trades for
# ---------------------------------------------------------------------------

function multires(label, exact, mk, x, ovs)
    ce = exact(x)
    te = bench(() -> exact(x))
    idx = findall(!iszero, ce.S2)
    Printf.@printf("%-24s exact %8.1f ms\n", label, te * 1e3)
    for ov in ovs
        st = mk(ov)
        cs = st(x)
        t = bench(() -> st(x))
        err = isempty(idx) ? 0.0 : sum(abs, cs.S2[idx] .- ce.S2[idx]) / sum(abs, ce.S2[idx])
        Printf.@printf("    oversampling=%d  %8.1f ms  %.2fx exact   S2 rel err %.2e\n",
                       ov, t * 1e3, te / t, err)
    end
    return nothing
end

# ---------------------------------------------------------------------------
# 4. Allocation: every `!` path, zero in steady state
# ---------------------------------------------------------------------------

_alloc3(f::F, a, b, c) where {F} = (f(a, b, c); @allocated f(a, b, c))

function allocations()
    header("Allocation (steady state, explicit serial + explicit spectral backend)")
    st1 = ST.Scattering1D.ScatteringTransform1D(512, 5; Q = 1, spectral = FB)
    c1 = ST.batch_coeffs(st1, Float64)
    Printf.@printf("  %-42s %6d B\n", "Scattering1D.scattering_transform!",
                   _alloc3(ST.Scattering1D.scattering_transform!, c1, st1, randn(512)))

    st2 = ST.Scattering2D.ScatteringTransform2D((64, 64), 3; L = 4, spectral = FB)
    c2 = ST.batch_coeffs(st2, Float64)
    Printf.@printf("  %-42s %6d B\n", "Scattering2D.scattering_transform2d!",
                   _alloc3(ST.Scattering2D.scattering_transform2d!, c2, st2, randn(64, 64)))

    st3 = ST.Scattering3D.ScatteringTransform3D((16, 16, 16), 2; n_orient = 4, spectral = FB)
    c3 = ST.batch_coeffs(st3, Float64)
    Printf.@printf("  %-42s %6d B\n", "Scattering3D.scattering_transform3d!",
                   _alloc3(ST.Scattering3D.scattering_transform3d!, c3, st3, randn(16, 16, 16)))

    mr = ST.SubsampledScattering.SubsampledScattering2D((64, 64), 3; L = 4, oversampling = 1,
                                                        spectral = FB)
    cm = ST.batch_coeffs(mr, Float64)
    Printf.@printf("  %-42s %6d B\n", "SubsampledScattering.subsampled_scattering!",
                   _alloc3(ST.SubsampledScattering.subsampled_scattering!, cm, mr, randn(64, 64)))

    X = randn(64, 64, 8)
    ws = ST.batch_workspace(st2, 8)
    out = ST.scattering_batch(st2, X)
    Printf.@printf("  %-42s %6d B\n", "Batched.batch_cascade!",
                   _alloc3(ST.Batched.batch_cascade!, out, ws, X))
    return nothing
end

# ---------------------------------------------------------------------------

# A timing that does not name the backend it resolved to is not a measurement. Every `Auto*` backend
# falls back to the in-core reference — `O(N²)` gridded, `O(M·K)` nonuniform — when its extension is
# absent, and that reads as a slow transform rather than as a missing dependency.
_ext(name) = Base.get_extension(ST, name) !== nothing

function loaded_fast_paths()
    have = String[]
    _ext(:ScatteringTransformsFFTWExt) && push!(have, "FFTW")
    _ext(:ScatteringTransformsFINUFFTExt) && push!(have, "FINUFFT")
    _ext(:ScatteringTransformsNonuniformFFTsExt) && push!(have, "NonuniformFFTs")
    _ext(:ScatteringTransformsFastSphericalHarmonicsExt) && push!(have, "FastSphericalHarmonics")
    _ext(:ScatteringTransformsNUFSHTExt) && push!(have, "NUFSHT")
    _ext(:ScatteringTransformsOhMyThreadsExt) && push!(have, "OhMyThreads")
    isempty(have) && push!(have, "NONE — every surface is on its in-core reference path")
    return have
end

function main()
    println("threads = ", Threads.nthreads(), "   FFTW threads = ", FFTW.get_num_threads(),
            "   tier = ", TIER)
    println("loaded fast paths: ", join(loaded_fast_paths(), ", "))

    header("Gridded surfaces — roofline and parallel efficiency")
    # `Q` is swept as well as `N`/`J`: it multiplies the wavelet count without changing the grid, so
    # it moves the transform along the work-per-byte axis rather than the size axis.
    grid = TIER == "full" ?
        ((1024, 6, 1, 64), (4096, 8, 1, 64), (4096, 8, 8, 16), (16384, 10, 1, 16)) :
        ((1024, 6, 1, 64), (4096, 8, 1, 32))
    for (N, J, Q, B) in grid
        st = ST.Scattering1D.ScatteringTransform1D(N, J; Q = Q, spectral = FB)
        gridded("1D N=$N J=$J Q=$Q", st, randn(N), randn(N, B), (N,))
    end
    for (N, J, B) in (TIER == "full" ? ((128, 4, 16), (256, 4, 8), (512, 5, 4)) : ((128, 4, 8),))
        st = ST.Scattering2D.ScatteringTransform2D((N, N), J; L = 8, spectral = FB)
        gridded("2D $(N)^2 J=$J L=8", st, randn(N, N), randn(N, N, B), (N, N))
    end
    for (N, J, B) in (TIER == "full" ? ((64, 3, 4), (128, 4, 2)) : ((64, 3, 2),))
        st = ST.Scattering3D.ScatteringTransform3D((N, N, N), J; n_orient = 6, spectral = FB)
        gridded("3D $(N)^3 J=$J", st, randn(N, N, N), randn(N, N, N, B), (N, N, N))
    end

    header("Multi-resolution second order — speedup vs the approximation it trades for")
    multires("1D N=4096 J=8",
             ST.Scattering1D.ScatteringTransform1D(4096, 8; Q = 1, spectral = FB),
             ov -> ST.SubsampledScattering.SubsampledScattering1D(4096, 8; Q = 1, oversampling = ov,
                                                                  spectral = FB), randn(4096), 1:3)
    multires("2D 128^2 J=4 L=8",
             ST.Scattering2D.ScatteringTransform2D((128, 128), 4; L = 8, spectral = FB),
             ov -> ST.SubsampledScattering.SubsampledScattering2D((128, 128), 4; L = 8,
                                                                  oversampling = ov, spectral = FB),
             randn(128, 128), 0:2)
    multires("3D 64^3 J=3",
             ST.Scattering3D.ScatteringTransform3D((64, 64, 64), 3; n_orient = 6, spectral = FB),
             ov -> ST.SubsampledScattering.SubsampledScattering3D((64, 64, 64), 3; n_orient = 6,
                                                                  oversampling = ov, spectral = FB),
             randn(64, 64, 64), 0:1)

    header("Nonuniform and spherical surfaces — batch reuse and threading")
    M = 2000
    Bp = 8
    px, py = rand(M), rand(M)
    sp = ST.scattered_planar_scattering(px, py, (32, 32), 3; L = 4)
    println("scattered planar plan: ", sp.plan)
    Xp = randn(M, Bp)
    ts = bench(() -> ST.scattering_batch(sp, Xp))
    tt = bench(() -> ST.scattering_batch(CB.ThreadedBackend(), sp, Xp))
    Printf.@printf("%-24s serial %8.1f ms   threaded %8.1f ms   %.2fx\n", "scattered planar M=$M",
                   ts * 1e3, tt * 1e3, ts / tt)
    # The batch axis is also a *transform* axis here: a plan built with `ntrans = B` runs the whole
    # stack through one NUFFT execution per cascade step instead of one per field. Reported against
    # the per-field loop above, since the two compose with threading rather than replacing it.
    spb = ST.ScatteredPlanar.build(Float64, px, py, (32, 32), 3; L = 4, ntrans = Bp)
    if ST.Plans.batch_width(spb.plan) == Bp
        tb = bench(() -> ST.scattering_batch(spb, Xp))
        tbt = bench(() -> ST.scattering_batch(CB.ThreadedBackend(), spb, Xp))
        Printf.@printf("%-24s batched %8.1f ms   %.2fx over per-field   threaded %8.1f ms   %.2fx\n",
                       "  ntrans=$Bp", tb * 1e3, ts / tb, tbt * 1e3, tb / tbt)
    else
        println("  ntrans=$Bp: unavailable — this NUFFT backend transforms one field per call")
    end
    for lmax in (TIER == "full" ? (16, 32, 64) : (24,))
        ss = ST.structured_spherical_scattering(lmax, 4)
        Θ, Φ = ST.SphericalCore.structured_grid(lmax, Float64)
        Xg = randn(length(Θ), length(Φ), 8)
        ts = bench(() -> ST.scattering_batch(ss, Xg))
        tt = bench(() -> ST.scattering_batch(CB.ThreadedBackend(), ss, Xg))
        Printf.@printf("%-24s serial %8.1f ms   threaded %8.1f ms   %.2fx\n",
                       "structured sphere lmax=$lmax", ts * 1e3, tt * 1e3, ts / tt)
    end

    # Scattered sphere reports per-field against batched because it is the one CPU surface where the
    # batched plan wins: the gridded cascade is bandwidth bound, so its `batch_cascade!` loses to a
    # per-slice loop, while here each step is an iterative solve whose transforms NUFSHT batches over
    # `ntrans`. Threading is reported against the batched time, since the two compose.
    if _ext(:ScatteringTransformsNUFSHTExt)
        for (lmax, M) in (TIER == "full" ? ((8, 500), (16, 1500), (32, 5000)) : ((8, 500),))
            ga = π * (3 - sqrt(5.0))
            θ = [acos(1 - 2 * (i - 0.5) / M) for i in 1:M]
            φ = [mod(ga * i, 2π) for i in 1:M]
            sc = ST.spherical_scattering(θ, φ, lmax, 3)
            Xc = randn(M, 8)
            println("scattered sphere plan: ", sc.plan)
            tb = bench(() -> ST.scattering_batch(sc, Xc))
            tf = bench(() -> ST._spherical_batch_perfield!(
                Matrix{Float64}(undef, ST.flat_rows(sc), size(Xc, 2)), sc, Xc))
            tt = bench(() -> ST.scattering_batch(CB.ThreadedBackend(), sc, Xc))
            Printf.@printf("%-24s per-field %8.1f ms   batched %8.1f ms   %.2fx   threaded %8.1f ms   %.2fx\n",
                           "scattered sphere lmax=$lmax", tf * 1e3, tb * 1e3, tf / tb, tt * 1e3, tb / tt)
        end
    else
        println("scattered sphere: skipped — NUFSHT not loaded (its in-core O(M·K) reference would " *
                "be reported as the transform's speed)")
    end

    allocations()
    println("\nA 'x floor' near 1 means the cascade costs its own arithmetic and memory traffic and")
    println("nothing more. The threaded floor divides by the thread count, which is optimistic on a")
    println("bandwidth-bound kernel once threads exceed the physical core count.")
    return nothing
end

main()
