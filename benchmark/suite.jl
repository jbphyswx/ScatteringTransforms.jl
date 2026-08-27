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

 4. **Solve overhead** — for the scattered least-squares path, the cost of one solver iteration
    against the type-2 plus type-1 it contains, and the whole solve against `iters` of those. Scale
    free for the same reason: it asks whether the solve costs more than the transforms it must do
    anyway, which is answerable on any machine. Swept over all three shape regimes, because `M` and
    `prod(ms)` are independent and the regime decides both the iteration count and whether the
    problem is determined at all.

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
# 4. Scattered least-squares solve — the three shape regimes
# ---------------------------------------------------------------------------

# `solve = true` replaces one adjoint application with an iterative solve, so its cost is two separate
# numbers and they fail differently. How many iterations the geometry needs is a property of the
# points. What one iteration costs *over* the two transforms inside it is a property of the solver:
# LSMR carries `2M + 9·prod(ms)` flops of vector work per iteration on top of a type-2 and a type-1,
# and that fraction is largest on small grids, where the transform has the least work to hide it
# behind. Both are measured rather than counted, because the vector work is bandwidth bound where the
# FFT is cache blocked, and because `M` and `ms` are independent: the caller may hand over more modes
# than samples, and the regime decides everything.
#
# Reported per regime: `M`, `prod(ms)`, iterations to tolerance, the cost of one type-2 plus one
# type-1, the measured cost of one solver iteration, and their ratio — the number that says whether
# the solve costs anything beyond the transforms it must do anyway.

# Iterations, driven through the plan's public transforms. `inverse_transform!` is `A` scaled by the
# synthesis `1/prod(ms)`, so the adjoint must carry the same scalar — a mismatched pair is not a
# scaled problem, it is a different one, and LSMR then runs to `maxiter` on an operator whose exact
# solution it should reach in a single step. With both scaled the factor cancels from every stopping
# test (`normA` falls by it exactly as `normx` rises, `normar` and `normr` together), so the count is
# the solve path's own.
function solve_probe(::Type{T}, base, b, ms, M, tol, maxiter) where {T}
    x = zeros(Complex{T}, ms)
    s = T(base.invN)
    info = ST.Plans.lsmr_solve!(x,
                                (dst, src) -> ST.Plans.inverse_transform!(dst, base, src),
                                (dst, src) -> (ST.Plans.forward_transform!(dst, base, src);
                                               dst .*= s),
                                Complex{T}.(b), zeros(Complex{T}, M), zeros(Complex{T}, M),
                                zeros(Complex{T}, ms), zeros(Complex{T}, ms), zeros(Complex{T}, ms),
                                zeros(Complex{T}, ms);
                                atol = tol, btol = tol, conlim = inv(eps(T)), maxiter = maxiter)
    return (info.iters, info.istop)
end

function solve_case(::Type{T}, label, xs, ys, ms, period, spec) where {T}
    M = length(xs)
    mk(solve, maxiter, rtol) = ST.Plans.make_scattered_plan(spec, xs, ys, ms, T; period = period,
                                                            solve = solve, maxiter = maxiter,
                                                            rtol = rtol, nufft_nthreads = 1)
    base = mk(false, 100, ST.Plans.default_solver_rtol(T, spec, nothing))
    b = T[sin(3xs[k]) * cos(2ys[k]) + T(0.5) for k in 1:M]
    X = zeros(Complex{T}, ms)
    pts = zeros(Complex{T}, M)

    # One application of `A` and one of `A†` — exactly what one LSMR iteration contains.
    ST.Plans.forward_transform!(X, base, b)
    t_pair = bench_kernel(() -> (ST.Plans.inverse_transform!(pts, base, X);
                                 ST.Plans.forward_transform!(X, base, b)))

    rtol = ST.Plans.default_solver_rtol(T, spec, nothing)
    iters, istop = solve_probe(T, base, b, ms, M, rtol, 100)

    # A solve the package refuses is a result, not a crash: at `conlim = 1/eps(T)` an operator whose
    # smallest singular direction carries nothing this precision can represent is rejected rather than
    # answered, and in `Float32` on a gappy set that is reachable. Reported, and the sweep goes on.
    solved = mk(true, 100, rtol)
    t_solve = try
        bench(() -> ST.Plans.forward_transform!(X, solved, b))
    catch err
        err isa ST.Plans.AnalysisNotConverged || rethrow()
        NaN
    end

    # Per-iteration cost, differenced across two forced budgets so that plan setup, the closing
    # `prod(ms)` rescale and the timer all cancel. Only meaningful while both budgets are actually
    # spent: `atol`/`btol` can be driven to zero but LSMR's machine-precision tests cannot, so on a
    # well-conditioned operator both runs stop at the same converged iterate and their difference is
    # noise rather than `k2 - k1` iterations. That case is reported as converged instead of quoting a
    # per-iteration cost the measurement cannot see.
    k1, k2 = 8, 24
    forced_iters, _ = solve_probe(T, base, b, ms, M, zero(T), k2)
    t_iter = if forced_iters == k2
        p1, p2 = mk(true, k1, zero(T)), mk(true, k2, zero(T))
        # More repetitions than elsewhere because differencing two nearby times amplifies their
        # scatter: at three reps this disagreed with `solve/(iters·pair)` by 25%, which is the
        # measurement's noise and not a property of the solver.
        d = try
            t1 = bench(() -> ST.Plans.forward_transform!(X, p1, b), 9)
            t2 = bench(() -> ST.Plans.forward_transform!(X, p2, b), 9)
            (t2 - t1) / (k2 - k1)
        catch err
            err isa ST.Plans.AnalysisNotConverged || rethrow()
            NaN
        end
        ST.Plans.close_plan!(p1); ST.Plans.close_plan!(p2)
        d
    else
        NaN
    end
    ST.Plans.close_plan!(base); ST.Plans.close_plan!(solved)

    Printf.@printf("%-24s M=%-8d n=%-7d iters %3d (istop %d)  pair %7.3f ms  solve %8.3f ms  %6s  iter %s\n",
                   label, M, prod(ms), iters, istop, t_pair * 1e3,
                   isnan(t_solve) ? NaN : t_solve * 1e3,
                   isnan(t_solve) ? "refused" :
                       Printf.@sprintf("%.2fx", t_solve / (max(iters, 1) * t_pair)),
                   isnan(t_iter) ? "converged" :
                       Printf.@sprintf("%7.3f ms  %.3fx", t_iter * 1e3, t_iter / t_pair))
    return nothing
end

function scattered_solve()
    ga = π * (3 - sqrt(5.0))
    # Sizes span the range where the overhead fraction is predicted to move: the vector work is `O(n)`
    # against a transform's `O(n log n)`, so the smallest grid is the hard case, not the afterthought.
    grids = TIER == "full" ? ((8, 8), (16, 16), (32, 32), (200, 200), (256, 256)) :
                             ((8, 8), (32, 32), (200, 200))
    spec = ST.Plans.FINUFFTBackend()
    for T in (TIER == "full" ? (Float64, Float32) : (Float64,))
        println("\n", T, ":")
        for ms in grids
            n = prod(ms)
            # Well sampled: the regime where the solver's extra vector work is pure overhead, and the
            # one that decides whether `solve = true` costs existing callers anything.
            M = 4n
            xs = [2π * mod(ga * k, 1.0) for k in 1:M]
            ys = [2π * (k - 0.5) / M for k in 1:M]
            solve_case(T, "well sampled $(ms[1])^2", xs, ys, ms, (2π, 2π), spec)

            # The exact DFT nodes: `A†A = n·I`, so any least-squares method is done in one iteration.
            # A count above 1 here means a scaling or convention has drifted, not that the solve is slow.
            gx = T[2π * (i - 1) / ms[1] for i in 1:ms[1], j in 1:ms[2]]
            gy = T[2π * (j - 1) / ms[2] for i in 1:ms[1], j in 1:ms[2]]
            solve_case(T, "  exact nodes $(ms[1])^2", vec(gx), vec(gy), ms, (2π, 2π), spec)

            # Fewer samples than modes, with a void — the geometry that made the old solver diverge.
            # Iterations here are the honest cost of the fix.
            Mg = n ÷ 2
            xg = [2π * mod(ga * k, 1.0) for k in 1:Mg]
            yg = [π * mod(sqrt(2.0) * k, 1.0) for k in 1:Mg]   # samples confined to half the domain
            solve_case(T, "  gappy $(ms[1])^2", xg, yg, ms, (2π, 2π), spec)
        end
    end

    # `M` at fixed `ms`: the point count enters the transform linearly and the solver's vector work
    # linearly too, so the ratio above should hold as `M` grows. This is where that is checked.
    println("\nM sweep at 32^2, well sampled:")
    for M in (TIER == "full" ? (100, 1_000, 10_000, 100_000, 1_000_000) : (100, 10_000, 1_000_000))
        xs = [2π * mod(ga * k, 1.0) for k in 1:M]
        ys = [2π * (k - 0.5) / M for k in 1:M]
        solve_case(Float64, "M=$M", xs, ys, (32, 32), (2π, 2π), spec)
    end

    # Threading and `ntrans` on the solve path, which is where a per-task plan or a per-column
    # recurrence would show up: the batched solve runs `B` LSMR recurrences through one execution per
    # transform, so its iteration count is the worst column's, not the average.
    println("\nSolve under a full cascade — threading and batching:")
    M, B = 4000, 8
    xs = [2π * mod(ga * k, 1.0) for k in 1:M]
    ys = [2π * (k - 0.5) / M for k in 1:M]
    Xb = randn(M, B)
    # Named explicitly and only when loaded: naming a backend whose extension is absent raises, and a
    # missing library is a gap to report, not a reason to abandon the sweep.
    _ext(:ScatteringTransformsFINUFFTExt) ?
        solve_cascade(ST.Plans.FINUFFTBackend(), xs, ys, Xb, B) :
        println("FINUFFT: not loaded")
    _ext(:ScatteringTransformsNonuniformFFTsExt) ?
        solve_cascade(ST.Plans.NonuniformFFTsBackend(), xs, ys, Xb, B) :
        println("NonuniformFFTs: not loaded")
    return nothing
end

function solve_cascade(spectral, xs, ys, Xb, B)
    sp = ST.scattered_planar_scattering(xs, ys, (32, 32), 3; L = 4, period = (2π, 2π),
                                        solve = true, spectral = spectral)
    adj = ST.scattered_planar_scattering(xs, ys, (32, 32), 3; L = 4, period = (2π, 2π),
                                         solve = false, spectral = spectral)
    ta = bench(() -> ST.scattering_batch(sp, Xb))
    tadj = bench(() -> ST.scattering_batch(adj, Xb))
    tt = bench(() -> ST.scattering_batch(CB.ThreadedBackend(), sp, Xb))
    Printf.@printf("%-26s adjoint %8.1f ms  solve %8.1f ms  %.1fx  threaded %8.1f ms  %.2fx\n",
                   string(nameof(typeof(spectral))), tadj * 1e3, ta * 1e3, ta / tadj,
                   tt * 1e3, ta / tt)
    spb = ST.ScatteredPlanar.build(Float64, xs, ys, (32, 32), 3; L = 4, period = (2π, 2π),
                                   solve = true, spectral = spectral, ntrans = B)
    if ST.Plans.batch_width(spb.plan) == B
        tb = bench(() -> ST.scattering_batch(spb, Xb))
        Printf.@printf("%-26s batched %8.1f ms  %.2fx over the per-field solve\n",
                       "  ntrans=$B", tb * 1e3, ta / tb)
    else
        println("  ntrans=$B: unavailable — this backend transforms one field per call")
    end
    return nothing
end

# ---------------------------------------------------------------------------
# 5. Allocation: every `!` path, zero in steady state
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

    header("Periodized cascade — speedup vs the approximation it trades for")
    multires("1D N=4096 J=8",
             ST.Scattering1D.ScatteringTransform1D(4096, 8; Q = 1, spectral = FB),
             ov -> ST.Scattering1D.ScatteringTransform1D(4096, 8; Q = 1, oversampling = ov,
                                                         spectral = FB), randn(4096), 1:3)
    multires("2D 128^2 J=4 L=8",
             ST.Scattering2D.ScatteringTransform2D((128, 128), 4; L = 8, spectral = FB),
             ov -> ST.Scattering2D.ScatteringTransform2D((128, 128), 4; L = 8,
                                                         oversampling = ov, spectral = FB),
             randn(128, 128), 0:2)
    multires("3D 64^3 J=3",
             ST.Scattering3D.ScatteringTransform3D((64, 64, 64), 3; n_orient = 6, spectral = FB),
             ov -> ST.Scattering3D.ScatteringTransform3D((64, 64, 64), 3; n_orient = 6,
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

    header("Scattered least-squares solve — cost per iteration against the transforms in it")
    scattered_solve()

    allocations()
    println("\nA 'x floor' near 1 means the cascade costs its own arithmetic and memory traffic and")
    println("nothing more. The threaded floor divides by the thread count, which is optimistic on a")
    println("bandwidth-bound kernel once threads exceed the physical core count.")
    return nothing
end

main()
