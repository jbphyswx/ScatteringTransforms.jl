"""
    suite.jl — the performance gate

Four scale-free criteria, because an absolute millisecond target says nothing without the machine
it was measured on:

 1. **Roofline ratio** — achieved time against separately timed forward transforms and band steps.
    A scalar band step is a spectral multiply, an in-place inverse transform, and a fused
    modulus+mean. A monogenic band step includes its Riesz components. The cascade uses
    `1 + nparents` forward transforms and `nw + npaths` bands per field.

    The floor is a full step: the multiply and the modulus each move `O(N)` words against the
    transform's `O(N log N)` of compute, so the cascade is memory-bandwidth bound, and a floor of
    FFTs alone sits below what any cascade reaches. The FFT's share of a step is printed beside it.

 2. **Parallel efficiency** — `t_serial / t_threaded`, against the *physical* core count. Threads
    beyond that share one memory controller, so on a bandwidth-bound kernel they do not scale.

 3. **Allocation** — zero, in steady state, on every `!` path.

 4. **Solve overhead** — for the scattered least-squares path, the cost of one solver iteration
    against the type-2 plus type-1 it contains, and the whole solve against `iters` of those. Scale
    free for the same reason: it asks whether the solve costs more than the transforms it must do
    anyway, which is answerable on any machine. Swept over all three shape regimes, because `M` and
    `prod(ms)` are independent and the regime decides both the iteration count and whether the
    problem is determined at all.

Run: julia --project=benchmark -t<threads> benchmark/suite.jl [quick|full|floors|solves|process|mpi]
"""

using FFTW: FFTW
using OhMyThreads: OhMyThreads
using FastSphericalHarmonics: FastSphericalHarmonics
# Loaded for their extensions: they give the nonuniform surfaces their fast paths.
using FINUFFT: FINUFFT
using NonuniformFFTs: NonuniformFFTs
using NUFSHT: NUFSHT
using FlowTransformBindings: FlowTransformBindings as FTB
using ComputationalBackends: ComputationalBackends as CB
using SpectralBackends: SpectralBackends as SB
using ScatteringTransforms: ScatteringTransforms as ST
using LinearAlgebra: LinearAlgebra
using Printf: Printf
using Distributed: Distributed
using MPI: MPI

const TIER = isempty(ARGS) ? "quick" : ARGS[1]
const FB = SB.FFTSpectralBackend()

bench(f, r = 3) = (f(); minimum(@elapsed(f()) for _ in 1:r))

# For the micro-kernels the floor is built from. A kernel of tens of microseconds timed on its own is
# mostly timer and call overhead, which inflates the floor until a transform appears to beat it; the
# kernel repeats inside the timed region until the measurement is well clear of that.
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

# Spectral-plan executions one cascade issues, read off its path tree.
function transform_count(st)
    nw = length(st.filter_bank.wavelets)
    npaths = length(ST.PathGraph.order_range(st.tree, 2))
    nparents = count(g -> !isempty(g[2]), st.groups)
    return (1 + nparents) + (nw + npaths)
end

function transform_count(st::ST.Monogenic.MonogenicScatteringTransform{D}) where {D}
    nw = length(st.filter_bank.wavelets)
    npaths = length(ST.PathGraph.order_range(st.tree, 2))
    nparents = count(g -> !isempty(g[2]), st.groups)
    return 1 + nparents + (D + 1) * (nw + npaths)
end

# The floor's plans must be built the way the package builds its own — pinned to one FFT thread.
# FFTW's count is process-global and FastSphericalHarmonics raises it to 4 simply by being loaded,
# and a plan built for 4 threads spawns 4 tasks on *every* execution. On a small grid that overhead
# exceeds the transform, which inflates the measured floor until the cascade appears to beat it.
plan_pinned(f) = ST.Plans.with_fft_nthreads(f, 1)

# One complete cascade step at a given grid size: multiply, inverse transform, modulus+mean.
function step_time(::Type{T}, spatial) where {T}
    xf, psi = randn(Complex{T}, spatial), randn(Complex{T}, spatial)
    c2 = similar(xf)
    p = plan_pinned(() -> FFTW.plan_ifft!(c2; flags = FFTW.MEASURE))
    return bench_kernel() do
        @. c2 = xf * psi
        p * c2
        s = zero(T)
        @inbounds @simd for i in eachindex(c2)
            s += ST.ScatteringCore._modulus(c2[i])
        end
        s
    end
end

function monogenic_step_time(::Type{T}, spatial::NTuple{D,Int}) where {T,D}
    xf, psi = randn(Complex{T}, spatial), randn(Complex{T}, spatial)
    c = similar(xf)
    energy = zeros(T, spatial)
    p = plan_pinned(() -> FFTW.plan_ifft!(c; flags = FFTW.MEASURE))
    return bench_kernel() do
        fill!(energy, zero(T))
        for _ in 0:D
            @. c = xf * psi
            p * c
            @. energy += abs2(c)
        end
        s = zero(T)
        @inbounds @simd for i in eachindex(energy)
            s += sqrt(energy[i])
        end
        s
    end
end

band_time(st, T, spatial) = step_time(T, spatial)
band_time(::ST.Monogenic.MonogenicScatteringTransform, T, spatial) = monogenic_step_time(T, spatial)

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
    tstep, tfft = band_time(st, T, spatial), fft_time(T, spatial)
    B = size(X)[end]

    tser = bench(() -> ST.scattering_batch(st, X))
    tthr = bench(() -> ST.scattering_batch(st, X; execution = CB.ThreadedBackend()))
    nf = 1 + count(g -> !isempty(g[2]), st.groups)
    nbands = length(st.filter_bank.wavelets) + length(ST.PathGraph.order_range(st.tree, 2))
    floor_serial = (nf * tfft + nbands * tstep) * B
    workers = min(B, Threads.nthreads())
    Printf.@printf("%-24s B=%-4d steps/slice=%-4d  step=%6.1f us (FFT is %2.0f%% of it)\n",
                   label, B, ntr, tstep * 1e6, 100 * tfft / tstep)
    Printf.@printf("    serial   %9.1f ms   %5.2fx floor\n", tser * 1e3, tser / floor_serial)
    Printf.@printf("    threaded %9.1f ms   %5.2fx floor   %.2fx serial on %d threads\n",
                   tthr * 1e3, tthr / (floor_serial / workers), tser / tthr, workers)

    # A single field has no batch to spread over, so threading takes the wavelet axis instead.
    t1s = bench(() -> st(x))
    Printf.@printf("    one field %8.1f ms   %5.2fx floor\n", t1s * 1e3, t1s / (floor_serial / B))
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

# Least-squares analysis uses an iterative solve, so its cost is two separate
# numbers and they fail differently. How many iterations the geometry needs is a property of the
# points. What one iteration costs *over* the two transforms inside it is a property of the solver:
# LSMR does vector work over the points and over the modes each iteration on top of a type-2 and a
# type-1, and that fraction is largest on small grids, where the transform has the least work to hide
# it behind. Both are measured: the vector work is bandwidth bound where the FFT is cache blocked, and
# `M` and `ms` are independent — the caller may hand over more modes than samples, and the regime
# decides both.
#
# Reported per regime: `M`, `prod(ms)`, iterations to tolerance, the cost of one type-2 plus one
# type-1, the measured cost of one solver iteration, and their ratio — the number that says whether
# the solve costs anything beyond the transforms it must do anyway.

# The real-input fit uses the plan's packed real-linear operator.
function solve_probe(plan, b, tol, maxiter)
    op, ws = plan.rsolve
    T = real(eltype(plan.hx))
    info = FTB.lsmr!(plan.hx, op, b, ws; rtol = tol,
                     conlim = inv(eps(T)), maxiter)
    return (info.iterations, ws.status[1])
end

function solve_case(::Type{T}, label, xs, ys, ms, period, spec) where {T}
    M = length(xs)
    mk(analysis, maxiter, rtol) = ST.Plans.make_scattered_plan(spec, xs, ys, ms, T; period,
                                                              analysis, maxiter, rtol, nthreads = 1)
    rtol = ST.Plans.default_solver_rtol(T, spec, nothing)
    solved = mk(ST.Plans.LeastSquares(), 100, rtol)
    b = T[sin(3xs[k]) * cos(2ys[k]) + T(0.5) for k in 1:M]
    X = zeros(Complex{T}, ms)

    # One application of `A` and one of `A†` — exactly what one LSMR iteration contains.
    ST.Plans.forward_transform!(X, solved, b)
    op, ws = solved.rsolve
    t_pair = bench_kernel() do
        FTB.lsmr_forward!(ws.u, op, solved.hx, nothing, 1)
        FTB.lsmr_adjoint!(ws.v, op, ws.u, nothing, 1)
    end
    iters, stop = solve_probe(solved, b, rtol, 100)
    t_solve = bench(() -> ST.Plans.forward_transform!(X, solved, b))

    # Per-iteration cost, differenced across two forced budgets so that plan setup, the closing
    # `prod(ms)` rescale and the timer all cancel. Only meaningful while both budgets are actually
    # spent: `atol`/`btol` can be driven to zero but LSMR's machine-precision tests cannot, so on a
    # well-conditioned operator both runs stop at the same converged iterate and their difference is
    # noise; that case is reported as converged, with no per-iteration cost.
    k1, k2 = 8, 24
    forced_iters, _ = solve_probe(solved, b, zero(T), k2)
    t_iter = if forced_iters == k2
        p1 = mk(ST.Plans.LeastSquares(), k1, zero(T))
        p2 = mk(ST.Plans.LeastSquares(), k2, zero(T))
        # Nine repetitions: differencing two nearby times amplifies their scatter.
        t1 = bench(() -> ST.Plans.forward_transform!(X, p1, b), 9)
        t2 = bench(() -> ST.Plans.forward_transform!(X, p2, b), 9)
        d = (t2 - t1) / (k2 - k1)
        ST.Plans.close_plan!(p1); ST.Plans.close_plan!(p2)
        d
    else
        NaN
    end
    ST.Plans.close_plan!(solved)

    Printf.@printf("%-24s M=%-8d n=%-7d iters %3d (%s)  pair %7.3f ms  solve %8.3f ms  %6s  iter %s\n",
                   label, M, prod(ms), iters, string(stop), t_pair * 1e3,
                   t_solve * 1e3,
                   Printf.@sprintf("%.2fx", t_solve / (max(iters, 1) * t_pair)),
                   isnan(t_iter) ? "early stop" :
                       Printf.@sprintf("%7.3f ms  %.3fx", t_iter * 1e3, t_iter / t_pair))
    return nothing
end

function scattered_solve()
    ga = π * (3 - sqrt(5.0))
    # The vector work is `O(n)` against a transform's `O(n log n)`, so the overhead fraction is largest on
    # the smallest grid.
    grids = TIER == "full" ? ((8, 8), (16, 16), (32, 32), (200, 200), (256, 256)) :
                             ((8, 8), (32, 32), (200, 200))
    for spec in nufft_libraries(), T in (TIER == "full" ? (Float64, Float32) : (Float64,))
        println("\n", nameof(typeof(spec)), " ", T, ":")
        for ms in grids
            n = prod(ms)
            # Well sampled: the regime where the solver's extra vector work is pure overhead, and the
            # one that exposes the cost of least-squares analysis.
            M = 4n
            xs = [2π * mod(ga * k, 1.0) for k in 1:M]
            ys = [2π * (k - 0.5) / M for k in 1:M]
            solve_case(T, "well sampled $(ms[1])^2", xs, ys, ms, (2π, 2π), spec)

            # The exact DFT nodes: `A†A = n·I`, so any least-squares method is done in one iteration.
            # A count above 1 here means a scaling or convention has drifted.
            gx = T[2π * (i - 1) / ms[1] for i in 1:ms[1], j in 1:ms[2]]
            gy = T[2π * (j - 1) / ms[2] for i in 1:ms[1], j in 1:ms[2]]
            solve_case(T, "  exact nodes $(ms[1])^2", vec(gx), vec(gy), ms, (2π, 2π), spec)

            # Fewer samples than modes, with a void: an underdetermined, ill-conditioned fit.
            Mg = n ÷ 2
            xg = [2π * mod(ga * k, 1.0) for k in 1:Mg]
            yg = [π * mod(sqrt(2.0) * k, 1.0) for k in 1:Mg]   # samples confined to half the domain
            solve_case(T, "  gappy $(ms[1])^2", xg, yg, ms, (2π, 2π), spec)
        end
    end

    # `M` at fixed `ms`: the point count enters the transform linearly and the solver's vector work
    # linearly too, so the ratio above should hold as `M` grows. This is where that is checked.
    for spec in nufft_libraries()
        println("\nM sweep at 32^2, well sampled, ", nameof(typeof(spec)), ":")
        for M in (TIER == "full" ? (100, 1_000, 10_000, 100_000, 1_000_000) : (100, 10_000, 1_000_000))
            xs = [2π * mod(ga * k, 1.0) for k in 1:M]
            ys = [2π * (k - 0.5) / M for k in 1:M]
            solve_case(Float64, "M=$M", xs, ys, (32, 32), (2π, 2π), spec)
        end
    end

    # Threading and `ntrans` on the solve path: the batched solve runs `B` LSMR recurrences through one
    # execution per transform, so its iteration count is the worst column's.
    println("\nSolve under a full cascade — threading and batching:")
    M, B = 4000, 8
    xs = [2π * mod(ga * k, 1.0) for k in 1:M]
    ys = [2π * (k - 0.5) / M for k in 1:M]
    Xb = randn(M, B)
    for spec in nufft_libraries()
        solve_cascade(spec, xs, ys, Xb, B)
    end
    return nothing
end

# The NUFFT libraries loaded here, each named explicitly: a library that is absent is reported as missing
# by `loaded_fast_paths` and skipped.
nufft_libraries() = filter(FTB.is_available, (FTB.NonuniformFFTsBackend(), FTB.FINUFFTBackend()))

function solve_cascade(spectral, xs, ys, Xb, B)
    sp = ST.scattered_planar_scattering(xs, ys, (32, 32), 3; L = 4, period = (2π, 2π),
                                        analysis = ST.Plans.LeastSquares(), transform = spectral)
    adj = ST.scattered_planar_scattering(xs, ys, (32, 32), 3; L = 4, period = (2π, 2π),
                                         analysis = ST.Plans.WeightedAdjoint(), transform = spectral)
    ta = bench(() -> ST.scattering_batch(sp, Xb))
    tadj = bench(() -> ST.scattering_batch(adj, Xb))
    tt = bench(() -> ST.scattering_batch(sp, Xb; execution = CB.ThreadedBackend()))
    Printf.@printf("%-26s adjoint %8.1f ms  solve %8.1f ms  %.1fx  threaded %8.1f ms  %.2fx\n",
                   string(nameof(typeof(spectral))), tadj * 1e3, ta * 1e3, ta / tadj,
                   tt * 1e3, ta / tt)
    spb = ST.ScatteredPlanar.build(Float64, xs, ys, (32, 32), 3; L = 4, period = (2π, 2π),
                                   analysis = ST.Plans.LeastSquares(), transform = spectral, ntrans = B)
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
# Scattered surfaces with every setting at their defaults, against their own plan's transforms
# ---------------------------------------------------------------------------

# Planar: one analysis of the samples onto the mode grid, followed by the gridded cascade.
function scattered_planar_defaults(M, m, J)
    xs, ys = rand(M), rand(M)
    st = ST.scattered_planar_scattering(xs, ys, (m, m), J; period = 1.0)
    f = randn(M)
    t = bench(() -> st(f), 2)
    X = zeros(ComplexF64, m, m)
    ta = bench_kernel(() -> ST.Plans.forward_transform!(X, st.plan, f))
    nf = 1 + count(g -> !isempty(g[2]), st.groups)
    nbands = length(st.filter_bank.wavelets) + length(ST.PathGraph.order_range(st.tree, 2))
    tstep = step_time(Float64, (m, m))
    tf = fft_time(Float64, (m, m))
    Printf.@printf("%-30s %8.1f ms = %5.2fx floor: analysis %.2f ms + %d FFTs + %d bands of %.1f us\n",
                   "scattered planar M=$M $(m)^2 J=$J", t * 1e3, t / (ta + nf * tf + nbands * tstep), ta * 1e3, nf, nbands,
                   tstep * 1e6)
    return nothing
end

# Sphere: the samples are fitted once; on the Gauss–Legendre grid the cascade then analyses every
# first-order field with children (`J − 1` of them) and synthesises `J + J(J−1)/2` bands, one transform
# each. The floor is that fit and those transforms.
function scattered_sphere_defaults(M, lmax, J)
    θ = acos.(1 .- 2 .* rand(M))
    φ = 2π .* rand(M)
    st = ST.spherical_scattering(θ, φ, lmax, J)
    f = randn(M)
    t = bench(() -> st(f), 2)
    C = ST.SphericalCore.sphere_coeffs_buffer(st.plan)
    u = ST.SphericalCore.sphere_field_buffer(st.grid)
    ta = bench(() -> ST.SphericalCore.sphere_coeffs!(C, st.plan, f), 2)
    ts = bench_kernel(() -> ST.SphericalCore.sphere_apply!(u, st.grid, C, Returns(1.0)))
    tg = bench_kernel(() -> ST.SphericalCore.sphere_coeffs!(C, st.grid, u))
    na, ns = J - 1, J + J * (J - 1) ÷ 2
    Printf.@printf("%-30s %8.1f ms = %5.2fx floor: fit %.1f ms + %d grid analyses of %.2f ms + %d syntheses of %.2f ms\n",
                   "scattered sphere M=$M lmax=$lmax J=$J", t * 1e3, t / (ta + na * tg + ns * ts), ta * 1e3,
                   na, tg * 1e3, ns, ts * 1e3)
    return nothing
end

function spherical_grid_study()
    SC = ST.SphericalCore
    M, lmax, J = 2500, 24, 4
    θ = [acos(1 - 2 * (k - 0.5) / M) for k in 1:M]
    φ = [2π * mod(k * (sqrt(5) - 1) / 2, 1) for k in 1:M]
    field(t, f) = cos(2t) + 0.4sin(t) * cos(3f)
    values = field.(θ, φ)
    grid = ST.spherical_scattering(θ, φ, lmax, J)
    point = SC.SphericalScatteringTransform(lmax, J, 2, ST.Plans.task_local_plan(grid.plan), grid.sigma2)
    θr, φr, weights = SC.gauss_legendre_grid(4lmax, Float64)
    fine = ST.spherical_scattering(θr, φr, lmax, J; weights,
        analysis = ST.Plans.WeightedAdjoint(), grid_lmax = 4lmax)
    reference = ST.Coefficients.flatten1d(fine(field.(θr, φr)))
    println("sphere grid accuracy: M=$M lmax=$lmax J=$J reference grid_lmax=$(4lmax)")
    for (label, st) in (("Gauss-Legendre cascade", grid), ("per-band point fits", point))
        t = bench(() -> st(values), 3)
        result = ST.Coefficients.flatten1d(st(values))
        error = maximum(abs, result .- reference) / max(maximum(abs, reference), eps())
        Printf.@printf("%-28s %.3f ms relative max error %.2e\n", label, t * 1e3, error)
    end
    foreach(ST.close_transform!, (grid, point, fine))
end

# ---------------------------------------------------------------------------
# 5. Allocation: every `!` path, zero in steady state
# ---------------------------------------------------------------------------

function _alloc3(f::F, a, b, c) where {F}
    f(a, b, c)
    return minimum(@allocated(f(a, b, c)) for _ in 1:3)
end

function allocations()
    header("Allocation (steady state, explicit serial + explicit spectral backend)")
    st1 = ST.ScatteringCore.ScatteringTransform(512, 5; Q = 1, transform = FB)
    c1 = ST.batch_coeffs(st1, Float64)
    Printf.@printf("  %-42s %6d B\n", "Scattering1D.scattering_transform!",
                   _alloc3(ST.Scattering1D.scattering_transform!, c1, st1, randn(512)))

    st2 = ST.ScatteringCore.ScatteringTransform((64, 64), 3; L = 4, transform = FB)
    c2 = ST.batch_coeffs(st2, Float64)
    Printf.@printf("  %-42s %6d B\n", "Scattering2D.scattering_transform2d!",
                   _alloc3(ST.Scattering2D.scattering_transform2d!, c2, st2, randn(64, 64)))

    st3 = ST.ScatteringCore.ScatteringTransform((16, 16, 16), 2; n_orient = 4, transform = FB)
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

# A timing names the backend it resolved to: every `Auto*` backend falls back to the in-core reference —
# `O(N²)` gridded, `O(M·K)` nonuniform — when its extension is absent.
_ext(name) = Base.get_extension(ST, name) !== nothing

function loaded_fast_paths()
    have = String[]
    _ext(:ScatteringTransformsFFTWExt) && push!(have, "FFTW")
    FTB.is_available(FTB.FINUFFTBackend()) && push!(have, "FINUFFT")
    FTB.is_available(FTB.NonuniformFFTsBackend()) && push!(have, "NonuniformFFTs")
    _ext(:ScatteringTransformsFastSphericalHarmonicsExt) && push!(have, "FastSphericalHarmonics")
    _ext(:ScatteringTransformsNUFSHTExt) && push!(have, "NUFSHT")
    _ext(:ScatteringTransformsOhMyThreadsExt) && push!(have, "OhMyThreads")
    isempty(have) && push!(have, "NONE — every surface is on its in-core reference path")
    return have
end

function main()
    println("project = ", Base.active_project())
    println("threads = ", Threads.nthreads(), "   FFTW threads = ", FFTW.get_num_threads(),
            "   tier = ", TIER)
    println("loaded fast paths: ", join(loaded_fast_paths(), ", "))

    header("Gridded surfaces — roofline and parallel efficiency")
    # `Q` is swept as well as `N`/`J`: it multiplies the wavelet count at a fixed grid, moving the
    # transform along the work-per-byte axis.
    grid = TIER == "full" ?
        ((1024, 6, 1, 64), (4096, 8, 1, 64), (4096, 8, 8, 16), (16384, 10, 1, 16)) :
        ((1024, 6, 1, 64), (4096, 8, 1, 32))
    for (N, J, Q, B) in grid
        st = ST.ScatteringCore.ScatteringTransform(N, J; Q = Q, transform = FB)
        gridded("1D N=$N J=$J Q=$Q", st, randn(N), randn(N, B), (N,))
    end
    for (N, J, B) in (TIER == "full" ? ((128, 4, 16), (256, 4, 8), (512, 5, 4)) : ((128, 4, 8),))
        st = ST.ScatteringCore.ScatteringTransform((N, N), J; L = 8, transform = FB)
        gridded("2D $(N)^2 J=$J L=8", st, randn(N, N), randn(N, N, B), (N, N))
    end
    for (N, J, B) in (TIER == "full" ? ((64, 3, 4), (128, 4, 2)) : ((64, 3, 2),))
        st = ST.ScatteringCore.ScatteringTransform((N, N, N), J; n_orient = 6, transform = FB)
        gridded("3D $(N)^3 J=$J", st, randn(N, N, N), randn(N, N, N, B), (N, N, N))
    end
    for (dims, J, B) in (((4096,), 8, 32), ((128, 128), 4, 8), ((32, 32, 32), 3, 4))
        st = ST.Monogenic.MonogenicScatteringTransform(dims, J; transform = FB)
        gridded("monogenic $dims J=$J", st, randn(dims...), randn(dims..., B), dims)
    end

    TIER == "floors" && return scattered_planar_defaults(10_000, 64, 4)

    header("Periodized cascade — speedup vs the approximation it trades for")
    multires("1D N=4096 J=8",
             ST.ScatteringCore.ScatteringTransform(4096, 8; Q = 1, transform = FB),
             ov -> ST.ScatteringCore.ScatteringTransform(4096, 8; Q = 1, oversampling = ov,
                                                         transform = FB), randn(4096), 1:3)
    multires("2D 128^2 J=4 L=8",
             ST.ScatteringCore.ScatteringTransform((128, 128), 4; L = 8, transform = FB),
             ov -> ST.ScatteringCore.ScatteringTransform((128, 128), 4; L = 8,
                                                         oversampling = ov, transform = FB),
             randn(128, 128), 0:2)
    multires("3D 64^3 J=3",
             ST.ScatteringCore.ScatteringTransform((64, 64, 64), 3; n_orient = 6, transform = FB),
             ov -> ST.ScatteringCore.ScatteringTransform((64, 64, 64), 3; n_orient = 6,
                                                         oversampling = ov, transform = FB),
             randn(64, 64, 64), 0:1)

    header("Nonuniform and spherical surfaces — batch reuse and threading")
    M = 2000
    Bp = 8
    px, py = rand(M), rand(M)
    sp = ST.scattered_planar_scattering(px, py, (32, 32), 3; period = 1.0, L = 4)
    println("scattered planar plan: ", sp.plan)
    Xp = randn(M, Bp)
    ts = bench(() -> ST.scattering_batch(sp, Xp))
    tt = bench(() -> ST.scattering_batch(sp, Xp; execution = CB.ThreadedBackend()))
    Printf.@printf("%-24s serial %8.1f ms   threaded %8.1f ms   %.2fx\n", "scattered planar M=$M",
                   ts * 1e3, tt * 1e3, ts / tt)
    # The batch axis is also a transform axis here: a plan built with `ntrans = B` analyses the whole
    # stack in one NUFFT execution. Reported against the per-field loop above; the two compose with
    # threading.
    spb = ST.ScatteredPlanar.build(Float64, px, py, (32, 32), 3; period = 1.0, L = 4, ntrans = Bp)
    if ST.Plans.batch_width(spb.plan) == Bp
        tb = bench(() -> ST.scattering_batch(spb, Xp))
        tbt = bench(() -> ST.scattering_batch(spb, Xp; execution = CB.ThreadedBackend()))
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
        tt = bench(() -> ST.scattering_batch(ss, Xg; execution = CB.ThreadedBackend()))
        Printf.@printf("%-24s serial %8.1f ms   threaded %8.1f ms   %.2fx\n",
                       "structured sphere lmax=$lmax", ts * 1e3, tt * 1e3, ts / tt)
    end

    # Scattered sphere reports per-field against batched because it is the one CPU surface where the
    # batched plan wins: the gridded cascade is bandwidth bound, so its `batch_cascade!` loses to a
    # per-slice loop, while here the fit at the points and every step on the grid are NUFSHT
    # transforms batched over `ntrans`. Threading is reported against the batched time, since the two
    # compose.
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
            tt = bench(() -> ST.scattering_batch(sc, Xc; execution = CB.ThreadedBackend()))
            Printf.@printf("%-24s per-field %8.1f ms   batched %8.1f ms   %.2fx   threaded %8.1f ms   %.2fx\n",
                           "scattered sphere lmax=$lmax", tf * 1e3, tb * 1e3, tf / tb, tt * 1e3, tb / tt)
        end
    else
        println("scattered sphere: skipped — NUFSHT not loaded (its in-core O(M·K) reference would " *
                "be reported as the transform's speed)")
    end

    header("Scattered surfaces at their defaults — cost in their own plan's transforms")
    TIER == "full" ? scattered_planar_defaults(100_000, 128, 5) : scattered_planar_defaults(10_000, 64, 4)
    if _ext(:ScatteringTransformsNUFSHTExt)
        TIER == "full" ? scattered_sphere_defaults(33_282, 128, 5) : scattered_sphere_defaults(8_450, 64, 4)
        spherical_grid_study()
    end

    header("Scattered least-squares solve — cost per iteration against the transforms in it")
    scattered_solve()

    allocations()
    println("\nA 'x floor' near 1 means the cascade costs its own arithmetic and memory traffic and")
    println("nothing more. The threaded floor divides by the thread count, which is optimistic on a")
    println("bandwidth-bound kernel once threads exceed the physical core count.")
    return nothing
end

function process_cases()
    M, B = 10_000, 8
    x = [mod(k * (sqrt(5) - 1) / 2, 1) for k in 1:M]
    y = [(k - 0.5) / M for k in 1:M]
    planar = ST.scattered_planar_scattering(x, y, (64, 64), 4; period = 1.0, L = 4, ntrans = 2)
    X = [cos(2π * x[k]) + 0.1b * sin(4π * y[k]) for k in 1:M, b in 1:B]
    θ, φ = acos.(1 .- 2 .* y), 2π .* x
    sphere = ST.spherical_scattering(θ, φ, 32, 4)
    F = [cos(θ[k]) + 0.1b * sin(θ[k]) * cos(φ[k]) for k in 1:M, b in 1:B]
    return (("scattered planar", planar, X), ("scattered sphere", sphere, F))
end

function process_study(execution)
    println("project=", Base.active_project(), " threads=", Threads.nthreads(), " execution=", execution)
    for (label, st, X) in process_cases()
        println(label, " ", st)
        reference = ST.scattering_batch(st, X)
        serial = bench(() -> ST.scattering_batch(st, X), 3)
        first_call = @elapsed result = ST.scattering_batch(st, X; execution)
        steady = bench(() -> ST.scattering_batch(st, X; execution), 3)
        error = maximum(abs, result .- reference) / max(maximum(abs, reference), eps())
        Printf.@printf("%-24s serial %.3f s first %.3f s steady %.3f s speedup %.2fx error %.2e\n",
                       label, serial, first_call, steady, serial / steady, error)
        ST.close_transform!(st)
    end
end

if TIER == "process"
    worker_ids = Distributed.addprocs(2;
        exeflags = `--project=$(Base.active_project()) --threads=1 --startup-file=no`)
    try
        Distributed.@everywhere begin
            using ScatteringTransforms: ScatteringTransforms
            using FFTW: FFTW
            using FINUFFT: FINUFFT
            using NonuniformFFTs: NonuniformFFTs
            using NUFSHT: NUFSHT
        end
        process_study(CB.DistributedBackend())
    finally
        Distributed.rmprocs(worker_ids)
    end
elseif TIER == "mpi"
    MPI.Init()
    process_study(CB.MPIBackend())
    MPI.Finalize()
elseif TIER == "solves"
    println("project=", Base.active_project(), " threads=", Threads.nthreads(),
            " FFTW threads=", FFTW.get_num_threads(), " libraries=", nufft_libraries())
    scattered_solve()
    allocations()
else
    main()
end
