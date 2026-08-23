# Nonuniform / scattered planar scattering via NUFFT. On a uniform grid the NUFFT
# analysis/synthesis reduce to fft/ifft, so scattered-planar scattering must reproduce the gridded
# ScatteringTransform2D exactly (to NUFFT tolerance); on irregular points the CG-solve path recovers
# a band-limited field.
using Random: Random

Test.@testset "Scattered / nonuniform planar scattering (NUFFT)" begin
    Ny, Nx, J, L = 24, 24, 3, 4

    # Uniform-grid points (axis-1 index fastest ⇒ matches column-major vec of the (Ny,Nx) image).
    n1 = vec([Float64(i) for i in 0:Ny-1, j in 0:Nx-1])
    n2 = vec([Float64(j) for i in 0:Ny-1, j in 0:Nx-1])
    f = randn(Ny, Nx)
    grid = ScatteringTransforms.Scattering2D.ScatteringTransform2D((Ny, Nx), J;
        L=L, max_order=2, spectral=SpectralBackends.FFTSpectralBackend())
    ref = ScatteringTransforms.Coefficients.flatten2d(grid(f))

    Test.@testset "uniform-grid parity reproduces the gridded FFT transform" begin
        for solve in (false, true)
            sca = ScatteringTransforms.scattered_planar_scattering(n1, n2, (Ny, Nx), J;
                L=L, max_order=2, period=(Ny, Nx), solve=solve)
            c = sca(vec(f))
            # coefficient container structure matches the gridded transform
            Test.@test size(ScatteringTransforms.Coefficients.second_order(c)) == (J * L, J * L)
            Test.@test ScatteringTransforms.Coefficients.flatten2d(c) ≈ ref rtol=1e-6
        end
    end

    Test.@testset "dependency-free direct NUDFT backend (no external library)" begin
        # The in-core DirectNUFFTBackend uses exact direct summation, no FINUFFT: on the uniform grid it
        # must reproduce the gridded FFT transform to machine precision, and match the FINUFFT path.
        dir = ScatteringTransforms.scattered_planar_scattering(n1, n2, (Ny, Nx), J;
            L=L, max_order=2, period=(Ny, Nx), spectral=SpectralBackends.DirectSumSpectralBackend())
        Test.@test dir.plan isa ScatteringTransforms.Plans.DirectNUFFTPlan
        cd = ScatteringTransforms.Coefficients.flatten2d(dir(vec(f)))
        Test.@test cd ≈ ref rtol=1e-9
        fin = ScatteringTransforms.scattered_planar_scattering(n1, n2, (Ny, Nx), J;
            L=L, max_order=2, period=(Ny, Nx), spectral=ScatteringTransforms.Plans.FINUFFTBackend())
        Test.@test cd ≈ ScatteringTransforms.Coefficients.flatten2d(fin(vec(f))) rtol=1e-6
    end

    Test.@testset "gappy, over-specified mode grid: the solve stays finite" begin
        # Fewer samples than modes, with a void: a coastal-band footprint over a 24×24 grid keeps 300
        # of 576 modes sampled, so at least 276 mode directions are constrained by no sample at all.
        # The least-squares problem is then rank-deficient, and an unguarded solve does not merely lose
        # accuracy — it diverges and writes NaN into the coefficients with nothing raised, which is how
        # a whole downstream product came out NaN before this was traced.
        ms_g = (24, 24)
        allx, ally = Float64[], Float64[]
        for i in 1:ms_g[1], j in 1:ms_g[2]
            j <= round(Int, 11 + 5 * sin(pi * i / ms_g[1])) && (push!(allx, i); push!(ally, j))
        end
        Random.seed!(0)
        keep = Random.randperm(length(allx))[1:300]
        for T in (Float32, Float64)
            xg, yg = T.(allx[keep]), T.(ally[keep])
            fg = T[abs(sin(3xg[k])) + 0.1 for k in eachindex(xg)]
            # Construction says so: this is the geometry the underdetermined warning exists for.
            st_g = Test.@test_logs (:warn, r"underdetermined") match_mode=:any (
                ScatteringTransforms.scattered_planar_scattering(T, xg, yg, ms_g, 3;
                    L = 4, max_order = 2, period = (T(ms_g[1]), T(ms_g[2])), solve = true))
            c = st_g(fg)
            Test.@test all(isfinite, ScatteringTransforms.Coefficients.first_order(c))
            Test.@test all(isfinite, ScatteringTransforms.Coefficients.second_order(c))
        end
    end

    Test.@testset "S0 is the (weighted) sample mean" begin
        sca = ScatteringTransforms.scattered_planar_scattering(n1, n2, (Ny, Nx), J; L=L, max_order=1, period=(Ny, Nx))
        Test.@test ScatteringTransforms.Coefficients.zeroth_order(sca(vec(f))) ≈ sum(f) / length(f)
    end

    Test.@testset "irregular points: CG solve recovers a band-limited field" begin
        Random.seed!(1)
        M = 6000
        px = 2π .* rand(M)
        py = 2π .* rand(M)
        g(x, y) = 1.0 + 0.7cos(x) + 0.5sin(2y) - 0.3cos(x) * sin(y)
        b = [g(px[k], py[k]) for k in 1:M]
        sc = ScatteringTransforms.scattered_planar_scattering(px, py, (Ny, Nx), J;
            L=L, max_order=2, period=(2π, 2π), solve=true)
        r = sc(b)
        Test.@test ScatteringTransforms.Coefficients.zeroth_order(r) ≈ sum(b) / M rtol=1e-6
        Test.@test all(isfinite, ScatteringTransforms.Coefficients.first_order(r))
        Test.@test all(ScatteringTransforms.Coefficients.first_order(r) .>= 0)
        # Planar admissible (scale-increasing) pairs populate the strict upper triangle (j2 > j1,
        # matching `flatten2d`); the diagonal and lower triangle stay exactly zero.
        S2 = ScatteringTransforms.Coefficients.second_order(r)
        n = J * L
        Test.@test all(S2[j1, j2] == 0 for j1 in 1:n for j2 in 1:j1)
        Test.@test any(S2[j1, j2] > 0 for j1 in 1:n for j2 in (j1 + 1):n)
    end

    Test.@testset "threaded batch, on every NUFFT backend" begin
        # A nonuniform plan carries the buffers each execution writes through, so tasks must not
        # share one. Sharing does not fail loudly — it silently corrupts concurrent transforms — so
        # this runs the threaded batch on each backend and requires it to reproduce the serial
        # result. The in-core plan alone would not catch it: the fast backends own the state.
        Random.seed!(2)
        M2, B = 800, 8
        xs, ys = rand(M2), rand(M2)
        Xb = randn(M2, B)
        for spec in (ScatteringTransforms.Plans.FINUFFTBackend(),
                     SpectralBackends.DirectSumSpectralBackend())
            sp = ScatteringTransforms.scattered_planar_scattering(xs, ys, (16, 16), 3;
                                                                  L = 4, spectral = spec)
            serial = ScatteringTransforms.scattering_batch(sp, Xb)
            threaded = ScatteringTransforms.scattering_batch(
                ComputationalBackends.ThreadedBackend(), sp, Xb)
            Test.@test all(isfinite, threaded)
            Test.@test threaded ≈ serial rtol=1e-10
        end
    end

    Test.@testset "ntrans batching matches the per-field loop" begin
        # FINUFFT transforms `ntrans` co-located fields per execution, so a transform built that way
        # runs each cascade step once for the whole stack instead of once per field. The coefficients
        # must be identical either way — the batching is an execution detail, not an approximation.
        Random.seed!(3)
        M2, B = 600, 4
        xs, ys = rand(M2), rand(M2)
        Xb = randn(M2, B)
        base = ScatteringTransforms.scattered_planar_scattering(xs, ys, (16, 16), 3;
            L = 4, spectral = ScatteringTransforms.Plans.FINUFFTBackend())
        batched = ScatteringTransforms.ScatteredPlanar.build(Float64, xs, ys, (16, 16), 3;
            L = 4, spectral = ScatteringTransforms.Plans.FINUFFTBackend(), ntrans = B)
        # Asserted, not assumed: if the plan silently came back single-field the comparison below
        # would pass while testing nothing.
        Test.@test ScatteringTransforms.Plans.batch_width(batched.plan) == B
        Test.@test ScatteringTransforms.scattering_batch(batched, Xb) ≈
                   ScatteringTransforms.scattering_batch(base, Xb) rtol=1e-12
        Test.@test ScatteringTransforms.scattering_batch(
                       ComputationalBackends.ThreadedBackend(), batched, Xb) ≈
                   ScatteringTransforms.scattering_batch(base, Xb) rtol=1e-12
        # Its buffers and guru plan are `B` wide, so any other stack size is refused, not reshaped.
        Test.@test_throws DimensionMismatch ScatteringTransforms.scattering_batch(batched,
                                                                                 Xb[:, 1:3])
    end

    Test.@testset "threaded batches with overlapping tasks match the serial batch" begin
        # The assertion above runs `B == ntrans`, which is a single chunk and therefore a single task —
        # it cannot see a race. These sizes give one task per column (per-field path) and eight
        # overlapping chunks (batched path), which is where a plan shared between tasks shows up: the
        # FINUFFT guru plan writes through its own buffers, so two tasks executing one would return
        # NaN rather than a slightly different answer.
        Random.seed!(13)
        M2, B2, W2 = 1500, 32, 4
        xs, ys = rand(M2), rand(M2)
        Xb = randn(M2, B2)
        st1 = ScatteringTransforms.scattered_planar_scattering(xs, ys, (32, 32), 3;
            L = 4, max_order = 2, spectral = ScatteringTransforms.Plans.FINUFFTBackend())
        serial = ScatteringTransforms.scattering_batch(ComputationalBackends.SerialBackend(), st1, Xb)
        Test.@test !any(isnan, serial)

        perfield = ScatteringTransforms.scattering_batch(
            ComputationalBackends.ThreadedBackend(), st1, Xb)
        Test.@test !any(isnan, perfield)
        Test.@test perfield ≈ serial rtol=1e-12

        stb = ScatteringTransforms.ScatteredPlanar.build(Float64, xs, ys, (32, 32), 3;
            L = 4, max_order = 2, spectral = ScatteringTransforms.Plans.FINUFFTBackend(),
            ntrans = W2)
        Test.@test ScatteringTransforms.Plans.batch_width(stb.plan) == W2
        chunked = ScatteringTransforms.scattering_batch(
            ComputationalBackends.ThreadedBackend(), stb, Xb)
        Test.@test !any(isnan, chunked)
        Test.@test chunked ≈ serial rtol=1e-12
    end

    Test.@testset "an explicit nufft_nthreads survives into the per-task plans" begin
        # A threaded backend applies task-local copies, so a thread count honoured only at construction
        # is a keyword that does nothing where it matters most. Unset, a per-task copy takes one thread
        # rather than a share of `Sys.CPU_THREADS`: those are logical cores, so on a hyperthreaded
        # machine already running one Julia task per physical core the arithmetic hands out a second
        # library thread per task.
        M2 = 500
        xs, ys = rand(M2), rand(M2)
        build(n) = ScatteringTransforms.scattered_planar_scattering(xs, ys, (16, 16), 3;
            L = 4, max_order = 2, spectral = ScatteringTransforms.Plans.FINUFFTBackend(),
            nufft_nthreads = n)
        for n in (1, 3)
            st = build(n)
            Test.@test st.plan.nthreads == n
            Test.@test ScatteringTransforms.Plans.task_local_plan(st.plan).nthreads == n
        end
        st0 = build(0)
        Test.@test st0.plan.nthreads == 0                                   # the library's own choice
        Test.@test ScatteringTransforms.Plans.task_local_plan(st0.plan).nthreads == 1
        Test.@test ScatteringTransforms.Plans.per_task_nthreads(0) == 1
        Test.@test ScatteringTransforms.Plans.per_task_nthreads(5) == 5
    end

    Test.@testset "the three solve paths agree on irregular points" begin
        # The existing cross-backend comparison runs `solve = false`, so a convention divergence in one
        # backend's least-squares path was invisible. Same points, same field, all three solvers.
        Random.seed!(17)
        M2 = 1200
        xs, ys = 2π .* rand(M2), 2π .* rand(M2)
        fs = [1.0 + 0.7cos(xs[k]) + 0.5sin(2ys[k]) - 0.3cos(xs[k]) * sin(ys[k]) for k in 1:M2]
        coeffs(spec) = ScatteringTransforms.Coefficients.flatten2d(
            ScatteringTransforms.scattered_planar_scattering(xs, ys, (16, 16), 3;
                L = 4, max_order = 2, period = (2π, 2π), solve = true, spectral = spec)(fs))
        direct = coeffs(SpectralBackends.DirectSumSpectralBackend())
        finufft = coeffs(ScatteringTransforms.Plans.FINUFFTBackend())
        nuffts = coeffs(ScatteringTransforms.Plans.NonuniformFFTsBackend())
        Test.@test all(isfinite, direct)
        Test.@test finufft ≈ direct rtol = 1e-5
        Test.@test nuffts ≈ direct rtol = 1e-5
        Test.@test nuffts ≈ finufft rtol = 1e-5
    end

    Test.@testset "the solve leaves its input untouched" begin
        # The cascade hands `forward_transform!` a buffer it overwrites immediately afterwards
        # (`ScatteredPlanar.jl:138,151`), so the solver must treat the samples as read-only. An
        # in-place "optimisation" that aliased them would corrupt the next cascade step silently.
        Random.seed!(19)
        M2 = 400
        xs, ys = 2π .* rand(M2), 2π .* rand(M2)
        b = [1.0 + 0.4cos(xs[k]) for k in 1:M2]
        before = copy(b)
        for spec in (SpectralBackends.DirectSumSpectralBackend(),
                     ScatteringTransforms.Plans.FINUFFTBackend(),
                     ScatteringTransforms.Plans.NonuniformFFTsBackend())
            plan = ScatteringTransforms.Plans.make_scattered_plan(spec, xs, ys, (16, 16), Float64;
                period = (2π, 2π), solve = true)
            X = zeros(ComplexF64, (16, 16))
            ScatteringTransforms.Plans.forward_transform!(X, plan, b)
            Test.@test b == before
        end
    end

    Test.@testset "solve=true under the threaded backend and batched over ntrans" begin
        # Two gaps at once: the threaded comparison ran `solve = false`, which is what would have hidden
        # a `task_local_plan` field left un-copied, and the batched solve was refused outright before.
        Random.seed!(23)
        M2, B2 = 800, 4
        xs, ys = 2π .* rand(M2), 2π .* rand(M2)
        X = hcat([[1.0 + 0.7cos(xs[k]) + 0.5sin(2ys[k]) + 0.05c for k in 1:M2] for c in 1:B2]...)
        FB = ScatteringTransforms.Plans.FINUFFTBackend()
        single = ScatteringTransforms.scattered_planar_scattering(xs, ys, (16, 16), 3;
            L = 4, max_order = 2, period = (2π, 2π), solve = true, spectral = FB)
        serial = ScatteringTransforms.scattering_batch(
            ComputationalBackends.SerialBackend(), single, X)
        Test.@test all(isfinite, serial)
        Test.@test ScatteringTransforms.scattering_batch(
            ComputationalBackends.ThreadedBackend(), single, X) ≈ serial rtol = 1e-10

        batched = ScatteringTransforms.ScatteredPlanar.build(Float64, xs, ys, (16, 16), 3;
            L = 4, max_order = 2, period = (2π, 2π), solve = true, spectral = FB, ntrans = B2)
        Test.@test ScatteringTransforms.Plans.batch_width(batched.plan) == B2
        # Asserted, not assumed: without the per-column bookkeeping the batched path is not running.
        Test.@test batched.plan.ls_batch !== nothing
        Test.@test ScatteringTransforms.scattering_batch(
            ComputationalBackends.SerialBackend(), batched, X) ≈ serial rtol = 1e-6
    end

    Test.@testset "a transform rebuilt from its spec analyses the same way" begin
        # A distributed worker cannot receive a plan, so it receives a spec and rebuilds. Everything
        # that decides what the analysis *is* has to survive that trip: `solve` selects a
        # least-squares inversion over the Type-1 adjoint, which on irregular points is a different
        # transform, and the tolerance and CG settings decide how exactly it is solved.
        Random.seed!(11)
        M2 = 800
        xs, ys = 2π .* rand(M2), 2π .* rand(M2)
        fs = [1.0 + 0.7cos(xs[k]) + 0.5sin(2ys[k]) for k in 1:M2]
        for spectral in (SpectralBackends.DirectSumSpectralBackend(),
                         ScatteringTransforms.Plans.FINUFFTBackend(),
                         ScatteringTransforms.Plans.NonuniformFFTsBackend())
            st = ScatteringTransforms.scattered_planar_scattering(xs, ys, (16, 16), 3;
                L = 4, max_order = 2, period = (2π, 2π), spectral = spectral,
                solve = true, maxiter = 37, rtol = 1.0e-7, eps = 1.0e-8, damp = 0.25)
            rb = ScatteringTransforms.rebuild_transform(ScatteringTransforms.transform_spec(st))
            a0 = ScatteringTransforms.Plans.plan_analysis(st.plan)
            a1 = ScatteringTransforms.Plans.plan_analysis(rb.plan)
            Test.@test a1.solve == a0.solve == true
            Test.@test a1.maxiter == 37
            Test.@test a1.rtol == a0.rtol
            Test.@test a1.eps == a0.eps
            # A rebuild that dropped the regulariser would solve a different problem on a worker than
            # on the caller. Asserted against the value passed, not just against `a0`, since the
            # default is zero and `0 == 0` would hold however badly the field were plumbed.
            Test.@test a1.damp == a0.damp == 0.25
            # And the same settings must give the same coefficients, not merely the same fields.
            Test.@test ScatteringTransforms.Coefficients.flatten2d(rb(fs)) ≈
                       ScatteringTransforms.Coefficients.flatten2d(st(fs))
        end
    end

    Test.@testset "NonuniformFFTs backend agrees with FINUFFT and direct summation" begin
        # Three independent transforms of the same field on the same points: exact direct
        # summation, FINUFFT, and NonuniformFFTs. They lay their modes out on the same fftfreq
        # lattice, so the coefficients must agree to the loosest of the three tolerances.
        Random.seed!(7)
        M2 = 600
        xs, ys = rand(M2), rand(M2)
        fs = randn(M2)
        coeffs(spec) = ScatteringTransforms.Coefficients.flatten2d(
            ScatteringTransforms.scattered_planar_scattering(xs, ys, (16, 16), 3;
                                                             L = 4, spectral = spec)(fs))
        direct = coeffs(SpectralBackends.DirectSumSpectralBackend())
        finufft = coeffs(ScatteringTransforms.Plans.FINUFFTBackend())
        nuffts = coeffs(ScatteringTransforms.Plans.NonuniformFFTsBackend())
        Test.@test finufft ≈ direct rtol=1e-6
        Test.@test nuffts ≈ direct rtol=1e-6
        Test.@test nuffts ≈ finufft rtol=1e-6
    end
end
