# Nonuniform / scattered planar scattering via NUFFT. On a uniform grid the analysis reduces to the
# fft, so scattered-planar scattering reproduces the gridded ScatteringTransform2D exactly (to NUFFT
# tolerance); on irregular points the least-squares solve recovers a band-limited field.
using Random: Random

Test.@testset "Scattered / nonuniform planar scattering (NUFFT)" begin
    Ny, Nx, J, L = 24, 24, 3, 4

    # Uniform-grid points (axis-1 index fastest ⇒ matches column-major vec of the (Ny,Nx) image).
    n1 = vec([Float64(i) for i in 0:Ny-1, j in 0:Nx-1])
    n2 = vec([Float64(j) for i in 0:Ny-1, j in 0:Nx-1])
    f = randn(Ny, Nx)
    grid = ScatteringTransforms.Scattering2D.ScatteringTransform2D((Ny, Nx), J;
        L=L, max_order=2, transform=SpectralBackends.FFTSpectralBackend())
    ref = ScatteringTransforms.Coefficients.flatten2d(grid(f))

    Test.@testset "uniform-grid parity reproduces the gridded FFT transform" begin
        for solve in (false, true)
            sca = ScatteringTransforms.scattered_planar_scattering(n1, n2, (Ny, Nx), J;
                L=L, max_order=2, period=(Ny, Nx), solve=solve)
            c = sca(vec(f))
            Test.@test ScatteringTransforms.Coefficients.flatten2d(c) ≈ ref rtol=1e-6
        end
    end

    Test.@testset "dependency-free direct NUDFT backend (no external library)" begin
        # Exact direct summation: on the uniform grid it reproduces the gridded FFT transform, and
        # matches the FINUFFT path.
        dir = ScatteringTransforms.scattered_planar_scattering(n1, n2, (Ny, Nx), J;
            L=L, max_order=2, period=(Ny, Nx), transform=SpectralBackends.DirectSumSpectralBackend())
        Test.@test dir.plan isa ScatteringTransforms.Plans.DirectNUFFTPlan
        cd = ScatteringTransforms.Coefficients.flatten2d(dir(vec(f)))
        Test.@test cd ≈ ref rtol=1e-9
        fin = ScatteringTransforms.scattered_planar_scattering(n1, n2, (Ny, Nx), J;
            L=L, max_order=2, period=(Ny, Nx), transform=FTB.FINUFFTBackend())
        Test.@test cd ≈ ScatteringTransforms.Coefficients.flatten2d(fin(vec(f))) rtol=1e-6
    end

    Test.@testset "gappy, over-specified mode grid: the solve stays finite" begin
        # Fewer samples than modes, with a void: a coastal-band footprint over a 24×24 grid keeps 300
        # samples for 576 modes, so the least-squares problem is rank-deficient. The coefficients stay
        # finite.
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
            # Construction warns that the solve is underdetermined.
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

    # A plane wave on mode `k` of a domain of size `P`: the analytic wavelets keep one of `±k`, so
    # `|x ⋆ ψ_j|` is the constant `½|ψ̂_j(k)| + ½|ψ̂_j(-k)|` (one term is zero) and so is `S1[j]`, and its
    # modulus has no oscillating part to pass on, so `S2 = 0`. Any points that determine the modes give
    # these values once the transform knows `P`.
    Pw, msw, Jw, Lw, kw = (3.0, 5.0), (16, 16), 3, 4, (3, 2)
    wave(x, y) = cos(2π * (kw[1] * x / Pw[1] + kw[2] * y / Pw[2]))
    fbw = ScatteringTransforms.FilterBanks.build_filter_bank2d(Float64, msw, Jw; L = Lw)
    S1wave = [(abs(ψ[kw[1] + 1, kw[2] + 1]) + abs(ψ[msw[1] - kw[1] + 1, msw[2] - kw[2] + 1])) / 2
              for ψ in fbw.wavelets]
    C = ScatteringTransforms.Coefficients

    Test.@testset "the period sets each mode's wavelength, whatever the point set" begin
        Random.seed!(41)
        Mw = 900
        x1, y1 = Pw[1] .* rand(Mw), Pw[2] .* rand(Mw)
        # Other points, every other one given a period away.
        x2 = [Pw[1] * rand() + (isodd(i) ? Pw[1] : 0.0) for i in 1:Mw]
        y2 = [Pw[2] * rand() - (isodd(i) ? Pw[2] : 0.0) for i in 1:Mw]
        for spec in (SpectralBackends.DirectSumSpectralBackend(), FTB.FINUFFTBackend(),
                     FTB.NonuniformFFTsBackend()), (x, y) in ((x1, y1), (x2, y2))
            st = ScatteringTransforms.scattered_planar_scattering(x, y, msw, Jw; period = Pw, L = Lw,
                transform = spec, solve = true, maxiter = 2000, rtol = 1e-12, eps = 1e-12,
                nufft_nthreads = 1)
            c = st(wave.(x, y))
            Test.@test C.first_order(c) ≈ S1wave rtol = 1e-6
            Test.@test maximum(abs, C.second_order(c)) < 1e-6 * maximum(S1wave)
        end
    end

    Test.@testset "a FlowGeometries grid gives the points, period, origin and weights" begin
        FGG = FlowGeometries.Grids
        geo = FlowGeometries.Geometry.CartesianGeometry()
        DS = SpectralBackends.DirectSumSpectralBackend()

        # A node set on a periodic domain.
        Random.seed!(43)
        Mw = 900
        x, y = Pw[1] .* rand(Mw), Pw[2] .* rand(Mw)
        nodes = FGG.UnstructuredGrid(geo, (x, y), ones(Mw); periodic = (true, true), period = Pw)
        st = ScatteringTransforms.scattered_planar_scattering(nodes, msw, Jw; L = Lw, transform = DS,
                                                              solve = true, maxiter = 2000, rtol = 1e-12)
        Test.@test C.first_order(st(wave.(x, y))) ≈ S1wave rtol = 1e-6

        # Its cell measure is the default quadrature of the mean.
        area = 0.5 .+ rand(Mw)
        weighted = FGG.UnstructuredGrid(geo, (x, y), area; periodic = (true, true), period = Pw)
        g = randn(Mw)
        stw = ScatteringTransforms.scattered_planar_scattering(weighted, msw, Jw; L = Lw, transform = DS,
                                                               max_order = 1)
        Test.@test C.zeroth_order(stw(g)) ≈ sum(area .* g) / sum(area) rtol = 1e-12

        # A bounded lattice away from zero: its length is `n` cell widths and its first cell centre the
        # origin, so its cells are the nodes of the DFT, where the adjoint is exact.
        x0, y0 = 10.0, -4.0
        xa = range(x0; step = Pw[1] / msw[1], length = msw[1])
        ya = range(y0; step = Pw[2] / msw[2], length = msw[2])
        lattice = FGG.StructuredGrid(geo, xa, ya)
        stl = ScatteringTransforms.scattered_planar_scattering(lattice, msw, Jw; L = Lw, transform = DS)
        fl = [wave(xi - x0, yj - y0) for xi in xa, yj in ya]
        Test.@test C.first_order(stl(vec(fl))) ≈ S1wave rtol = 1e-10
    end

    Test.@testset "irregular points: the least-squares solve recovers a band-limited field" begin
        Random.seed!(1)
        M = 6000
        px = 2π .* rand(M)
        py = 2π .* rand(M)
        g(x, y) = 1.0 + 0.7cos(x) + 0.5sin(2y) - 0.3cos(x) * sin(y)
        b = [g(px[k], py[k]) for k in 1:M]
        sc = ScatteringTransforms.scattered_planar_scattering(px, py, (Ny, Nx), J;
            L=L, max_order=2, period=(2π, 2π), solve=true)
        r = sc(b)
        # `S0` is the mean of the fitted field, `g`'s own mean.
        Test.@test ScatteringTransforms.Coefficients.zeroth_order(r) ≈ 1.0 rtol=1e-6
        Test.@test all(ScatteringTransforms.Coefficients.first_order(r) .>= 0)
        # Planar admissible (scale-increasing) pairs populate the strict upper triangle (j2 > j1,
        # matching `flatten2d`); the diagonal and lower triangle stay exactly zero.
        S2 = ScatteringTransforms.Coefficients.second_order(r)
        n = J * L
        Test.@test all(S2[j1, j2] == 0 for j1 in 1:n for j2 in 1:j1)
        Test.@test any(S2[j1, j2] > 0 for j1 in 1:n for j2 in (j1 + 1):n)
    end

    Test.@testset "threaded batch, on every NUFFT backend" begin
        # A nonuniform plan carries the buffers each execution writes through, and each task gets its
        # own.
        Random.seed!(2)
        M2, B = 800, 8
        xs, ys = rand(M2), rand(M2)
        Xb = randn(M2, B)
        for spec in (FTB.FINUFFTBackend(), FTB.NonuniformFFTsBackend(),
                     SpectralBackends.DirectSumSpectralBackend())
            sp = ScatteringTransforms.scattered_planar_scattering(xs, ys, (16, 16), 3;
                                                                  period = 1.0, L = 4, transform = spec)
            serial = ScatteringTransforms.scattering_batch(sp, Xb)
            threaded = ScatteringTransforms.scattering_batch(
                sp, Xb; execution = ComputationalBackends.ThreadedBackend())
            Test.@test threaded ≈ serial rtol=1e-10
        end
    end

    Test.@testset "ntrans batching matches the per-field loop" begin
        # FINUFFT transforms `ntrans` co-located fields per execution, so a transform built that way
        # runs each cascade step once for the whole stack.
        Random.seed!(3)
        M2, B = 600, 4
        xs, ys = rand(M2), rand(M2)
        Xb = randn(M2, B)
        base = ScatteringTransforms.scattered_planar_scattering(xs, ys, (16, 16), 3;
            period = 1.0, L = 4, transform = FTB.FINUFFTBackend())
        batched = ScatteringTransforms.ScatteredPlanar.build(Float64, xs, ys, (16, 16), 3;
            period = 1.0, L = 4, transform = FTB.FINUFFTBackend(), ntrans = B)
        # Precondition: the plan is `B` wide.
        Test.@test ScatteringTransforms.Plans.batch_width(batched.plan) == B
        Test.@test ScatteringTransforms.scattering_batch(batched, Xb) ≈
                   ScatteringTransforms.scattering_batch(base, Xb) rtol=1e-12
        Test.@test ScatteringTransforms.scattering_batch(
                       batched, Xb; execution = ComputationalBackends.ThreadedBackend()) ≈
                   ScatteringTransforms.scattering_batch(base, Xb) rtol=1e-12
        # Stacks of any width run `B` columns at a time, the last group padded: three columns (one
        # short group), seven (a full group and a short one), and none, as an MPI rank's empty block.
        Xw = randn(M2, 7)
        for cols in (1:3, 1:7, 1:0)
            want = ScatteringTransforms.scattering_batch(base, Xw[:, cols])
            for backend in (ComputationalBackends.SerialBackend(), ComputationalBackends.ThreadedBackend())
                got = ScatteringTransforms.scattering_batch(batched, Xw[:, cols]; execution = backend)
                Test.@test size(got) == size(want)
                Test.@test got ≈ want rtol=1e-12
            end
        end
    end

    Test.@testset "the exact direct sum transforms ntrans fields per call" begin
        # Each column of a `B`-wide direct sum is the same arithmetic as a single-field one.
        Random.seed!(4)
        M2, B = 300, 3
        xs, ys = 2π .* rand(M2), 2π .* rand(M2)
        Xb = randn(M2, 5)
        for solve in (false, true)
            opts = (; period = 2π, L = 4, solve = solve, maxiter = 1000, rtol = 1e-12,
                    transform = SpectralBackends.DirectSumSpectralBackend())
            single = ScatteringTransforms.ScatteredPlanar.build(Float64, xs, ys, (12, 12), 3; opts...)
            wide = ScatteringTransforms.ScatteredPlanar.build(Float64, xs, ys, (12, 12), 3;
                                                              ntrans = B, opts...)
            Test.@test ScatteringTransforms.Plans.batch_width(wide.plan) == B
            Test.@test ScatteringTransforms.scattering_batch(wide, Xb) ≈
                       ScatteringTransforms.scattering_batch(single, Xb) rtol = 1e-12
        end
    end

    Test.@testset "a complex stack gives complex S0 rows, on every backend and width" begin
        # `S1`/`S2` are moduli; `S0` is the complex field's mean, here checked against `sum/M`.
        Random.seed!(8)
        M2 = 400
        xs, ys = rand(M2), rand(M2)
        Z = randn(ComplexF64, M2, 5)
        for (spec, ntrans) in ((SpectralBackends.DirectSumSpectralBackend(), 1),
                               (FTB.FINUFFTBackend(), 1), (FTB.FINUFFTBackend(), 2))
            st = ScatteringTransforms.ScatteredPlanar.build(Float64, xs, ys, (12, 12), 3;
                period = 1.0, L = 4, transform = spec, ntrans = ntrans, nufft_nthreads = 1)
            single = ScatteringTransforms.scattered_planar_scattering(xs, ys, (12, 12), 3;
                period = 1.0, L = 4, transform = spec, nufft_nthreads = 1)
            for backend in (ComputationalBackends.SerialBackend(), ComputationalBackends.ThreadedBackend())
                out = ScatteringTransforms.scattering_batch(st, Z; execution = backend)
                Test.@test eltype(out) == ComplexF64
                Test.@test out[ScatteringTransforms.Coefficients.flat_row_s0(), :] ≈
                           vec(sum(Z; dims = 1)) ./ M2 rtol = 1e-12
                for b in axes(Z, 2)
                    Test.@test out[:, b] ≈
                               ScatteringTransforms.Coefficients.flatten2d(single(Z[:, b])) rtol = 1e-10
                end
            end
        end
    end

    Test.@testset "threaded batches with overlapping tasks match the serial batch" begin
        # One task per column (per-field path) and eight overlapping chunks (batched path); the
        # FINUFFT guru plan writes through its own buffers.
        Random.seed!(13)
        M2, B2, W2 = 1500, 32, 4
        xs, ys = rand(M2), rand(M2)
        Xb = randn(M2, B2)
        st1 = ScatteringTransforms.scattered_planar_scattering(xs, ys, (32, 32), 3;
            period = 1.0, L = 4, max_order = 2, transform = FTB.FINUFFTBackend())
        serial = ScatteringTransforms.scattering_batch(st1, Xb; execution = ComputationalBackends.SerialBackend())

        perfield = ScatteringTransforms.scattering_batch(
            st1, Xb; execution = ComputationalBackends.ThreadedBackend())
        Test.@test perfield ≈ serial rtol=1e-12

        stb = ScatteringTransforms.ScatteredPlanar.build(Float64, xs, ys, (32, 32), 3;
            period = 1.0, L = 4, max_order = 2, transform = FTB.FINUFFTBackend(),
            ntrans = W2)
        Test.@test ScatteringTransforms.Plans.batch_width(stb.plan) == W2
        chunked = ScatteringTransforms.scattering_batch(
            stb, Xb; execution = ComputationalBackends.ThreadedBackend())
        Test.@test chunked ≈ serial rtol=1e-12
    end

    Test.@testset "an explicit nufft_nthreads survives into the per-task plans" begin
        # A threaded backend applies task-local copies. Unset (`0`), a per-task copy takes one thread.
        M2 = 500
        xs, ys = rand(M2), rand(M2)
        build(n) = ScatteringTransforms.scattered_planar_scattering(xs, ys, (16, 16), 3;
            period = 1.0, L = 4, max_order = 2, transform = FTB.FINUFFTBackend(),
            nufft_nthreads = n)
        for n in (1, 3)
            st = build(n)
            Test.@test st.plan.nthreads == n
            Test.@test ScatteringTransforms.Plans.task_local_plan(st.plan).nthreads == n
        end
        Test.@test ScatteringTransforms.Plans.task_local_plan(build(0).plan).nthreads == 1
    end

    Test.@testset "every solve path recovers a band-limited field it was given" begin
        # Real samples have a Hermitian spectrum: the nonuniform FFTs fit them with the real series of the
        # half grid, both `±N/2` included, and the direct sum fits the complex `fftfreq` grid. The two
        # models differ at the unpaired `±N/2` frequency.
        #
        # The field here lies in both: Hermitian coefficients, zero on the modes an `fftfreq` grid cannot
        # pair, giving a real band-limited field with a known answer to recover.
        Random.seed!(17)
        ms_r, M2 = (16, 16), 1200
        xs, ys = 2π .* rand(M2), 2π .* rand(M2)
        neg(j, N) = (hi = (N - 1) ÷ 2; f = j - 1 <= hi ? j - 1 : j - 1 - N;
                     -f > hi ? 0 : (-f >= 0 ? -f + 1 : -f + N + 1))
        F0 = zeros(ComplexF64, ms_r)
        for j in 1:ms_r[2], i in 1:ms_r[1]
            ii, jj = neg(i, ms_r[1]), neg(j, ms_r[2])
            (ii == 0 || jj == 0) && continue            # unpaired: a real field cannot use it
            iszero(F0[i, j]) || continue
            z = randn(ComplexF64) / (1 + abs(i - 1) + abs(j - 1))
            (i, j) == (ii, jj) ? (F0[i, j] = real(z)) : (F0[i, j] = z; F0[ii, jj] = conj(z))
        end
        ref = ScatteringTransforms.Plans.make_scattered_plan(
            SpectralBackends.DirectSumSpectralBackend(), xs, ys, ms_r, Float64; period = (2π, 2π))
        pts = zeros(ComplexF64, M2)
        ScatteringTransforms.Plans.inverse_transform!(pts, ref, F0)
        Test.@test maximum(abs ∘ imag, pts) < 1e-12 * maximum(abs ∘ real, pts)   # it is real
        fs = real.(pts)
        for spec in (SpectralBackends.DirectSumSpectralBackend(),
                     FTB.FINUFFTBackend(),
                     FTB.NonuniformFFTsBackend())
            # Converged settings, compared against the exact coefficients.
            plan = ScatteringTransforms.Plans.make_scattered_plan(spec, xs, ys, ms_r, Float64;
                period = (2π, 2π), solve = true, maxiter = 2000, rtol = 1.0e-12,
                nufft_nthreads = 1)
            F = zeros(ComplexF64, ms_r)
            ScatteringTransforms.Plans.forward_transform!(F, plan, fs)
            Test.@test maximum(abs, F .- F0) < 1.0e-6 * maximum(abs, F0)
        end
    end

    Test.@testset "a real solve on the uniform nodes is the DFT, Nyquist included" begin
        # On the nodes `2πj/m` the solve fits the samples exactly and `±m/2` coincide, so the
        # coefficients are `fft` of the samples, the even axes' `-m/2` row and column included.
        Random.seed!(29)
        for ms_u in ((24, 22), (23, 22), (24, 21), (23, 21)), spec in (FTB.FINUFFTBackend(), FTB.NonuniformFFTsBackend())
            gx = [2π * (i - 1) / ms_u[1] for i in 1:ms_u[1], j in 1:ms_u[2]]
            gy = [2π * (j - 1) / ms_u[2] for i in 1:ms_u[1], j in 1:ms_u[2]]
            fu = randn(ms_u)
            plan = ScatteringTransforms.Plans.make_scattered_plan(spec, vec(gx), vec(gy), ms_u, Float64;
                period = (2π, 2π), solve = true, maxiter = 2000, rtol = 1.0e-12, eps = 1.0e-12,
                nufft_nthreads = 1)
            F = zeros(ComplexF64, ms_u)
            ScatteringTransforms.Plans.forward_transform!(F, plan, vec(fu))
            Test.@test F ≈ FFTW.fft(fu) rtol = 1.0e-9
            ScatteringTransforms.Plans.close_plan!(plan)
        end
    end

    Test.@testset "the solve leaves its input untouched" begin
        # The cascade reuses the sample buffer after `forward_transform!`, so the solve leaves it
        # unchanged.
        Random.seed!(19)
        M2 = 400
        xs, ys = 2π .* rand(M2), 2π .* rand(M2)
        b = [1.0 + 0.4cos(xs[k]) for k in 1:M2]
        before = copy(b)
        for spec in (SpectralBackends.DirectSumSpectralBackend(),
                     FTB.FINUFFTBackend(),
                     FTB.NonuniformFFTsBackend())
            plan = ScatteringTransforms.Plans.make_scattered_plan(spec, xs, ys, (16, 16), Float64;
                period = (2π, 2π), solve = true)
            X = zeros(ComplexF64, (16, 16))
            ScatteringTransforms.Plans.forward_transform!(X, plan, b)
            Test.@test b == before
        end
    end

    Test.@testset "solve=true under the threaded backend and batched over ntrans" begin
        # Both fast backends; `batch_width` is asserted on each batched plan.
        Random.seed!(23)
        M2, B2 = 800, 4
        xs, ys = 2π .* rand(M2), 2π .* rand(M2)
        X = hcat([[1.0 + 0.7cos(xs[k]) + 0.5sin(2ys[k]) + 0.05c for k in 1:M2] for c in 1:B2]...)
        # Converged settings: serial, threaded and batched solves stopped on a tolerance end an
        # iteration or two apart, and differ by about that tolerance.
        conv = (; maxiter = 2000, rtol = 1.0e-12)
        for spec in (FTB.FINUFFTBackend(),
                     FTB.NonuniformFFTsBackend())
            single = ScatteringTransforms.scattered_planar_scattering(xs, ys, (16, 16), 3;
                L = 4, max_order = 2, period = (2π, 2π), solve = true, transform = spec, conv...)
            serial = ScatteringTransforms.scattering_batch(
                single, X; execution = ComputationalBackends.SerialBackend())
            Test.@test ScatteringTransforms.scattering_batch(
                single, X; execution = ComputationalBackends.ThreadedBackend()) ≈ serial rtol = 1e-8

            batched = ScatteringTransforms.ScatteredPlanar.build(Float64, xs, ys, (16, 16), 3;
                L = 4, max_order = 2, period = (2π, 2π), solve = true, transform = spec,
                ntrans = B2, conv...)
            Test.@test ScatteringTransforms.Plans.batch_width(batched.plan) == B2
            Test.@test ScatteringTransforms.scattering_batch(
                batched, X; execution = ComputationalBackends.SerialBackend()) ≈ serial rtol = 1e-8
            # A per-task rebuild keeps the width.
            Test.@test ScatteringTransforms.Plans.batch_width(
                ScatteringTransforms.Plans.task_local_plan(batched.plan)) == B2
            Test.@test ScatteringTransforms.scattering_batch(
                batched, X; execution = ComputationalBackends.ThreadedBackend()) ≈ serial rtol = 1e-8
        end
    end

    Test.@testset "a transform rebuilt from its spec analyses the same way" begin
        # A distributed worker receives a spec and rebuilds the plan. `solve`, the tolerances and
        # `damp` all survive the trip.
        Random.seed!(11)
        M2 = 800
        xs, ys = 2π .* rand(M2), 2π .* rand(M2)
        fs = [1.0 + 0.7cos(xs[k]) + 0.5sin(2ys[k]) for k in 1:M2]
        for spectral in (SpectralBackends.DirectSumSpectralBackend(),
                         FTB.FINUFFTBackend(),
                         FTB.NonuniformFFTsBackend())
            st = ScatteringTransforms.scattered_planar_scattering(xs, ys, (16, 16), 3;
                L = 4, max_order = 2, period = (2π, 2π), transform = spectral,
                solve = true, maxiter = 37, rtol = 1.0e-7, eps = 1.0e-8, damp = 0.25)
            rb = ScatteringTransforms.rebuild_transform(ScatteringTransforms.transform_spec(st))
            a0 = ScatteringTransforms.Plans.plan_analysis(st.plan)
            a1 = ScatteringTransforms.Plans.plan_analysis(rb.plan)
            Test.@test a1.solve == a0.solve == true
            Test.@test a1.maxiter == 37
            Test.@test a1.rtol == a0.rtol
            Test.@test a1.eps == a0.eps
            # Against the value passed: the default is zero.
            Test.@test a1.damp == a0.damp == 0.25
            Test.@test ScatteringTransforms.Coefficients.flatten2d(rb(fs)) ≈
                       ScatteringTransforms.Coefficients.flatten2d(st(fs))
        end
    end

    Test.@testset "NonuniformFFTs backend agrees with FINUFFT and direct summation" begin
        # Three independent transforms of the same field on the same points: exact direct
        # summation, FINUFFT, and NonuniformFFTs. They lay their modes out on the same fftfreq
        # lattice, so the coefficients agree to the NUFFT tolerance.
        Random.seed!(7)
        M2 = 600
        xs, ys = rand(M2), rand(M2)
        fs = randn(M2)
        coeffs(spec) = ScatteringTransforms.Coefficients.flatten2d(
            ScatteringTransforms.scattered_planar_scattering(xs, ys, (16, 16), 3;
                                                             period = 1.0, L = 4, transform = spec)(fs))
        direct = coeffs(SpectralBackends.DirectSumSpectralBackend())
        finufft = coeffs(FTB.FINUFFTBackend())
        nuffts = coeffs(FTB.NonuniformFFTsBackend())
        Test.@test finufft ≈ direct rtol=1e-6
        Test.@test nuffts ≈ direct rtol=1e-6
    end

    Test.@testset "A plan carries scratch only for the path it takes" begin
        # `task_local_plan` builds one plan per task, so anything a plan holds and never uses is
        # multiplied by the thread count. Gated by counting bytes, with `M ≫ prod(ms)` so point-space
        # scratch dominates.
        Random.seed!(21)
        ms = (32, 32)
        M2 = 16 * prod(ms)
        xs, ys = rand(M2), rand(M2)
        B = 4
        pt_real = M2 * B * sizeof(Float64)             # one (M, B) real point array
        mode_bytes = prod(ms) * B * sizeof(ComplexF64) # one (ms…, B) full mode array
        for spec in (FTB.NonuniformFFTsBackend(),
                     FTB.FINUFFTBackend())
            plan(solve) = ScatteringTransforms.Plans.make_scattered_plan(
                spec, xs, ys, ms, Float64; period = (1.0, 1.0), solve = solve, eps = 1e-8,
                ntrans = B, nufft_nthreads = 1)
            # A non-solving plan holds none of the solver's six vectors, four of them mode-space.
            Test.@test Base.summarysize(plan(false)) + 4 * mode_bytes <=
                       Base.summarysize(plan(true))
            # Past the NUFFT plans underneath, a non-solving plan holds its two coordinate vectors
            # and mode-space buffers only.
            p0 = plan(false)
            own = Base.summarysize(p0) - Base.summarysize(p0.cplan) - Base.summarysize(p0.rplan)
            Test.@test own < 2 * M2 * sizeof(Float64) + mode_bytes

            # The cascade's own arrays reach the transform without a copy, so running one through a
            # plan adds no buffer. NonuniformFFTs' plan holds a `TimerOutput` that grows the first time
            # each section is entered, so this is a bound below one point array.
            p = plan(false)
            F = zeros(ComplexF64, ms..., B)
            pc = zeros(ComplexF64, M2, B)
            base = Base.summarysize(p)
            ScatteringTransforms.Plans.forward_transform!(F, p, pc)
            ScatteringTransforms.Plans.inverse_transform!(pc, p, F)
            Test.@test Base.summarysize(p) - base < pt_real
            # An array the transform cannot take lands in a buffer built on demand.
            ScatteringTransforms.Plans.forward_transform!(F, p, zeros(Float32, M2, B))
            Test.@test Base.summarysize(p) - base >= pt_real
        end

        # The cascade analyses real fields after its first step; on the real-data backend that path
        # needs no complex point buffer.
        nr = ScatteringTransforms.Plans.make_scattered_plan(
            FTB.NonuniformFFTsBackend(), xs, ys, ms, Float64;
            period = (1.0, 1.0), solve = false, eps = 1e-8, ntrans = B, nufft_nthreads = 1)
        Fr = zeros(ComplexF64, ms..., B)
        base_r = Base.summarysize(nr)
        ScatteringTransforms.Plans.forward_transform!(Fr, nr, zeros(Float64, M2, B))
        Test.@test Base.summarysize(nr) - base_r < pt_real

        # A real field solves on the half grid, so the full-grid solver set is built only for a
        # complex field.
        nu = ScatteringTransforms.Plans.make_scattered_plan(
            FTB.NonuniformFFTsBackend(), xs, ys, ms, Float64;
            period = (1.0, 1.0), solve = true, eps = 1e-8, maxiter = 20, nufft_nthreads = 1)
        Xm = zeros(ComplexF64, ms)
        ScatteringTransforms.Plans.forward_transform!(Xm, nu, randn(M2))
        Test.@test nu.csolve[] === nothing
        ScatteringTransforms.Plans.forward_transform!(Xm, nu, randn(ComplexF64, M2))
        Test.@test nu.csolve[] !== nothing
    end
end
