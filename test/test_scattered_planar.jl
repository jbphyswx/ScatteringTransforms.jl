# Nonuniform / scattered planar scattering via NUFFT. On a uniform grid the NUFFT
# analysis/synthesis reduce to fft/ifft, so scattered-planar scattering must reproduce the gridded
# ScatteringTransform2D exactly (to NUFFT tolerance); on irregular points the CG-solve path recovers
# a band-limited field.
using FINUFFT: FINUFFT
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
end
