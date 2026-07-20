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
        L=L, max_order=2, spectral=ScatteringTransforms.Plans.FFTBackend())
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
end
