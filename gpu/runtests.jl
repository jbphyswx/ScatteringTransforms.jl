# Tier-2 GPU tests — real CUDA hardware (not part of the default `Pkg.test()` CI, which validates the
# vendor-neutral path on `KA.CPU()`). Run manually / on a GPU node:  julia --project=gpu gpu/runtests.jl
#
# Exercises the exact same device-resident constructors + batched-throughput path as the CPU-parity
# tests, but on an actual CUDA GPU via `GPUBackend(CUDA.CUDABackend())`, and checks parity against the
# serial CPU (FFTW) reference. Exits cleanly (0) when no functional CUDA device is present.

using CUDA: CUDA
using KernelAbstractions: KernelAbstractions as KA
using AbstractFFTs: AbstractFFTs
using FFTW: FFTW
using FINUFFT: FINUFFT
using Test: Test
using ScatteringTransforms: ScatteringTransforms as ST
using ComputationalBackends: ComputationalBackends as CB
using SpectralBackends: SpectralBackends as SB

# Asserted before the device check, because it needs no device and is the one failure this tier can
# catch on any machine: an extension that names a symbol from a package loaded only at runtime does
# not precompile, and the default suite cannot see it because the extension never loads there.
Test.@testset "Device extensions load" begin
    Test.@test Base.get_extension(ST, :ScatteringTransformsKernelAbstractionsExt) !== nothing
    Test.@test Base.get_extension(ST, :ScatteringTransformscuFINUFFTExt) !== nothing
    # And the device seam really has a method for device points, rather than falling back to the host
    # one and silently building a CPU plan for a device transform.
    Test.@test hasmethod(ST.Plans.nufft_guru_make,
                         Tuple{CUDA.CuArray, Int, NTuple{2, Int}, Int, Int, Float64, Type{Float64}})
end

if !CUDA.functional()
    @info "No functional CUDA device — skipping the Tier-2 GPU tests that need one (the KA.CPU() " *
          "parity tests in the default suite cover the vendor-neutral paths)."
    exit(0)
end

Test.@testset "ScatteringTransforms CUDA GPU (Tier-2)" begin
    gpu = CB.GPUBackend(CUDA.CUDABackend())
    FB = SB.FFTSpectralBackend()

    Test.@testset "1D/2D single-image parity vs serial CPU" begin
        N, J = 128, 4
        x = randn(Float32, N)
        ref = ST.Scattering1D.ScatteringTransform1D(Float32, N, J; Q=1, max_order=2, spectral=FB)
        gst = ST.Scattering1D.ScatteringTransform1D(Float32, N, J, gpu; Q=1, max_order=2)
        Test.@test ST.Coefficients.flatten1d(gst(CUDA.CuVector(x))) ≈ ST.Coefficients.flatten1d(ref(x)) rtol=1e-4

        Ny, Nx, L = 32, 32, 4
        y = randn(Float32, Ny, Nx)
        ref2 = ST.Scattering2D.ScatteringTransform2D(Float32, (Ny, Nx), 3; L=L, max_order=2, spectral=FB)
        gst2 = ST.Scattering2D.ScatteringTransform2D(Float32, (Ny, Nx), 3, gpu; L=L, max_order=2)
        Test.@test ST.Coefficients.flatten2d(gst2(CUDA.CuMatrix(y))) ≈ ST.Coefficients.flatten2d(ref2(y)) rtol=1e-4
    end

    Test.@testset "batched throughput parity vs serial CPU" begin
        Ny, Nx, J, L, B = 24, 24, 3, 4, 8
        X = randn(Float32, Ny, Nx, B)
        ref = ST.Scattering2D.ScatteringTransform2D(Float32, (Ny, Nx), J; L=L, max_order=2, spectral=FB)
        gst = ST.Scattering2D.ScatteringTransform2D(Float32, (Ny, Nx), J, gpu; L=L, max_order=2)
        got = Array(ST.scattering_batch(gpu, gst, CUDA.CuArray(X)))
        Test.@test got ≈ ST.scattering_batch(ref, X) rtol=1e-4
    end

    Test.@testset "scattered planar on device points (cuFINUFFT) matches the host FINUFFT path" begin
        # A scattered transform follows the array type of its points, so device points give a
        # device-resident transform whose NUFFT comes from the cuFINUFFT seam. Same points, same
        # field, same mode grid as the host transform — only where it runs differs.
        M, ms, J, L = 2000, (32, 32), 3, 4
        x = rand(Float64, M) .* 2π
        y = rand(Float64, M) .* 2π
        f = [1.0 + 0.7cos(x[k]) + 0.5sin(2y[k]) - 0.3cos(x[k]) * sin(y[k]) for k in 1:M]
        host = ST.scattered_planar_scattering(x, y, ms, J; L = L, max_order = 2, period = (2π, 2π),
                                              spectral = ST.Plans.FINUFFTBackend())
        dev = ST.scattered_planar_scattering(CUDA.CuArray(x), CUDA.CuArray(y), ms, J;
                                             L = L, max_order = 2, period = (2π, 2π),
                                             spectral = ST.Plans.FINUFFTBackend())
        # Asserted, not assumed: if the host seam had answered for device points the plan would hold
        # a `finufft_plan` and this transform would have quietly run on the CPU.
        Test.@test dev.plan.guru1 isa FINUFFT.cufinufft_plan
        got = Array(ST.Coefficients.flatten2d(dev(CUDA.CuArray(f))))
        Test.@test got ≈ ST.Coefficients.flatten2d(host(f)) rtol = 1e-5
    end
end
