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
using Test: Test
using ScatteringTransforms: ScatteringTransforms as ST
using ComputationalBackends: ComputationalBackends as CB
using SpectralBackends: SpectralBackends as SB

if !CUDA.functional()
    @info "No functional CUDA device — skipping Tier-2 GPU tests (the KA.CPU() parity tests in the " *
          "default suite cover the same code paths)."
    exit(0)
end

Test.@testset "ScatteringTransforms CUDA GPU (Tier-2)" begin
    gpu = CB.GPUBackend(CUDA.CUDABackend())
    FB = SB.FFTSpectralBackend()

    Test.@testset "1D/2D single-image parity vs serial CPU" begin
        N, J = 128, 4
        x = randn(Float32, N)
        ref = ST.Scattering1D.ScatteringTransform1D(Float32, N, J; Q=1, max_order=2spectral=FB)
        gst = ST.Scattering1D.ScatteringTransform1D(Float32, N, J, gpu; Q=1, max_order=2)
        Test.@test ST.Coefficients.flatten1d(gst(CUDA.CuVector(x))) ≈ ST.Coefficients.flatten1d(ref(x)) rtol=1e-4

        Ny, Nx, L = 32, 32, 4
        y = randn(Float32, Ny, Nx)
        ref2 = ST.Scattering2D.ScatteringTransform2D(Float32, (Ny, Nx), 3; L=L, max_order=2spectral=FB)
        gst2 = ST.Scattering2D.ScatteringTransform2D(Float32, (Ny, Nx), 3, gpu; L=L, max_order=2)
        Test.@test ST.Coefficients.flatten2d(gst2(CUDA.CuMatrix(y))) ≈ ST.Coefficients.flatten2d(ref2(y)) rtol=1e-4
    end

    Test.@testset "batched throughput parity vs serial CPU" begin
        Ny, Nx, J, L, B = 24, 24, 3, 4, 8
        X = randn(Float32, Ny, Nx, B)
        ref = ST.Scattering2D.ScatteringTransform2D(Float32, (Ny, Nx), J; L=L, max_order=2spectral=FB)
        gst = ST.Scattering2D.ScatteringTransform2D(Float32, (Ny, Nx), J, gpu; L=L, max_order=2)
        got = Array(ST.scattering_batch(gpu, gst, CUDA.CuArray(X)))
        Test.@test got ≈ ST.scattering_batch(ref, X) rtol=1e-4
    end
end
