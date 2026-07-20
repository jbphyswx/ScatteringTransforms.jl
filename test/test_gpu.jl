# Vendor-neutral GPU path, validated on the KernelAbstractions CPU backend.
#
# `GPUBackend(KA.CPU())` runs the *exact* device-resident constructors + batched-throughput path used
# on CUDA/ROCm/…, differing only in the concrete array type — so this exercises the full
# KA.allocate + AbstractFFTs-plan + generic-engine chain with no GPU hardware. `KA.allocate(KA.CPU(),…)`
# returns a plain `Array`, so AbstractFFTs builds a real FFTW plan (FFTW is loaded in the test env).
using KernelAbstractions: KernelAbstractions as KA
using AbstractFFTs: AbstractFFTs

Test.@testset "GPU vendor-neutral path (GPUBackend(KA.CPU()))" begin
    gpu = ScatteringTransforms.Backends.GPUBackend(KA.CPU())
    FB = ScatteringTransforms.Plans.FFTBackend()

    Test.@testset "1D single-image parity vs serial" begin
        N, J = 128, 4
        x = randn(Float32, N)
        ref = ScatteringTransforms.Scattering1D.ScatteringTransform1D(N, J; Q=1, max_order=2, T=Float32, spectral=FB)
        gst = ScatteringTransforms.Scattering1D.ScatteringTransform1D(N, J, gpu; Q=1, max_order=2, T=Float32)
        Test.@test ScatteringTransforms.Coefficients.flatten1d(gst(x)) ≈ ScatteringTransforms.Coefficients.flatten1d(ref(x)) rtol=1e-4
    end

    Test.@testset "2D single-image parity vs serial" begin
        Ny, Nx, J, L = 32, 32, 3, 4
        x = randn(Float32, Ny, Nx)
        ref = ScatteringTransforms.Scattering2D.ScatteringTransform2D((Ny, Nx), J; L=L, max_order=2, T=Float32, spectral=FB)
        gst = ScatteringTransforms.Scattering2D.ScatteringTransform2D((Ny, Nx), J, gpu; L=L, max_order=2, T=Float32)
        Test.@test ScatteringTransforms.Coefficients.flatten2d(gst(x)) ≈ ScatteringTransforms.Coefficients.flatten2d(ref(x)) rtol=1e-4
    end

    Test.@testset "3D single-image parity vs serial" begin
        N3, J = (8, 8, 8), 2
        x = randn(Float32, N3...)
        ref = ScatteringTransforms.Scattering3D.ScatteringTransform3D(N3, J; n_orient=4, max_order=2, T=Float32, spectral=FB)
        gst = ScatteringTransforms.Scattering3D.ScatteringTransform3D(N3, J, gpu; n_orient=4, max_order=2, T=Float32)
        # 3D uses the scales×orientations (2D) coefficient container.
        Test.@test ScatteringTransforms.Coefficients.flatten2d(gst(x)) ≈ ScatteringTransforms.Coefficients.flatten2d(ref(x)) rtol=1e-4
    end

    Test.@testset "batched throughput parity (1D + 2D)" begin
        # 1D batch
        N, J, B = 96, 4, 8
        X = randn(Float32, N, B)
        ref = ScatteringTransforms.Scattering1D.ScatteringTransform1D(N, J; Q=1, max_order=2, T=Float32, spectral=FB)
        gst = ScatteringTransforms.Scattering1D.ScatteringTransform1D(N, J, gpu; Q=1, max_order=2, T=Float32)
        Test.@test Array(ScatteringTransforms.scattering_batch(gpu, gst, X)) ≈
                   ScatteringTransforms.scattering_batch(ref, X) rtol=1e-4

        # 2D batch
        Ny, Nx, J2, L, B2 = 24, 24, 3, 4, 6
        X2 = randn(Float32, Ny, Nx, B2)
        ref2 = ScatteringTransforms.Scattering2D.ScatteringTransform2D((Ny, Nx), J2; L=L, max_order=2, T=Float32, spectral=FB)
        gst2 = ScatteringTransforms.Scattering2D.ScatteringTransform2D((Ny, Nx), J2, gpu; L=L, max_order=2, T=Float32)
        Test.@test Array(ScatteringTransforms.scattering_batch(gpu, gst2, X2)) ≈
                   ScatteringTransforms.scattering_batch(ref2, X2) rtol=1e-4

        # `!` variant writes into a preallocated output and agrees with the allocating form.
        flen = size(ScatteringTransforms.scattering_batch(gpu, gst2, X2), 1)
        out = zeros(Float32, flen, B2)
        ScatteringTransforms.scattering_batch!(out, gpu, gst2, X2)
        Test.@test out ≈ ScatteringTransforms.scattering_batch(ref2, X2) rtol=1e-4
    end

    Test.@testset "batched `!` does no data-proportional allocation" begin
        # With a preallocated output + workspace, a repeated `!` call must not allocate a temporary
        # that scales with the data — a data-proportional temporary would exceed `sizeof(X)`.
        kaext = Base.get_extension(ScatteringTransforms, :ScatteringTransformsKernelAbstractionsExt)
        Ny, Nx, J, L, B = 24, 24, 3, 4, 6
        X = randn(Float32, Ny, Nx, B)
        gst = ScatteringTransforms.Scattering2D.ScatteringTransform2D((Ny, Nx), J, gpu; L=L, max_order=2, T=Float32)
        ws = kaext.GPUBatchWorkspace(gpu, gst, B)
        flen = size(ScatteringTransforms.scattering_batch(gpu, gst, X), 1)
        out = zeros(Float32, flen, B)
        ScatteringTransforms.scattering_batch!(out, gpu, gst, X; workspace=ws)   # warm up (compile)
        alloc = @allocated ScatteringTransforms.scattering_batch!(out, gpu, gst, X; workspace=ws)
        Test.@test alloc < sizeof(X)
    end

    Test.@testset "scattering_field! order-2 runs on device arrays (guards the scalar-index fix)" begin
        Ny, Nx, J, L = 32, 32, 3, 4
        x = randn(Float32, Ny, Nx)
        gst = ScatteringTransforms.Scattering2D.ScatteringTransform2D((Ny, Nx), J, gpu; L=L, max_order=2, T=Float32)
        field = ScatteringTransforms.ScatteringFields.scattering_field(gst, x)
        Test.@test all(isfinite, field.data)
        # Parity with the serial localized field.
        ref = ScatteringTransforms.Scattering2D.ScatteringTransform2D((Ny, Nx), J; L=L, max_order=2, T=Float32, spectral=FB)
        fref = ScatteringTransforms.ScatteringFields.scattering_field(ref, x)
        Test.@test Array(field.data) ≈ Array(fref.data) rtol=1e-4
    end
end
