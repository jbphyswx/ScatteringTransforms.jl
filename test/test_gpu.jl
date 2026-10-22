# Vendor-neutral GPU path, validated on the KernelAbstractions CPU backend.
#
# `GPUBackend(KA.CPU())` runs the device-resident constructors and batched path used on CUDA/ROCm,
# with `Array` as the device array type, so AbstractFFTs builds an FFTW plan.
using KernelAbstractions: KernelAbstractions as KA
using AbstractFFTs: AbstractFFTs

Test.@testset "GPU vendor-neutral path (GPUBackend(KA.CPU()))" begin
    gpu = ComputationalBackends.GPUBackend(KA.CPU())
    FB = SpectralBackends.FFTSpectralBackend()

    Test.@testset "1D single-image parity vs serial" begin
        N, J = 128, 4
        x = randn(Float32, N)
        ref = ScatteringTransforms.Scattering1D.ScatteringTransform1D(Float32, N, J; Q=1, max_order=2, transform=FB)
        gst = ScatteringTransforms.Scattering1D.ScatteringTransform1D(Float32, N, J, gpu; Q=1, max_order=2)
        Test.@test ScatteringTransforms.Coefficients.flatten1d(gst(x)) ≈ ScatteringTransforms.Coefficients.flatten1d(ref(x)) rtol=1e-4
    end

    Test.@testset "2D single-image parity vs serial" begin
        Ny, Nx, J, L = 32, 32, 3, 4
        x = randn(Float32, Ny, Nx)
        ref = ScatteringTransforms.Scattering2D.ScatteringTransform2D(Float32, (Ny, Nx), J; L=L, max_order=2, transform=FB)
        gst = ScatteringTransforms.Scattering2D.ScatteringTransform2D(Float32, (Ny, Nx), J, gpu; L=L, max_order=2)
        Test.@test ScatteringTransforms.Coefficients.flatten2d(gst(x)) ≈ ScatteringTransforms.Coefficients.flatten2d(ref(x)) rtol=1e-4
    end

    Test.@testset "3D single-image parity vs serial" begin
        N3, J = (8, 8, 8), 2
        x = randn(Float32, N3...)
        ref = ScatteringTransforms.Scattering3D.ScatteringTransform3D(Float32, N3, J; n_orient=4, max_order=2, transform=FB)
        gst = ScatteringTransforms.Scattering3D.ScatteringTransform3D(Float32, N3, J, gpu; n_orient=4, max_order=2)
        # 3D uses the scales×orientations (2D) coefficient container.
        Test.@test ScatteringTransforms.Coefficients.flatten2d(gst(x)) ≈ ScatteringTransforms.Coefficients.flatten2d(ref(x)) rtol=1e-4
    end

    Test.@testset "batched throughput parity (1D + 2D + 3D)" begin
        # 1D batch
        N, J, B = 96, 4, 8
        X = randn(Float32, N, B)
        ref = ScatteringTransforms.Scattering1D.ScatteringTransform1D(Float32, N, J; Q=1, max_order=2, transform=FB)
        gst = ScatteringTransforms.Scattering1D.ScatteringTransform1D(Float32, N, J, gpu; Q=1, max_order=2)
        Test.@test Array(ScatteringTransforms.scattering_batch(gst, X; execution = gpu)) ≈
                   ScatteringTransforms.scattering_batch(ref, X) rtol=1e-4

        # 2D batch
        Ny, Nx, J2, L, B2 = 24, 24, 3, 4, 6
        X2 = randn(Float32, Ny, Nx, B2)
        ref2 = ScatteringTransforms.Scattering2D.ScatteringTransform2D(Float32, (Ny, Nx), J2; L=L, max_order=2, transform=FB)
        gst2 = ScatteringTransforms.Scattering2D.ScatteringTransform2D(Float32, (Ny, Nx), J2, gpu; L=L, max_order=2)
        Test.@test Array(ScatteringTransforms.scattering_batch(gst2, X2; execution = gpu)) ≈
                   ScatteringTransforms.scattering_batch(ref2, X2) rtol=1e-4

        # 3D batch
        N3, J3, B3 = (8, 8, 8), 2, 4
        X3 = randn(Float32, N3..., B3)
        ref3 = ScatteringTransforms.Scattering3D.ScatteringTransform3D(Float32, N3, J3; n_orient=4, max_order=2, transform=FB)
        gst3 = ScatteringTransforms.Scattering3D.ScatteringTransform3D(Float32, N3, J3, gpu; n_orient=4, max_order=2)
        Test.@test Array(ScatteringTransforms.scattering_batch(gst3, X3; execution = gpu)) ≈
                   ScatteringTransforms.scattering_batch(ref3, X3) rtol=1e-4
    end

    Test.@testset "scattering_field! order-2 runs on device arrays (guards the scalar-index fix)" begin
        Ny, Nx, J, L = 32, 32, 3, 4
        x = randn(Float32, Ny, Nx)
        gst = ScatteringTransforms.Scattering2D.ScatteringTransform2D(Float32, (Ny, Nx), J, gpu; L=L, max_order=2)
        field = ScatteringTransforms.ScatteringFields.scattering_field(gst, x)
        ref = ScatteringTransforms.Scattering2D.ScatteringTransform2D(Float32, (Ny, Nx), J; L=L, max_order=2, transform=FB)
        fref = ScatteringTransforms.ScatteringFields.scattering_field(ref, x)
        Test.@test Array(field.data) ≈ Array(fref.data) rtol=1e-4
    end
    Test.@testset "scattered planar builds on a device backend and matches the host" begin
        # Everything downstream of the points follows their array type, so the device constructor
        # only relocates the points. NonuniformFFTs runs on any KernelAbstractions backend.
        SP = ScatteringTransforms.ScatteredPlanar
        M, ms, J, L = 400, (16, 16), 3, 4
        Random.seed!(11)
        px, py, f = rand(M), rand(M), randn(M)
        spec = FTB.NonuniformFFTsBackend()
        host = SP.build(Float64, px, py, ms, J; period = 1.0, L = L, transform = spec)
        dev = SP.build(Float64, px, py, ms, J, gpu; period = 1.0, L = L, transform = spec)
        Test.@test Array(ScatteringTransforms.Coefficients.flatten2d(dev(f))) ≈
                   Array(ScatteringTransforms.Coefficients.flatten2d(host(f)))
        X = randn(M, 4)
        Test.@test Array(ScatteringTransforms.scattering_batch(dev, X)) ≈
                   Array(ScatteringTransforms.scattering_batch(host, X))

        # `solve = true` on device points: the solve runs in `copyto!`, `fill!`, `norm` and fused
        # broadcasts, and the batched form reduces through `mapreducedim!` into preallocated slices.
        #
        # The field is band limited, so the solve has an exact answer, and `maxiter` exceeds
        # `prod(ms) = 256`, the step count at which the Krylov process terminates. A threaded NUFFT
        # spreads in nondeterministic order, and an unconverged solve amplifies that by about
        # `cond(A)²`.
        fb = [1 + 0.7cos(2π * px[k]) + 0.5sin(4π * py[k]) for k in 1:M]
        host_s = SP.build(Float64, px, py, ms, J; period = 1.0, L = L, transform = spec, solve = true,
                          maxiter = 400)
        dev_s = SP.build(Float64, px, py, ms, J, gpu; period = 1.0, L = L, transform = spec,
                         solve = true, maxiter = 400)
        Test.@test Array(ScatteringTransforms.Coefficients.flatten2d(dev_s(fb))) ≈
                   Array(ScatteringTransforms.Coefficients.flatten2d(host_s(fb))) rtol = 1e-8
        Xb = hcat([[1 + 0.7cos(2π * px[k]) + 0.5c * sin(4π * py[k]) for k in 1:M] for c in 1:4]...)
        Test.@test Array(ScatteringTransforms.scattering_batch(dev_s, Xb)) ≈
                   Array(ScatteringTransforms.scattering_batch(host_s, Xb)) rtol = 1e-8
    end

    Test.@testset "monogenic scattering on the device matches the host" begin
        Mono = ScatteringTransforms.Monogenic
        for (dims, J) in (((64,), 4), ((32, 32), 3), ((12, 12, 12), 2)), α in (J, 1)
            x = randn(Float32, dims...)
            ref = Mono.MonogenicScattering(Float32, dims, J; Q = 1, max_order = 2, oversampling = α,
                                           transform = FB)
            gst = Mono.MonogenicScattering(Float32, dims, J, gpu; Q = 1, max_order = 2, oversampling = α)
            flat(c) = vcat(c.S0, vec(c.S1), vec(c.S2))
            Test.@test flat(gst(x)) ≈ flat(ref(x)) rtol = 1e-4
            Test.@test Array(Mono.monogenic_components(gst, x, 1).amplitude) ≈
                       Mono.monogenic_components(ref, x, 1).amplitude rtol = 1e-4
        end
    end

    Test.@testset "phase retrieval and the frame inverse run on a device transform" begin
        Inv = ScatteringTransforms.Inverse
        N, J = 128, 5
        x = randn(Float32, N)
        ref = ScatteringTransforms.Scattering1D.ScatteringTransform1D(Float32, N, J; Q = 1,
                                                                      max_order = 1, transform = FB)
        gst = ScatteringTransforms.Scattering1D.ScatteringTransform1D(Float32, N, J, gpu; Q = 1,
                                                                      max_order = 1)
        Test.@test Array(Inv.iwavelet(gst, Inv.wavelet_transform(gst, x))) ≈ x rtol = 1e-4
        # The true field is a fixed point of the projections; a random seed is drawn for the device.
        wt = Inv.wavelet_transform(ref, x)
        moduli = [abs.(w) for w in wt.wavelet]
        Test.@test Array(Inv.reconstruct_phase(gst, moduli; iters = 5, init = x,
                                               seed_lowpass = wt.lowpass)) ≈ x rtol = 1e-4
        xr = Array(Inv.reconstruct_phase(gst, moduli; iters = 2))
        Test.@test size(xr) == (N,)
        Test.@test all(isfinite, xr)
    end

    Test.@testset "a device transform builds its own batch workspace; complex stacks keep S0" begin
        Ny, Nx, J, L, B = 24, 24, 3, 4, 5
        ref = ScatteringTransforms.Scattering2D.ScatteringTransform2D(Float32, (Ny, Nx), J; L = L,
                                                                      max_order = 2, transform = FB)
        gst = ScatteringTransforms.Scattering2D.ScatteringTransform2D(Float32, (Ny, Nx), J, gpu; L = L,
                                                                      max_order = 2)
        X = randn(Float32, Ny, Nx, B)
        ws = ScatteringTransforms.batch_workspace(gst, B)
        out = similar(gst.buffer_input, Float32, (size(ScatteringTransforms.scattering_batch(ref, X), 1), B))
        ScatteringTransforms.scattering_batch!(out, gst, X; workspace = ws)
        Test.@test Array(out) ≈ ScatteringTransforms.scattering_batch(ref, X) rtol = 1e-4

        Z = randn(ComplexF32, Ny, Nx, B)
        got = Array(ScatteringTransforms.scattering_batch(gst, Z; execution = gpu))
        Test.@test eltype(got) <: Complex
        Test.@test got ≈ ScatteringTransforms.scattering_batch(ref, Z) rtol = 1e-4
    end
end
