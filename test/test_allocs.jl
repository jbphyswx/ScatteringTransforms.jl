# Allocation discipline — proves the hot paths are zero-allocation as designed, and that the paths
# which must allocate do so minimally and independently of the data size (not per-element/per-batch).
#
# Every measurement is function-wrapped (so `@allocated` doesn't see global-variable boxing) and
# warmed up once before measuring. Covers: the core in-place building blocks, the headline in-place
# transforms (1D/2D/3D), the localized field, the batched `!` (in-core + GPU), and the documented
# allocating entry points (`st(x)` allocates only its coefficient container).
using KernelAbstractions: KernelAbstractions as KA
using AbstractFFTs: AbstractFFTs

# f is warmed up once, then measured on a second identical call.
_alloc(f, args...) = (f(args...); @allocated f(args...))

Test.@testset "Allocation discipline" begin
    C   = ScatteringTransforms.Coefficients
    S1D = ScatteringTransforms.Scattering1D
    S2D = ScatteringTransforms.Scattering2D
    S3D = ScatteringTransforms.Scattering3D
    SC  = ScatteringTransforms.ScatteringCore
    P   = ScatteringTransforms.Plans
    SF  = ScatteringTransforms.ScatteringFields
    FB  = ScatteringTransforms.Plans.FFTBackend()

    N, J = 512, 6
    st1 = S1D.ScatteringTransform1D(N, J; Q=1, max_order=2, spectral=FB)
    nw1 = length(st1.filter_bank.wavelets)
    x1  = randn(N)

    Test.@testset "core in-place ops are exactly zero-allocation" begin
        sfft = P.forward_transform(st1.plan, complex.(x1))
        ψ = st1.filter_bank.wavelets[3]
        bufc, outc = similar(sfft), similar(sfft)
        outm = Vector{Float64}(undef, N)
        bi = complex.(x1); bo = similar(bi)
        Test.@test _alloc(SC.wavelet_convolve!, outc, sfft, ψ, st1.plan, bufc) == 0
        Test.@test _alloc(SC.apply_modulus!, outm, outc) == 0
        Test.@test _alloc(SC.spatial_average, outm) == 0
        Test.@test _alloc(P.forward_transform!, bo, st1.plan, bi) == 0
        Test.@test _alloc(P.inverse_transform!, bo, st1.plan, bi) == 0
    end

    Test.@testset "headline in-place transforms are zero-allocation (mutable-S0 container)" begin
        c1 = C.ScatteringCoefficients1D(Vector{Float64}(undef, nw1), zeros(nw1, nw1); S0=[0.0])
        Test.@test _alloc(S1D.scattering_transform!, c1, st1, x1) == 0

        st2 = S2D.ScatteringTransform2D((32, 32), 3; L=4, max_order=2, spectral=FB)
        J2, L2 = st2.filter_bank.J, st2.filter_bank.L
        c2 = C.ScatteringCoefficients2D(Vector{Float64}(undef, J2 * L2), zeros(J2 * L2, J2 * L2);
                                        S0=[0.0], n_scales=J2, n_orientations=L2)
        x2 = randn(32, 32)
        Test.@test _alloc(S2D.scattering_transform2d!, c2, st2, x2) == 0

        st3 = S3D.ScatteringTransform3D((8, 8, 8), 2; n_orient=4, max_order=2, spectral=FB)
        J3, no3 = st3.filter_bank.J, st3.filter_bank.n_orient
        c3 = C.ScatteringCoefficients2D(Vector{Float64}(undef, J3 * no3), zeros(J3 * no3, J3 * no3);
                                        S0=[0.0], n_scales=J3, n_orientations=no3)
        x3 = randn(8, 8, 8)
        Test.@test _alloc(S3D.scattering_transform3d!, c3, st3, x3) == 0
    end

    Test.@testset "localized scattering_field! is zero-allocation" begin
        f1 = SF.scattering_field(st1, x1)
        Test.@test _alloc(SF.scattering_field!, f1, st1, x1) == 0
        st2 = S2D.ScatteringTransform2D((32, 32), 3; L=4, max_order=2, spectral=FB)
        x2 = randn(32, 32)
        f2 = SF.scattering_field(st2, x2)
        Test.@test _alloc(SF.scattering_field!, f2, st2, x2) == 0
    end

    Test.@testset "batched `!` allocation is independent of the batch size" begin
        flen1 = C.flatten_length(C.ScatteringCoefficients1D(nw1, Float64; compute_S2=true))
        a16 = _alloc(ScatteringTransforms.scattering_batch!, Matrix{Float64}(undef, flen1, 16), st1, randn(N, 16))
        a64 = _alloc(ScatteringTransforms.scattering_batch!, Matrix{Float64}(undef, flen1, 64), st1, randn(N, 64))
        Test.@test a16 == a64            # one-time workspace only — no per-column allocation
        Test.@test a16 < 4096            # and minimal

        st2 = S2D.ScatteringTransform2D((24, 24), 3; L=4, max_order=2, spectral=FB)
        flen2 = C.flatten_length(C.ScatteringCoefficients2D(st2.filter_bank.J, st2.filter_bank.L, Float64; compute_S2=true))
        b8  = _alloc(ScatteringTransforms.scattering_batch!, Matrix{Float64}(undef, flen2, 8),  st2, randn(24, 24, 8))
        b32 = _alloc(ScatteringTransforms.scattering_batch!, Matrix{Float64}(undef, flen2, 32), st2, randn(24, 24, 32))
        Test.@test b8 == b32
    end

    Test.@testset "GPU batched `!` allocates nothing data-proportional (GPUBackend(KA.CPU()))" begin
        gpu = ScatteringTransforms.Backends.GPUBackend(KA.CPU())
        kaext = Base.get_extension(ScatteringTransforms, :ScatteringTransformsKernelAbstractionsExt)
        Ny, Nx, B = 24, 24, 8
        gst = S2D.ScatteringTransform2D((Ny, Nx), 3, gpu; L=4, max_order=2, T=Float32)
        ws = kaext.GPUBatchWorkspace(gpu, gst, B)
        X = randn(Float32, Ny, Nx, B)
        flen = size(ScatteringTransforms.scattering_batch(gpu, gst, X), 1)
        out = zeros(Float32, flen, B)
        a = _alloc((o, X) -> ScatteringTransforms.scattering_batch!(o, gpu, gst, X; workspace=ws), out, X)
        Test.@test a < sizeof(X)         # no temporary that scales with the field data
    end

    Test.@testset "st(x) allocates only its coefficient container (size-independent)" begin
        # The non-mutating callable is documented to allocate coefficient storage once; that cost
        # depends on the number of wavelets, not the signal length.
        st_small = S1D.ScatteringTransform1D(256,  J; Q=1, max_order=2, spectral=FB)
        st_big   = S1D.ScatteringTransform1D(4096, J; Q=1, max_order=2, spectral=FB)
        Test.@test _alloc(st_small, randn(256)) == _alloc(st_big, randn(4096))
    end
end
