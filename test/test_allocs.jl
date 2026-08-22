# Allocation discipline — proves the hot paths are zero-allocation as designed, and that the paths
# which must allocate do so minimally and independently of the data size (not per-element/per-batch).
#
# Every measurement is function-wrapped (so `@allocated` doesn't see global-variable boxing) and
# warmed up once before measuring. Covers: the core in-place building blocks, the headline in-place
# transforms (1D/2D/3D), the localized field, the batched `!` (in-core + GPU), and the documented
# allocating entry points (`st(x)` allocates only its coefficient container).
using KernelAbstractions: KernelAbstractions as KA
using AbstractFFTs: AbstractFFTs

# f is warmed up once, then measured on a second identical call. The `f::F where {F}` forces Julia to
# specialize this helper on the concrete function type, so `f(args...)` is a statically-known call with
# an inferred return type. Without it, a scalar (bitstype) return is boxed by the higher-order call — a
# spurious 16-byte allocation the measurement would otherwise attribute to `f` (observed on Julia 1.11;
# the 1.12 compiler elides it). Array-returning `!` ops are unaffected (they return an existing pointer).
_alloc(f::F, args...) where {F} = (f(args...); @allocated f(args...))

Test.@testset "Allocation discipline" begin
    C   = ScatteringTransforms.Coefficients
    S1D = ScatteringTransforms.Scattering1D
    S2D = ScatteringTransforms.Scattering2D
    S3D = ScatteringTransforms.Scattering3D
    SC  = ScatteringTransforms.ScatteringCore
    P   = ScatteringTransforms.Plans
    SF  = ScatteringTransforms.ScatteringFields
    FB  = SpectralBackends.FFTSpectralBackend()

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

    Test.@testset "the batched-plan cascade is exactly zero-allocation" begin
        # The opt-in whole-stack path, measured on the cascade itself: `scattering_batch!` reaches it
        # through a keyword argument, and the keyword call allocates independently of the cascade.
        st2 = S2D.ScatteringTransform2D((24, 24), 3; L=4, max_order=2, spectral=FB)
        X = randn(24, 24, 8)
        ws = ScatteringTransforms.batch_workspace(st2, 8)
        out = ScatteringTransforms.scattering_batch(st2, X)
        Test.@test _alloc(ScatteringTransforms.Batched.batch_cascade!, out, ws, X) == 0
    end

    Test.@testset "GPU batched `!` allocates nothing data-proportional (GPUBackend(KA.CPU()))" begin
        gpu = ComputationalBackends.GPUBackend(KA.CPU())
        kaext = Base.get_extension(ScatteringTransforms, :ScatteringTransformsKernelAbstractionsExt)
        Ny, Nx, B = 24, 24, 8
        gst = S2D.ScatteringTransform2D(Float32, (Ny, Nx), 3, gpu; L=4, max_order=2)
        ws = kaext.gpu_batch_workspace(gpu, gst, B)
        X = randn(Float32, Ny, Nx, B)
        flen = size(ScatteringTransforms.scattering_batch(gpu, gst, X), 1)
        out = zeros(Float32, flen, B)
        a = _alloc((o, X) -> ScatteringTransforms.scattering_batch!(o, gpu, gst, X; workspace=ws), out, X)
        Test.@test a < sizeof(X)         # no temporary that scales with the field data
    end

    Test.@testset "batched scattered-planar cascade allocates nothing per field" begin
        # The batched cascade multiplies a `(ms…, B)` mode array by a `(ms)` filter, which only
        # broadcasts if the filter carries a trailing singleton. Building that view per wavelet and
        # per path would allocate an array header on every call, so the views are built once with the
        # transform; this asserts they stay there.
        #
        # `nufft_nthreads = 1` is what makes zero reachable, and it is a property of the library, not
        # of this cascade: FINUFFT executes its internal FFT through the same libfftw3 FFTW.jl has
        # installed a Julia-task thread callback into, so a multi-threaded FINUFFT plan spawns a task
        # per thread on every transform (measured: 276 tasks, 139104 B for `B = 4`). Single-threaded it
        # takes libfftw3's serial path and the cascade's own allocation — which is what is under test —
        # is exactly zero.
        SP = ScatteringTransforms.ScatteredPlanar
        M, ms = 500, (16, 16)
        px, py = rand(M), rand(M)
        function batched_alloc(B)
            stb = SP.build(Float64, px, py, ms, 3; L = 4, max_order = 2,
                           spectral = ScatteringTransforms.Plans.FINUFFTBackend(), ntrans = B,
                           nufft_nthreads = 1)
            # Asserted, not assumed: a plan that quietly came back single-field would make the
            # measurement below time the per-field cascade and agree for the wrong reason.
            Test.@test ScatteringTransforms.Plans.batch_width(stb.plan) == B
            nwp = length(stb.filter_bank.wavelets)
            S0 = Vector{Float64}(undef, B)
            S1 = Matrix{Float64}(undef, nwp, B)
            S2 = zeros(Float64, nwp, nwp, B)
            return _alloc(SP.scattered_planar_scattering_batch!, S0, S1, S2, stb, randn(M, B))
        end
        Test.@test batched_alloc(4) == 0
        Test.@test batched_alloc(8) == 0
    end

    Test.@testset "st(x) allocates only its coefficient container (size-independent)" begin
        # The non-mutating callable is documented to allocate coefficient storage once; that cost
        # depends on the number of wavelets, not the signal length.
        st_small = S1D.ScatteringTransform1D(256,  J; Q=1, max_order=2, spectral=FB)
        st_big   = S1D.ScatteringTransform1D(4096, J; Q=1, max_order=2, spectral=FB)
        Test.@test _alloc(st_small, randn(256)) == _alloc(st_big, randn(4096))
    end
end
