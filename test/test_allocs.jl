# Allocation gates: hot paths allocate nothing, and the paths that allocate do so independently of
# the data size. Each measurement runs behind a function barrier after one warm-up call.
using KernelAbstractions: KernelAbstractions as KA
using AbstractFFTs: AbstractFFTs

# `f::F` specializes the harness on the function type, so a bitstype return is not boxed.
_alloc(f::F, args...) where {F} = (f(args...); @allocated f(args...))
# Fixed arity: a splatted `args...` can allocate inside the harness.
_alloc4(f::F, a, b, c, d) where {F} = (f(a, b, c, d); @allocated f(a, b, c, d))

Test.@testset "Allocation discipline" begin
    C   = ScatteringTransforms.Coefficients
    S1D = ScatteringTransforms.Scattering1D
    S2D = ScatteringTransforms.Scattering2D
    SC  = ScatteringTransforms.ScatteringCore
    P   = ScatteringTransforms.Plans
    SF  = ScatteringTransforms.ScatteringFields
    FB  = SpectralBackends.FFTSpectralBackend()

    N, J = 512, 6
    st1 = S1D.ScatteringTransform1D(N, J; Q=1, max_order=2, transform=FB)
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

    Test.@testset "localized scattering_field! is zero-allocation" begin
        f1 = SF.scattering_field(st1, x1)
        Test.@test _alloc(SF.scattering_field!, f1, st1, x1) == 0
        st2 = S2D.ScatteringTransform2D((32, 32), 3; L=4, max_order=2, transform=FB)
        x2 = randn(32, 32)
        f2 = SF.scattering_field(st2, x2)
        Test.@test _alloc(SF.scattering_field!, f2, st2, x2) == 0
    end

    Test.@testset "batched `!` allocation is independent of the batch size" begin
        flen1 = C.flatten_length(C.ScatteringCoefficients1D(nw1, Float64; compute_S2=true))
        a16 = _alloc(ScatteringTransforms.scattering_batch!, Matrix{Float64}(undef, flen1, 16), st1, randn(N, 16))
        a64 = _alloc(ScatteringTransforms.scattering_batch!, Matrix{Float64}(undef, flen1, 64), st1, randn(N, 64))
        Test.@test a16 == a64            # one-time workspace only, nothing per column
        Test.@test a16 < 4096

        st2 = S2D.ScatteringTransform2D((24, 24), 3; L=4, max_order=2, transform=FB)
        flen2 =C.flatten_length(C.ScatteringCoefficients2D(st2.filter_bank.J, st2.filter_bank.L, Float64; compute_S2=true))
        b8  = _alloc(ScatteringTransforms.scattering_batch!, Matrix{Float64}(undef, flen2, 8),  st2, randn(24, 24, 8))
        b32 = _alloc(ScatteringTransforms.scattering_batch!, Matrix{Float64}(undef, flen2, 32), st2, randn(24, 24, 32))
        Test.@test b8 == b32
    end

    Test.@testset "the batched-plan cascade is exactly zero-allocation" begin
        # The whole-stack cascade, called directly; `scattering_batch!` reaches it through a keyword
        # argument.
        st2 = S2D.ScatteringTransform2D((24, 24), 3; L=4, max_order=2, transform=FB)
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
        flen = size(ScatteringTransforms.scattering_batch(gst, X; execution = gpu), 1)
        out = zeros(Float32, flen, B)
        a = _alloc((o, X) -> ScatteringTransforms.scattering_batch!(o, gst, X; execution = gpu,
                                                                    workspace = ws), out, X)
        Test.@test a < sizeof(X)         # no temporary that scales with the field data
    end

    Test.@testset "batched scattered-planar cascade allocates nothing per field" begin
        # The batched cascade multiplies a `(ms…, B)` mode array by `(ms…, 1)` filter views built once
        # with the transform. FINUFFT runs its FFT through the libfftw3 that FFTW.jl gives a
        # Julia-task thread callback; at `nufft_nthreads = 1` the library allocates nothing.
        SP = ScatteringTransforms.ScatteredPlanar
        M, ms = 500, (16, 16)
        px, py = rand(M), rand(M)
        function batched_alloc(B)
            stb = SP.build(Float64, px, py, ms, 3; period = 1.0, L = 4, max_order = 2,
                           transform = FTB.FINUFFTBackend(), ntrans = B, nufft_nthreads = 1)
            # Precondition: the plan is `B` wide.
            Test.@test ScatteringTransforms.Plans.batch_width(stb.plan) == B
            nwp = length(stb.filter_bank.wavelets)
            S0 = Vector{Float64}(undef, B)
            S1 = Matrix{Float64}(undef, nwp, B)
            S2 = zeros(Float64, nwp, nwp, B)
            return _alloc(SP.scattered_planar_scattering_batch!, S0, S1, S2, stb, randn(M, B))
        end
        a4, a8 = batched_alloc(4), batched_alloc(8)
        Test.@test a4 == a8      # nothing per field
        # The bound covers the returned `(; S0, S1, S2)` itself.
        Test.@test a4 <= 32
    end

    Test.@testset "the scattered least-squares solve allocates nothing per call" begin
        # `solve = true` runs an iterative solve per analysis, once per field and wavelet path; its
        # workspace lives on the plan. NonuniformFFTs is excluded: its `exec_type1!`/`exec_type2!`
        # open a `Threads.@threads` region on every execution, even at one thread.
        Msc, mssc = 500, (16, 16)
        Random.seed!(31)
        pxs, pys = 2π .* rand(Msc), 2π .* rand(Msc)
        bs = randn(Msc)
        Xs = zeros(ComplexF64, mssc)
        for spec in (SpectralBackends.DirectSumSpectralBackend(), FTB.FINUFFTBackend())
            plan = P.make_scattered_plan(spec, pxs, pys, mssc, Float64; period = (2π, 2π),
                                         solve = true, maxiter = 20, nufft_nthreads = 1)
            # Precondition: the plan solves.
            Test.@test P.plan_analysis(plan).solve
            Test.@test _alloc(P.forward_transform!, Xs, plan, bs) == 0
        end
        # Batched: the per-column recurrence runs host-side over `B` states, reducing into mirrors
        # preallocated on the plan.
        Bsc = 4
        pb = P.make_scattered_plan(FTB.FINUFFTBackend(), pxs, pys, mssc, Float64; period = (2π, 2π),
                                   solve = true, maxiter = 20, ntrans = Bsc, nufft_nthreads = 1)
        Test.@test P.batch_width(pb) == Bsc
        Test.@test _alloc(P.forward_transform!, zeros(ComplexF64, mssc..., Bsc), pb,
                          randn(Msc, Bsc)) == 0
    end

    Test.@testset "the spherical least-squares analysis reuses its plan's workspace" begin
        # The cascade analyses a field, then each first-order field, through `sphere_coeffs!`, and every
        # analysis is an iterative solve whose workspace is the plan's. NUFSHT runs on one thread over
        # FINUFFT, where the library's solve allocates nothing.
        SC = ScatteringTransforms.SphericalCore
        ext = Base.get_extension(ScatteringTransforms, :ScatteringTransformsNUFSHTExt)
        lmax, Ms = 6, 200
        θ = acos.([1 - 2 * (i - 0.5) / Ms for i in 1:Ms])
        φ = [mod(π * (3 - sqrt(5)) * i, 2π) for i in 1:Ms]
        np = NUFSHT.make_plan(Float64, θ, φ, lmax; nufft = FTB.FINUFFTBackend(), nthreads = 1)
        sp = ext.NUSHTSphericalPlan(np, Ms, lmax, θ, φ, 1e-8, 500, FTB.FINUFFTBackend())
        C = SC.sphere_coeffs_buffer(sp)
        Test.@test _alloc(SC.sphere_coeffs!, C, sp, randn(Ms)) == 0

        # The in-core plan solves through its own operator scratch and workspace, single-field and
        # batched, and synthesizes into the caller's array.
        for B in (1, 3)
            dp = SC.make_spherical_plan(SpectralBackends.DirectSumSpectralBackend(), θ, φ, lmax,
                                        Float64; ntrans = B)
            F = B == 1 ? randn(Ms) : randn(Ms, B)
            Cd = SC.sphere_coeffs_buffer(dp)
            Test.@test _alloc3(SC.sphere_coeffs!, Cd, dp, F) == 0
            out = similar(F)
            SC.sphere_coeffs!(Cd, dp, F)
            Test.@test _alloc4(SC.sphere_apply!, out, dp, Cd, Returns(1.0)) == 0
        end
    end

    Test.@testset "st(x) allocates only its coefficient container (size-independent)" begin
        # The non-mutating callable allocates coefficient storage, whose size depends on the number of
        # wavelets only.
        st_small = S1D.ScatteringTransform1D(256,  J; Q=1, max_order=2, transform=FB)
        st_big   = S1D.ScatteringTransform1D(4096, J; Q=1, max_order=2, transform=FB)
        Test.@test _alloc(st_small, randn(256)) == _alloc(st_big, randn(4096))
    end
end
