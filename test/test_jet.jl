using ScatteringTransforms: ScatteringTransforms as ST
using SpectralBackends: SpectralBackends as SB
using JET: JET
using Test: Test

Test.@testset "JET type stability (hot path)" begin
    # JET refuses to run on pre-release Julia (`@test_opt` throws on nightly/rc).
    if !isempty(VERSION.prerelease)
        @info "Skipping JET type-stability checks on pre-release Julia $(VERSION)"
        Test.@test_skip true
    else
        # The transform functors. The direct-sum backend keeps the whole call inside
        # ScatteringTransforms.
        st1 = ST.ScatteringCore.ScatteringTransform(64, 4; Q=2, max_order=2, transform=SB.DirectSumSpectralBackend())
        x1 = randn(64)
        JET.@test_opt st1(x1)
        JET.@test_opt ST.scattering_batch(st1, randn(64, 2))

        st2 = ST.ScatteringCore.ScatteringTransform((16, 16), 2; L=4, max_order=2, transform=SB.DirectSumSpectralBackend())
        x2 = randn(16, 16)
        JET.@test_opt st2(x2)
        JET.@test_opt ST.scattering_batch(st2, randn(16, 16, 2))

        st3 = ST.ScatteringCore.ScatteringTransform((8, 8, 8), 2; n_orient=6, max_order=2, transform=SB.DirectSumSpectralBackend())
        x3 = randn(8, 8, 8)
        JET.@test_opt st3(x3)
        JET.@test_opt ST.scattering_batch(st3, randn(8, 8, 8, 2))

        # Monogenic (Riesz) production functor.
        stm = ST.Monogenic.MonogenicScatteringTransform((16, 16), 2; Q=1, max_order=2, transform=SB.DirectSumSpectralBackend())
        JET.@test_opt stm(x2)
        JET.@test_opt ST.scattering_batch(stm, randn(16, 16, 2))

        points = collect(range(0.1, 6.0; length = 64))
        planar = ST.scattered_planar_scattering(points, reverse(points), (4, 4), 2;
            period = 2π, transform = SB.DirectSumSpectralBackend(),
            analysis = ST.Plans.LeastSquares())
        JET.@test_opt target_modules = (ST,) planar(zeros(64))
        JET.@test_opt target_modules = (ST,) ST.scattering_batch(planar, zeros(64, 2))

        θ = acos.(collect(range(-0.95, 0.95; length = 64)))
        sphere = ST.spherical_scattering(θ, points, 3, 2;
            transform = SB.DirectSumSpectralBackend(), analysis = ST.Plans.LeastSquares())
        JET.@test_opt target_modules = (ST,) sphere(zeros(64))
        spherical_batch = ST.SphericalBatchWorkspace(sphere, 2)
        JET.@test_opt target_modules = (ST,) ST._apply_batch!(zeros(ST.flat_rows(sphere), 2),
                                                            spherical_batch, zeros(64, 2))
        structured = ST.structured_spherical_scattering(3, 2; transform = SB.DirectSumSpectralBackend())
        JET.@test_opt target_modules = (ST,) structured(zeros(4, 7))
        structured_batch = ST.SphericalBatchWorkspace(structured, 2)
        JET.@test_opt target_modules = (ST,) ST._apply_batch!(zeros(ST.flat_rows(structured), 2),
                                                            structured_batch, zeros(4, 7, 2))

        # Filter frequency responses (built once per bank).
        JET.@test_opt ST.Filters.frequency_response(ST.Filters.Morlet1D(64, 0; Q=1))
        JET.@test_opt ST.Filters.frequency_response(ST.Filters.Morlet2D((16, 16), 1, 0.0; L=4))
        JET.@test_opt ST.Filters.frequency_response(ST.Filters.Morlet3D((8, 8, 8), 1, (0.0, 0.0, 1.0)))

        # Exact linear wavelet-frame inverse round trip (non-mutating, eltype-generic).
        sti = ST.ScatteringCore.ScatteringTransform(64, 4; Q=1, max_order=1, transform=SB.DirectSumSpectralBackend())
        JET.@test_opt ST.Inverse.iwavelet(sti, ST.Inverse.wavelet_transform(sti, x1))
    end
end
