using ScatteringTransforms: ScatteringTransforms as ST
using JET: JET
using Test: Test

Test.@testset "JET type stability (hot path)" begin
    # JET tracks compiler internals and explicitly refuses to run on pre-release Julia
    # (`@test_opt` throws on nightly/rc — it loads as a no-op stub). Skip there; the type-stability
    # gate runs on every released version in CI.
    if !isempty(VERSION.prerelease)
        @info "Skipping JET type-stability checks on pre-release Julia $(VERSION)"
        Test.@test_skip true
    else
        # Audit the in-place, zero-alloc PRODUCTION functors. The in-core direct-sum backend keeps
        # the whole call inside ScatteringTransforms (no FFTW/CUFFT internals to analyze).
        st1 = ST.Scattering1D.ScatteringTransform1D(64, 4; Q=2, max_order=2, spectral=ST.Plans.DirectSumBackend())
        x1 = randn(64)
        JET.@test_opt st1(x1)
        JET.@test_call st1(x1)

        st2 = ST.Scattering2D.ScatteringTransform2D((16, 16), 2; L=4, max_order=2, spectral=ST.Plans.DirectSumBackend())
        x2 = randn(16, 16)
        JET.@test_opt st2(x2)
        JET.@test_call st2(x2)

        st3 = ST.Scattering3D.ScatteringTransform3D((8, 8, 8), 2; n_orient=6, max_order=2, spectral=ST.Plans.DirectSumBackend())
        x3 = randn(8, 8, 8)
        JET.@test_opt st3(x3)

        # Monogenic (Riesz) production functor.
        stm = ST.Monogenic.MonogenicScattering((16, 16), 2; Q=1, max_order=2, spectral=ST.Plans.DirectSumBackend())
        JET.@test_opt stm(x2)

        # Filter frequency responses (built once per bank).
        JET.@test_opt ST.Filters.frequency_response(ST.Filters.Morlet1D(64, 0; Q=1))
        JET.@test_opt ST.Filters.frequency_response(ST.Filters.Morlet2D((16, 16), 1, 0.0; L=4))
        JET.@test_opt ST.Filters.frequency_response(ST.Filters.Morlet3D((8, 8, 8), 1, (0.0, 0.0, 1.0)))

        # Exact linear wavelet-frame inverse round trip (non-mutating, eltype-generic).
        sti = ST.Scattering1D.ScatteringTransform1D(64, 4; Q=1, max_order=1, spectral=ST.Plans.DirectSumBackend())
        JET.@test_opt ST.Inverse.iwavelet(sti, ST.Inverse.wavelet_transform(sti, x1))
    end
end
