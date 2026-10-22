# Reverse-mode autodiff coverage (Mooncake backend, via DifferentiationInterface).
#
# Loading Mooncake errors on pre-release Julia, so the load and the tests are gated on a released
# version.
using ScatteringTransforms: ScatteringTransforms
using DifferentiationInterface: DifferentiationInterface as DI
using ADTypes: AutoMooncake
using Random: Random
using Test: Test

@static if isempty(VERSION.prerelease)
    using Mooncake: Mooncake   # loads DifferentiationInterface's Mooncake extension used below

    Test.@testset "Autodiff through scattering(st,x): Mooncake gradient matches finite differences" begin
        Random.seed!(1)
        N, J = 32, 3
        st = ScatteringTransforms.Scattering1D.ScatteringTransform1D(N, J; Q=1, max_order=2,
                                                        transform=SpectralBackends.DirectSumSpectralBackend())
        xt = randn(N)
        target = ScatteringTransforms.ScatteringCore.scattering(st, xt)
        t1 = ScatteringTransforms.Coefficients.first_order(target)
        t2 = ScatteringTransforms.Coefficients.second_order(target)
        loss(x) = sum(abs2, ScatteringTransforms.Coefficients.first_order(ScatteringTransforms.ScatteringCore.scattering(st, x)) .- t1) +
                  sum(abs2, ScatteringTransforms.Coefficients.second_order(ScatteringTransforms.ScatteringCore.scattering(st, x)) .- t2)

        backend = AutoMooncake()
        x0 = randn(N)
        prep = DI.prepare_gradient(loss, backend, x0)
        val, grad = DI.value_and_gradient(loss, prep, backend, x0)
        Test.@test val ≈ loss(x0)

        # Directional finite-difference check.
        d = randn(N); h = 1e-6
        fd = (loss(x0 .+ h .* d) - loss(x0 .- h .* d)) / (2h)
        Test.@test isapprox(sum(grad .* d), fd; rtol=1e-3, atol=1e-8)
    end

    Test.@testset "Gradient-descent synthesis (DifferentiationInterface ext, Mooncake)" begin
        # synthesize: from noise, descend ‖S(x̂) − S(target)‖² so the coefficients converge.
        # The coefficient error decreases monotonically with iterations, and `iters = 400` puts any
        # random init under the 0.1 relative-error bar.
        Random.seed!(1)
        N, J = 48, 4
        iters = 400
        st = ScatteringTransforms.Scattering1D.ScatteringTransform1D(N, J; Q=1, max_order=2,
                                                        transform=SpectralBackends.DirectSumSpectralBackend())
        xtarget = cumsum(randn(N)); xtarget .-= sum(xtarget) / N
        res = ScatteringTransforms.synthesize(st, xtarget;
                                              backend=AutoMooncake(), init=randn(N), iters=iters, lr=0.05)
        Test.@test length(res.losses) == iters
        Test.@test res.losses[end] < res.losses[1] / 5           # objective drops substantially

        # Synthesized coefficients approach the target's.
        ct = ScatteringTransforms.ScatteringCore.scattering(st, xtarget)
        cs = ScatteringTransforms.ScatteringCore.scattering(st, res.field)
        rel = sqrt(sum(abs2, ScatteringTransforms.Coefficients.first_order(cs) .- ScatteringTransforms.Coefficients.first_order(ct))) /
              sqrt(sum(abs2, ScatteringTransforms.Coefficients.first_order(ct)))
        Test.@test rel < 0.1

        # Target may also be passed as a precomputed coefficient container.
        res2 = ScatteringTransforms.synthesize(st, ct; backend=AutoMooncake(), init=randn(N), iters=20)
        Test.@test res2.losses[end] < res2.losses[1]
    end
else
    Test.@testset "Mooncake autodiff (skipped on pre-release Julia)" begin
        @info "Skipping Mooncake autodiff checks: Mooncake fails to precompile on pre-release Julia $(VERSION)"
        Test.@test_skip true
    end
end
