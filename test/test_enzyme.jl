# Reverse-mode autodiff coverage (Enzyme backend, via DifferentiationInterface).
#
# Gated on released Julia: Enzyme compiles against LLVM internals and does not reliably build on
# pre-release versions, so an unconditional load would take the whole suite down on nightly. The
# gate mirrors test_jet.jl, so nightly still exercises everything except these checks.
using ScatteringTransforms: ScatteringTransforms
using SpectralBackends: SpectralBackends
using DifferentiationInterface: DifferentiationInterface as DI
using ADTypes: AutoEnzyme
using Random: Random
using Test: Test

@static if isempty(VERSION.prerelease)
    using Enzyme: Enzyme   # loads DifferentiationInterface's Enzyme extension used below

    # Reverse mode: the objective is scalar in a length-N vector, so one reverse pass gives the
    # whole gradient.
    #
    # No custom AD rule is needed: the direct-sum plan's non-mutating transform is plain scalar
    # arithmetic over its twiddle table. A dense-matrix formulation would need one, because Enzyme
    # has no derivative for complex `gemm`.
    #
    # Runtime activity is on because `scattering` maps closures over the filter bank, so each
    # closure captures constant filters alongside the active input and static activity analysis
    # cannot separate them. This is Enzyme's documented remedy for mixed activity, not a workaround
    # for a defect in the transform — and it is the whole configuration `synthesize` needs, which is
    # the minimum this test should pin: `synthesize` declares the transform and target as
    # `DI.Constant` arguments rather than capturing them, so no `function_annotation` is required.
    enzyme_backend() = AutoEnzyme(; mode = Enzyme.set_runtime_activity(Enzyme.Reverse))

    # A hand-written closure over the transform additionally needs `Const`: the closure is a struct
    # holding the filter bank and plan tables, and reverse mode cannot prove such an argument
    # read-only. `synthesize` avoids the annotation by passing those as `DI.Constant` arguments; a
    # caller differentiating their own closure has to say so themselves.
    enzyme_closure_backend() = AutoEnzyme(; mode = Enzyme.set_runtime_activity(Enzyme.Reverse),
                                          function_annotation = Enzyme.Const)

    Test.@testset "Autodiff through scattering(st,x): Enzyme gradient matches finite differences" begin
        Random.seed!(1)   # reproducibility (this AD-vs-FD check is seed-robust anyway)
        N, J = 32, 3
        st = ScatteringTransforms.Scattering1D.ScatteringTransform1D(N, J; Q=1, max_order=2,
                                                        spectral=SpectralBackends.DirectSumSpectralBackend())
        xt = randn(N)
        target = ScatteringTransforms.ScatteringCore.scattering(st, xt)
        t1 = ScatteringTransforms.Coefficients.first_order(target)
        t2 = ScatteringTransforms.Coefficients.second_order(target)
        loss(x) = sum(abs2, ScatteringTransforms.Coefficients.first_order(ScatteringTransforms.ScatteringCore.scattering(st, x)) .- t1) +
                  sum(abs2, ScatteringTransforms.Coefficients.second_order(ScatteringTransforms.ScatteringCore.scattering(st, x)) .- t2)

        backend = enzyme_closure_backend()
        x0 = randn(N)
        prep = DI.prepare_gradient(loss, backend, x0)
        val, grad = DI.value_and_gradient(loss, prep, backend, x0)
        Test.@test all(isfinite, grad)
        Test.@test val ≈ loss(x0)

        # Directional finite-difference check.
        d = randn(N); h = 1e-6
        fd = (loss(x0 .+ h .* d) - loss(x0 .- h .* d)) / (2h)
        Test.@test isapprox(sum(grad .* d), fd; rtol=1e-3, atol=1e-8)
    end

    Test.@testset "Gradient-descent synthesis (DifferentiationInterface ext, Enzyme)" begin
        # synthesize: from noise, descend ‖S(x̂) − S(target)‖² so the coefficients converge.
        # `iters=400` is a converged budget: the coefficient error decreases monotonically with
        # iterations, so at 400 steps any random init lands well under the 0.1 relative-error bar —
        # the seed below is only for reproducibility, not to dodge slow-converging inits.
        Random.seed!(1)
        N, J = 48, 4
        iters = 400
        st = ScatteringTransforms.Scattering1D.ScatteringTransform1D(N, J; Q=1, max_order=2,
                                                        spectral=SpectralBackends.DirectSumSpectralBackend())
        xtarget = cumsum(randn(N)); xtarget .-= sum(xtarget) / N
        res = ScatteringTransforms.synthesize(st, xtarget;
                                              backend=enzyme_backend(), init=randn(N), iters=iters, lr=0.05)
        Test.@test length(res.losses) == iters
        Test.@test all(isfinite, res.field)
        Test.@test res.losses[end] < res.losses[1] / 5           # objective drops substantially

        # Synthesized coefficients approach the target's.
        ct = ScatteringTransforms.ScatteringCore.scattering(st, xtarget)
        cs = ScatteringTransforms.ScatteringCore.scattering(st, res.field)
        rel = sqrt(sum(abs2, ScatteringTransforms.Coefficients.first_order(cs) .- ScatteringTransforms.Coefficients.first_order(ct))) /
              sqrt(sum(abs2, ScatteringTransforms.Coefficients.first_order(ct)))
        Test.@test rel < 0.1

        # Target may also be passed as a precomputed coefficient container.
        res2 = ScatteringTransforms.synthesize(st, ct; backend=enzyme_backend(), init=randn(N), iters=20)
        Test.@test res2.losses[end] < res2.losses[1]
    end
else
    Test.@testset "Enzyme autodiff (skipped on pre-release Julia)" begin
        @info "Skipping Enzyme autodiff checks: Enzyme does not build on pre-release Julia $(VERSION)"
        Test.@test_skip true
    end
end
