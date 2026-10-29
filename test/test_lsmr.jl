# Policy around the least-squares solve `FlowTransformBindings.lsmr!`: the tolerance default, the
# notice for an underdetermined mode grid.

Test.@testset "Least-squares solve policy" begin
    P = ScatteringTransforms.Plans

    Test.@testset "the tolerance default follows the transform's accuracy" begin
        SBk = SpectralBackends
        # An exact operator is limited by the arithmetic; an approximate one by its own tolerance,
        # since Type-1 and Type-2 at tolerance `eps` are adjoints only to about `eps`.
        Test.@test P.default_solver_rtol(Float64, SBk.DirectSumSpectralBackend(), nothing) ≈
                   sqrt(eps(Float64))
        Test.@test P.default_solver_rtol(Float32, SBk.DirectSumSpectralBackend(), nothing) ≈
                   sqrt(eps(Float32))
        Test.@test P.default_solver_rtol(Float64, FTB.FINUFFTBackend(), 1.0e-9) ≈ 1.49011612e-8 rtol = 1e-6
        Test.@test P.default_solver_rtol(Float32, FTB.FINUFFTBackend(), nothing) >= 1.0f-5
    end

    Test.@testset "an underdetermined solve warns and picks no λ" begin
        Test.@test_logs (:warn, r"underdetermined") respect_maxlog = true begin
            P.warn_underdetermined(300, (24, 24), P.LeastSquares(), 0)
            P.warn_underdetermined(300, (24, 24), P.LeastSquares(), 0)
        end
        # Silent when the samples determine the modes, when damping was asked for, and when the plan
        # does no solve at all.
        Test.@test_logs P.warn_underdetermined(1000, (16, 16), P.LeastSquares(), 0)
        Test.@test_logs P.warn_underdetermined(300, (24, 24), P.LeastSquares(), 0.1)
        Test.@test_logs P.warn_underdetermined(300, (24, 24), P.WeightedAdjoint(), 0)
    end
end
