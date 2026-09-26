# What ScatteringTransforms decides around its least-squares solve, `FlowTransformBindings.lsmr!`,
# which FlowTransformBindings tests against LAPACK: the tolerance default, the refusal of an unusable
# answer, and the notice for an underdetermined mode grid.

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

    Test.@testset "the guard refuses only an unusable answer" begin
        # A rank-deficient point set does not reach this: `A†b` lies in the range of `A†A`, so the
        # iteration never enters the singular directions (10 points against 81 spherical modes, and every
        # planar point duplicated, both converge and return). The contract is asserted on a workspace.
        function outcome(status; normr = 1.0, condA = 1.0e3)
            op = FTB.FunctionOperator((y, x) -> y, (x, y) -> x, zeros(ComplexF64, 4, 1),
                                      zeros(ComplexF64, 8, 1))
            ws = FTB.LSMRWorkspace(op)
            ws.status[1] = status
            ws.normr[1] = normr
            ws.condA[1] = condA
            ws.iterations[1] = 7
            return ws
        end
        # Stopping at `maxiter` is not a failure: both error measures are monotone, so the iterate is the
        # best one seen.
        for s in (FTB.LSMR_MAXITER, FTB.LSMR_RESIDUAL, FTB.LSMR_OPTIMAL, FTB.LSMR_PRECISION)
            Test.@test P._check_solve(outcome(s), 1000, (16, 16), 1.0e-8, 100) === nothing
        end
        # A condition estimate at `conlim` or `1/eps` means the smallest resolved direction carries
        # nothing, and a non-finite residual means nothing at all.
        Test.@test_throws P.AnalysisNotConverged P._check_solve(outcome(FTB.LSMR_CONDITION), 1000,
                                                                (16, 16), 1.0e-8, 100)
        for r in (NaN, Inf)
            Test.@test_throws P.AnalysisNotConverged P._check_solve(outcome(FTB.LSMR_MAXITER; normr = r),
                                                                    1000, (16, 16), 1.0e-8, 100)
        end
        # The message names the geometry, since "did not converge" alone does not say which of `ms`, the
        # point count or `damp` to change.
        err = try
            P._check_solve(outcome(FTB.LSMR_CONDITION), 300, (24, 24), 1.0e-8, 100)
        catch e
            e
        end
        msg = sprint(Base.showerror, err)
        Test.@test occursin("576 modes", msg) && occursin("300 points", msg)
        Test.@test occursin("damp", msg)
    end

    Test.@testset "an underdetermined solve warns and picks no λ" begin
        # `maxlog = 1` caps the warning per logger, and `@test_logs` installs a fresh one, so each case
        # below is seen.
        Test.@test_logs (:warn, r"underdetermined") P.warn_underdetermined(300, (24, 24), true, 0)
        # Silent when the samples determine the modes, when damping was asked for, and when the plan
        # does no solve at all.
        Test.@test_logs P.warn_underdetermined(1000, (16, 16), true, 0)
        Test.@test_logs P.warn_underdetermined(300, (24, 24), true, 0.1)
        Test.@test_logs P.warn_underdetermined(300, (24, 24), false, 0)
    end
end
