# The least-squares solver on its own, against LAPACK on dense operators — so a failure here is the
# solver's and not a transform convention's. This is the ground truth the scattered-planar solve never
# had: its only exact-reference test ran on the uniform DFT nodes, where `A†A = prod(ms)·I` and any
# solver converges in one iteration.
using LinearAlgebra: LinearAlgebra

Test.@testset "LSMR least-squares solver" begin
    P = ScatteringTransforms.Plans

    # `x`, plus the six work arrays: two over point space and four over mode space.
    function lsmr(A, b; damp = 0.0, atol = 1e-13, btol = 1e-13, conlim = 1e14, maxiter = 2000)
        m, n = size(A)
        CT = complex(eltype(A))
        Aadj = collect(adjoint(A))
        x = zeros(CT, n)
        info = P.lsmr_solve!(x,
                             (dst, src) -> LinearAlgebra.mul!(dst, A, src),
                             (dst, src) -> LinearAlgebra.mul!(dst, Aadj, src),
                             b, zeros(CT, m), zeros(CT, m),
                             zeros(CT, n), zeros(CT, n), zeros(CT, n), zeros(CT, n);
                             damp = damp, atol = atol, btol = btol, conlim = conlim,
                             maxiter = maxiter)
        return x, info
    end
    relerr(a, b) = LinearAlgebra.norm(a - b) / LinearAlgebra.norm(b)

    Test.@testset "_sym_ortho satisfies its defining identities" begin
        # A wrong sign branch still converges on easy problems and corrupts only hard ones, so this is
        # checked directly rather than through a solve. `1e30` is where `sqrt(a^2 + b^2)` overflows in
        # Float32 and `hypot` does not.
        for T in (Float64, Float32)
            for a in T[0, -0.0, 1, -1, 3, -4, 1e-30, -1e-30, 1e30, -1e30],
                b in T[0, -0.0, 1, -1, 3, -4, 1e-30, -1e-30, 1e30, -1e30]
                c, s, r = P._sym_ortho(a, b)
                Test.@test r >= 0
                Test.@test isfinite(r)
                if r > 0
                    tol = sqrt(eps(T)) * max(r, one(T))
                    Test.@test abs(c * a + s * b - r) <= tol
                    Test.@test abs(-s * a + c * b) <= tol
                end
            end
        end
        Test.@test P._sym_ortho(1.0f30, 1.0f30)[3] ≈ 1.4142135f30
    end

    Test.@testset "overdetermined and well conditioned: matches LAPACK" begin
        Random.seed!(7)
        A = randn(ComplexF64, 200, 50)
        b = randn(ComplexF64, 200)
        x, info = lsmr(A, b)
        Test.@test relerr(x, A \ b) < 1e-10
        Test.@test info.istop in (1, 2)
        # The residual the recurrence reports is the true one — the property the previous solver
        # violated, where a recursively updated residual reached 8e-24 in Float32.
        Test.@test info.normr ≈ LinearAlgebra.norm(b - A * x) rtol = 1e-8
        # `normA` accumulates `Σ(α² + β²)` over the bidiagonalisation, so it estimates the *Frobenius*
        # norm of the part built so far — bounded above by `‖A‖_F`, and not the spectral norm.
        Test.@test 0 < info.normA <= LinearAlgebra.norm(A) * (1 + 1e-8)
    end

    Test.@testset "underdetermined: converges to the minimum-norm solution" begin
        Random.seed!(7)
        A = randn(ComplexF64, 50, 200)
        b = randn(ComplexF64, 50)
        x, info = lsmr(A, b)
        Test.@test relerr(x, LinearAlgebra.pinv(A) * b) < 1e-10
        Test.@test LinearAlgebra.norm(b - A * x) < 1e-9
    end

    Test.@testset "damping matches the augmented system [A; λI]" begin
        Random.seed!(7)
        A = randn(ComplexF64, 50, 200)
        b = randn(ComplexF64, 50)
        nx, nr = Float64[], Float64[]
        for λ in (1.0e-3, 1.0e-1, 1.0)
            x, _ = lsmr(A, b; damp = λ)
            aug = [A; λ * Matrix(LinearAlgebra.I, 200, 200)]
            Test.@test relerr(x, aug \ [b; zeros(ComplexF64, 200)]) < 1e-10
            push!(nx, LinearAlgebra.norm(x))
            push!(nr, LinearAlgebra.norm(b - A * x))
        end
        # Tikhonov is a contraction in λ: the solution shrinks and the data misfit grows.
        Test.@test issorted(nx; rev = true)
        Test.@test issorted(nr)
    end

    Test.@testset "‖r‖ and ‖A†r‖ decrease monotonically" begin
        # The property the whole choice of LSMR rests on, checked on a rank-deficient operator — where
        # conjugate gradients on the normal equations instead grew from 2e-16 to 70 over 100 iterations.
        Random.seed!(3)
        U = LinearAlgebra.qr(randn(ComplexF64, 60, 60)).Q * Matrix(LinearAlgebra.I, 60, 60)
        V = LinearAlgebra.qr(randn(ComplexF64, 80, 80)).Q * Matrix(LinearAlgebra.I, 80, 80)
        A = U * [LinearAlgebra.diagm([exp10.(range(0, -6; length = 40)); zeros(20)]) zeros(60, 20)] *
            adjoint(V)
        b = randn(ComplexF64, 60)
        prev_r, prev_ar = Inf, Inf
        for k in 1:40
            _, info = lsmr(A, b; maxiter = k, atol = 0.0, btol = 0.0, conlim = 0.0)
            Test.@test info.normr <= prev_r * (1 + 1e-10)
            Test.@test info.normar <= prev_ar * (1 + 1e-10)
            prev_r, prev_ar = info.normr, info.normar
        end
    end

    Test.@testset "a zero right-hand side is the zero solution, not a division by zero" begin
        Random.seed!(7)
        A = randn(ComplexF64, 30, 10)
        x, info = lsmr(A, zeros(ComplexF64, 30))
        Test.@test all(iszero, x)
        Test.@test info.iters == 0
    end

    Test.@testset "the tolerance default follows the transform's accuracy" begin
        SBk = SpectralBackends
        # An exact operator is limited by the arithmetic; an approximate one by its own tolerance,
        # since Type-1 and Type-2 at tolerance `eps` are adjoints only to about `eps`.
        Test.@test P.default_solver_rtol(Float64, SBk.DirectSumSpectralBackend(), nothing) ≈
                   sqrt(eps(Float64))
        Test.@test P.default_solver_rtol(Float32, SBk.DirectSumSpectralBackend(), nothing) ≈
                   sqrt(eps(Float32))
        Test.@test P.default_solver_rtol(Float64, P.FINUFFTBackend(), 1.0e-9) ≈ 1.49011612e-8 rtol = 1e-6
        Test.@test P.default_solver_rtol(Float32, P.FINUFFTBackend(), nothing) >= 1.0f-5
    end

    Test.@testset "the guard refuses only an unusable answer" begin
        # What the solve is allowed to return is a decision, and it is the one thing here with no
        # natural failing input to reach it: a rank-deficient point set does not make this fire,
        # because `A†b` lies in the range of `A†A` by construction, so the iteration never enters the
        # singular directions (checked: 10 points against 81 spherical modes, and every planar point
        # duplicated, both converge and return). The contract is therefore asserted directly.
        info(; istop, normr = 1.0, condA = 1.0e3) =
            (istop = istop, iters = 7, normr = normr, normar = 1.0e-9, normA = 10.0, condA = condA)
        # Stopping at `maxiter` is not a failure: both error measures are monotone, so the iterate is
        # the best one seen. This is the case a `rtol`-based guard would wrongly reject.
        Test.@test P._check_solve(info(istop = 7), 1000, (16, 16), 1.0e-8, 100) === nothing
        Test.@test P._check_solve(info(istop = 1), 1000, (16, 16), 1.0e-8, 100) === nothing
        Test.@test P._check_solve(info(istop = 2), 1000, (16, 16), 1.0e-8, 100) === nothing
        # A conditioning estimate past the limit, or past what the precision can express, means the
        # smallest resolved direction carries nothing — and a broken recurrence means nothing at all.
        Test.@test_throws P.AnalysisNotConverged P._check_solve(info(istop = 3), 1000, (16, 16),
                                                                1.0e-8, 100)
        Test.@test_throws P.AnalysisNotConverged P._check_solve(info(istop = 6), 1000, (16, 16),
                                                                1.0e-8, 100)
        Test.@test_throws P.AnalysisNotConverged P._check_solve(info(istop = 7, normr = NaN), 1000,
                                                                (16, 16), 1.0e-8, 100)
        Test.@test_throws P.AnalysisNotConverged P._check_solve(info(istop = 7, normr = Inf), 1000,
                                                                (16, 16), 1.0e-8, 100)
        # The message names the geometry, because "did not converge" alone tells the caller nothing
        # about which of `ms`, the point count, or `damp` to change.
        err = try
            P._check_solve(info(istop = 3), 300, (24, 24), 1.0e-8, 100)
        catch e
            e
        end
        msg = sprint(Base.showerror, err)
        Test.@test occursin("576 modes", msg) && occursin("300 points", msg)
        Test.@test occursin("damp", msg)
    end

    Test.@testset "an underdetermined solve warns rather than picking a λ" begin
        # No `λ` is chosen for the caller, so the one thing owed them is notice that the problem they
        # posed has no unique answer. `maxlog = 1` caps the warning per logger, and `@test_logs`
        # installs a fresh one, so each case below is seen.
        Test.@test_logs (:warn, r"underdetermined") P.warn_underdetermined(300, (24, 24), true, 0)
        # Silent when the samples determine the modes, when damping was asked for, and when the plan
        # does no solve at all.
        Test.@test_logs P.warn_underdetermined(1000, (16, 16), true, 0)
        Test.@test_logs P.warn_underdetermined(300, (24, 24), true, 0.1)
        Test.@test_logs P.warn_underdetermined(300, (24, 24), false, 0)
    end
end
