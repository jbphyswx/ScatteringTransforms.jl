# Plan construction and transforms from concurrent tasks. Every fast backend here plans through one
# process-global FFTW planner, which `Plans.PLANNER_LOCK` guards; FastTransforms' OpenMP thread count is
# set and restored per call on the OS thread making it. A build racing another library's build faults
# inside the planner.
using FastSphericalHarmonics: FastSphericalHarmonics as FSH
using NUFSHT: NUFSHT
using OhMyThreads: OhMyThreads

Test.@testset "Concurrent plan construction" begin
    Test.@testset "concurrent scattered-sphere builds match serial ones" begin
        lmax, J, M, ntask = 8, 2, 400, 4
        gr = (sqrt(5) - 1) / 2
        θs = [[acos(1 - 2 * (k - 0.5) / M) for k in 1:M] for _ in 1:ntask]
        φs = [[2π * mod((k + t) * gr, 1) for k in 1:M] for t in 1:ntask]
        fields = [[cos(2 * θs[t][k]) + 0.4 * cos(3 * φs[t][k]) * sin(θs[t][k]) for k in 1:M]
                  for t in 1:ntask]

        ft_count() = Int(ccall((:omp_get_max_threads, NUFSHT.FastTransforms.libfasttransforms),
                               Cint, ()))
        before = ft_count()

        serial = [ScatteringTransforms.spherical_scattering(θs[t], φs[t], lmax, J)(fields[t])
                  for t in 1:ntask]
        concurrent = Vector{Any}(undef, ntask)
        OhMyThreads.@tasks for t in 1:ntask
            concurrent[t] = ScatteringTransforms.spherical_scattering(θs[t], φs[t], lmax, J)(fields[t])
        end
        for t in 1:ntask
            Test.@test concurrent[t].S0 ≈ serial[t].S0
            Test.@test concurrent[t].S1 ≈ serial[t].S1
            Test.@test concurrent[t].S2 ≈ serial[t].S2
        end
        # And the calling thread's FastTransforms count is left as it was found.
        Test.@test ft_count() == before
    end

    Test.@testset "one reused spherical plan applied concurrently matches the serial batch" begin
        # One column per thread makes every chunk one wide, which is the width this plan already has,
        # so the threaded batch asks for a plan of the width it is holding. Any wider and each task
        # builds its own plan, where no sharing can occur — that sizing is what the earlier threaded
        # tests use, and why they never saw this. (At one thread there is nothing to race either way.)
        lmax, J, M = 8, 3, 800
        B = max(2, Threads.nthreads())
        gr = (sqrt(5) - 1) / 2
        θ = [acos(1 - 2 * (k - 0.5) / M) for k in 1:M]
        φ = [2π * mod(k * gr, 1) for k in 1:M]
        st = ScatteringTransforms.spherical_scattering(θ, φ, lmax, J; max_order = 2)
        X = [cos(2 * θ[k]) + 0.4 * cos(3 * φ[k]) * sin(θ[k]) + 0.1 * b for k in 1:M, b in 1:B]

        # The NUFSHT plan's own width, not `Plans.batch_width`: that falls back to 1 for any argument
        # it has no method for, so it would report 1 here whatever the plan actually is.
        Test.@test st.plan.plan.B == 1
        serial = ScatteringTransforms.scattering_batch(ComputationalBackends.SerialBackend(), st, X)
        threaded = ScatteringTransforms.scattering_batch(ComputationalBackends.ThreadedBackend(), st, X)
        # A raced solve diverges rather than erroring, so the comparison is against the serial values,
        # not merely against `isfinite`.
        Test.@test all(isfinite, threaded)
        Test.@test threaded ≈ serial rtol = 1e-8
        # The plan handed to a task is never the shared one, whatever width is asked for.
        for k in (1, B)
            Test.@test ScatteringTransforms.SphericalCore.task_local_batch_plan(st.plan, k) !== st.plan
        end
    end

    Test.@testset "concurrent structured-sphere builds match serial ones" begin
        lmax, J, ntask = 12, 2, 4
        Θ, Φ = ScatteringTransforms.structured_sphere_points(lmax)
        f = [cos(θ)^2 - 1/3 + 0.5 * sin(θ) * cos(φ) for θ in Θ, φ in Φ]
        serial = ScatteringTransforms.structured_spherical_scattering(lmax, J)(f)
        concurrent = Vector{Any}(undef, ntask)
        OhMyThreads.@tasks for t in 1:ntask
            concurrent[t] = ScatteringTransforms.structured_spherical_scattering(lmax, J)(f)
        end
        for t in 1:ntask
            Test.@test concurrent[t].S1 ≈ serial.S1
            Test.@test concurrent[t].S2 ≈ serial.S2
        end
    end

    Test.@testset "concurrent scattered-planar builds match serial ones" begin
        # The NUFFT backends plan through the same libfftw3 as the spherical ones, so their builds
        # take the same lock; a spherical build racing a planar one was the observed crash.
        Ny, Nx, J, L, M, ntask = 16, 16, 3, 4, 800, 4
        xs = [2π .* rand(M) for _ in 1:ntask]
        ys = [2π .* rand(M) for _ in 1:ntask]
        g(x, y) = 1.0 + 0.7cos(x) + 0.5sin(2y)
        fields = [[g(xs[t][k], ys[t][k]) for k in 1:M] for t in 1:ntask]
        for spectral in (ScatteringTransforms.Plans.FINUFFTBackend(),
                         ScatteringTransforms.Plans.NonuniformFFTsBackend())
            build(t) = ScatteringTransforms.scattered_planar_scattering(
                xs[t], ys[t], (Ny, Nx), J; L = L, max_order = 2, period = (2π, 2π),
                spectral = spectral)
            serial = [build(t)(fields[t]) for t in 1:ntask]
            concurrent = Vector{Any}(undef, ntask)
            OhMyThreads.@tasks for t in 1:ntask
                concurrent[t] = build(t)(fields[t])
            end
            for t in 1:ntask
                Test.@test ScatteringTransforms.Coefficients.first_order(concurrent[t]) ≈
                           ScatteringTransforms.Coefficients.first_order(serial[t])
                Test.@test ScatteringTransforms.Coefficients.second_order(concurrent[t]) ≈
                           ScatteringTransforms.Coefficients.second_order(serial[t])
            end
        end
    end
end
