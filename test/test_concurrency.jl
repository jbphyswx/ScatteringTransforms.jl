# Plan construction and transforms from concurrent tasks. Every fast backend here plans through one
# process-global FFTW planner, which FFTW.jl's planner lock guards; FastTransforms' OpenMP thread count
# is set and restored per call on the OS thread making it. A build racing another library's build
# faults inside the planner.
using FastSphericalHarmonics: FastSphericalHarmonics as FSH
using NUFSHT: NUFSHT
using OhMyThreads: OhMyThreads

# Tallies the plans `task_local_plan` and `batch_plan` build from a plan, and the `close_plan!` calls
# they receive; every other call passes through to `inner`.
struct LifecyclePlan{P, A} <: ScatteringTransforms.Plans.AbstractScatteringPlan
    inner::P
    built::A
    closed::A
end
LifecyclePlan(inner) = LifecyclePlan(inner, Threads.Atomic{Int}(0), Threads.Atomic{Int}(0))
_derived(p::LifecyclePlan, inner) =
    (Threads.atomic_add!(p.built, 1); LifecyclePlan(inner, p.built, p.closed))

ScatteringTransforms.Plans.forward_transform!(out, p::LifecyclePlan, x) =
    ScatteringTransforms.Plans.forward_transform!(out, p.inner, x)
ScatteringTransforms.Plans.inverse_transform!(out, p::LifecyclePlan, x) =
    ScatteringTransforms.Plans.inverse_transform!(out, p.inner, x)
ScatteringTransforms.Plans.batch_width(p::LifecyclePlan) =
    ScatteringTransforms.Plans.batch_width(p.inner)
ScatteringTransforms.Plans.plan_analysis(p::LifecyclePlan) =
    ScatteringTransforms.Plans.plan_analysis(p.inner)
ScatteringTransforms.Plans.task_local_plan(p::LifecyclePlan) =
    _derived(p, ScatteringTransforms.Plans.task_local_plan(p.inner))
ScatteringTransforms.Plans.close_plan!(p::LifecyclePlan) =
    (Threads.atomic_add!(p.closed, 1); ScatteringTransforms.Plans.close_plan!(p.inner))
ScatteringTransforms.SphericalCore.supports_batch(p::LifecyclePlan) =
    ScatteringTransforms.SphericalCore.supports_batch(p.inner)
ScatteringTransforms.SphericalCore.batch_plan(p::LifecyclePlan, B::Integer) =
    ScatteringTransforms.Plans.batch_width(p) == B ? p :
    _derived(p, ScatteringTransforms.SphericalCore.batch_plan(p.inner, B))
ScatteringTransforms.SphericalCore.sphere_coeffs!(C, p::LifecyclePlan, f) =
    ScatteringTransforms.SphericalCore.sphere_coeffs!(C, p.inner, f)
ScatteringTransforms.SphericalCore.sphere_coeffs_buffer(p::LifecyclePlan) =
    ScatteringTransforms.SphericalCore.sphere_coeffs_buffer(p.inner)
ScatteringTransforms.SphericalCore.sphere_field_buffer(p::LifecyclePlan) =
    ScatteringTransforms.SphericalCore.sphere_field_buffer(p.inner)
ScatteringTransforms.SphericalCore.sphere_npoints(p::LifecyclePlan) =
    ScatteringTransforms.SphericalCore.sphere_npoints(p.inner)
ScatteringTransforms.SphericalCore.sphere_apply!(out, p::LifecyclePlan, C, h) =
    ScatteringTransforms.SphericalCore.sphere_apply!(out, p.inner, C, h)
ScatteringTransforms.SphericalCore.sphere_mean(p::LifecyclePlan, f) =
    ScatteringTransforms.SphericalCore.sphere_mean(p.inner, f)
ScatteringTransforms.SphericalCore.sphere_means!(out, p::LifecyclePlan, f) =
    ScatteringTransforms.SphericalCore.sphere_means!(out, p.inner, f)
ScatteringTransforms.SphericalCore.coeffs_means!(out, p::LifecyclePlan, C) =
    ScatteringTransforms.SphericalCore.coeffs_means!(out, p.inner, C)
ScatteringTransforms.SphericalCore.riesz_scratch(p::LifecyclePlan, f) =
    ScatteringTransforms.SphericalCore.riesz_scratch(p.inner, f)
ScatteringTransforms.SphericalCore.sphere_riesz_energy!(out, p::LifecyclePlan, C, h, scr) =
    ScatteringTransforms.SphericalCore.sphere_riesz_energy!(out, p.inner, C, h, scr)
ScatteringTransforms.SphericalCore.plan_points(p::LifecyclePlan) =
    ScatteringTransforms.SphericalCore.plan_points(p.inner)
ScatteringTransforms.SphericalCore.coeffs_mean(p::LifecyclePlan, C, b::Int = 1) =
    ScatteringTransforms.SphericalCore.coeffs_mean(p.inner, C, b)

# `st` with its plan, and its grid where it has one, wrapped in a `LifecyclePlan`, everything else
# shared.
function with_lifecycle(st)
    T = typeof(st)
    wrap(i) = fieldname(T, i) === :workspaces ? ScatteringTransforms.ScatteringCore.Workspaces(T.name.wrapper) :
              (fieldname(T, i) in (:plan, :grid) && getfield(st, i) !== nothing) ?
              LifecyclePlan(getfield(st, i)) : getfield(st, i)
    return T.name.wrapper(ntuple(wrap, fieldcount(T))...)
end

Test.@testset "Batch plans are reused and closed with the transform" begin
    CBk = ComputationalBackends
    M, ms, J = 300, (12, 12), 2
    Random.seed!(17)
    x, y = 2π .* rand(M), 2π .* rand(M)
    X = randn(M, 4)
    for (spectral, ntrans) in ((SpectralBackends.DirectSumSpectralBackend(), 1),
                               (FTB.FINUFFTBackend(), 2), (FTB.NonuniformFFTsBackend(), 2))
        st = ScatteringTransforms.scattered_planar_scattering(x, y, ms, J; L = 4,
                 period = (2π, 2π), transform = spectral, ntrans = ntrans)
        stc = with_lifecycle(st)
        serial = ntrans == 1 ? ScatteringTransforms.scattering_batch(st, X) :
                 hcat(ScatteringTransforms.scattering_batch(st, X[:, 1:2]),
                      ScatteringTransforms.scattering_batch(st, X[:, 3:4]))
        Test.@test ScatteringTransforms.scattering_batch(stc, X; execution = CBk.ThreadedBackend()) ≈ serial
        built = stc.plan.built[]
        ScatteringTransforms.scattering_batch(stc, X; execution = CBk.ThreadedBackend())
        Test.@test stc.plan.built[] == built >= 1
        ScatteringTransforms.close_transform!(stc)
        Test.@test stc.plan.closed[] == built + 1
    end

    lmax, Ms = 6, 200
    θ = [acos(1 - 2 * (k - 0.5) / Ms) for k in 1:Ms]
    φ = [2π * mod(k * (sqrt(5) - 1) / 2, 1) for k in 1:Ms]
    F = [cos(2θ[k]) + 0.3b * sin(θ[k]) * cos(φ[k]) for k in 1:Ms, b in 1:3]
    for spectral in (SpectralBackends.DirectSumSpectralBackend(), SpectralBackends.NUFSHTSpectralBackend())
        for build in (ScatteringTransforms.spherical_scattering,
                      ScatteringTransforms.spherical_monogenic_scattering)
            st = build(θ, φ, lmax, J; transform = spectral)
            stc = with_lifecycle(st)
            serial = ScatteringTransforms.scattering_batch(st, F)
            for backend in (CBk.SerialBackend(), CBk.ThreadedBackend())
                Test.@test ScatteringTransforms.scattering_batch(stc, F; execution = backend) ≈ serial rtol = 1e-8
            end
            if build === ScatteringTransforms.spherical_scattering
                retained = length(stc.workspaces.batches)
                ScatteringTransforms.scattering_batch(stc, view(F, :, 1:2))
                built = stc.plan.built[]
                ScatteringTransforms.scattering_batch(stc, view(F, :, 1:2))
                Test.@test stc.plan.built[] == built
                Test.@test length(stc.workspaces.batches) == retained
            end
            ScatteringTransforms.close_transform!(stc)
            Test.@test stc.plan.closed[] == stc.plan.built[] + 1
            Test.@test stc.grid.built[] >= 1
            Test.@test stc.grid.closed[] == stc.grid.built[] + 1
            ScatteringTransforms.close_transform!(st)
        end
    end
end

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

        function transform_once(t)
            st = ScatteringTransforms.spherical_scattering(θs[t], φs[t], lmax, J)
            try
                return st(fields[t])
            finally
                ScatteringTransforms.close_transform!(st)
            end
        end
        serial = [transform_once(t) for t in 1:ntask]
        concurrent = Vector{Any}(undef, ntask)
        OhMyThreads.@tasks for t in 1:ntask
            concurrent[t] = transform_once(t)
        end
        for t in 1:ntask
            Test.@test concurrent[t].S0 ≈ serial[t].S0
            Test.@test concurrent[t].S1 ≈ serial[t].S1
            Test.@test concurrent[t].S2 ≈ serial[t].S2
        end
        # The calling thread's FastTransforms count is restored.
        Test.@test ft_count() == before
    end

    Test.@testset "one reused spherical plan applied concurrently matches the serial batch" begin
        # One column per thread makes every chunk one wide, the width this plan already has, so the
        # threaded batch asks for a plan of the width it is holding.
        lmax, J, M = 8, 3, 800
        B = max(2, Threads.nthreads())
        gr = (sqrt(5) - 1) / 2
        θ = [acos(1 - 2 * (k - 0.5) / M) for k in 1:M]
        φ = [2π * mod(k * gr, 1) for k in 1:M]
        st = ScatteringTransforms.spherical_scattering(θ, φ, lmax, J; max_order = 2)
        X = [cos(2 * θ[k]) + 0.4 * cos(3 * φ[k]) * sin(θ[k]) + 0.1 * b for k in 1:M, b in 1:B]

        Test.@test ScatteringTransforms.Plans.batch_width(st.plan) == st.plan.plan.B == 1
        serial = ScatteringTransforms.scattering_batch(st, X; execution = ComputationalBackends.SerialBackend())
        threaded = ScatteringTransforms.scattering_batch(st, X; execution = ComputationalBackends.ThreadedBackend())
        Test.@test threaded ≈ serial rtol = 1e-8
        # The plan handed to a task is never the shared one, whatever width is asked for.
        for k in (1, B)
            p = ScatteringTransforms.SphericalCore.task_local_batch_plan(st.plan, k)
            Test.@test p !== st.plan
            Test.@test ScatteringTransforms.Plans.batch_width(p) == k
            ScatteringTransforms.Plans.close_plan!(p)
        end
        ScatteringTransforms.close_transform!(st)
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
        # take the same lock.
        Ny, Nx, J, L, M, ntask = 16, 16, 3, 4, 800, 4
        xs = [2π .* rand(M) for _ in 1:ntask]
        ys = [2π .* rand(M) for _ in 1:ntask]
        g(x, y) = 1.0 + 0.7cos(x) + 0.5sin(2y)
        fields = [[g(xs[t][k], ys[t][k]) for k in 1:M] for t in 1:ntask]
        for spectral in (FTB.FINUFFTBackend(), FTB.NonuniformFFTsBackend())
            function transform_once(t)
                st = ScatteringTransforms.scattered_planar_scattering(
                    xs[t], ys[t], (Ny, Nx), J; L = L, max_order = 2, period = (2π, 2π),
                    transform = spectral)
                try
                    return st(fields[t])
                finally
                    ScatteringTransforms.close_transform!(st)
                end
            end
            serial = [transform_once(t) for t in 1:ntask]
            concurrent = Vector{Any}(undef, ntask)
            OhMyThreads.@tasks for t in 1:ntask
                concurrent[t] = transform_once(t)
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
