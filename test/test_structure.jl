# Structural gates: operation counts, workspace sizes and object identity.

include("counting_plan.jl")

Test.@testset "Cascade issues each first-order transform exactly once" begin
    # The cascade walks `(j1, children)` groups, so a first-order field is convolved once and its
    # spectrum taken once, then reused by every child.
    for (build, input) in (
            (() -> ScatteringTransforms.ScatteringCore.ScatteringTransform(64, 4; Q = 2,
                       max_order = 2, transform = SpectralBackends.FFTSpectralBackend()), randn(64)),
            (() -> ScatteringTransforms.ScatteringCore.ScatteringTransform((16, 16), 3; L = 4,
                       max_order = 2, transform = SpectralBackends.FFTSpectralBackend()),
             randn(16, 16)),
            (() -> ScatteringTransforms.ScatteringCore.ScatteringTransform((8, 8, 8), 2;
                       n_orient = 4, max_order = 2,
                       transform = SpectralBackends.FFTSpectralBackend()), randn(8, 8, 8)))
        st = build()
        nw = length(st.filter_bank.wavelets)
        npaths2 = length(ScatteringTransforms.PathGraph.order_range(st.tree, 2))
        nparents = count(g -> !isempty(g[2]), st.groups)
        counts = count_executions((s, x) -> s(x), st, input)

        # One forward for the input, one per wavelet that has children; one inverse per wavelet and
        # per order-2 path.
        Test.@test counts.forward == 1 + nparents
        Test.@test counts.inverse == nw + npaths2
    end
end

Test.@testset "Decimation reduces cascade work, not just its shape" begin
    # Every `oversampling` issues the same transforms, at smaller sizes, so each execution is
    # weighed by its butterfly count.
    dims = (64, 64)
    input = randn(dims)
    build(α) = ScatteringTransforms.ScatteringCore.ScatteringTransform(dims, 4; L = 4,
                   oversampling = α, transform = SpectralBackends.FFTSpectralBackend())
    ref = count_executions((s, x) -> s(x), build(4), input)
    Test.@test ref.work == ref.total * prod(dims) * 12      # 64² · log₂(64²), i.e. nothing decimated
    prev = ref.work
    for α in (2, 1, 0)
        counts = count_executions((s, x) -> s(x), build(α), input)
        Test.@test counts.total == ref.total                # same transforms ...
        Test.@test counts.work < prev                       # ... strictly cheaper at each step
        prev = counts.work
    end
    Test.@test prev < ref.work ÷ 5
end

Test.@testset "Workspace is O(field), not O(nw * field)" begin
    # Only one first-order field and one spectrum are live at a time, so a transform's scratch must
    # not grow with the number of wavelets.
    #
    # Arrays are counted once by identity: where a plan inverts in place, a resolution's inverse
    # destination is its own working array.
    function ws_bytes(s)
        seen = Base.IdSet{Any}()
        for a in (s.buffer_input, s.buffer_signal_fft)
            push!(seen, a)
        end
        for d in (s.pw.work, s.pw.out, s.pw.spec), a in values(d)
            push!(seen, a)
        end
        return sum(sizeof, seen)
    end
    small = ScatteringTransforms.ScatteringCore.ScatteringTransform((32, 32), 2; L = 2,
                transform = SpectralBackends.FFTSpectralBackend())
    big = ScatteringTransforms.ScatteringCore.ScatteringTransform((32, 32), 4; L = 8,
              transform = SpectralBackends.FFTSpectralBackend())
    Test.@test length(big.filter_bank.wavelets) > 7 * length(small.filter_bank.wavelets)
    Test.@test ws_bytes(small) == ws_bytes(big)

    # Decimation adds one buffer set per live resolution, but those shrink geometrically, so total
    # scratch stays a small multiple of one field however deep the cascade goes.
    field = 2 * sizeof(ComplexF64) * 32 * 32           # buffer_input + buffer_signal_fft
    deep = ScatteringTransforms.ScatteringCore.ScatteringTransform((32, 32), 4; L = 8,
               oversampling = 0, transform = SpectralBackends.FFTSpectralBackend())
    Test.@test ws_bytes(deep) < 3 * field
end

Test.@testset "task_workspace shares read-only state, copies only scratch" begin
    # The filter bank dominates a transform's memory, and every task shares it.
    st = ScatteringTransforms.ScatteringCore.ScatteringTransform((32, 32), 3; L = 4,
             transform = SpectralBackends.FFTSpectralBackend())
    ws = ScatteringTransforms.ScatteringCore.task_workspace(st)
    Test.@test ws.filter_bank === st.filter_bank
    Test.@test ws.tree === st.tree
    Test.@test ws.groups === st.groups
    Test.@test ws.buffer_input !== st.buffer_input
    Test.@test ws.buffer_signal_fft !== st.buffer_signal_fft
    Test.@test ws.plan === first(ws.pw.plans)

    # The cascade workspace holds the mutable state: buffers per task, filters shared, at every
    # resolution.
    Test.@test ws.pw !== st.pw
    Test.@test ws.pw.filters === st.pw.filters
    Test.@test keys(ws.pw.work) == keys(st.pw.work)
    Test.@test keys(ws.pw.spec) == keys(st.pw.spec)
    for r in keys(st.pw.work)
        Test.@test ws.pw.work[r] !== st.pw.work[r]
    end
    # Aliasing survives the copy: an out that was its own work stays so, one that was not stays
    # distinct.
    for r in keys(st.pw.out)
        Test.@test (ws.pw.out[r] === ws.pw.work[r]) == (st.pw.out[r] === st.pw.work[r])
    end
end

Test.@testset "Every backend returns the serial result" begin
    # Each task owns disjoint outputs, so there is no reduction to reorder and threaded results are
    # identical to serial.
    for (st, X) in (
            (ScatteringTransforms.ScatteringCore.ScatteringTransform(64, 3; Q = 1,
                 transform = SpectralBackends.FFTSpectralBackend()), randn(64, 5)),
            (ScatteringTransforms.ScatteringCore.ScatteringTransform((16, 16), 2; L = 4,
                 transform = SpectralBackends.FFTSpectralBackend()), randn(16, 16, 4)),
            (ScatteringTransforms.ScatteringCore.ScatteringTransform((8, 8, 8), 2; n_orient = 4,
                 transform = SpectralBackends.FFTSpectralBackend()), randn(8, 8, 8, 3)),
            (ScatteringTransforms.Monogenic.MonogenicScatteringTransform((16, 16), 2;
                 transform = SpectralBackends.FFTSpectralBackend()), randn(16, 16, 3)))
        ref = ScatteringTransforms.scattering_batch(st, X)
        Test.@test ScatteringTransforms.scattering_batch(st, X; execution = ComputationalBackends.SerialBackend()) == ref
        Test.@test ScatteringTransforms.scattering_batch(st, X; execution = ComputationalBackends.ThreadedBackend()) == ref
        Test.@test ScatteringTransforms.scattering_batch(st, X; execution = ComputationalBackends.AutoBackend()) == ref
        out = similar(ref)
        Test.@test ScatteringTransforms.scattering_batch!(out, st, X;
                                                          execution = ComputationalBackends.ThreadedBackend()) == ref
    end
end

Test.@testset "Unavailable backends refuse rather than downgrade" begin
    # A request is honoured exactly or it errors.
    Test.@test_throws ArgumentError ScatteringTransforms.Execution.check_available(
        ComputationalBackends.AutoBackend())
end

Test.@testset "Plan-owning structs print one line" begin
    # The default `show` of a struct holding an FFTW plan can reach `fftw_sprint_plan`, which can
    # segfault. Every plan-owning struct prints a one-line summary.
    for spec in (SpectralBackends.DirectSumSpectralBackend(), SpectralBackends.FFTSpectralBackend())
        sts = (ScatteringTransforms.ScatteringCore.ScatteringTransform(32, 3; Q = 1, transform = spec),
               ScatteringTransforms.ScatteringCore.ScatteringTransform((16, 16), 2; L = 4, transform = spec),
               ScatteringTransforms.ScatteringCore.ScatteringTransform((8, 8, 8), 2; n_orient = 4,
                                                                      transform = spec),
               ScatteringTransforms.Monogenic.MonogenicScatteringTransform((16, 16), 2; transform = spec))
        for st in sts
            for x in (st, st.plan)
                s = sprint(show, x)
                Test.@test !occursin('\n', s)
                Test.@test !isempty(s)
            end
            pw = st isa ScatteringTransforms.Monogenic.MonogenicScatteringTransform ? st.mw.pw : st.pw
            Test.@test !occursin('\n', sprint(show, pw))
        end
        st2 = ScatteringTransforms.ScatteringCore.ScatteringTransform((16, 16), 2; L = 4,
                                                                      transform = spec)
        Test.@test !occursin('\n', sprint(show, ScatteringTransforms.batch_workspace(st2, 2)))
        Test.@test !occursin('\n', sprint(show, ScatteringTransforms.Inverse.ReconstructionWorkspace(st2)))
    end
    # The scattered and spherical transforms hold library plans and point-sized buffers.
    M = 200
    sp = ScatteringTransforms.scattered_planar_scattering(rand(M), rand(M), (8, 8), 2; period = 1.0,
                                                          L = 4)
    θ = [acos(1 - 2 * (k - 0.5) / M) for k in 1:M]
    φ = [2π * mod(k * (sqrt(5) - 1) / 2, 1) for k in 1:M]
    for st in (sp, ScatteringTransforms.spherical_scattering(θ, φ, 4, 2),
               ScatteringTransforms.spherical_monogenic_scattering(θ, φ, 4, 2),
               ScatteringTransforms.structured_spherical_scattering(4, 2))
        Test.@test !occursin('\n', sprint(show, st))
        Test.@test !occursin('\n', sprint(show, st.plan))
    end
end

Test.@testset "A transform plans its full grid once" begin
    # The cascade's level 1 is the full grid, which the transform's own plan covers.
    FB = SpectralBackends.FFTSpectralBackend()
    for st in (ScatteringTransforms.ScatteringCore.ScatteringTransform(64, 3; Q = 2, transform = FB),
               ScatteringTransforms.ScatteringCore.ScatteringTransform((32, 32), 3; L = 4, transform = FB),
               ScatteringTransforms.ScatteringCore.ScatteringTransform((8, 8, 8), 2; n_orient = 4,
                                                                      transform = FB))
        Test.@test st.plan === first(st.pw.plans)
    end
    mono = ScatteringTransforms.Monogenic.MonogenicScatteringTransform((32, 32), 3; transform = FB)
    Test.@test mono.plan === first(mono.mw.pw.plans)
end

# Fixed arity, with the function passed as a value: a captured module local or a splatted `args...`
# allocates inside the harness.
_alloc3(f::F, a, b, c) where {F} = (f(a, b, c); @allocated f(a, b, c))

Test.@testset "Periodized cascade converges to the undecimated one (1D/2D/3D)" begin
    # Producing each convolution on its own decimated grid is exact in the frequency domain; `U₁`'s
    # spectrum is then the aliased spectrum of the full `U₁`, and `oversampling` reduces that. The
    # error falls monotonically and vanishes once no scale is decimated. The coarse filters are
    # periodizations of the full-grid ones.
    Random.seed!(21)
    FB = SpectralBackends.FFTSpectralBackend()

    s2err(cs, ce) = (idx = findall(!iszero, ce.S2);
                     isempty(idx) ? 0.0 :
                     sum(abs, cs.S2[idx] .- ce.S2[idx]) / sum(abs, ce.S2[idx]))

    for (mk, x, ov_exact, run!) in (
            (ov -> ScatteringTransforms.ScatteringCore.ScatteringTransform(512, 5; Q = 1,
                       oversampling = ov, transform = FB),
             randn(512), 5, ScatteringTransforms.Scattering1D.scattering_transform!),
            (ov -> ScatteringTransforms.ScatteringCore.ScatteringTransform((64, 64), 3; L = 4,
                       oversampling = ov, transform = FB),
             randn(64, 64), 3, ScatteringTransforms.Scattering2D.scattering_transform2d!),
            (ov -> ScatteringTransforms.ScatteringCore.ScatteringTransform((16, 16, 16), 2;
                       n_orient = 4, oversampling = ov, transform = FB),
             randn(16, 16, 16), 2, ScatteringTransforms.Scattering3D.scattering_transform3d!))
        ce = mk(ov_exact)(x)
        Test.@test s2err(mk(ov_exact + 2)(x), ce) == 0         # nothing decimated: bit-identical
        Test.@test s2err(mk(1)(x), ce) <= s2err(mk(0)(x), ce) + 1e-12

        # The `r > 1` path builds an index range and a view per alias block, off the heap.
        for ov in (ov_exact, 1, 0)
            st = mk(ov)
            coeffs = ScatteringTransforms.batch_coeffs(st, Float64)
            Test.@test _alloc3(run!, coeffs, st, x) == 0
        end
    end
end

Test.@testset "Every surface has a batch entry point, matching its per-field result" begin
    # A batch reuses one plan, one filter bank and one workspace, and returns what transforming each
    # field alone returns.
    Random.seed!(20)
    FB = SpectralBackends.FFTSpectralBackend()

    # Decimating: the batched cascade carries one divisor per resolution.
    sub = ScatteringTransforms.ScatteringCore.ScatteringTransform(256, 4; Q = 1, max_order = 2,
              oversampling = 1, transform = FB)
    Xs = randn(256, 5)
    outs = ScatteringTransforms.scattering_batch(sub, Xs)
    for b in 1:5
        Test.@test view(outs, :, b) == ScatteringTransforms.Coefficients.flatten1d(sub(Xs[:, b]))
    end

    M = 200
    ppx, ppy = rand(M), rand(M)
    sp = ScatteringTransforms.scattered_planar_scattering(ppx, ppy, (16, 16), 3;
                                                          period = 1.0, L = 4, max_order = 2)
    Xp = randn(M, 4)
    outp = ScatteringTransforms.scattering_batch(sp, Xp)
    for b in 1:4
        Test.@test view(outp, :, b) ≈ ScatteringTransforms.Coefficients.flatten2d(sp(Xp[:, b]))
    end

    lmax, J, B = 8, 3, 4
    ss = ScatteringTransforms.structured_spherical_scattering(lmax, J)
    Θ, Φ = ScatteringTransforms.SphericalCore.structured_grid(lmax, Float64)
    Xg = randn(length(Θ), length(Φ), B)
    outg = ScatteringTransforms.scattering_batch(ss, Xg)
    for b in 1:B
        Test.@test view(outg, :, b) ≈ ScatteringTransforms.Coefficients.flatten1d(ss(Xg[:, :, b]))
    end

    # Analysis is an iterative solve; at `rtol = 1e-12` a batched stack and the fields solved one at a
    # time agree to round-off.
    sc = ScatteringTransforms.spherical_scattering(acos.(2 .* rand(150) .- 1), 2π .* rand(150),
                                                   lmax, J; rtol = 1.0e-12, maxiter = 2000)
    Xc = randn(150, B)
    outc = ScatteringTransforms.scattering_batch(sc, Xc)
    for b in 1:B
        Test.@test view(outc, :, b) ≈ ScatteringTransforms.Coefficients.flatten1d(sc(Xc[:, b])) rtol=1e-6
    end

    # A task gets its own workspace and its own copy of the plan's analysis scratch. The subsampled
    # path is FFT-only and agrees bit-for-bit. The nonuniform and spherical paths go through BLAS,
    # whose `gemm` reduction order depends on its thread count, and the threaded backend pins BLAS to
    # one thread, so those agree to round-off. Each serial baseline is computed in the same
    # expression as its threaded call.
    TB = ComputationalBackends.ThreadedBackend()
    Test.@test ScatteringTransforms.scattering_batch(sub, Xs; execution = TB) == outs
    Test.@test ScatteringTransforms.scattering_batch(sp, Xp; execution = TB) ≈
               ScatteringTransforms.scattering_batch(sp, Xp) rtol=1e-12
    Test.@test ScatteringTransforms.scattering_batch(ss, Xg; execution = TB) ≈
               ScatteringTransforms.scattering_batch(ss, Xg) rtol=1e-12
    Test.@test ScatteringTransforms.scattering_batch(sc, Xc; execution = TB) ≈
               ScatteringTransforms.scattering_batch(sc, Xc) rtol=1e-12
end
