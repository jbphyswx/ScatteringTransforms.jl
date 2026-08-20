# Structural gates: properties that are cheap to check exactly and expensive to lose silently.
#
# These are counting and identity assertions, never wall-clock ones — timing in a test suite is GC-
# and load-flaky, while an operation count is deterministic and instant. Each one pins a specific
# regression that would otherwise only show up as "it got slower".

include("counting_plan.jl")

Test.@testset "Cascade issues each first-order transform exactly once" begin
    # The cascade walks `(j1, children)` groups, so a first-order field is convolved once and its
    # spectrum taken once, then reused by every child. Computing S1 and S2 in separate passes would
    # convolve every wavelet twice — `nw` wasted inverse transforms per field.
    for (build, input) in (
            (() -> ScatteringTransforms.Scattering1D.ScatteringTransform1D(64, 4; Q = 2,
                       max_order = 2, spectral = SpectralBackends.FFTSpectralBackend()), randn(64)),
            (() -> ScatteringTransforms.Scattering2D.ScatteringTransform2D((16, 16), 3; L = 4,
                       max_order = 2, spectral = SpectralBackends.FFTSpectralBackend()),
             randn(16, 16)),
            (() -> ScatteringTransforms.Scattering3D.ScatteringTransform3D((8, 8, 8), 2;
                       n_orient = 4, max_order = 2,
                       spectral = SpectralBackends.FFTSpectralBackend()), randn(8, 8, 8)))
        st = build()
        nw = length(st.filter_bank.wavelets)
        npaths2 = length(ScatteringTransforms.PathGraph.order_range(st.tree, 2))
        nparents = count(g -> !isempty(g[2]), st.groups)
        counts = count_executions((s, x) -> s(x), st, input)

        # One forward for the input, one per wavelet that has children; one inverse per wavelet and
        # per order-2 path. Anything more means a pass was duplicated.
        Test.@test counts.forward == 1 + nparents
        Test.@test counts.inverse == nw + npaths2
        Test.@test counts.total == 1 + nparents + nw + npaths2
        Test.@test counts.total < 3nw + npaths2 + 1     # the two-pass form's count
    end
end

Test.@testset "Workspace is O(field), not O(nw * field)" begin
    # Only one first-order field and one spectrum are live at a time, so a transform's scratch must
    # not grow with the number of wavelets. It used to hold `nw` of each.
    ws_bytes(s) = sum(sizeof, (s.buffer_input, s.buffer_signal_fft, s.buffer_conv,
                               s.buffer_mod, s.buffer_u1, s.buffer_u1_fft))
    small = ScatteringTransforms.Scattering2D.ScatteringTransform2D((32, 32), 2; L = 2,
                spectral = SpectralBackends.FFTSpectralBackend())
    big = ScatteringTransforms.Scattering2D.ScatteringTransform2D((32, 32), 4; L = 8,
              spectral = SpectralBackends.FFTSpectralBackend())
    Test.@test length(big.filter_bank.wavelets) > 7 * length(small.filter_bank.wavelets)
    Test.@test ws_bytes(small) == ws_bytes(big)
end

Test.@testset "task_workspace shares read-only state, copies only scratch" begin
    # This is what makes threading cheap: the filter bank dominates a transform's memory, and every
    # task must share it rather than take a copy.
    st = ScatteringTransforms.Scattering2D.ScatteringTransform2D((32, 32), 3; L = 4,
             spectral = SpectralBackends.FFTSpectralBackend())
    ws = ScatteringTransforms.ScatteringCore.task_workspace(st)
    Test.@test ws.filter_bank === st.filter_bank
    Test.@test ws.tree === st.tree
    Test.@test ws.groups === st.groups
    Test.@test ws.buffer_conv !== st.buffer_conv
    Test.@test ws.buffer_u1 !== st.buffer_u1
    Test.@test ws.buffer_signal_fft !== st.buffer_signal_fft
end

Test.@testset "Every backend returns the serial result" begin
    # A backend must be honoured exactly, so its output is compared against serial rather than
    # merely checked for being finite. Threaded is required to be *identical*, not approximate:
    # each task owns disjoint outputs, so there is no reduction to reorder.
    for (st, X) in (
            (ScatteringTransforms.Scattering1D.ScatteringTransform1D(64, 3; Q = 1,
                 spectral = SpectralBackends.FFTSpectralBackend()), randn(64, 5)),
            (ScatteringTransforms.Scattering2D.ScatteringTransform2D((16, 16), 2; L = 4,
                 spectral = SpectralBackends.FFTSpectralBackend()), randn(16, 16, 4)),
            (ScatteringTransforms.Scattering3D.ScatteringTransform3D((8, 8, 8), 2; n_orient = 4,
                 spectral = SpectralBackends.FFTSpectralBackend()), randn(8, 8, 8, 3)))
        ref = ScatteringTransforms.scattering_batch(st, X)
        Test.@test ScatteringTransforms.scattering_batch(ComputationalBackends.SerialBackend(), st, X) == ref
        Test.@test ScatteringTransforms.scattering_batch(ComputationalBackends.ThreadedBackend(), st, X) == ref
        Test.@test ScatteringTransforms.scattering_batch(ComputationalBackends.AutoBackend(), st, X) == ref
        out = similar(ref)
        Test.@test ScatteringTransforms.scattering_batch!(out, ComputationalBackends.ThreadedBackend(), st, X) == ref
    end
end

Test.@testset "Unavailable backends refuse rather than downgrade" begin
    # A request is honoured exactly or it errors; nothing silently falls back to a slower path.
    Test.@test_throws ArgumentError ScatteringTransforms.Execution.check_available(
        ComputationalBackends.AutoBackend())
    # AutoBackend resolves on real capability, and only to something that can actually run.
    Test.@test ScatteringTransforms.Execution.resolve_backend(ComputationalBackends.AutoBackend()) isa
               ComputationalBackends.AbstractLocalBackend
    Test.@test ScatteringTransforms.Execution.resolve_backend(ComputationalBackends.SerialBackend()) ===
               ComputationalBackends.SerialBackend()
end

Test.@testset "Plan-owning structs print one line" begin
    # The default `show` of a struct holding an FFTW plan can reach `fftw_sprint_plan`, which can
    # segfault; a device plan would print its whole buffer. Every such struct prints a summary.
    for spec in (SpectralBackends.DirectSumSpectralBackend(), SpectralBackends.FFTSpectralBackend())
        st = ScatteringTransforms.Scattering1D.ScatteringTransform1D(32, 3; Q = 1, spectral = spec)
        s = sprint(show, st.plan)
        Test.@test !occursin('\n', s)
        Test.@test !isempty(s)
    end
end

# Fixed arity, and the function comes in as a value: a captured module local or a splatted `args...`
# both cost bytes that `@allocated` would attribute to the callee.
_alloc3(f::F, a, b, c) where {F} = (f(a, b, c); @allocated f(a, b, c))

Test.@testset "Multi-resolution second order converges to the exact transform (1D/2D/3D)" begin
    # Decimating `U₁` before the second wavelet transform is exact in the frequency domain; what is
    # approximate is assuming `U₁` carries no energy above the reduced Nyquist, and `oversampling`
    # buys that down. So the error must fall monotonically, and vanish once no scale is decimated.
    #
    # This also pins the reduced-resolution banks to the *normalised* builder: bare frequency
    # responses omit the tight-frame factor (`0.90` in 1D, `0.38` in 2D), which would rescale every
    # decimated second-order coefficient and show up here as a floor the error never drops below.
    Random.seed!(21)
    FB = SpectralBackends.FFTSpectralBackend()
    MR = ScatteringTransforms.SubsampledScattering

    s2err(cs, ce) = (idx = findall(!iszero, ce.S2);
                     isempty(idx) ? 0.0 :
                     sum(abs, cs.S2[idx] .- ce.S2[idx]) / sum(abs, ce.S2[idx]))

    for (mk, exact, x, ov_exact) in (
            (ov -> MR.SubsampledScattering1D(512, 5; Q = 1, oversampling = ov, spectral = FB),
             ScatteringTransforms.Scattering1D.ScatteringTransform1D(512, 5; Q = 1, spectral = FB),
             randn(512), 5),
            (ov -> MR.SubsampledScattering2D((64, 64), 3; L = 4, oversampling = ov, spectral = FB),
             ScatteringTransforms.Scattering2D.ScatteringTransform2D((64, 64), 3; L = 4, spectral = FB),
             randn(64, 64), 3),
            (ov -> MR.SubsampledScattering3D((16, 16, 16), 2; n_orient = 4, oversampling = ov, spectral = FB),
             ScatteringTransforms.Scattering3D.ScatteringTransform3D((16, 16, 16), 2; n_orient = 4, spectral = FB),
             randn(16, 16, 16), 2))
        ce = exact(x)
        Test.@test s2err(mk(ov_exact)(x), ce) < 1e-12          # nothing decimated: exact
        Test.@test s2err(mk(1)(x), ce) <= s2err(mk(0)(x), ce) + 1e-12
        Test.@test isapprox(mk(0)(x).S1, ce.S1; rtol = 1e-10)  # first order is never decimated

        # One shared cascade, so the `!` form is allocation-free at every dimension.
        st = mk(1)
        coeffs = ScatteringTransforms.batch_coeffs(st, Float64)
        Test.@test _alloc3(MR.subsampled_scattering!, coeffs, st, x) == 0
    end
end

# The flat column of a spherical result. Pairs are `(j1, j2 < j1)`, and the flat layout stores one
# row per unordered pair, so the transposed index is the same slot.
function _spherical_ref_column(r, J)
    col = zeros(Float64, ScatteringTransforms.Coefficients.flat_length(J))
    col[ScatteringTransforms.Coefficients.flat_row_s0()] = r.S0
    for j in 1:J
        col[ScatteringTransforms.Coefficients.flat_row_s1(j, J)] = r.S1[j]
    end
    for j1 in 1:J, j2 in 1:(j1 - 1)
        col[ScatteringTransforms.Coefficients.flat_row_s2(j2, j1, J)] =
            isempty(r.S2) ? 0.0 : r.S2[j1, j2]
    end
    return col
end

Test.@testset "Every surface has a batch entry point, matching its per-field result" begin
    # A batch reuses one plan, one filter bank and one workspace, so it must return exactly what
    # transforming each field alone returns. On the nonuniform and spherical surfaces the plan is
    # most of the construction cost, which is what makes the reuse worth having.
    Random.seed!(20)
    FB = SpectralBackends.FFTSpectralBackend()

    sub = ScatteringTransforms.SubsampledScattering.SubsampledScattering1D(256, 4; Q = 1,
              max_order = 2, spectral = FB)
    Xs = randn(256, 5)
    outs = ScatteringTransforms.scattering_batch(sub, Xs)
    for b in 1:5
        Test.@test view(outs, :, b) ≈ ScatteringTransforms.Coefficients.flatten1d(sub(Xs[:, b]))
    end

    M = 200
    ppx, ppy = rand(M), rand(M)
    sp = ScatteringTransforms.scattered_planar_scattering(ppx, ppy, (16, 16), 3;
                                                          L = 4, max_order = 2)
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
        Test.@test view(outg, :, b) ≈ _spherical_ref_column(ss(Xg[:, :, b]), J)
    end

    # `rtol` is tightened well past the backend default here: analysis is an iterative solve, so at
    # the default tolerance a batched stack and the same fields solved one at a time are both merely
    # "converged", and differ at that tolerance rather than at round-off. Converging hard is what
    # makes this a parity assertion about the cascade instead of a restatement of the solver's
    # stopping rule.
    sc = ScatteringTransforms.spherical_scattering(acos.(2 .* rand(150) .- 1), 2π .* rand(150),
                                                   lmax, J; rtol = 1.0e-12, maxiter = 2000)
    Xc = randn(150, B)
    outc = ScatteringTransforms.scattering_batch(sc, Xc)
    for b in 1:B
        Test.@test view(outc, :, b) ≈ _spherical_ref_column(sc(Xc[:, b]), J) rtol=1e-6
    end

    # A task gets its own workspace *and* its own copy of the plan's analysis scratch, so nothing
    # written to is shared. The subsampled path is FFT-only and must therefore agree bit-for-bit.
    #
    # The nonuniform and spherical paths go through BLAS, and the threaded backend pins BLAS to one
    # thread while the serial one leaves the user's setting alone (it is worth up to 1.3x there). A
    # `gemm` reduces in an order that depends on its thread count, so these agree to round-off rather
    # than exactly — a property of BLAS, not of the cascade.
    if Base.get_extension(ScatteringTransforms, :ScatteringTransformsOhMyThreadsExt) !== nothing
        TB = ComputationalBackends.ThreadedBackend()
        #
        # Each serial baseline is taken in the same expression as the threaded call it is compared
        # against, rather than reused from earlier in the testset, so the assertion cannot depend on
        # anything established in between.
        Test.@test ScatteringTransforms.scattering_batch(TB, sub, Xs) == outs
        Test.@test ScatteringTransforms.scattering_batch(TB, sp, Xp) ≈
                   ScatteringTransforms.scattering_batch(sp, Xp) rtol=1e-12
        Test.@test ScatteringTransforms.scattering_batch(TB, ss, Xg) ≈
                   ScatteringTransforms.scattering_batch(ss, Xg) rtol=1e-12
        Test.@test ScatteringTransforms.scattering_batch(TB, sc, Xc) ≈
                   ScatteringTransforms.scattering_batch(sc, Xc) rtol=1e-12
    end
end
