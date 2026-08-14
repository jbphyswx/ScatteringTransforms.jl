"""
    batch_topology.jl — how should a batch of slices be laid across threads?

A batch can be transformed anywhere on a spectrum:

  * **chunk = 1** — every task takes one slice and runs the whole cascade on it. Each task's working
    set is one slice plus its buffers; the cascade's `1 + nparents` forward and `nw + paths` inverse
    transforms then all reuse that resident data.
  * **chunk = B** — one batched plan transforms the whole stack per operation. Each individual
    operation is `B×` larger, so per-call overhead is amortised, but every operation sweeps the whole
    stack instead of reusing one slice.
  * **anything between** — the chunk sets the working-set size, which is the quantity that decides
    whether those `O(nw + paths)` operations hit cache or stream memory.

This sweeps that axis against a roofline, so the choice is measured rather than assumed. The
roofline is the same cascade's transform count times the cost of one bare FFT at that size: the time
the transform would take if everything except the FFTs were free.

Run:  julia --project=benchmark -t<threads> benchmark/batch_topology.jl
"""

using FFTW: FFTW
using OhMyThreads: OhMyThreads
using ComputationalBackends: ComputationalBackends as CB
using SpectralBackends: SpectralBackends as SB
using ScatteringTransforms: ScatteringTransforms as ST
using LinearAlgebra: LinearAlgebra
using Printf: Printf

bench(f, r = 3) = (f(); minimum(@elapsed(f()) for _ in 1:r))

# Transforms one application of the cascade must issue, from the tree itself.
function transform_count(st)
    nw = length(st.filter_bank.wavelets)
    npaths2 = length(ST.PathGraph.order_range(st.tree, 2))
    nparents = count(g -> !isempty(g[2]), st.groups)
    return (1 + nparents) + (nw + npaths2)
end

# Cost of one bare in-place FFT of the transform's slice size.
function fft_unit_time(::Type{T}, spatial) where {T}
    a = zeros(Complex{T}, spatial)
    b = similar(a)
    p = FFTW.plan_fft(a; flags = FFTW.MEASURE)
    return bench(() -> LinearAlgebra.mul!(b, p, a), 20)
end

# Cost of one *complete* cascade step at that size: the spectral multiply, the inverse transform,
# and the fused modulus+mean. This is the real floor — an FFT-only roofline is badly optimistic,
# because the multiply and the modulus move ~6N words against the transform's N log N of compute,
# and the cascade is memory-bandwidth bound rather than FFT bound.
function step_unit_time(::Type{T}, spatial) where {T}
    xf = randn(Complex{T}, spatial)
    psi = randn(Complex{T}, spatial)
    c1 = similar(xf)
    c2 = similar(xf)
    p = FFTW.plan_ifft(xf; flags = FFTW.MEASURE)
    step! = () -> begin
        @. c1 = xf * psi
        LinearAlgebra.mul!(c2, p, c1)
        s = zero(T)
        @inbounds @simd for i in eachindex(c2)
            s += abs(c2[i])
        end
        return s
    end
    return bench(step!, 20)
end

slice_bytes(::Type{T}, spatial) where {T} = prod(spatial) * 2 * sizeof(T)

function study(label, st, X, spatial)
    B = size(X)[end]
    T = real(eltype(st.filter_bank.averaging))
    ntr = transform_count(st)
    tfft = fft_unit_time(T, spatial)
    tstep = step_unit_time(T, spatial)
    # Floor: every cascade step done back to back with nothing else, spread over the threads.
    roof_serial = ntr * tstep * B
    roofline = roof_serial / Threads.nthreads()
    out = ST.scattering_batch(st, X)

    println("\n", label, "   B=$B  transforms/slice=$ntr  slice=",
            Base.format_bytes(slice_bytes(T, spatial)))
    println("    one FFT = ", round(tfft * 1e6; digits = 1),
            " us   one full step (multiply + ifft + modulus + mean) = ",
            round(tstep * 1e6; digits = 1), " us   -> the FFT is ",
            round(100 * tfft / tstep; digits = 0), "% of a step")
    println("    serial floor ", round(roof_serial * 1e3; digits = 1), " ms  /  ",
            Threads.nthreads(), " threads = ", round(roofline * 1e3; digits = 1), " ms")
    Printf.@printf("  %-28s %10s %9s %12s\n", "layout", "time", "vs roof", "working set")

    function row(name, t, wsbytes)
        Printf.@printf("  %-28s %8.1f ms %8.2fx %12s\n", name, t * 1e3, t / roofline,
                wsbytes === nothing ? "-" : Base.format_bytes(wsbytes))
    end
    row("floor (steps / threads)", roofline, nothing)
    row("serial, per-slice", bench(() -> ST.scattering_batch!(out, st, X), 2), slice_bytes(T, spatial) * 5)
    row("threaded, per-slice", bench(() -> ST.scattering_batch(CB.ThreadedBackend(), st, X), 2),
        slice_bytes(T, spatial) * 5)

    ext = Base.get_extension(ST, :ScatteringTransformsOhMyThreadsExt)
    for chunk in unique(clamp.([1, 2, 4, 8, 16, 32, B], 1, B))
        t = bench(() -> ext.chunked_batch!(out, st, X; chunk = chunk), 2)
        row("threaded, chunk=$chunk", t, slice_bytes(T, spatial) * 5 * chunk)
    end
    # One batched plan, letting the FFT library thread underneath instead of tasks above.
    wsN = ST.batch_workspace(st, B; fft_nthreads = Threads.nthreads())
    row("batched, FFT threads", bench(() -> ST.Batched.batch_cascade!(out, wsN, X), 2),
        slice_bytes(T, spatial) * 5 * B)
    return nothing
end

function main()
    println("threads=", Threads.nthreads(), "  FFTW threads available=", FFTW.get_num_threads())
    FB = SB.FFTSpectralBackend()
    for (label, st, X, spatial) in (
        ("1D N=1024 J=6 Q=1",
         ST.Scattering1D.ScatteringTransform1D(1024, 6; Q = 1, spectral = FB), randn(1024, 64), (1024,)),
        ("1D N=4096 J=8 Q=1",
         ST.Scattering1D.ScatteringTransform1D(4096, 8; Q = 1, spectral = FB), randn(4096, 64), (4096,)),
        ("2D 64^2 J=4 L=8",
         ST.Scattering2D.ScatteringTransform2D((64, 64), 4; L = 8, spectral = FB), randn(64, 64, 32), (64, 64)),
        ("2D 128^2 J=4 L=8",
         ST.Scattering2D.ScatteringTransform2D((128, 128), 4; L = 8, spectral = FB), randn(128, 128, 16), (128, 128)))
        study(label, st, X, spatial)
    end
    println("\n  'vs roof' is time / (transform count x one-full-step time x B / threads).")
    println("  1.0 would mean the cascade costs exactly its own arithmetic and memory traffic, with")
    println("  perfect thread scaling. The gap above 1.0 is scheduling and imperfect scaling, not")
    println("  the transform: on a bandwidth-bound kernel, threads beyond the physical core count")
    println("  contend for one memory controller, so the floor's /threads divisor is optimistic.")
    return nothing
end

main()