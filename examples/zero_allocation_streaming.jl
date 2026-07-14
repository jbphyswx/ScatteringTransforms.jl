"""
    zero_allocation_streaming.jl

Process many signals with minimal allocation — the pattern for TB-scale datasets.
Run with: `julia --project=. zero_allocation_streaming.jl`
"""

using ScatteringTransforms: ScatteringTransforms as ST
using FFTW: FFTW                       # O(N log N) fast path (without this it falls back to the slow direct sum)
using Statistics: Statistics
using Test: Test

println("="^64)
println("Zero-allocation streaming")
println("="^64)

N, J, nsig = 1024, 6, 200
st = ST.Scattering1D.ScatteringTransform1D(N, J; Q=1, max_order=2)
nw = length(st.filter_bank.wavelets)
signals = [randn(N) for _ in 1:nsig]

# ── naive: `st(x)` allocates fresh coefficient storage every call ─────────────
naive(sigs) = [ST.Coefficients.flatten1d(st(x)) for x in sigs]

# ── streaming: pre-allocate ONE coefficient buffer (mutable S0 container) and
#    reuse it via the in-place `scattering_transform!` — steady-state ~0 alloc ──
function streaming(sigs)
    coeffs = ST.Coefficients.ScatteringCoefficients1D(Vector{Float64}(undef, nw), zeros(nw, nw); S0=[0.0])
    out = Vector{Vector{Float64}}(undef, length(sigs))
    for (i, x) in enumerate(sigs)
        ST.Scattering1D.scattering_transform!(coeffs, st, x)
        out[i] = ST.Coefficients.flatten1d(coeffs)
    end
    return out
end

# measure steady-state allocation of one in-place transform (no I/O in the timed call)
coeffs = ST.Coefficients.ScatteringCoefficients1D(Vector{Float64}(undef, nw), zeros(nw, nw); S0=[0.0])
ST.Scattering1D.scattering_transform!(coeffs, st, signals[1])                       # warm up
bytes = @allocated ST.Scattering1D.scattering_transform!(coeffs, st, signals[1])
println("\nin-place scattering_transform! steady-state allocation: ", bytes, " bytes")
println("(buffers + plan reused; only a mutable S0 container is updated)")

# ── batched: one plan + workspace reused across the whole stack ───────────────
X = reduce(hcat, signals)
B = ST.scattering_batch(st, X)
println("\nscattering_batch(st, $(size(X))) → ", size(B), "  (flattened coeffs × signals)")

# correctness: all three agree
rn, rs = naive(signals), streaming(signals)
Test.@testset "streaming consistency" begin
    Test.@test all(rn[i] ≈ rs[i] for i in eachindex(signals))
    Test.@test all(B[:, i] ≈ rn[i] for i in eachindex(signals))
    Test.@test bytes < 4096          # essentially zero (no per-signal workspace allocation)
end

println("\n", "="^64, "\nDone.\n", "="^64)
