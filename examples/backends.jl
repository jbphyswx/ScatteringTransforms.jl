"""
    backends.jl

Spectral backends (in-core direct sum vs FFTW fast path) and compute backends
(serial / threaded / GPU). Run with: `julia --project=. backends.jl`
(Use `julia -t4` to see threading.)
"""

using ScatteringTransforms: ScatteringTransforms as ST
using FFTW: FFTW                 # enables the FFTW fast path
using OhMyThreads: OhMyThreads    # enables ThreadedBackend
using Statistics: Statistics
using Test: Test

println("="^64)
println("Backends")
println("="^64)

N, J = 2048, 7
x = randn(N)

# ── spectral backend: in-core direct sum (default-available) vs FFTW fast path ─
st_direct = ST.ScatteringTransform1D(N, J; Q=1, max_order=2, spectral=:direct)
st_fftw   = ST.ScatteringTransform1D(N, J; Q=1, max_order=2, spectral=:fftw)
cd, cf = st_direct(x), st_fftw(x)
println("\nspectral backends agree: ",
        ST.first_order(cd) ≈ ST.first_order(cf) && ST.second_order(cd) ≈ ST.second_order(cf))
st_direct(x); st_fftw(x)        # warm up
td = minimum(@elapsed(st_direct(x)) for _ in 1:3)
tf = minimum(@elapsed(st_fftw(x)) for _ in 1:3)
println("direct sum: ", round(td * 1e3, digits=2), " ms   FFTW: ", round(tf * 1e3, digits=2),
        " ms   (", round(td / tf, digits=1), "× faster)")

# ── compute backends over a batch ─────────────────────────────────────────────
st = ST.ScatteringTransform1D(N, J; Q=1, max_order=2)   # spectral=:auto → FFTW (loaded)
X = randn(N, 64)
serial   = ST.scattering_batch(st, X)
threaded = ST.scattering_batch(ST.ThreadedBackend(), st, X)   # OhMyThreads, $(Threads.nthreads()) threads
println("\nthreaded batch == serial batch: ", threaded ≈ serial,
        "   (", Threads.nthreads(), " threads)")

# ── GPU: only if a CUDA device is available; skip gracefully otherwise ────────
println()
gpu_ok = false
try
    @eval using CUDA
    gpu_ok = CUDA.functional()
catch
    gpu_ok = false
end
if gpu_ok
    st_gpu = ST.ScatteringTransform1D(N, J; T=Float32, device=CUDA.device())
    cg = st_gpu(CUDA.CuVector{Float32}(Float32.(x)))
    println("GPU transform ran; S1 length = ", length(ST.first_order(cg)))
else
    println("CUDA not available — skipping GPU demo (the same `st(x)` runs on CuArray buffers ",
            "when a CUDA device is present; load `using CUDA` and build with `device=CUDA.device()`).")
end

Test.@testset "backends" begin
    Test.@test ST.first_order(cd) ≈ ST.first_order(cf)
    Test.@test ST.second_order(cd) ≈ ST.second_order(cf)
    Test.@test threaded ≈ serial
end

println("\n", "="^64, "\nDone.\n", "="^64)
