"""
    backends.jl

Spectral backends (in-core direct sum vs FFTW fast path) and compute backends
(serial / threaded / GPU). Run with: `julia --project=. backends.jl`
(Use `julia -t4` to see threading.)
"""

using ScatteringTransforms: ScatteringTransforms as ST
using ComputationalBackends: ComputationalBackends as CB
using SpectralBackends: SpectralBackends as SB
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
st_direct = ST.Scattering1D.ScatteringTransform1D(N, J; Q=1, max_order=2, spectral=SB.DirectSumSpectralBackend())
st_fftw   = ST.Scattering1D.ScatteringTransform1D(N, J; Q=1, max_order=2, spectral=SB.FFTSpectralBackend())
cd, cf = st_direct(x), st_fftw(x)
println("\nspectral backends agree: ",
        ST.Coefficients.first_order(cd) ≈ ST.Coefficients.first_order(cf) && ST.Coefficients.second_order(cd) ≈ ST.Coefficients.second_order(cf))
st_direct(x); st_fftw(x)        # warm up
td = minimum(@elapsed(st_direct(x)) for _ in 1:3)
tf = minimum(@elapsed(st_fftw(x)) for _ in 1:3)
println("direct sum: ", round(td * 1e3, digits=2), " ms   FFTW: ", round(tf * 1e3, digits=2),
        " ms   (", round(td / tf, digits=1), "× faster)")

# ── compute backends over a batch ─────────────────────────────────────────────
st = ST.Scattering1D.ScatteringTransform1D(N, J; Q=1, max_order=2)   # AutoSpectralBackend → FFTW (loaded)
X = randn(N, 64)
serial   = ST.scattering_batch(st, X)
threaded = ST.scattering_batch(CB.ThreadedBackend(), st, X)   # OhMyThreads, $(Threads.nthreads()) threads
println("\nthreaded batch == serial batch: ", threaded ≈ serial,
        "   (", Threads.nthreads(), " threads)")

# ── GPU: vendor-neutral via KernelAbstractions; run on CUDA if available ──────
# The GPU path is device-agnostic: build the transform with `GPUBackend(<any KA backend>)`. The KA
# CPU backend always works (used here as a portable fallback / for CI parity); on a machine with a
# functional CUDA GPU it runs on the GPU via `GPUBackend(CUDA.CUDABackend())` with no code change.
println()
using KernelAbstractions: KernelAbstractions as KA
using AbstractFFTs: AbstractFFTs

# Pick a device backend + matching input array: CUDA GPU if functional, else the KA CPU backend.
gpu_backend, xdev = try
    @eval using CUDA
    CUDA.functional() ? (CB.GPUBackend(CUDA.CUDABackend()), CUDA.CuVector{Float32}(Float32.(x))) :
                        (CB.GPUBackend(KA.CPU()), Float32.(x))
catch
    (CB.GPUBackend(KA.CPU()), Float32.(x))
end
st_gpu = ST.Scattering1D.ScatteringTransform1D(Float32, N, J, gpu_backend;)
cg = st_gpu(xdev)
println("GPU (", typeof(gpu_backend.backend), ") transform ran; S1 length = ",
        length(ST.Coefficients.first_order(cg)))

Test.@testset "backends" begin
    Test.@test ST.Coefficients.first_order(cd) ≈ ST.Coefficients.first_order(cf)
    Test.@test ST.Coefficients.second_order(cd) ≈ ST.Coefficients.second_order(cf)
    Test.@test threaded ≈ serial
end

println("\n", "="^64, "\nDone.\n", "="^64)
