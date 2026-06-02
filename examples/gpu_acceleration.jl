"""
    gpu_acceleration.jl

Demonstrates GPU acceleration with CUDA arrays.
Requires: CUDA.jl and a CUDA-capable GPU.

Run with: julia --project=. gpu_acceleration.jl
"""

using ScatteringTransforms

println("="^60)
println("GPU Acceleration Demo (CUDA)")
println("="^60)

# Check if CUDA is available
try
    using CUDA
    println("\nCUDA is available!")
    println("  Device: $(CUDA.name(CUDA.device()))")
catch e
    println("\nCUDA not available. Install with: using Pkg; Pkg.add(\"CUDA\")")
    println("Error: $e")
    exit(0)
end

# Configuration
N = 4096
J = 6

println("\nConfiguration:")
println("  Signal length: $N")
println("  Number of scales: $J")

# ============================================================================
# CPU Version (Reference)
# ============================================================================
println("\n" * "-"^60)
println("CPU Version (Reference)")
println("-"^60)

signal_cpu = randn(Float32, N)  # Float32 for GPU efficiency
st_cpu = ScatteringTransform1D(N, J; Q=1, max_order=2, T=Float32)

# Warmup
_ = st_cpu(signal_cpu)

# Benchmark
start_time = time()
for i in 1:10
    coeffs_cpu = st_cpu(signal_cpu)
end
elapsed_cpu = (time() - start_time) / 10

println("  Average time: $(round(elapsed_cpu * 1000, digits=2)) ms")
println("  S1 type: $(typeof(coeffs_cpu.S1))")

# ============================================================================
# GPU Version
# ============================================================================
println("\n" * "-"^60)
println("GPU Version (CUDA)")
println("-"^60)

# Transfer to GPU
signal_gpu = CuVector{Float32}(signal_cpu)

# Note: For full GPU support, the filter bank and buffers would also need
# to be on the GPU. Currently the package is GPU-ready in terms of type
# parameters but requires explicit CuArray creation.

println("  Signal on GPU: $(typeof(signal_gpu))")
println("  Device memory: $(CUDA.memory_status())")

# For a complete GPU example, the transform would need:
# - Filter bank on GPU (CuVector/CuMatrix for filters)
# - FFT plans for GPU (CUDA.CUFFT)
# - Workspace buffers on GPU

println("\n  ⚠️  Full GPU implementation requires:")
println("     - CuVector/CuMatrix for filter bank")
println("     - CUDA.CUFFT for FFT plans")
println("     - GPU workspace buffers")
println("\n  The package type system is GPU-ready!")
println("  Parametric types: {T,V,M} work with CuArray")

# ============================================================================
# Type Verification
# ============================================================================
println("\n" * "-"^60)
println("Type System Verification")
println("-"^60)

# Create CuArray-based coefficients (conceptual)
num_w = length(st_cpu.filter_bank.wavelets)
S1_cu = CuVector{Float32}(undef, num_w)
S2_cu = CuMatrix{Float32}(undef, num_w, num_w)

# This would work if we had GPU FFT plans
coeffs_cu = ScatteringCoefficients1D(S1_cu, S2_cu; S0=Float32(0))

println("  CuVector S1: $(typeof(coeffs_cu.S1))")
println("  CuMatrix S2: $(typeof(coeffs_cu.S2))")
println("  ✓ Parametric types accept CuArrays")

println("\n" * "="^60)
println("GPU Status: Type system ready, needs GPU FFT backend")
println("="^60)
