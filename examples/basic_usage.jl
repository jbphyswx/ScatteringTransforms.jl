"""
    basic_usage.jl

A guided tour of ScatteringTransforms.jl: averaged coefficients, the localized (Mallat) field,
reduced descriptors, batching, multithreading, the FFTW fast path, and 3D volumes.
Run with: `julia --project=. basic_usage.jl`
"""

using ScatteringTransforms: ScatteringTransforms as ST
using FFTW: FFTW                      # loading FFTW enables the O(N log N) fast path (spectral=:auto)
using OhMyThreads: OhMyThreads        # loading this enables ThreadedBackend batched transforms
using Statistics: Statistics
using Test: Test

println("="^64)
println("ScatteringTransforms.jl — guided tour")
println("="^64)

# ---------------------------------------------------------------------------
# 1. 1D averaged scattering coefficients
# ---------------------------------------------------------------------------
println("\n1. 1D averaged coefficients  S0=⟨x⟩, S1[λ]=⟨|x⋆ψ_λ|⟩, S2[λ1,λ2]=⟨||x⋆ψ_λ1|⋆ψ_λ2|⟩")
N = 1024
t = range(0, 2π, length = N)
x = sin.(10 .* t) .+ 0.5 .* sin.(50 .* t) .+ 0.1 .* randn(N)
st = ST.ScatteringTransform1D(N, 6; Q = 1, max_order = 2)   # spectral=:auto → FFTW (loaded above)
c = st(x)
println("   S0 = ", round(ST.zeroth_order(c), digits = 4),
        " | S1: ", length(ST.first_order(c)), " coeffs | S2: ", size(ST.second_order(c)))

# ---------------------------------------------------------------------------
# 2. Localized (Mallat) field — and its consistency with the averaged coefficients
# ---------------------------------------------------------------------------
println("\n2. Localized field  S_p x = (|U_p x| ⋆ φ_J) ↓ s   (mean over space == averaged coeff)")
sf = ST.scattering_field(st, x; subsample = 1)
root = first(ST.PathGraph.order_range(st.tree, 0))
println("   mean(field[root]) = ", round(Statistics.mean(ST.path_field(sf, root)), digits = 6),
        "   vs  S0 = ", round(ST.zeroth_order(c), digits = 6))

# ---------------------------------------------------------------------------
# 3. Reduced descriptors (amplitude-normalized, log)
# ---------------------------------------------------------------------------
println("\n3. Reduced descriptors")
nc = ST.normalized_coefficients(c)
println("   normalized s1 (S1/S0), first 3: ", round.(nc.s1[1:3], digits = 3))

# ---------------------------------------------------------------------------
# 4. Batched transform — one plan + workspace reused across many signals
# ---------------------------------------------------------------------------
println("\n4. Batched transform (plan + workspace reused)")
X = randn(N, 64)
coeffs_batch = ST.scattering_batch(st, X)
println("   scattering_batch(st, $(size(X))) → ", size(coeffs_batch), " (coeffs × batch)")
threaded = ST.scattering_batch(ST.ThreadedBackend(), st, X)
println("   threaded == serial: ", threaded ≈ coeffs_batch)

# ---------------------------------------------------------------------------
# 5. 2D oriented scattering + anisotropy / sparsity
# ---------------------------------------------------------------------------
println("\n5. 2D scattering + reduced sparsity/shape")
M = 128
xs = range(0, 8π, length = M)'
oriented = repeat(sin.(xs), M, 1)                  # horizontal stripes → anisotropic
st2 = ST.ScatteringTransform2D((M, M), 3; L = 8, max_order = 2)
c2 = st2(oriented)
red = ST.compute_shape_sparsity(ST.first_order(c2), ST.second_order(c2), st2.filter_bank.meta)
println("   max sparsity s21 = ", round(maximum(red.sparsity), digits = 4),
        " | max |shape s22| = ", round(maximum(abs, red.shape), digits = 4), " (≠0 ⇒ oriented)")

# ---------------------------------------------------------------------------
# 6. 3D volumetric scattering
# ---------------------------------------------------------------------------
println("\n6. 3D volumetric scattering")
st3 = ST.ScatteringTransform3D((16, 16, 16), 2; n_orient = 6, max_order = 2)
c3 = st3(randn(16, 16, 16))
println("   3D S1: ", length(ST.first_order(c3)), " coeffs (J × n_orient)")

# ---------------------------------------------------------------------------
# 7. Element types: Float32 end-to-end
# ---------------------------------------------------------------------------
stf = ST.ScatteringTransform1D(N, 6; Q = 1, max_order = 2, T = Float32)
cf = stf(Float32.(x))
println("\n7. Float32 in → ", eltype(ST.first_order(cf)), " out")

# light sanity checks so the example doubles as a smoke test
Test.@testset "basic_usage smoke" begin
    Test.@test ST.zeroth_order(c) ≈ Statistics.mean(x)
    Test.@test Statistics.mean(ST.path_field(sf, root)) ≈ ST.zeroth_order(c) atol = 1e-8
    Test.@test threaded ≈ coeffs_batch
    Test.@test eltype(ST.first_order(cf)) == Float32
end

println("\n", "="^64, "\nDone.\n", "="^64)
