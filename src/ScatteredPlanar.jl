module ScatteredPlanar

"""
    ScatteredPlanar.jl — scattered / nonuniform planar scattering

Scattering of a scalar field sampled at arbitrary points `(x, y)` on the plane. Analysis lifts the
samples onto a uniform Fourier **mode grid** of size `ms` (a nonuniform DFT), where the oriented-Morlet
wavelet bank lives; the wavelet multiply happens on that grid and synthesis evaluates the filtered
field back at the points. The modulus + a (quadrature-weighted) spatial mean give the S0/S1/S2
coefficients, mirroring the gridded `ScatteringTransform2D` cascade (same `build_filter_bank2d`, same
`PathGraph` admissible-path tree, same `wavelet_convolve!` / `apply_modulus!`).

The spectral plan is chosen by `spectral`: the in-core `Plans.DirectNUFFTBackend` (exact
direct-summation NUDFT, no dependencies) is the always-available default; `Plans.NUFFTBackend` (or
`Plans.AutoSpectral` once the FINUFFT extension is loaded) selects the FINUFFT fast path. Both satisfy
the same `AbstractScatteringPlan` interface, so the cascade is identical either way.
"""

using ..Plans: Plans
using ..FilterBanks: FilterBanks
using ..PathGraph: PathGraph
using ..Coefficients: Coefficients
using ..ScatteringCore: ScatteringCore
using LinearAlgebra: LinearAlgebra

# Standalone scattered cascade (point-buffers ≠ mode-buffers). All array/container fields are type
# parameters, matching `ScatteringTransform2D`'s style.
struct ScatteredPlanarScattering{T, FB, Tree, P, WV<:AbstractVector{T},
                                 CV<:AbstractVector{Complex{T}}, MM<:AbstractMatrix{Complex{T}},
                                 RV<:AbstractVector{T}, U1V<:AbstractVector, U1M<:AbstractVector}
    filter_bank::FB          # oriented Morlet bank on the `ms` fftfreq lattice
    tree::Tree               # admissible scattering paths
    max_order::Int
    plan::P                  # scattered spectral plan (DirectNUFFTPlan or FINUFFT NUFFTScatteringPlan)
    weights::WV              # (M) spatial-mean quadrature weights, summing to 1
    buf_input_pts::CV        # (M) complexified input / U1
    X_modes::MM              # (ms) signal mode coefficients
    buf_modes::MM            # (ms) wavelet-multiply scratch
    buf_conv_pts::CV         # (M) synthesis output at points
    buf_mod_pts::RV          # (M) modulus at points
    U1_pts::U1V              # per-wavelet first-order moduli (each (M))
    U1_modes::U1M            # per-wavelet U1 mode coefficients (each (ms))
end

function build(x::AbstractVector, y::AbstractVector, ms::NTuple{2, Int}, J::Int;
               L::Int = 8, max_order::Int = 2, T::Type = Float64,
               spectral::Plans.AbstractSpectralBackend = Plans.AutoSpectral(),
               period = nothing, solve::Bool = false, weights = nothing,
               eps = nothing, maxiter::Int = 100, rtol::Real = 1.0e-8)
    M = length(x)
    fb = FilterBanks.build_filter_bank2d(ms, J; L = L, T = T)
    tree = PathGraph.build_tree([m.j_eff for m in fb.meta], max_order)
    plan = Plans.make_scattered_plan(spectral, x, y, ms, T;
                                     period = period, solve = solve, maxiter = maxiter,
                                     rtol = rtol, eps = eps)
    w = weights === nothing ? fill(one(T) / M, M) : T.(weights) ./ sum(weights)
    nw = length(fb.wavelets)
    order2 = max_order >= 2
    U1_pts   = order2 ? [Vector{T}(undef, M) for _ in 1:nw] : Vector{T}[]
    U1_modes = order2 ? [Matrix{Complex{T}}(undef, ms) for _ in 1:nw] : Matrix{Complex{T}}[]
    return ScatteredPlanarScattering(
        fb, tree, max_order, plan, w,
        Vector{Complex{T}}(undef, M), Matrix{Complex{T}}(undef, ms), Matrix{Complex{T}}(undef, ms),
        Vector{Complex{T}}(undef, M), Vector{T}(undef, M), U1_pts, U1_modes)
end

_wmean(st::ScatteredPlanarScattering, v::AbstractVector) = LinearAlgebra.dot(st.weights, v)

"""
    (st::ScatteredPlanarScattering)(x) -> ScatteringCoefficients2D

Apply the scattered planar scattering transform to a length-`M` vector of samples at the plan's points.
"""
function (st::ScatteredPlanarScattering{T})(x::AbstractVector) where {T}
    fb, plan, tree = st.filter_bank, st.plan, st.tree
    nw = length(fb.wavelets)
    coeffs = Coefficients.ScatteringCoefficients2D(fb.J, fb.L, T; compute_S2 = st.max_order >= 2)

    st.buf_input_pts .= complex.(x)
    Plans.forward_transform!(st.X_modes, plan, st.buf_input_pts)         # signal mode coeffs

    @inbounds for (j, ψ) in enumerate(fb.wavelets)
        ScatteringCore.wavelet_convolve!(st.buf_conv_pts, st.X_modes, ψ, plan, st.buf_modes)
        ScatteringCore.apply_modulus!(st.buf_mod_pts, st.buf_conv_pts)
        coeffs.S1[j] = _wmean(st, st.buf_mod_pts)
    end

    if st.max_order >= 2
        @inbounds for (j1, ψ1) in enumerate(fb.wavelets)
            ScatteringCore.wavelet_convolve!(st.buf_conv_pts, st.X_modes, ψ1, plan, st.buf_modes)
            ScatteringCore.apply_modulus!(st.U1_pts[j1], st.buf_conv_pts)
        end
        @inbounds for j1 in 1:nw
            st.buf_input_pts .= complex.(st.U1_pts[j1])
            Plans.forward_transform!(st.U1_modes[j1], plan, st.buf_input_pts)
        end
        @inbounds for p in PathGraph.order_range(tree, 2)
            idx = PathGraph.path_indices(tree, p)
            j1, j2 = idx[1], idx[2]
            ScatteringCore.wavelet_convolve!(st.buf_conv_pts, st.U1_modes[j1], fb.wavelets[j2],
                                             plan, st.buf_modes)
            ScatteringCore.apply_modulus!(st.buf_mod_pts, st.buf_conv_pts)
            coeffs.S2[j1, j2] = _wmean(st, st.buf_mod_pts)
        end
    end

    return Coefficients.update_S0(coeffs, _wmean(st, x))
end

end # module ScatteredPlanar
