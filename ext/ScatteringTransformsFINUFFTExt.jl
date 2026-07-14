module ScatteringTransformsFINUFFTExt

"""
    ScatteringTransformsFINUFFTExt — nonuniform / scattered planar scattering (NUFFT)

The **scattered/nonuniform Cartesian** backend, completing the planar side of the grid-support matrix
(#4a). A scalar field sampled at arbitrary points `(x, y)` is scattered via a Type-1 NUFFT onto a
uniform Fourier **mode grid** of size `ms`, where the ordinary oriented-Morlet wavelet bank lives; the
wavelet multiply happens on that grid and a Type-2 NUFFT evaluates the filtered field back at the
points. The modulus + a (quadrature-weighted) spatial mean then give the S0/S1/S2 coefficients, exactly
mirroring the gridded [`ScatteringTransform2D`] cascade (same `build_filter_bank2d`, same `PathGraph`
admissible-path tree, same `wavelet_convolve!`/`apply_modulus!`).

Plans use FFT mode ordering (`modeord=1`) so the mode grid matches the filter bank's `fftfreq` lattice.
With analysis = Type-1 and synthesis = Type-2/`prod(ms)`, a uniform `0:m-1` grid makes these exactly
`fft`/`ifft`, so the transform reduces to the gridded FFT transform (validated in tests).

**Correctness note (no shortcut).** The Type-1 adjoint (`solve=false`) equals the true DFT on a uniform
grid and is accurate for adequately-sampled band-limited fields, but on gappy/irregular sampling it is
only the adjoint, not the inverse. `solve=true` runs a conjugate-gradient least-squares inversion (the
planar analogue of NUFSHT's `nusht_solve!`) for the true band-limited coefficients — slower, but the
principled choice for irregular data. The caller picks per their sampling.
"""

using FINUFFT: FINUFFT
using LinearAlgebra: LinearAlgebra
using ScatteringTransforms: ScatteringTransforms

const ST = ScatteringTransforms
const Plans = ST.Plans
const FilterBanks = ST.FilterBanks
const PathGraph = ST.PathGraph
const Coefficients = ST.Coefficients
const ScatteringCore = ST.ScatteringCore

# ---------------------------------------------------------------------------
# NUFFT spectral plan: analysis (points → modes, Type-1 or CG-solve) + synthesis (modes → points).
# Implements the `Plans.AbstractScatteringPlan` interface so the cascade reuses `wavelet_convolve!`.
# `guru1`/`guru2` are opaque FINUFFT C-plan handles (kept untyped, as in FlowFieldSpectra); the
# array buffers are type parameters so the struct stays generic and concretely typed.
# ---------------------------------------------------------------------------

mutable struct NUFFTScatteringPlan{T, CV<:AbstractVector{Complex{T}},
                                   MM<:AbstractMatrix{Complex{T}}} <: Plans.AbstractScatteringPlan
    guru1                         # Type-1 (points → modes), iflag −1, FFT mode order
    guru2                         # Type-2 (modes → points), iflag +1, FFT mode order
    ms::NTuple{2,Int}
    M::Int
    invN::T                       # 1/prod(ms); makes synthesis the ifft-convention inverse
    solve::Bool
    maxiter::Int
    rtol::T
    cj::CV                        # (M) nonuniform exec buffer (shared by Type-1/Type-2)
    r::MM                         # (ms) CG residual / rhs
    p::MM                         # (ms) CG search direction
    Ap::MM                        # (ms) CG A†A·p
    tmp_pts::CV                   # (M) CG scratch
end

function _make_plan(x, y, ms::NTuple{2,Int}, ::Type{T}, period, eps, solve, maxiter, rtol) where {T}
    M = length(x)
    length(y) == M || throw(DimensionMismatch("x and y must have equal length"))
    xmin, ymin = T(minimum(x)), T(minimum(y))
    # Default period so a uniform 0:m-1 grid (span m-1) maps to the exact DFT nodes 2π·(0:m-1)/m.
    px = period === nothing ? (T(maximum(x)) - xmin) * ms[1] / (ms[1] - 1) : T(period[1])
    py = period === nothing ? (T(maximum(y)) - ymin) * ms[2] / (ms[2] - 1) : T(period[2])
    sx = T(2π) .* (T.(x) .- xmin) ./ px
    sy = T(2π) .* (T.(y) .- ymin) ./ py
    guru1 = FINUFFT.finufft_makeplan(1, collect(ms), -1, 1, T(eps); dtype = T, modeord = 1)
    guru2 = FINUFFT.finufft_makeplan(2, collect(ms), +1, 1, T(eps); dtype = T, modeord = 1)
    FINUFFT.finufft_setpts!(guru1, sx, sy)
    FINUFFT.finufft_setpts!(guru2, sx, sy)
    plan = NUFFTScatteringPlan(
        guru1, guru2, ms, M, one(T) / prod(ms), solve, maxiter, T(rtol),
        Vector{Complex{T}}(undef, M),
        Matrix{Complex{T}}(undef, ms), Matrix{Complex{T}}(undef, ms), Matrix{Complex{T}}(undef, ms),
        Vector{Complex{T}}(undef, M))
    finalizer(pl -> (FINUFFT.finufft_destroy!(pl.guru1); FINUFFT.finufft_destroy!(pl.guru2)), plan)
    return plan
end

# Synthesis: modes → points, scaled by 1/prod(ms) (ifft convention). For a Type-2 plan
# `finufft_exec!(plan, input, output)` takes input=modes, output=points.
function Plans.inverse_transform!(out_pts::AbstractVector, plan::NUFFTScatteringPlan, Xmodes::AbstractMatrix)
    FINUFFT.finufft_exec!(plan.guru2, Xmodes, plan.cj)
    @. out_pts = plan.cj * plan.invN
    return out_pts
end

# Analysis: points → modes. Type-1 adjoint (fft-equivalent on a uniform grid) unless `solve`.
function Plans.forward_transform!(Xmodes::AbstractMatrix, plan::NUFFTScatteringPlan, x_pts::AbstractVector)
    if plan.solve
        _cg_solve!(Xmodes, plan, x_pts)
    else
        copyto!(plan.cj, x_pts)
        FINUFFT.finufft_exec!(plan.guru1, plan.cj, Xmodes)
    end
    return Xmodes
end

# CG least-squares inversion: find modes `f` with Type2(f) ≈ prod(ms)·x (so synthesis, = Type2/N,
# recovers x). Solves the normal equations (A†A) f = A† (N·x) with A = Type-2, A† = Type-1.
function _cg_solve!(f::AbstractMatrix, plan::NUFFTScatteringPlan{T}, x_pts::AbstractVector) where {T}
    N = one(T) / plan.invN
    copyto!(plan.cj, x_pts)
    FINUFFT.finufft_exec!(plan.guru1, plan.cj, plan.r)         # r = A†x  (modes)
    plan.r .*= N                                               # r = A†(N·x) = rhs
    fill!(f, zero(Complex{T}))
    copyto!(plan.p, plan.r)
    rsold = real(LinearAlgebra.dot(vec(plan.r), vec(plan.r)))
    rs0 = rsold
    rs0 == 0 && return f
    @inbounds for _ in 1:plan.maxiter
        FINUFFT.finufft_exec!(plan.guru2, plan.p, plan.tmp_pts)    # tmp = A·p (Type-2: modes→points)
        FINUFFT.finufft_exec!(plan.guru1, plan.tmp_pts, plan.Ap)   # Ap  = A†A·p (Type-1: points→modes)
        α = rsold / real(LinearAlgebra.dot(vec(plan.p), vec(plan.Ap)))
        f .+= α .* plan.p
        plan.r .-= α .* plan.Ap
        rsnew = real(LinearAlgebra.dot(vec(plan.r), vec(plan.r)))
        sqrt(rsnew) <= plan.rtol * sqrt(rs0) && break
        plan.p .= plan.r .+ (rsnew / rsold) .* plan.p
        rsold = rsnew
    end
    return f
end

# ---------------------------------------------------------------------------
# Scattered planar scattering transform (standalone; point-buffers ≠ mode-buffers).
# All array/container fields are type parameters (matching `ScatteringTransform2D`'s style).
# ---------------------------------------------------------------------------

struct ScatteredPlanarScattering{T, FB, Tree, P, WV<:AbstractVector{T},
                                 CV<:AbstractVector{Complex{T}}, MM<:AbstractMatrix{Complex{T}},
                                 RV<:AbstractVector{T}, U1V<:AbstractVector, U1M<:AbstractVector}
    filter_bank::FB          # oriented Morlet bank on the `ms` fftfreq lattice
    tree::Tree               # admissible scattering paths
    max_order::Int
    plan::P                  # NUFFTScatteringPlan
    weights::WV              # (M) spatial-mean quadrature weights, summing to 1
    buf_input_pts::CV        # (M) complexified input / U1
    X_modes::MM              # (ms) signal mode coefficients
    buf_modes::MM            # (ms) wavelet-multiply scratch
    buf_conv_pts::CV         # (M) synthesis output at points
    buf_mod_pts::RV          # (M) modulus at points
    U1_pts::U1V              # per-wavelet first-order moduli (each (M))
    U1_modes::U1M            # per-wavelet U1 mode coefficients (each (ms))
end

function ST.scattered_planar_scattering(x::AbstractVector, y::AbstractVector, ms::NTuple{2,Int}, J::Int;
                                        L::Int = 8, max_order::Int = 2, T::Type = Float64,
                                        period = nothing, solve::Bool = false, weights = nothing,
                                        eps::Real = (T === Float32 ? 1.0e-6 : 1.0e-9),
                                        maxiter::Int = 100, rtol::Real = 1.0e-8)
    M = length(x)
    fb = FilterBanks.build_filter_bank2d(ms, J; L = L, T = T)
    tree = PathGraph.build_tree([m.j_eff for m in fb.meta], max_order)
    plan = _make_plan(x, y, ms, T, period, eps, solve, maxiter, rtol)
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

end # module ScatteringTransformsFINUFFTExt
