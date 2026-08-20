module ScatteredPlanar

"""
    ScatteredPlanar.jl — scattered / nonuniform planar scattering

Scattering of a scalar field sampled at arbitrary points `(x, y)` on the plane. Analysis lifts the
samples onto a uniform Fourier **mode grid** of size `ms` (a nonuniform DFT), where the oriented-Morlet
wavelet bank lives; the wavelet multiply happens on that grid and synthesis evaluates the filtered
field back at the points. The modulus + a (quadrature-weighted) spatial mean give the S0/S1/S2
coefficients, mirroring the gridded `ScatteringTransform2D` cascade (same `build_filter_bank2d`, same
`PathGraph` admissible-path tree, same `wavelet_convolve!` / `apply_modulus!`).

The spectral plan is chosen by `spectral`: the in-core `SB.DirectSumSpectralBackend` (exact
direct-summation NUDFT, no dependencies) is the always-available default;
`SB.NUFFTSpectralBackend` (or `SB.AutoSpectralBackend` once a NUFFT extension is loaded) selects the
fast path. Both satisfy the same `AbstractScatteringPlan` interface, so the cascade is identical
either way.
"""

using ..Plans: Plans
using SpectralBackends: SpectralBackends as SB
using ..FilterBanks: FilterBanks
using ..PathGraph: PathGraph
using ..Coefficients: Coefficients
using ..ScatteringCore: ScatteringCore
using LinearAlgebra: LinearAlgebra

# Standalone scattered cascade (point-buffers ≠ mode-buffers). All array/container fields are type
# parameters, matching `ScatteringTransform2D`'s style.
# The point and mode buffers carry no fixed rank: a transform built with `ntrans = B` holds
# `(M, B)` / `(ms…, B)` buffers and runs the whole stack through one execution per cascade step,
# where a single-field one holds `(M)` / `(ms)`. The plan's batch width is fixed at construction,
# so a transform is either single-field or batched.
struct ScatteredPlanarScattering{T, FB, Tree, G<:AbstractVector, P, WV<:AbstractVector{T},
                                 WB<:AbstractVector, CV<:AbstractArray{Complex{T}},
                                 MM<:AbstractArray{Complex{T}}, RV<:AbstractArray{T}}
    filter_bank::FB          # oriented Morlet bank on the `ms` fftfreq lattice
    tree::Tree               # admissible scattering paths
    groups::G                # (j1, children, path ids) from the tree, longest-first
    max_order::Int
    plan::P                  # scattered spectral plan (DirectNUFFTPlan or FINUFFT NUFFTScatteringPlan)
    weights::WV              # (M) spatial-mean quadrature weights, summing to 1
    # The bank's filters viewed as `(ms…, 1)` so they broadcast across a batch axis. These share the
    # bank's memory — no filter is copied — and exist only so the batched cascade does not build a
    # reshape per wavelet and per path on every call, which is pure allocation.
    wav_b::WB
    buf_input_pts::CV        # (M) complexified input / U1
    X_modes::MM              # (ms) signal mode coefficients
    buf_modes::MM            # (ms) wavelet-multiply scratch
    buf_conv_pts::CV         # (M) synthesis output at points
    buf_mod_pts::RV          # (M) modulus at points
    buf_u1_pts::RV           # (M) first-order modulus of the current j1
    buf_u1_modes::MM         # (ms) its mode coefficients, reused across that j1's children
end

# A copy of `A` living wherever `ref` lives. `similar` is the only device hook the cascade needs:
# it keeps this file free of any device package while still producing a device-resident transform
# when the caller hands in device points.
function _like(ref::AbstractArray, A::AbstractArray)
    d = similar(ref, eltype(A), size(A))
    copyto!(d, A)
    return d
end

function build(::Type{T}, x::AbstractVector, y::AbstractVector, ms::NTuple{2, Int}, J::Int;
               L::Int = 8, max_order::Int = 2,
               spectral::SB.AbstractSpectralBackend = SB.AutoSpectralBackend(),
               period = nothing, solve::Bool = false, weights = nothing,
               eps = nothing, maxiter::Int = 100, rtol::Real = 1.0e-8,
               ntrans::Int = 1) where {T}
    M = length(x)
    cpu_fb = FilterBanks.build_filter_bank2d(T, ms, J; L = L)
    # The transform lives wherever its points do. Every array is `similar` to `x`, so host points
    # give a host transform and device points a device-resident one through the same code — there is
    # no separate device constructor, and no device package is referenced here.
    fb = FilterBanks.FilterBank2D([_like(x, ψ) for ψ in cpu_fb.wavelets],
                                  _like(x, cpu_fb.averaging), cpu_fb.meta, cpu_fb.J, cpu_fb.L)
    tree = PathGraph.build_tree([m.j_eff for m in fb.meta], max_order)
    plan = Plans.make_scattered_plan(spectral, x, y, ms, T;
                                     period = period, solve = solve, maxiter = maxiter,
                                     rtol = rtol, eps = eps, ntrans = ntrans)
    # The buffers must match what the plan will actually execute, which is its own width — a backend
    # that cannot batch reports 1 regardless of what was asked for.
    B = Plans.batch_width(plan)
    w = _like(x, weights === nothing ? fill(one(T) / M, M) : T.(weights) ./ sum(weights))
    nw = length(fb.wavelets)
    groups = max_order >= 2 ? PathGraph.order2_groups(tree, nw) :
             [(j, Int[], Int[]) for j in 1:nw]
    # One first-order field and one mode array, not `nw` of each: the cascade finishes every child of
    # a wavelet before starting the next.
    cpts() = B == 1 ? similar(x, Complex{T}, M) : similar(x, Complex{T}, M, B)
    rpts() = B == 1 ? similar(x, T, M) : similar(x, T, M, B)
    modes() = B == 1 ? similar(x, Complex{T}, ms) : similar(x, Complex{T}, (ms..., B))
    wav_b = [reshape(ψ, ms..., 1) for ψ in fb.wavelets]
    return ScatteredPlanarScattering(
        fb, tree, groups, max_order, plan, w, wav_b,
        cpts(), modes(), modes(), cpts(), rpts(), rpts(), modes())
end
build(x::AbstractVector, y::AbstractVector, ms::NTuple{2, Int}, J::Int; kwargs...) =
    build(Float64, x, y, ms, J; kwargs...)

_wmean(st::ScatteredPlanarScattering, v::AbstractVector) = LinearAlgebra.dot(st.weights, v)

# Shares the read-only bank, tree, work list and weights; copies only the buffers and the plan's
# scratch. The filter bank and the plan's interpolation tables are the bulk of a scattered
# transform, so a task pays `O(M + prod(ms))`, not the whole transform.
ScatteringCore.task_workspace(st::ScatteredPlanarScattering) = ScatteredPlanarScattering(
    st.filter_bank, st.tree, st.groups, st.max_order, Plans.task_local_plan(st.plan), st.weights,
    st.wav_b,
    similar(st.buf_input_pts), similar(st.X_modes), similar(st.buf_modes),
    similar(st.buf_conv_pts), similar(st.buf_mod_pts),
    similar(st.buf_u1_pts), similar(st.buf_u1_modes))

"""
    (st::ScatteredPlanarScattering)(x) -> ScatteringCoefficients2D

Apply the scattered planar scattering transform to a length-`M` vector of samples at the plan's points.
Allocates a coefficient container per call; use [`scattered_planar_scattering!`](@ref) to reuse one.
"""
function (st::ScatteredPlanarScattering{T})(x::AbstractVector) where {T}
    fb = st.filter_bank
    coeffs = Coefficients.ScatteringCoefficients2D(fb.J, fb.L, T; compute_S2 = st.max_order >= 2)
    return scattered_planar_scattering!(coeffs, st, x)
end

"""
    scattered_planar_scattering!(coeffs, st, x) -> coeffs

In-place counterpart of the callable: write into a preallocated `ScatteringCoefficients2D`.
Allocation-free when `coeffs` carries a 1-element `S0` (so `update_S0` writes rather than rewraps).
"""
function scattered_planar_scattering!(coeffs::Coefficients.ScatteringCoefficients2D,
                                      st::ScatteredPlanarScattering, x::AbstractVector)
    fb, plan = st.filter_bank, st.plan

    st.buf_input_pts .= complex.(x)
    Plans.forward_transform!(st.X_modes, plan, st.buf_input_pts)         # signal mode coeffs

    # Both orders in one grouped pass, as in the gridded `cascade!`: each first-order field is
    # synthesised once and reused by its children, rather than once for S1 and again for S2, and
    # only one first-order field and one mode array are live at a time.
    isempty(coeffs.S2) || fill!(coeffs.S2, zero(eltype(coeffs.S2)))
    @inbounds for (j1, children, _) in st.groups
        ScatteringCore.wavelet_convolve!(st.buf_conv_pts, st.X_modes, fb.wavelets[j1],
                                         plan, st.buf_modes)
        ScatteringCore.apply_modulus!(st.buf_u1_pts, st.buf_conv_pts)
        coeffs.S1[j1] = _wmean(st, st.buf_u1_pts)
        isempty(children) && continue
        st.buf_input_pts .= complex.(st.buf_u1_pts)
        Plans.forward_transform!(st.buf_u1_modes, plan, st.buf_input_pts)
        for j2 in children
            ScatteringCore.wavelet_convolve!(st.buf_conv_pts, st.buf_u1_modes, fb.wavelets[j2],
                                             plan, st.buf_modes)
            ScatteringCore.apply_modulus!(st.buf_mod_pts, st.buf_conv_pts)
            coeffs.S2[j1, j2] = _wmean(st, st.buf_mod_pts)
        end
    end

    return Coefficients.update_S0(coeffs, _wmean(st, x))
end

# Weighted spatial mean of every column at once: `dst[b] = Σ_m w[m]·V[m, b]`, i.e. `Vᵀw`.
_wmean_batch!(dst::AbstractVector, st::ScatteredPlanarScattering, V::AbstractMatrix) =
    LinearAlgebra.mul!(dst, transpose(V), st.weights)

"""
    scattered_planar_scattering_batch!(S0, S1, S2, st, X) -> (; S0, S1, S2)

Cascade over a stack of `B` fields sampled at the same points: `X` is `(M, B)`, `S0` is `(B)`, `S1`
is `(nw, B)` and `S2` is `(nw, nw, B)`.

Every transform covers the whole stack in one execution, so the cascade issues its
`1 + nparents + nw + npaths` transforms once rather than once per field. `st` must have been built
with a matching `ntrans`, since a guru plan's batch width is fixed when it is made.
"""
function scattered_planar_scattering_batch!(S0::AbstractVector, S1::AbstractMatrix,
                                            S2::AbstractArray{<:Any, 3},
                                            st::ScatteredPlanarScattering, X::AbstractMatrix)
    plan = st.plan
    st.buf_input_pts .= complex.(X)
    Plans.forward_transform!(st.X_modes, plan, st.buf_input_pts)

    isempty(S2) || fill!(S2, zero(eltype(S2)))
    @inbounds for (j1, children, _) in st.groups
        ScatteringCore.wavelet_convolve!(st.buf_conv_pts, st.X_modes, st.wav_b[j1], plan,
                                         st.buf_modes)
        ScatteringCore.apply_modulus!(st.buf_u1_pts, st.buf_conv_pts)
        _wmean_batch!(view(S1, j1, :), st, st.buf_u1_pts)
        isempty(children) && continue
        st.buf_input_pts .= complex.(st.buf_u1_pts)
        Plans.forward_transform!(st.buf_u1_modes, plan, st.buf_input_pts)
        for j2 in children
            ScatteringCore.wavelet_convolve!(st.buf_conv_pts, st.buf_u1_modes, st.wav_b[j2],
                                             plan, st.buf_modes)
            ScatteringCore.apply_modulus!(st.buf_mod_pts, st.buf_conv_pts)
            _wmean_batch!(view(S2, j1, j2, :), st, st.buf_mod_pts)
        end
    end
    _wmean_batch!(S0, st, X)
    return (S0 = S0, S1 = S1, S2 = S2)
end

end # module ScatteredPlanar
