module ScatteredPlanar

using ..Plans: Plans
using SpectralBackends: SpectralBackends as SB
using ..FilterBanks: FilterBanks
using ..PathGraph: PathGraph
using ..Coefficients: Coefficients
using ..ScatteringCore: ScatteringCore
using LinearAlgebra: LinearAlgebra

"""
    ScatteredPlanarScattering{T,FB,Tree,G,P,WV,WB,CV,MM,RV}

Scattering transform of a field sampled at scattered planar points, as built by
`scattered_planar_scattering`: an oriented Morlet bank on the `ms` mode grid, the spectral plan
between the points and that grid, and the buffers the cascade runs in.

A transform built with `ntrans = B > 1` holds `(M, B)` and `(ms…, B)` buffers and transforms exactly
`B` fields per call; a single-field one holds `(M)` and `(ms)` buffers.
"""
struct ScatteredPlanarScattering{T, FB, Tree, G<:AbstractVector, P, WV<:AbstractVector{T},
                                 WB<:AbstractVector, CV<:AbstractArray{Complex{T}},
                                 MM<:AbstractArray{Complex{T}}, RV<:AbstractArray{T}}
    filter_bank::FB          # oriented Morlet bank on the `ms` fftfreq lattice
    tree::Tree               # admissible scattering paths
    groups::G                # (j1, children, path ids) from the tree, longest-first
    max_order::Int
    plan::P                  # scattered spectral plan: `DirectNUFFTPlan` or `NUFFTScatteringPlan`
    weights::WV              # (M) spatial-mean quadrature weights, summing to 1
    # The bank's filters viewed as `(ms…, 1)`, sharing its memory, so they broadcast across a batch
    # axis.
    wav_b::WB
    X_modes::MM              # (ms) signal mode coefficients
    buf_modes::MM            # (ms) wavelet-multiply scratch
    buf_conv_pts::CV         # (M) synthesis output at points
    buf_mod_pts::RV          # (M) modulus at points
    buf_u1_pts::RV           # (M) first-order modulus of the current j1
    buf_u1_modes::MM         # (ms) its mode coefficients, reused across that j1's children
end

# A copy of `A` on the array type of `ref`.
function _like(ref::AbstractArray, A::AbstractArray)
    d = similar(ref, eltype(A), size(A))
    copyto!(d, A)
    return d
end

# The bank on the array type of the points. A computed bank moves only its low-pass and scratch; its
# wavelets are parameters.
_to_points(ref::AbstractArray, fb::FilterBanks.FilterBank2D) =
    FilterBanks.FilterBank2D([_like(ref, ψ) for ψ in fb.wavelets], _like(ref, fb.averaging),
                             fb.meta, fb.J, fb.L)
_to_points(ref::AbstractArray, fb::FilterBanks.ComputedFilterBank2D) =
    FilterBanks.ComputedFilterBank2D(fb.morlets, fb.rescale, _like(ref, fb.averaging),
                                     _like(ref, fb.scratch), fb.meta, fb.J, fb.L)

function build(::Type{T}, x::AbstractVector, y::AbstractVector, ms::NTuple{2, Int}, J::Int;
               period, origin = (0, 0), L::Int = 8, max_order::Int = 2,
               transform::SB.AbstractSpectralBackend = SB.AutoSpectralBackend(),
               solve::Bool = false, weights = nothing,
               eps = nothing, maxiter::Int = 100,
               rtol::Real = Plans.default_solver_rtol(T, transform, eps), damp::Real = 0,
               ntrans::Int = 1, nufft_nthreads::Int = 0, cache::Bool = true) where {T}
    M = length(x)
    cpu_fb = FilterBanks.build_filter_bank2d(T, ms, J, Val(cache); L = L)
    # Every array is `similar` to `x`, so device points give a device-resident transform.
    fb = _to_points(x, cpu_fb)
    tree = PathGraph.build_tree([m.j_eff for m in fb.meta], max_order)
    plan = Plans.make_scattered_plan(transform, x, y, ms, T;
                                     period = period, origin = origin, solve = solve,
                                     maxiter = maxiter, rtol = rtol, damp = damp, eps = eps,
                                     ntrans = ntrans, nufft_nthreads = nufft_nthreads)
    # Buffers follow the plan's own batch width; a backend that cannot batch reports 1 whatever
    # `ntrans` asked for.
    B = Plans.batch_width(plan)
    w = _like(x, weights === nothing ? fill(one(T) / M, M) : T.(weights) ./ sum(weights))
    nw = FilterBanks.nwavelets(fb)
    groups = max_order >= 2 ? PathGraph.order2_groups(tree, nw) :
             [(j, Int[], Int[]) for j in 1:nw]
    # One first-order field and one mode array suffice: the cascade finishes every child of a
    # wavelet before starting the next.
    cpts() = B == 1 ? similar(x, Complex{T}, M) : similar(x, Complex{T}, M, B)
    rpts() = B == 1 ? similar(x, T, M) : similar(x, T, M, B)
    modes() = B == 1 ? similar(x, Complex{T}, ms) : similar(x, Complex{T}, (ms..., B))
    wav_b = FilterBanks.batch_views(fb, ms)
    # `buf_mod_pts` and `buf_u1_pts` are one array: `U₁` is dead once its analysis has read it, which
    # is before any child writes its own modulus.
    u1pts = rpts()
    return ScatteredPlanarScattering(
        fb, tree, groups, max_order, plan, w, wav_b,
        modes(), modes(), cpts(), u1pts, u1pts, modes())
end
build(x::AbstractVector, y::AbstractVector, ms::NTuple{2, Int}, J::Int; kwargs...) =
    build(Float64, x, y, ms, J; kwargs...)

Base.show(io::IO, st::ScatteredPlanarScattering{T}) where {T} =
    print(io, "ScatteredPlanarScattering{", T, "}(M=", length(st.weights), ", ms=", size(st.X_modes)[1:2],
          ", J=", st.filter_bank.J, ", L=", st.filter_bank.L, ", max_order=", st.max_order, ", ",
          st.plan, ")")

_wmean(st::ScatteredPlanarScattering, v::AbstractVector) = LinearAlgebra.dot(st.weights, v)

# Shares the read-only bank, tree, work list and weights; copies only the buffers and the plan's
# scratch.
function ScatteringCore.task_workspace(st::ScatteredPlanarScattering)
    fb = FilterBanks.task_bank(st.filter_bank)
    # A computed bank gives the task its own scratch, so its batch views are rebuilt against it.
    ms = size(st.X_modes)[1:2]
    wav_b = fb === st.filter_bank ? st.wav_b : FilterBanks.batch_views(fb, ms)
    u1pts = similar(st.buf_u1_pts)   # shared with `buf_mod_pts`, as in the original
    return ScatteredPlanarScattering(
        fb, st.tree, st.groups, st.max_order, Plans.task_local_plan(st.plan), st.weights, wav_b,
        similar(st.X_modes), similar(st.buf_modes),
        similar(st.buf_conv_pts), u1pts, u1pts, similar(st.buf_u1_modes))
end

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
Allocation-free when `coeffs` carries a 1-element `S0`, which [`Coefficients.update_S0`](@ref)
writes in place.
"""
function scattered_planar_scattering!(coeffs::Coefficients.ScatteringCoefficients2D,
                                      st::ScatteredPlanarScattering, x::AbstractVector)
    fb, plan = st.filter_bank, st.plan

    # `x` reaches the plan unconverted: a real field has a Hermitian spectrum, which a NUFFT plan
    # analyses on half the mode grid.
    Plans.forward_transform!(st.X_modes, plan, x)                        # signal mode coeffs

    # Both orders in one grouped pass, as in the gridded `cascade!`.
    isempty(coeffs.S2) || fill!(coeffs.S2, zero(eltype(coeffs.S2)))
    @inbounds for (j1, children, _) in st.groups
        ScatteringCore.wavelet_convolve!(st.buf_conv_pts, st.X_modes, FilterBanks.filter_at(fb, j1),
                                         plan, st.buf_modes)
        ScatteringCore.apply_modulus!(st.buf_u1_pts, st.buf_conv_pts)
        coeffs.S1[j1] = _wmean(st, st.buf_u1_pts)
        isempty(children) && continue
        # A modulus, so real whatever the input was — and passed as such.
        Plans.forward_transform!(st.buf_u1_modes, plan, st.buf_u1_pts)
        for j2 in children
            ScatteringCore.wavelet_convolve!(st.buf_conv_pts, st.buf_u1_modes, FilterBanks.filter_at(fb, j2),
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

Each transform covers the whole stack in one execution; `st` must have been built with
`ntrans = B`.
"""
function scattered_planar_scattering_batch!(S0::AbstractVector, S1::AbstractMatrix,
                                            S2::AbstractArray{<:Any, 3},
                                            st::ScatteredPlanarScattering, X::AbstractMatrix)
    plan = st.plan
    Plans.forward_transform!(st.X_modes, plan, X)

    isempty(S2) || fill!(S2, zero(eltype(S2)))
    @inbounds for (j1, children, _) in st.groups
        FilterBanks.filter_at(st.filter_bank, j1)
        ScatteringCore.wavelet_convolve!(st.buf_conv_pts, st.X_modes, st.wav_b[j1], plan,
                                         st.buf_modes)
        ScatteringCore.apply_modulus!(st.buf_u1_pts, st.buf_conv_pts)
        _wmean_batch!(view(S1, j1, :), st, st.buf_u1_pts)
        isempty(children) && continue
        # A modulus, so real whatever the input was — and passed as such.
        Plans.forward_transform!(st.buf_u1_modes, plan, st.buf_u1_pts)
        for j2 in children
            FilterBanks.filter_at(st.filter_bank, j2)
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
