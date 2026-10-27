module ScatteredPlanar

using ..Plans: Plans
using SpectralBackends: SpectralBackends as SB
using ..FilterBanks: FilterBanks
using ..PathGraph: PathGraph
using ..Coefficients: Coefficients
using ..ScatteringCore: ScatteringCore
using ..Cascade: Cascade

"""
    ScatteredPlanarScattering{T,FB,Tree,G,P,WV,RS,CS,MM,PW}

Scattering transform of a field sampled at scattered planar points, as built by
`scattered_planar_scattering`: the samples analysed onto the `ms` mode grid by the spectral plan, then
the cascade of the gridded `Scattering2D` transform on that grid, with its oriented Morlet bank.

A transform built with `ntrans = B > 1` analyses `B` fields per call: its sample and mode buffers are
`(M, B)` and `(ms…, B)`; a single-field one holds `(M)` and `(ms)` buffers.
"""
struct ScatteredPlanarScattering{T, FB, Tree, G<:AbstractVector, P, WV<:AbstractVector{T},
                                 RS<:AbstractArray{T}, CS<:AbstractArray{Complex{T}},
                                 MM<:AbstractArray{Complex{T}}, PW}
    filter_bank::FB          # oriented Morlet bank on the `ms` fftfreq lattice
    tree::Tree               # admissible scattering paths
    groups::G                # (j1, children, path ids) from the tree, longest-first
    max_order::Int
    plan::P                  # scattered spectral plan: `DirectNUFFTPlan` or `NUFFTScatteringPlan`
    weights::WV              # (M) quadrature weights of the domain mean, summing to 1
    wreal::RS                # (M[, B]) weighted real samples, the adjoint's input
    wcplx::CS                # (M[, B]) weighted complex samples
    X_modes::MM              # (ms[, B]) the analysed field's spectrum, fft convention
    dims::NTuple{2, Int}     # the mode grid `ms`
    pw::PW                   # the gridded cascade's workspace on the mode grid
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

# The gridded cascade on the mode grid, in the memory of the mode buffer `modes`: on the host with the
# FFT the gridded transform takes, on a device with the device's FFT library and the means summed there.
_mode_cascade(::Array, fb, groups, ms::NTuple{2, Int}, ::Type{T}, α::Int) where {T} =
    Cascade.build(fb, groups, ms, T, α, Plans.make_plan(SB.AutoSpectralBackend(), T, ms))

function _mode_cascade(modes::AbstractArray, fb, groups, ms::NTuple{2, Int}, ::Type{T}, α::Int) where {T}
    alloc(rd) = fill!(similar(modes, Complex{T}, rd), zero(Complex{T}))
    return Cascade.build(fb, groups, ms, T, α; alloc = alloc,
                         makeplan = rd -> Plans.abstractffts_plan(alloc(rd)),
                         device_zeros = (S, d) -> fill!(similar(modes, S, d), zero(S)))
end

function build(::Type{T}, x::AbstractVector, y::AbstractVector, ms::NTuple{2, Int}, J::Int;
               period, origin = (0, 0), L::Int = 8, max_order::Int = 2,
               transform::SB.AbstractSpectralBackend = SB.AutoSpectralBackend(),
               solve::Bool = false, weights = nothing,
               eps = nothing, maxiter::Int = 100,
               rtol::Real = Plans.default_solver_rtol(T, transform, eps), damp::Real = 0,
               ntrans::Int = 1, nufft_nthreads::Int = 0,
               filters::FilterBanks.AbstractFilterStorage = FilterBanks.StoredFilters(),
               oversampling::Int = J) where {T}
    M = length(x)
    cpu_fb = FilterBanks.build_filter_bank2d(T, ms, J, filters; L = L)
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
    pts(S) = B == 1 ? similar(x, S, M) : similar(x, S, M, B)
    modes = B == 1 ? similar(x, Complex{T}, ms) : similar(x, Complex{T}, (ms..., B))
    pw = _mode_cascade(modes, fb, groups, ms, T, oversampling)
    return ScatteredPlanarScattering(fb, tree, groups, max_order, plan, w, pts(T), pts(Complex{T}), modes,
                                     ms, pw)
end
build(x::AbstractVector, y::AbstractVector, ms::NTuple{2, Int}, J::Int; kwargs...) =
    build(Float64, x, y, ms, J; kwargs...)

Base.show(io::IO, st::ScatteredPlanarScattering{T}) where {T} =
    print(io, "ScatteredPlanarScattering{", T, "}(M=", length(st.weights), ", ms=", st.dims,
          ", J=", st.filter_bank.J, ", L=", st.filter_bank.L, ", max_order=", st.max_order,
          ", oversampling=", st.pw.oversampling, ", ", st.plan, ")")

# Shares the read-only bank, tree, work list and weights; copies only the buffers and the plans' scratch.
function ScatteringCore.task_workspace(st::ScatteredPlanarScattering)
    fb = FilterBanks.task_bank(st.filter_bank)
    return ScatteredPlanarScattering(
        fb, st.tree, st.groups, st.max_order, Plans.task_local_plan(st.plan), st.weights,
        similar(st.wreal), similar(st.wcplx), similar(st.X_modes), st.dims, Cascade.task_copy(st.pw, fb))
end

# The spectrum of the field the samples `x` determine, in `st.X_modes`: the least-squares fit, or the
# adjoint weighted by the quadrature, `X_k = N Σ_n w_n x_n e^{−ik·s_n}`, the domain integral of
# `x e^{−ik·s}` scaled to the fft convention, `N = prod(ms)`.
function _analyse!(st::ScatteredPlanarScattering{T}, x::AbstractArray) where {T}
    st.plan.solve && return Plans.forward_transform!(st.X_modes, st.plan, x)
    wx = _weighted!(_samples(st, x), st.weights, x)
    Plans.forward_transform!(st.X_modes, st.plan, wx)
    st.X_modes .*= T(prod(st.dims))
    return st.X_modes
end

_samples(st::ScatteredPlanarScattering, ::AbstractArray{<:Real}) = st.wreal
_samples(st::ScatteredPlanarScattering, ::AbstractArray{<:Complex}) = st.wcplx

# The samples copied to the transform's memory, then weighted.
_weighted!(buf, w, x) = (copyto!(buf, x); buf .*= w; buf)

# The mean of the analysed field, `X_0/N`; real for a real field, whose `X_0` is real up to rounding.
_field_mean(::AbstractArray{<:Real}, X0) = real(X0)
_field_mean(::AbstractArray{<:Complex}, X0) = X0

# Every field's mean into `S0`, from the `X_0` row `X0` of its spectrum and `s = 1/N`.
_put_means!(S0, ::AbstractArray{<:Real}, X0, s) = (S0 .= real.(X0) .* s; S0)
_put_means!(S0, ::AbstractArray{<:Complex}, X0, s) = (S0 .= X0 .* s; S0)

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
                                      st::ScatteredPlanarScattering{T}, x::AbstractVector) where {T}
    X = _analyse!(st, x)
    Cascade.cascade!(coeffs.S1, coeffs.S2, st.pw, st.filter_bank, st.groups, X)
    return Coefficients.update_S0(coeffs, _field_mean(x, sum(view(X, 1:1, 1:1))) / T(prod(st.dims)))
end

"""
    scattered_planar_scattering_batch!(S0, S1, S2, st, X) -> (; S0, S1, S2)

Cascade over a stack of `B` fields sampled at the same points: `X` is `(M, B)`, `S0` is `(B)`, `S1`
is `(nw, B)` and `S2` is `(nw, nw, B)`, `(0, 0, B)` without a second order.

The analysis covers the whole stack in one execution, and each field's cascade then runs on the mode
grid; `st` must have been built with `ntrans = B`.
"""
function scattered_planar_scattering_batch!(S0::AbstractVector, S1::AbstractMatrix,
                                            S2::AbstractArray{<:Any, 3},
                                            st::ScatteredPlanarScattering{T}, X::AbstractMatrix) where {T}
    Xm = _analyse!(st, X)
    @inbounds for b in axes(X, 2)
        Cascade.cascade!(view(S1, :, b), view(S2, :, :, b), st.pw, st.filter_bank, st.groups,
                         view(Xm, :, :, b))
    end
    _put_means!(S0, X, view(Xm, 1, 1, :), inv(T(prod(st.dims))))
    return (S0 = S0, S1 = S1, S2 = S2)
end

end # module ScatteredPlanar
