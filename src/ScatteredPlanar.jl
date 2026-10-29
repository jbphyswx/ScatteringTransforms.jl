module ScatteredPlanar

using ..Plans: Plans
using SpectralBackends: SpectralBackends as SB
using ..FilterBanks: FilterBanks
using ..PathGraph: PathGraph
using ..Coefficients: Coefficients
using ..ScatteringCore: ScatteringCore
using ..Batched: Batched

"""
    ScatteredPlanarScatteringTransform

Scattering transform of a field sampled at scattered planar points, as built by
`scattered_planar_scattering`: the samples analysed onto the `ms` mode grid by the spectral plan, then
the cascade of the gridded `Scattering2D` transform on that grid, with its oriented Morlet bank.

A transform built with `ntrans = B > 1` analyses `B` fields per call: its sample and mode buffers are
`(M, B)` and `(ms…, B)`, including a trailing batch axis at width 1.
"""
struct ScatteredPlanarScatteringTransform{T, FB, Tree, G<:AbstractVector, P, WV<:AbstractVector{T},
                                 RS<:AbstractArray{T}, CS<:AbstractArray{Complex{T}},
                                 MM<:AbstractArray{Complex{T}}, PW, BW, FM, HM, W} <: ScatteringCore.AbstractScatteringTransform
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
    batch::BW
    flat::FM
    hostflat::HM
    workspaces::W
end

ScatteredPlanarScatteringTransform(fb, tree, groups, max_order, plan, weights, wreal, wcplx,
                          modes, dims, pw, batch, flat, hostflat) =
    ScatteredPlanarScatteringTransform(fb, tree, groups, max_order, plan, weights, wreal, wcplx,
                               modes, dims, pw, batch, flat, hostflat,
                               ScatteringCore.Workspaces(ScatteredPlanarScatteringTransform))

# A copy of `A` on the array type of `ref`.
function _like(ref::AbstractArray, A::AbstractArray)
    d = similar(ref, eltype(A), size(A))
    copyto!(d, A)
    return d
end

_host_storage(a::Array) = a
_host_storage(a::AbstractMatrix{T}) where {T} = Matrix{T}(undef, size(a))

function _host_coefficients!(st::ScatteredPlanarScatteringTransform)
    st.hostflat === st.flat || copyto!(st.hostflat, st.flat)
    return st.hostflat
end

# The bank on the array type of the points. A computed bank moves only its low-pass and scratch; its
# wavelets are parameters.
_to_points(ref::AbstractArray, fb::FilterBanks.FilterBank2D) =
    FilterBanks.FilterBank2D([_like(ref, ψ) for ψ in fb.wavelets], _like(ref, fb.averaging),
                             fb.meta, fb.J, fb.L)
_to_points(ref::AbstractArray, fb::FilterBanks.ComputedFilterBank2D) =
    FilterBanks.ComputedFilterBank2D(fb.morlets, fb.rescale, _like(ref, fb.averaging),
                                     similar(ref, eltype(fb.scratch), size(fb.scratch)), fb.meta, fb.J, fb.L)

# The gridded cascade on the mode grid, in the memory of the mode buffer `modes`: on the host with the
# FFT the gridded transform takes, on a device with the device's FFT library and the means summed there.
_mode_plan(::Array, ::Type{T}, rd, B) where {T} =
    Plans.make_plan(SB.AutoSpectralBackend(), T, rd; batch = B)
_mode_plan(modes::AbstractArray, ::Type{T}, rd, B) where {T} =
    Plans.abstractffts_plan(similar(modes, Complex{T}, (rd..., B)); region = (1, 2))

function _mode_workspace(modes, fb, groups, ms::NTuple{2,Int}, ::Type{T}, α::Int, B::Int) where {T}
    zeros_like(::Type{S}, dims) where {S} = fill!(similar(modes, S, dims), zero(S))
    return Batched.build_spectral_workspace(fb, groups, ms, α, B, T; zeros_like,
        makeplan = rd -> _mode_plan(modes, T, rd, B),
        shapefilter = f -> reshape(f, size(f)..., 1))
end

function build(::Type{T}, x::AbstractVector, y::AbstractVector, ms::NTuple{2, Int}, J::Int;
               period, origin = (0, 0), L::Int = 8, max_order::Int = 2,
               transform::SB.AbstractSpectralBackend = SB.AutoSpectralBackend(),
               analysis::Plans.AbstractAnalysis = Plans.WeightedAdjoint(), weights = nothing,
               tol = nothing, maxiter::Int = 100,
               rtol::Real = Plans.default_solver_rtol(T, transform, tol), damp::Real = 0,
               ntrans::Int = 1, nthreads::Int = 0,
               filters::FilterBanks.AbstractFilterStorage = FilterBanks.StoredFilters(),
               oversampling::Int = J) where {T}
    M = length(x)
    cpu_fb = FilterBanks.build_filter_bank2d(T, ms, J, filters; L = L)
    # Every array is `similar` to `x`, so device points give a device-resident transform.
    fb = _to_points(x, cpu_fb)
    tree = PathGraph.build_tree([m.j_eff for m in fb.meta], max_order)
    plan = Plans.make_scattered_plan(transform, x, y, ms, T;
                                     period = period, origin = origin, analysis,
                                     maxiter, rtol, damp, tol, ntrans, nthreads)
    # Buffers follow the plan's own batch width; a backend that cannot batch reports 1 whatever
    # `ntrans` asked for.
    B = Plans.batch_width(plan)
    w = _like(x, weights === nothing ? fill(one(T) / M, M) : T.(weights) ./ sum(weights))
    nw = FilterBanks.nwavelets(fb)
    groups = max_order >= 2 ? PathGraph.order2_groups(tree, nw) :
             [(j, Int[], Int[]) for j in 1:nw]
    pts(::Type{S}) where {S} = similar(x, S, M, B)
    modes = similar(x, Complex{T}, ms..., B)
    batch = _mode_workspace(modes, fb, groups, ms, T, oversampling, B)
    flat = similar(x, Complex{T}, Coefficients.flat_length(nw), B)
    return ScatteredPlanarScatteringTransform(fb, tree, groups, max_order, plan, w, pts(T), pts(Complex{T}), modes,
                                     ms, batch.pw, batch, flat, _host_storage(flat))
end
build(x::AbstractVector, y::AbstractVector, ms::NTuple{2, Int}, J::Int; kwargs...) =
    build(Float64, x, y, ms, J; kwargs...)

Base.show(io::IO, st::ScatteredPlanarScatteringTransform{T}) where {T} =
    print(io, "ScatteredPlanarScatteringTransform{", T, "}(M=", length(st.weights), ", ms=", st.dims,
          ", J=", st.filter_bank.J, ", L=", st.filter_bank.L, ", max_order=", st.max_order,
          ", oversampling=", st.pw.oversampling, ", ", st.plan, ")")

# Shares the read-only bank, tree, work list and weights; copies only the buffers and the plans' scratch.
function ScatteringCore.task_workspace(st::ScatteredPlanarScatteringTransform)
    batch = Batched.task_copy(st.batch)
    flat = similar(st.flat)
    return ScatteredPlanarScatteringTransform(
        batch.bank, st.tree, st.groups, st.max_order, Plans.task_local_plan(st.plan), st.weights,
        similar(st.wreal), similar(st.wcplx), similar(st.X_modes), st.dims, batch.pw, batch,
        flat, _host_storage(flat))
end

# The spectrum of the field the samples `x` determine, in `st.X_modes`: the least-squares fit, or the
# adjoint weighted by the quadrature, `X_k = N Σ_n w_n x_n e^{−ik·s_n}`, the domain integral of
# `x e^{−ik·s}` scaled to the fft convention, `N = prod(ms)`.
function _analyse!(st::ScatteredPlanarScatteringTransform{T}, x::AbstractArray) where {T}
    return _analyse!(st, x, Plans.plan_analysis(st.plan).analysis)
end

_analyse!(st::ScatteredPlanarScatteringTransform, x, ::Plans.LeastSquares) =
    Plans.forward_transform!(st.X_modes, st.plan, x)

function _analyse!(st::ScatteredPlanarScatteringTransform{T}, x, ::Plans.WeightedAdjoint) where {T}
    wx = _weighted!(_samples(st, x), st.weights, x)
    Plans.forward_transform!(st.X_modes, st.plan, wx)
    st.X_modes .*= T(prod(st.dims))
    return st.X_modes
end

_samples(st::ScatteredPlanarScatteringTransform, ::AbstractArray{<:Real}) = st.wreal
_samples(st::ScatteredPlanarScatteringTransform, ::AbstractArray{<:Complex}) = st.wcplx

# The samples copied to the transform's memory, then weighted.
_weighted!(buf, w, x) = (copyto!(buf, x); buf .*= w; buf)

# The mean of the analysed field, `X_0/N`; real for a real field, whose `X_0` is real up to rounding.
_field_mean(::AbstractArray{<:Real}, X0) = real(X0)
_field_mean(::AbstractArray{<:Complex}, X0) = X0

# Every field's mean into `S0`, from the `X_0` row `X0` of its spectrum and `s = 1/N`.
_put_means!(S0, ::AbstractArray{<:Real}, X0, s) = (S0 .= real.(X0) .* s; S0)
_put_means!(S0, ::AbstractArray{<:Complex}, X0, s) = (S0 .= X0 .* s; S0)

"""
    (st::ScatteredPlanarScatteringTransform)(x) -> ScatteringCoefficients2D

Apply the scattered planar scattering transform to a length-`M` vector of samples at the plan's points.
Allocates a coefficient container per call; use [`scattered_planar_scattering!`](@ref) to reuse one.
"""
function (st::ScatteredPlanarScatteringTransform{T})(x::AbstractVector) where {T}
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
                                      st::ScatteredPlanarScatteringTransform{T}, x::AbstractVector) where {T}
    X = _analyse!(st, x)
    Batched.batch_spectral_cascade!(st.flat, st.batch, X)
    h = _host_coefficients!(st)
    nw = length(coeffs.S1)
    coeffs.S1 .= real.(view(h, 2:(nw + 1), 1))
    if !isempty(coeffs.S2)
        fill!(coeffs.S2, zero(eltype(coeffs.S2)))
        for j1 in 1:nw, j2 in (j1 + 1):nw
            row = Coefficients.flat_row_s2(j1, j2, nw)
            coeffs.S2[j1, j2] = real(h[row, 1])
        end
    end
    return Coefficients.update_S0(coeffs, _field_mean(x, h[1, 1]))
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
                                            st::ScatteredPlanarScatteringTransform{T}, X::AbstractMatrix) where {T}
    Xm = _analyse!(st, X)
    Batched.batch_spectral_cascade!(st.flat, st.batch, Xm)
    nw = size(S1, 1)
    S1 .= real.(view(st.flat, 2:(nw + 1), :))
    if !isempty(S2)
        fill!(S2, zero(eltype(S2)))
        for j1 in 1:nw, j2 in (j1 + 1):nw
            row = Coefficients.flat_row_s2(j1, j2, nw)
            view(S2, j1:j1, j2:j2, :) .= real.(reshape(view(st.flat, row:row, :), 1, 1, :))
        end
    end
    _put_means!(S0, X, view(Xm, 1, 1, :), inv(T(prod(st.dims))))
    return (S0 = S0, S1 = S1, S2 = S2)
end

end # module ScatteredPlanar
