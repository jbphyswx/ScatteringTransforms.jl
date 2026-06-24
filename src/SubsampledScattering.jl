module SubsampledScattering

"""
    SubsampledScattering.jl — fast second order via intermediate subsampling

The first-order modulus field `U₁ = |x ⋆ ψ_{j₁}|` is a low-frequency envelope (the modulus
demodulates the narrow-band wavelet response), so it is oversampled at full resolution `N` and
can be decimated by `2^(scale - oversampling)` before the second wavelet transform — making the
order-2 FFTs run on shorter arrays. This needs the second-stage wavelets at the reduced
resolution (a multi-resolution bank), and decimation is mildly approximate (controlled by
`oversampling`). Opt-in; with a large `oversampling` it reproduces the exact transform.
"""

using ..Plans: Plans
using ..Filters: Filters
using ..FilterBanks: FilterBanks
using ..ScatteringCore: ScatteringCore
using ..Coefficients: Coefficients
using ..PathGraph: PathGraph

export SubsampledScattering1D

"""
    SubsampledScattering1D(N, J; Q=1, max_order=2, oversampling=1, T=Float64, spectral=:auto)

A 1D scattering transform with fast second order via intermediate subsampling: the first-order
modulus envelope is decimated by `2^(scale - oversampling)` before the second wavelet transform
(multi-resolution wavelet bank). Opt-in and approximate — a large `oversampling` reproduces the
exact [`ScatteringTransform1D`](@ref); aggressive values trade a little accuracy for speed.
"""
struct SubsampledScattering1D{T, Tree<:PathGraph.ScatteringTree}
    N::Int
    J::Int
    Q::Int
    max_order::Int
    oversampling::Int
    spectral::Symbol
    tree::Tree
    meta::Vector{FilterBanks.WaveletMeta{T}}
    wavelets_full::Vector{Vector{Complex{T}}}        # ψ_i at full N
    plan_full::Plans.AbstractScatteringPlan
    # per decimation factor ds > 1: wavelets at N÷ds and a plan at N÷ds
    levels::Dict{Int, Tuple{Vector{Vector{Complex{T}}}, Plans.AbstractScatteringPlan}}
end

# Decimation factor for a first-order wavelet of octave `scale`.
_ds(scale::Int, oversampling::Int) = 1 << max(0, scale - oversampling)

function SubsampledScattering1D(N::Int, J::Int; Q::Int = 1, max_order::Int = 2,
                                oversampling::Int = 1, T::Type = Float64, spectral::Symbol = :auto)
    fb = FilterBanks.build_filter_bank1d(N, J; Q = Q, T = T)
    meta = fb.meta
    wavelets_full = fb.wavelets
    tree = PathGraph.build_tree([m.j_eff for m in meta], max_order)
    plan_full = Plans.make_plan(spectral, T, (N,))

    # Build a reduced-resolution wavelet bank + plan for each distinct ds > 1 that occurs.
    levels = Dict{Int, Tuple{Vector{Vector{Complex{T}}}, Plans.AbstractScatteringPlan}}()
    for m in meta
        ds = _ds(m.scale, oversampling)
        (ds == 1 || haskey(levels, ds)) && continue
        N1 = N ÷ ds
        N1 * ds == N || throw(ArgumentError("oversampling/scale gives ds=$ds not dividing N=$N"))
        wl = [Filters.frequency_response(Filters.Morlet1D{T}(N1, mm.scale * Q + mm.q; Q = Q)) for mm in meta]
        levels[ds] = (wl, Plans.make_plan(spectral, T, (N1,)))
    end
    return SubsampledScattering1D{T, typeof(tree)}(N, J, Q, max_order, oversampling, spectral,
                                                   tree, meta, wavelets_full, plan_full, levels)
end

function (st::SubsampledScattering1D{T})(signal::AbstractVector) where {T}
    N = st.N
    num_w = length(st.wavelets_full)
    coeffs = Coefficients.ScatteringCoefficients1D(num_w, T; compute_S2 = st.max_order >= 2)

    xc = complex.(T.(signal))
    xfft = similar(xc)
    Plans.forward_transform!(xfft, st.plan_full, xc)

    conv = similar(xc)
    mult = similar(xc)
    modbuf = Vector{T}(undef, N)

    # first order (full resolution)
    U1 = [Vector{T}(undef, N) for _ in 1:num_w]
    @inbounds for i in 1:num_w
        ScatteringCore.wavelet_convolve!(conv, xfft, st.wavelets_full[i], st.plan_full, mult)
        ScatteringCore.apply_modulus!(U1[i], conv)
        coeffs.S1[i] = ScatteringCore.spatial_average(U1[i])
    end

    # second order with intermediate subsampling of U1
    if st.max_order >= 2
        for p in PathGraph.order_range(st.tree, 2)
            idx = PathGraph.path_indices(st.tree, p)
            i1, i2 = idx[1], idx[2]
            ds = _ds(st.meta[i1].scale, st.oversampling)
            if ds == 1
                ψ = st.wavelets_full[i2]; plan = st.plan_full; u = U1[i1]
            else
                wl, plan = st.levels[ds]
                ψ = wl[i2]
                u = @view U1[i1][1:ds:(1 + (N ÷ ds - 1) * ds)]   # decimate the envelope
            end
            n1 = length(u)
            uc = complex.(u)
            uf = similar(uc)
            Plans.forward_transform!(uf, plan, uc)
            c2 = similar(uc); m2 = similar(uc)
            ScatteringCore.wavelet_convolve!(c2, uf, ψ, plan, m2)
            mb = Vector{T}(undef, n1)
            ScatteringCore.apply_modulus!(mb, c2)
            coeffs.S2[i1, i2] = ScatteringCore.spatial_average(mb)
        end
    end

    S0 = ScatteringCore.spatial_average(signal)
    return Coefficients.update_S0(coeffs, S0)
end

end # module SubsampledScattering
