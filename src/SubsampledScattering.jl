module SubsampledScattering

"""
    SubsampledScattering.jl — fast second order via intermediate subsampling

The first-order modulus field `U₁ = |x ⋆ ψ_{j₁}|` is a low-frequency envelope (the modulus
demodulates the narrow-band wavelet response), so it is oversampled at the full grid and can be
decimated by `2^(scale - oversampling)` on every axis before the second wavelet transform — making
the order-2 transforms run on a smaller grid. This needs the second-stage wavelets at the reduced
resolution (a multi-resolution bank), and decimation is mildly approximate (controlled by
`oversampling`). Opt-in; with a large `oversampling` it reproduces the exact transform.

Decimating in space is periodising in frequency, so the strided view below is the exact operation;
what is approximate is the assumption that `U₁` carries no energy above the reduced Nyquist. That
error is what `oversampling` buys down.

The saving grows with dimension: decimating by `ds` shrinks the grid by `ds^D`, so on a 3D volume a
factor of two costs the second order eight times less work.
"""

using ..Plans: Plans
using ComputationalBackends: ComputationalBackends as CB
using SpectralBackends: SpectralBackends as SB
using ..FilterBanks: FilterBanks
using ..ScatteringCore: ScatteringCore
using ..Coefficients: Coefficients
using ..PathGraph: PathGraph

export MultiResolutionScattering
export SubsampledScattering1D, SubsampledScattering2D, SubsampledScattering3D
export subsampled_scattering!

"""
    Level{WV,P,CA,RA}

One resolution of the multi-resolution cascade: the wavelet bank and spectral plan on the decimated
grid, plus the buffers the second order runs in there. Buffers live on the level, so a path costs no
allocation.
"""
struct Level{WV, P, CA, RA}
    ds::Int
    wavelets::WV
    plan::P
    uc::CA        # complexified (decimated) first-order field
    uf::CA        # its spectrum — computed once per j1, reused by every child
    conv::CA      # inverse-transform output
    mult::CA      # multiply scratch
    u::RA         # decimated first-order field, gathered from the strided view
end

"""
    MultiResolutionScattering{T,D}

Scattering transform with a multi-resolution second order, on a `D`-dimensional grid. Build one with
[`SubsampledScattering1D`](@ref), [`SubsampledScattering2D`](@ref) or
[`SubsampledScattering3D`](@ref); they differ only in which filter bank they build, and share this
type and one cascade.
"""
struct MultiResolutionScattering{T, D, FB, Tree <: PathGraph.ScatteringTree, G <: AbstractVector,
                            PF <: Plans.AbstractScatteringPlan, CA <: AbstractArray{Complex{T}},
                            RA <: AbstractArray{T}, LV <: AbstractDict}
    dims::NTuple{D, Int}
    J::Int
    max_order::Int
    oversampling::Int
    filter_bank::FB                # full-resolution bank (wavelets, averaging, meta, J, Q/L/n_orient)
    tree::Tree
    groups::G                      # (j1, children, path ids), longest-first
    plan_full::PF
    xc::CA                         # complexified input
    xfft::CA                       # its spectrum, read-only across the cascade
    conv::CA                       # full-resolution inverse-transform output
    mult::CA                       # full-resolution multiply scratch
    u1::RA                         # full-resolution first-order modulus
    levels::LV                     # ds => Level, covering every ds the tree uses (including 1)
end

# Decimation factor for a first-order wavelet of octave `scale`.
_ds(scale::Int, oversampling::Int) = 1 << max(0, scale - oversampling)

# Orientations per scale, however the bank spells it — the second index of the 2D/3D container.
_orient_count(fb::FilterBanks.FilterBank1D) = fb.Q
_orient_count(fb::FilterBanks.FilterBank2D) = fb.L
_orient_count(fb::FilterBanks.FilterBank3D) = fb.n_orient

"""
    _build(T, dims, bank_at; J, max_order, oversampling, spectral) -> MultiResolutionScattering

Shared construction. `bank_at(dims)` builds the filter bank on a given grid, so the caller supplies
the dimension-specific bank and everything else is common.

Reduced-resolution banks go through the same builder as the full one, which is what applies the
tight-frame normalisation. Building them from bare frequency responses instead leaves out that
factor (measured `0.90` in 1D and `0.38` in 2D, i.e. not close to 1), which would rescale every
decimated second-order coefficient.
"""
function _build(::Type{T}, dims::NTuple{D, Int}, bank_at; J::Int, max_order::Int,
                oversampling::Int, spectral::SB.AbstractSpectralBackend) where {T, D}
    fb = bank_at(dims)
    tree = PathGraph.build_tree([m.j_eff for m in fb.meta], max_order)
    nw = length(fb.wavelets)
    groups = max_order >= 2 ? PathGraph.order2_groups(tree, nw) : [(j, Int[], Int[]) for j in 1:nw]
    plan_full = Plans.make_plan(spectral, T, dims)

    lvl(ds, wl, pl, rd) = Level(ds, wl, pl, Array{Complex{T}, D}(undef, rd),
                                Array{Complex{T}, D}(undef, rd), Array{Complex{T}, D}(undef, rd),
                                Array{Complex{T}, D}(undef, rd), Array{T, D}(undef, rd))
    LT = typeof(lvl(1, fb.wavelets, plan_full, dims))
    levels = Dict{Int, LT}()
    for m in fb.meta
        ds = _ds(m.scale, oversampling)
        haskey(levels, ds) && continue
        if ds == 1
            levels[1] = lvl(1, fb.wavelets, plan_full, dims)
            continue
        end
        rd = map(n -> n ÷ ds, dims)
        all(map((r, n) -> r * ds == n, rd, dims)) ||
            throw(ArgumentError("oversampling/scale gives ds=$ds, which does not divide dims=$dims"))
        levels[ds] = lvl(ds, bank_at(rd).wavelets, Plans.make_plan(spectral, T, rd), rd)
    end

    return MultiResolutionScattering{T, D, typeof(fb), typeof(tree), typeof(groups), typeof(plan_full),
                                Array{Complex{T}, D}, Array{T, D}, typeof(levels)}(
        dims, J, max_order, oversampling, fb, tree, groups, plan_full,
        Array{Complex{T}, D}(undef, dims), Array{Complex{T}, D}(undef, dims),
        Array{Complex{T}, D}(undef, dims), Array{Complex{T}, D}(undef, dims),
        Array{T, D}(undef, dims), levels)
end

"""
    SubsampledScattering1D([T=Float64,] N, J; Q=1, max_order=2, oversampling=1,
                           spectral=AutoSpectralBackend())

1D scattering with a multi-resolution second order: the first-order modulus envelope is decimated by
`2^(scale - oversampling)` before the second wavelet transform. Opt-in and approximate — a large
`oversampling` reproduces the exact
[`ScatteringTransform1D`](@ref ScatteringTransforms.Scattering1D.ScatteringTransform1D); aggressive
values trade a little accuracy for speed.
"""
SubsampledScattering1D(::Type{T}, N::Int, J::Int; Q::Int = 1, max_order::Int = 2,
                       oversampling::Int = 1,
                       spectral::SB.AbstractSpectralBackend = SB.AutoSpectralBackend()) where {T} =
    _build(T, (N,), d -> FilterBanks.build_filter_bank1d(T, d[1], J; Q = Q);
           J = J, max_order = max_order, oversampling = oversampling, spectral = spectral)
SubsampledScattering1D(N::Int, J::Int; kwargs...) = SubsampledScattering1D(Float64, N, J; kwargs...)

"""
    SubsampledScattering2D([T=Float64,] (Ny, Nx), J; L=8, max_order=2, oversampling=1,
                           spectral=AutoSpectralBackend())

2D counterpart of [`SubsampledScattering1D`](@ref). Decimating by `ds` shrinks the second order's
grid by `ds²`.
"""
SubsampledScattering2D(::Type{T}, N::NTuple{2, Int}, J::Int; L::Int = 8, max_order::Int = 2,
                       oversampling::Int = 1,
                       spectral::SB.AbstractSpectralBackend = SB.AutoSpectralBackend()) where {T} =
    _build(T, N, d -> FilterBanks.build_filter_bank2d(T, d, J; L = L);
           J = J, max_order = max_order, oversampling = oversampling, spectral = spectral)
SubsampledScattering2D(N::NTuple{2, Int}, J::Int; kwargs...) =
    SubsampledScattering2D(Float64, N, J; kwargs...)

"""
    SubsampledScattering3D([T=Float64,] (Nz, Ny, Nx), J; n_orient=6, max_order=2, oversampling=1,
                           spectral=AutoSpectralBackend())

3D counterpart of [`SubsampledScattering1D`](@ref). Decimating by `ds` shrinks the second order's
grid by `ds³`, which is where this path pays off most.
"""
SubsampledScattering3D(::Type{T}, N::NTuple{3, Int}, J::Int; n_orient::Int = 6,
                       max_order::Int = 2, oversampling::Int = 1,
                       spectral::SB.AbstractSpectralBackend = SB.AutoSpectralBackend()) where {T} =
    _build(T, N, d -> FilterBanks.build_filter_bank3d(T, d, J; n_orient = n_orient);
           J = J, max_order = max_order, oversampling = oversampling, spectral = spectral)
SubsampledScattering3D(N::NTuple{3, Int}, J::Int; kwargs...) =
    SubsampledScattering3D(Float64, N, J; kwargs...)

Base.show(io::IO, st::MultiResolutionScattering{T, D}) where {T, D} =
    print(io, "MultiResolutionScattering{", T, ",", D, "}(dims=", st.dims, ", J=", st.J,
          ", oversampling=", st.oversampling, ")")

# Decimation in space == periodisation in frequency: a strided read on every axis.
@inline _decimated(u::AbstractArray{<:Any, D}, ds::Int, rd::NTuple{D, Int}) where {D} =
    view(u, ntuple(d -> 1:ds:(1 + (rd[d] - 1) * ds), D)...)

"""
    subsampled_scattering!(coeffs, st, field) -> coeffs

In-place multi-resolution scattering into a pre-allocated coefficient container. Allocation-free.

Grouped by first-order wavelet, so each `U₁` is decimated and transformed **once** and reused by
every one of its children, rather than once per order-2 path.
"""
function subsampled_scattering!(coeffs, st::MultiResolutionScattering{T, D},
                                field::AbstractArray{<:Any, D}) where {T, D}
    st.xc .= complex.(field)
    Plans.forward_transform!(st.xfft, st.plan_full, st.xc)
    isempty(coeffs.S2) || fill!(coeffs.S2, zero(eltype(coeffs.S2)))
    wavelets, meta = st.filter_bank.wavelets, st.filter_bank.meta

    @inbounds for (j1, children, _) in st.groups
        ScatteringCore.wavelet_convolve!(st.conv, st.xfft, wavelets[j1], st.plan_full, st.mult)
        coeffs.S1[j1] = ScatteringCore.modulus_mean!(st.u1, st.conv)
        isempty(children) && continue

        lev = st.levels[_ds(meta[j1].scale, st.oversampling)]
        if lev.ds == 1
            copyto!(lev.u, st.u1)
        else
            copyto!(lev.u, _decimated(st.u1, lev.ds, size(lev.u)))
        end
        lev.uc .= complex.(lev.u)
        Plans.forward_transform!(lev.uf, lev.plan, lev.uc)
        for j2 in children
            ScatteringCore.wavelet_convolve!(lev.conv, lev.uf, lev.wavelets[j2],
                                             lev.plan, lev.mult)
            coeffs.S2[j1, j2] = ScatteringCore.modulus_mean(lev.conv)
        end
    end
    return Coefficients.update_S0(coeffs, ScatteringCore.spatial_average(field))
end

# 1D reports scales along one axis; 2D and 3D report scales × orientations, as their exact
# counterparts do.
coeff_container(st::MultiResolutionScattering{T, 1}) where {T} =
    Coefficients.ScatteringCoefficients1D(length(st.filter_bank.wavelets), T;
                                          compute_S2 = st.max_order >= 2)
coeff_container(st::MultiResolutionScattering{T}) where {T} =
    Coefficients.ScatteringCoefficients2D(st.filter_bank.J, _orient_count(st.filter_bank), T;
                                          compute_S2 = st.max_order >= 2)

(st::MultiResolutionScattering{T, D})(field::AbstractArray{<:Any, D}) where {T, D} =
    subsampled_scattering!(coeff_container(st), st, field)

# Shares the read-only banks, tree and plan tables; copies only the buffers.
function ScatteringCore.task_workspace(st::MultiResolutionScattering)
    lv = typeof(st.levels)()
    for (ds, l) in st.levels
        lv[ds] = Level(l.ds, l.wavelets, Plans.task_local_plan(l.plan), similar(l.uc),
                       similar(l.uf), similar(l.conv), similar(l.mult), similar(l.u))
    end
    return typeof(st)(st.dims, st.J, st.max_order, st.oversampling, st.filter_bank, st.tree,
                      st.groups, Plans.task_local_plan(st.plan_full), similar(st.xc),
                      similar(st.xfft), similar(st.conv), similar(st.mult), similar(st.u1), lv)
end

end # module SubsampledScattering
