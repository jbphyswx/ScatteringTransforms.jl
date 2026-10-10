module Scattering3D

"""
    Scattering3D.jl — 3D volumetric scattering transform

Oriented 3D Morlet wavelets (`J` scales × `n_orient` sphere directions), run through the periodized
cascade (`Cascade`) over the scattering path tree (second order = strictly coarser scale, all
orientation pairs), with the scales×orientations coefficient container.
"""

using ..Plans: Plans
using ComputationalBackends: ComputationalBackends as CB
using SpectralBackends: SpectralBackends as SB
using ..FilterBanks: FilterBanks
using ..ScatteringCore: ScatteringCore
using ..Coefficients: Coefficients
using ..PathGraph: PathGraph
using ..Cascade: Cascade

export ScatteringTransform3D, scattering_transform3d!, cascade!

"""
    ScatteringTransform3D{T,M,P,Tree,FB,G,PW}

3D volumetric scattering transform; the fields are those of [`ScatteringTransform2D`](@ref), with
`dims = (Nz, Ny, Nx)`.
"""
struct ScatteringTransform3D{T, M<:AbstractArray{Complex{T},3},
                             P<:Plans.AbstractScatteringPlan, Tree<:PathGraph.ScatteringTree,
                             FB, G<:AbstractVector, PW}
    filter_bank::FB
    tree::Tree
    groups::G
    max_order::Int
    plan::P
    buffer_input::M
    buffer_signal_fft::M
    dims::NTuple{3,Int}
    pw::PW
end

"""
    ScatteringTransform3D([T=Float64,] N, J; n_orient=6, max_order=2, spectral=AutoSpectralBackend())

Build a 3D volumetric scattering transform for `N = (Nz, Ny, Nx)` volumes over `J` scales and
`n_orient` sphere directions. The element type is positional, as for `zeros(T, …)`; omit it for
`Float64`.
"""
function ScatteringTransform3D(::Type{T}, N::NTuple{3,Int}, J::Int;
                               n_orient::Int=6,
                               max_order::Int=2,
                               cache::Bool=true,
                               oversampling::Int=J,
                               spectral::SB.AbstractSpectralBackend=SB.AutoSpectralBackend()) where {T}
    filter_bank = FilterBanks.build_filter_bank3d(T, N, J, Val(cache); n_orient=n_orient)
    tree = PathGraph.build_tree([m.j_eff for m in filter_bank.meta], max_order)
    nw = FilterBanks.nwavelets(filter_bank)
    groups = max_order >= 2 ? PathGraph.order2_groups(tree, nw) :
             [(j, Int[], Int[]) for j in 1:nw]
    plan = Plans.make_plan(spectral, T, N)
    pw = Cascade.build(filter_bank, groups, N, T, oversampling, plan)
    return ScatteringTransform3D(filter_bank, tree, groups, max_order, plan,
                                 zeros(Complex{T}, N), zeros(Complex{T}, N), N, pw)
end
ScatteringTransform3D(N::NTuple{3,Int}, J::Int; kwargs...) =
    ScatteringTransform3D(Float64, N, J; kwargs...)

Base.show(io::IO, st::ScatteringTransform3D{T}) where {T} =
    print(io, "ScatteringTransform3D{", T, "}(N=", st.dims, ", J=", st.filter_bank.J, ", n_orient=",
          st.filter_bank.n_orient, ", max_order=", st.max_order, ", oversampling=", st.pw.oversampling,
          ", ", st.plan, ")")

# Shares filter bank / tree / groups; copies only the buffers and the plans' scratch. The workspace's
# level-1 plan is the transform's own.
function ScatteringCore.task_workspace(st::ScatteringTransform3D)
    fb = FilterBanks.task_bank(st.filter_bank)
    pw = Cascade.task_copy(st.pw, fb)
    return ScatteringTransform3D(fb, st.tree, st.groups, st.max_order, first(pw.plans),
                                 similar(st.buffer_input), similar(st.buffer_signal_fft), st.dims, pw)
end

"""
    (st::ScatteringTransform3D)(volume) -> ScatteringCoefficients2D

Apply the 3D scattering transform. (Coefficients use the scales×orientations container.)
"""
function (st::ScatteringTransform3D)(volume::AbstractArray{<:Any,3})
    J = st.filter_bank.J
    n_orient = st.filter_bank.n_orient
    T = real(eltype(st.buffer_input))
    coeffs = Coefficients.ScatteringCoefficients2D(J, n_orient, T; compute_S2 = st.max_order >= 2)
    return scattering_transform3d!(coeffs, st, volume)
end

"""
    scattering_transform3d!(coeffs, st, volume) -> coeffs

In-place 3D volumetric scattering transform; fills `coeffs` (a scales×orientations container)
and returns it with `S0` updated.
"""
function scattering_transform3d!(coeffs::Coefficients.ScatteringCoefficients2D,
                                 st::ScatteringTransform3D,
                                 volume::AbstractArray{<:Any,3})
    st.buffer_input .= complex.(volume)
    Plans.forward_transform!(st.buffer_signal_fft, st.plan, st.buffer_input)
    cascade!(coeffs.S1, coeffs.S2, st, st.buffer_signal_fft)
    return Coefficients.update_S0(coeffs, ScatteringCore.spatial_average(volume))
end

"""
    scattering_transform3d!(coeffs, backend, st, volume)

Transform one volume on an explicit execution backend — see the 2D counterpart.
"""
function scattering_transform3d!(coeffs::Coefficients.ScatteringCoefficients2D,
                                 backend::CB.AbstractExecutionBackend,
                                 st::ScatteringTransform3D, volume::AbstractArray{<:Any,3})
    st.buffer_input .= complex.(volume)
    Plans.forward_transform!(st.buffer_signal_fft, st.plan, st.buffer_input)
    cascade!(coeffs.S1, coeffs.S2, backend, st, st.buffer_signal_fft)
    return Coefficients.update_S0(coeffs, ScatteringCore.spatial_average(volume))
end

cascade!(S1::AbstractVector, S2::AbstractMatrix, ::CB.AbstractSerialBackend,
         st::ScatteringTransform3D, vol_fft::AbstractArray{<:Any,3}) = cascade!(S1, S2, st, vol_fft)

"""
    cascade!(S1, S2, st, vol_fft) -> (S1, S2)

Both scattering orders in one grouped pass — see the 1D `cascade!` for the scheme.
"""
cascade!(S1::AbstractVector, S2::AbstractMatrix, st::ScatteringTransform3D,
         vol_fft::AbstractArray{<:Any,3}) =
    Cascade.cascade!(S1, S2, st.pw, st.filter_bank, st.groups, vol_fft)

# ============================================================================
# Non-mutating, autodiff-friendly forward — see Scattering1D for the rationale.
# ============================================================================

function ScatteringCore.scattering(st::ScatteringTransform3D, volume::AbstractArray{<:Any,3})
    S0, S1, S2 = Cascade.scattering_values(st, volume)
    fb = st.filter_bank
    return Coefficients.ScatteringCoefficients2D(S1, S2; S0=S0,
        n_scales=fb.J, n_orientations=fb.n_orient)
end

end # module Scattering3D
