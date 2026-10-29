module Scattering3D

using ..Plans: Plans
using ComputationalBackends: ComputationalBackends as CB
using SpectralBackends: SpectralBackends as SB
using ..FilterBanks: FilterBanks
using ..ScatteringCore: ScatteringCore
using ..Coefficients: Coefficients
using ..PathGraph: PathGraph
using ..ScatteringFields: ScatteringFields
using ..Cascade: Cascade

export scattering_transform3d!, cascade!

"""
    ScatteringCore.ScatteringTransform{3}([T=Float64,] N, J; n_orient=6, max_order=2, filters=StoredFilters(),
                          oversampling=J, transform=AutoSpectralBackend())

Build a 3D volumetric scattering transform for `N = (Nz, Ny, Nx)` volumes over `J` scales and
`n_orient` sphere directions. The element type is positional, as for `zeros(T, …)`; omit it for
`Float64`. `filters`, `oversampling` and `transform` are as for `ScatteringCore.ScatteringTransform{1}`.
"""
function ScatteringCore.ScatteringTransform{3}(::Type{T}, N::NTuple{3,Int}, J::Int;
                               n_orient::Int=6,
                               max_order::Int=2,
                               filters::FilterBanks.AbstractFilterStorage=FilterBanks.StoredFilters(),
                               oversampling::Int=J,
                               transform::SB.AbstractSpectralBackend=SB.AutoSpectralBackend()) where {T}
    filter_bank = FilterBanks.build_filter_bank3d(T, N, J, filters; n_orient=n_orient)
    tree = PathGraph.build_tree([m.j_eff for m in filter_bank.meta], max_order)
    nw = FilterBanks.nwavelets(filter_bank)
    groups = max_order >= 2 ? PathGraph.order2_groups(tree, nw) :
             [(j, Int[], Int[]) for j in 1:nw]
    plan = Plans.make_plan(transform, T, N)
    pw = Cascade.build(filter_bank, groups, N, T, oversampling, plan)
    return ScatteringCore.ScatteringTransform{3}(filter_bank, tree, groups, max_order, plan,
                                 zeros(Complex{T}, N), zeros(Complex{T}, N), N, pw)
end
"""
    (st::ScatteringCore.ScatteringTransform{3})(volume) -> ScatteringCoefficients2D

The scattering coefficients of `volume`, in a new scales × orientations container.
"""
function (st::ScatteringCore.ScatteringTransform{3})(volume::AbstractArray{<:Any,3})
    J = st.filter_bank.J
    n_orient = st.filter_bank.n_orient
    T = real(eltype(st.buffer_input))
    coeffs = Coefficients.ScatteringCoefficients2D(J, n_orient, T; compute_S2 = st.max_order >= 2)
    return scattering_transform3d!(coeffs, st, volume)
end

"""
    scattering_transform3d!(coeffs, st, volume; execution = SerialBackend()) -> coeffs

In-place 3D volumetric scattering transform; fills `coeffs` (a scales×orientations container)
and returns it with `S0` updated. `execution` is as for `Scattering2D.scattering_transform2d!`.
"""
function scattering_transform3d!(coeffs::Coefficients.ScatteringCoefficients2D,
                                 st::ScatteringCore.ScatteringTransform{3}, volume::AbstractArray{<:Any,3};
                                 execution::CB.AbstractExecutionBackend = CB.SerialBackend())
    st.buffer_input .= complex.(volume)
    Plans.forward_transform!(st.buffer_signal_fft, st.plan, st.buffer_input)
    cascade!(coeffs.S1, coeffs.S2, execution, st, st.buffer_signal_fft)
    return Coefficients.update_S0(coeffs, ScatteringCore.spatial_average(volume))
end

cascade!(S1::AbstractVector, S2::AbstractMatrix, ::CB.AbstractSerialBackend,
         st::ScatteringCore.ScatteringTransform{3}, vol_fft::AbstractArray{<:Any,3}) = cascade!(S1, S2, st, vol_fft)

"""
    cascade!(S1, S2, st, vol_fft) -> (S1, S2)

Both scattering orders in one grouped pass, as in the 1D `cascade!`.
"""
cascade!(S1::AbstractVector, S2::AbstractMatrix, st::ScatteringCore.ScatteringTransform{3},
         vol_fft::AbstractArray{<:Any,3}) =
    Cascade.cascade!(S1, S2, st.pw, st.filter_bank, st.groups, vol_fft)

function ScatteringFields.scattering_field(st::ScatteringCore.ScatteringTransform{3}, volume::AbstractArray{<:Any,3};
        subsample::Int = ScatteringFields.default_subsample(st.dims, st.filter_bank.J))
    all(n -> n % subsample == 0, st.dims) ||
        throw(ArgumentError("subsample must divide each volume dimension"))
    T = promote_type(real(eltype(st.buffer_input)), eltype(volume))
    dims = map(n -> n ÷ subsample, st.dims)
    data = fill!(similar(st.buffer_input, T, (dims..., PathGraph.npaths(st.tree))), zero(T))
    field = ScatteringFields.ScatteringField3D(st.tree, data, subsample,
                                                Cascade.field_workspace(st, subsample))
    return ScatteringFields.scattering_field!(field, st, volume)
end

function ScatteringFields.scattering_field!(field::ScatteringFields.ScatteringField3D,
        st::ScatteringCore.ScatteringTransform{3}, volume::AbstractArray{<:Any,3})
    st.buffer_input .= complex.(volume)
    Plans.forward_transform!(st.buffer_signal_fft, st.plan, st.buffer_input)
    Cascade.field_cascade!(field.data, field.ws, st.filter_bank, st.groups, st.buffer_signal_fft,
                           first(PathGraph.order_range(st.tree, 0)),
                           first(PathGraph.order_range(st.tree, 1)))
    return field
end

# ============================================================================
# Non-mutating forward for autodiff; see `ScatteringCore.scattering`.
# ============================================================================

function ScatteringCore.scattering(st::ScatteringCore.ScatteringTransform{3}, volume::AbstractArray{<:Any,3})
    S0, S1, S2 = Cascade.scattering_values(st, volume)
    fb = st.filter_bank
    return Coefficients.ScatteringCoefficients2D(S1, S2; S0=S0,
        n_scales=fb.J, n_orientations=fb.n_orient)
end

end # module Scattering3D
