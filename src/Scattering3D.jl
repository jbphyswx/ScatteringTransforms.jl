module Scattering3D

"""
    Scattering3D.jl — 3D volumetric scattering transform

Oriented 3D Morlet wavelets (`J` scales × `n_orient` sphere directions). Reuses the
dimension-agnostic core ops (`wavelet_convolve!`, `apply_modulus!`, `spatial_average`), the
scattering path tree (second order = strictly coarser scale, all orientation pairs), and the
scales×orientations coefficient container.
"""

using ..Plans: Plans
using ..FilterBanks: FilterBanks
using ..ScatteringCore: ScatteringCore
using ..Coefficients: Coefficients
using ..PathGraph: PathGraph

export ScatteringTransform3D, scattering_transform3d!
export compute_S1_3d!, compute_S2_3d!

struct ScatteringTransform3D{T,M<:AbstractArray{Complex{T},3},R<:AbstractArray{T,3},P<:Plans.AbstractScatteringPlan,Tree<:PathGraph.ScatteringTree}
    filter_bank::FilterBanks.FilterBank3D{T,M}
    tree::Tree
    max_order::Int
    plan::P
    buffer_input::M
    buffer_signal_fft::M
    buffer_conv::M
    buffer_mod::R
    U1_buffers::Vector{R}
    U1_fft_buffers::Vector{M}

    function ScatteringTransform3D(N::NTuple{3,Int}, J::Int;
                                   n_orient::Int=6,
                                   max_order::Int=2,
                                   T::Type=Float64,
                                   spectral::Symbol=:auto)
        filter_bank = FilterBanks.build_filter_bank3d(N, J; n_orient=n_orient, T=T)
        tree = PathGraph.build_tree([m.j_eff for m in filter_bank.meta], max_order)
        plan = Plans.make_plan(spectral, T, N)

        dummy = zeros(Complex{T}, N)
        num_w = length(filter_bank.wavelets)
        buffer_input      = similar(dummy)
        buffer_signal_fft = similar(dummy)
        buffer_conv       = similar(dummy)
        buffer_mod        = zeros(T, N)

        if max_order >= 2
            U1_buffers     = [zeros(T, N) for _ in 1:num_w]
            U1_fft_buffers = [similar(dummy) for _ in 1:num_w]
        else
            U1_buffers     = Array{T,3}[]
            U1_fft_buffers = Array{Complex{T},3}[]
        end

        new{T, typeof(buffer_conv), typeof(buffer_mod), typeof(plan), typeof(tree)}(
            filter_bank, tree, max_order, plan,
            buffer_input, buffer_signal_fft, buffer_conv, buffer_mod,
            U1_buffers, U1_fft_buffers,
        )
    end
end

"""
    (st::ScatteringTransform3D)(volume) -> ScatteringCoefficients2D

Apply the 3D scattering transform. (Coefficients use the scales×orientations container.)
"""
function (st::ScatteringTransform3D)(volume::AbstractArray{<:Any,3})
    J = st.filter_bank.J
    n_orient = st.filter_bank.n_orient
    T = eltype(st.buffer_mod)
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
    compute_S1_3d!(coeffs.S1, st, st.buffer_signal_fft)
    if st.max_order >= 2
        compute_S2_3d!(coeffs.S2, st, st.buffer_signal_fft)
    end
    S0_val = ScatteringCore.spatial_average(volume)
    return Coefficients.update_S0(coeffs, S0_val)
end

function compute_S1_3d!(S1::AbstractVector, st::ScatteringTransform3D, vol_fft::AbstractArray{<:Any,3})
    @inbounds for (j, ψ_fft) in enumerate(st.filter_bank.wavelets)
        ScatteringCore.wavelet_convolve!(st.buffer_conv, vol_fft, ψ_fft, st.plan, st.buffer_input)
        ScatteringCore.apply_modulus!(st.buffer_mod, st.buffer_conv)
        S1[j] = ScatteringCore.spatial_average(st.buffer_mod)
    end
    return S1
end

function compute_S2_3d!(S2::AbstractMatrix, st::ScatteringTransform3D, vol_fft::AbstractArray{<:Any,3})
    num_w = length(st.filter_bank.wavelets)
    @inbounds for (j1, ψ1_fft) in enumerate(st.filter_bank.wavelets)
        ScatteringCore.wavelet_convolve!(st.buffer_conv, vol_fft, ψ1_fft, st.plan, st.buffer_input)
        ScatteringCore.apply_modulus!(st.U1_buffers[j1], st.buffer_conv)
    end
    @inbounds for j1 in 1:num_w
        st.buffer_input .= complex.(st.U1_buffers[j1])
        Plans.forward_transform!(st.U1_fft_buffers[j1], st.plan, st.buffer_input)
    end
    tree = st.tree
    @inbounds for p in PathGraph.order_range(tree, 2)
        idx = PathGraph.path_indices(tree, p)
        j1, j2 = idx[1], idx[2]
        ScatteringCore.wavelet_convolve!(st.buffer_conv, st.U1_fft_buffers[j1],
            st.filter_bank.wavelets[j2], st.plan, st.buffer_input)
        ScatteringCore.apply_modulus!(st.buffer_mod, st.buffer_conv)
        S2[j1, j2] = ScatteringCore.spatial_average(st.buffer_mod)
    end
    return S2
end

end # module Scattering3D
