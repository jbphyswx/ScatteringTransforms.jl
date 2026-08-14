module Scattering3D

"""
    Scattering3D.jl — 3D volumetric scattering transform

Oriented 3D Morlet wavelets (`J` scales × `n_orient` sphere directions). Reuses the
dimension-agnostic core ops (`wavelet_convolve!`, `apply_modulus!`, `spatial_average`), the
scattering path tree (second order = strictly coarser scale, all orientation pairs), and the
scales×orientations coefficient container.
"""

using ..Plans: Plans
using ComputationalBackends: ComputationalBackends as CB
using SpectralBackends: SpectralBackends as SB
using ..FilterBanks: FilterBanks
using ..ScatteringCore: ScatteringCore
using ..Coefficients: Coefficients
using ..PathGraph: PathGraph

export ScatteringTransform3D, scattering_transform3d!, cascade!

struct ScatteringTransform3D{T, M<:AbstractArray{Complex{T},3}, R<:AbstractArray{T,3},
                             P<:Plans.AbstractScatteringPlan, Tree<:PathGraph.ScatteringTree,
                             FB<:FilterBanks.FilterBank3D, G<:AbstractVector}
    filter_bank::FB
    tree::Tree
    groups::G               # (j1, children, path ids) from the tree, longest-first
    max_order::Int
    plan::P
    buffer_input::M
    buffer_signal_fft::M
    buffer_conv::M
    buffer_mod::R
    buffer_u1::R            # first-order modulus of the current j1
    buffer_u1_fft::M        # its spectrum, reused across that j1's children
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
                               spectral::SB.AbstractSpectralBackend=SB.AutoSpectralBackend()) where {T}
    filter_bank = FilterBanks.build_filter_bank3d(T, N, J; n_orient=n_orient)
    tree = PathGraph.build_tree([m.j_eff for m in filter_bank.meta], max_order)
    groups = max_order >= 2 ? PathGraph.order2_groups(tree, length(filter_bank.wavelets)) :
             [(j, Int[], Int[]) for j in 1:length(filter_bank.wavelets)]
    plan = Plans.make_plan(spectral, T, N)

    dummy = zeros(Complex{T}, N)
    # O(prod(N)) workspace, not O(nw·prod(N)): one first-order volume and one spectrum are live at
    # a time, which matters most in 3D where a single volume is already large.
    return ScatteringTransform3D(filter_bank, tree, groups, max_order, plan,
                                 similar(dummy), similar(dummy), similar(dummy),
                                 zeros(T, N), zeros(T, N), similar(dummy))
end
ScatteringTransform3D(N::NTuple{3,Int}, J::Int; kwargs...) =
    ScatteringTransform3D(Float64, N, J; kwargs...)

# Shares filter bank / tree / groups; copies only the buffers and the plan's scratch.
ScatteringCore.task_workspace(st::ScatteringTransform3D) =
    ScatteringTransform3D(st.filter_bank, st.tree, st.groups, st.max_order,
                          Plans.task_local_plan(st.plan),
                          similar(st.buffer_input), similar(st.buffer_signal_fft),
                          similar(st.buffer_conv), similar(st.buffer_mod),
                          similar(st.buffer_u1), similar(st.buffer_u1_fft))

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
function cascade!(S1::AbstractVector, S2::AbstractMatrix, st::ScatteringTransform3D,
                  vol_fft::AbstractArray{<:Any,3})
    isempty(S2) || fill!(S2, zero(eltype(S2)))
    wavelets = st.filter_bank.wavelets
    @inbounds for (j1, children, _) in st.groups
        ScatteringCore.wavelet_convolve!(st.buffer_conv, vol_fft, wavelets[j1],
                                         st.plan, st.buffer_input)
        if isempty(children)
            S1[j1] = ScatteringCore.modulus_mean(st.buffer_conv)
            continue
        end
        S1[j1] = ScatteringCore.modulus_mean!(st.buffer_u1, st.buffer_conv)
        st.buffer_input .= complex.(st.buffer_u1)
        Plans.forward_transform!(st.buffer_u1_fft, st.plan, st.buffer_input)
        for j2 in children
            ScatteringCore.wavelet_convolve!(st.buffer_conv, st.buffer_u1_fft, wavelets[j2],
                                             st.plan, st.buffer_input)
            S2[j1, j2] = ScatteringCore.modulus_mean(st.buffer_conv)
        end
    end
    return S1, S2
end

# ============================================================================
# Non-mutating, autodiff-friendly forward — see Scattering1D for the rationale.
# ============================================================================

function ScatteringCore.scattering(st::ScatteringTransform3D, volume::AbstractArray{<:Any,3})
    plan = st.plan
    fb = st.filter_bank
    tree = st.tree
    n = length(fb.wavelets)

    Xf = Plans.forward_transform(plan, complex.(volume))
    U1 = map(ψ -> abs.(Plans.inverse_transform(plan, Xf .* ψ)), fb.wavelets)
    S1 = map(u -> sum(u) / length(u), U1)
    S0 = sum(volume) / length(volume)

    if st.max_order >= 2 && length(tree.by_order) >= 3
        U1f = map(u -> Plans.forward_transform(plan, complex.(u)), U1)
        r2 = PathGraph.order_range(tree, 2)
        s2vals = map(collect(r2)) do p
            idx = PathGraph.path_indices(tree, p)
            m = abs.(Plans.inverse_transform(plan, U1f[idx[1]] .* fb.wavelets[idx[2]]))
            sum(m) / length(m)
        end
        pos = zeros(Int, n, n)
        for (k, p) in enumerate(r2)
            idx = PathGraph.path_indices(tree, p)
            pos[idx[1], idx[2]] = k
        end
        Tc = eltype(s2vals)
        S2 = [pos[j1, j2] == 0 ? zero(Tc) : s2vals[pos[j1, j2]] for j1 in 1:n, j2 in 1:n]
    else
        S2 = Matrix{eltype(S1)}(undef, 0, 0)
    end
    return Coefficients.ScatteringCoefficients2D(S1, S2; S0=S0,
        n_scales=fb.J, n_orientations=fb.n_orient)
end

end # module Scattering3D
