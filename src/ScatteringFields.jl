module ScatteringFields

# The localized scattering transform holds, for each path `p`, the low-passed and subsampled field
#
#     S_p x = (|U_p x| ⋆ φ_J) ↓ s
#
# at resolution `N / s`, in one array whose trailing axis is the scattering tree's path id. Its
# spatial mean is the global coefficient of `p`, which under `↓ s` needs `φ̂_J(0) = 1` and `φ̂_J` to
# vanish on the subsampling lattice; `φ_J` is `Filters.gaussian_lowpass!`, as is every bank's
# `averaging`. A field owns the cascade workspace that fills it, so refilling it allocates nothing.

using ..PathGraph: PathGraph

export ScatteringField1D, ScatteringField2D, ScatteringField3D, path_field, subsample_factor
export scattering_field, scattering_field!

"""
    scattering_field(st, x; subsample) -> ScatteringField{1,2,3}D

Localized (Mallat) scattering transform: returns the per-path low-passed, subsampled fields.
Methods are added by the per-dimension transform modules.
"""
function scattering_field end

"""
    scattering_field!(field, st, x) -> field

In-place localized scattering transform into a pre-allocated `ScatteringField`.
"""
function scattering_field! end

"""
    ScatteringField1D{T,A,Tree,WS}

Localized 1D scattering field. `data` is `(M, npaths)`; column `p` is the localized field of
path `p` at the subsampled resolution `M = N ÷ s`. `ws` is the `Cascade.FieldWorkspace` that fills
it.
"""
struct ScatteringField1D{T, A<:AbstractMatrix{T}, Tree<:PathGraph.ScatteringTree, WS}
    tree::Tree
    data::A
    subsample::Int
    ws::WS
end

"""
    ScatteringField2D{T,A,Tree,WS}

Localized 2D scattering field. `data` is `(My, Mx, npaths)`; slice `[:, :, p]` is the localized
field of path `p` at subsampled resolution `(My, Mx) = (Ny, Nx) ÷ s`.
"""
struct ScatteringField2D{T, A<:AbstractArray{T,3}, Tree<:PathGraph.ScatteringTree, WS}
    tree::Tree
    data::A
    subsample::Int
    ws::WS
end

"""
    ScatteringField3D

Localized volumetric scattering with `data` shaped `(Mz, My, Mx, npaths)`.
"""
struct ScatteringField3D{T, A<:AbstractArray{T,4}, Tree<:PathGraph.ScatteringTree, WS}
    tree::Tree
    data::A
    subsample::Int
    ws::WS
end

field_container(tree, data::AbstractMatrix, subsample, ws) = ScatteringField1D(tree, data, subsample, ws)
field_container(tree, data::AbstractArray{<:Any,3}, subsample, ws) = ScatteringField2D(tree, data, subsample, ws)
field_container(tree, data::AbstractArray{<:Any,4}, subsample, ws) = ScatteringField3D(tree, data, subsample, ws)

function default_subsample(dims::Tuple, J::Int)
    s = 1 << max(0, J - 1)
    while s > 1 && !_divides_dims(dims, s)
        s >>= 1
    end
    return s
end

_divides_dims(dims, s) = all(n -> n % s == 0, dims)

"""
    path_field(sf, p) -> view

View of path `p`'s localized field, with the transform's spatial rank.
"""
@inline path_field(sf::ScatteringField1D, p::Integer) = view(sf.data, :, p)
@inline path_field(sf::ScatteringField2D, p::Integer) = view(sf.data, :, :, p)
@inline path_field(sf::ScatteringField3D, p::Integer) = view(sf.data, :, :, :, p)

"""
    subsample_factor(sf) -> Int

The decimation factor `s` applied to produce the field resolution.
"""
@inline subsample_factor(sf::Union{ScatteringField1D,ScatteringField2D,ScatteringField3D}) = sf.subsample

end # module ScatteringFields
