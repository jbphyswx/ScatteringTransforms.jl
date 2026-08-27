module ScatteringFields

"""
    ScatteringFields.jl — Localized (Mallat) scattering field storage

Unlike the globally-averaged coefficients (one scalar per path), the *localized* scattering
transform returns, for each path `p`, the low-pass-smoothed, subsampled field

    S_p x = (|U_p x| ⋆ φ_J) ↓ s

at resolution `N / s` (`s` = subsample factor). Every path shares the same output resolution
(set by the φ_J low-pass), so the natural storage is a single dense array with a trailing
"path" axis indexed by the scattering tree's path ids (order-0 root first, then order 1, …).

The globally-averaged coefficient of a path is the spatial mean of its localized field. Keeping that
true under `↓ s` requires `φ̂_J(0) = 1` *and* `φ̂_J` to vanish on the subsampling lattice, which is why
`φ_J` is [`Filters.gaussian_lowpass!`](@ref) and not the bank's `averaging`.

A field owns the cascade workspace it was built for, so filling it repeatedly allocates nothing.
"""

using ..PathGraph: PathGraph

export ScatteringField1D, ScatteringField2D, path_field, subsample_factor
export scattering_field, scattering_field!

"""
    scattering_field(st, x; subsample) -> ScatteringField{1,2}D

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
    path_field(sf, p) -> view

Non-allocating view of path `p`'s localized field (a vector for 1D, a matrix for 2D).
"""
@inline path_field(sf::ScatteringField1D, p::Integer) = view(sf.data, :, p)
@inline path_field(sf::ScatteringField2D, p::Integer) = view(sf.data, :, :, p)

"""
    subsample_factor(sf) -> Int

The decimation factor `s` applied to produce the field resolution.
"""
@inline subsample_factor(sf::Union{ScatteringField1D,ScatteringField2D}) = sf.subsample

end # module ScatteringFields
