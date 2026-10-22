module PathGraph

# A path `(λ₁, λ₂, …)` of wavelet indices is admissible iff its effective log-scale strictly
# increases, `j_eff(λ_{k+1}) > j_eff(λ_k)`, so each wavelet is strictly coarser than the last. In 1D
# this is `j₂ > j₁`; in 2D and 3D it admits every orientation pair across strictly coarser scales and
# excludes same-scale pairs, which share `j_eff`.

export ScatteringTree, build_tree, npaths, path_indices, order_range, order2_groups

"""
    ScatteringTree{IV,RV}

Flat CSR description of the scattering tree: wavelet indices into the filter bank and per-path
order labels. The containers are parametric; [`build_tree`](@ref) produces `Vector{Int}` and
`Vector{UnitRange{Int}}`.

# Fields
- `path_data`: concatenated wavelet-index lists for all paths
- `path_ptr`: CSR offsets (length `npaths+1`); path `p` is `path_data[path_ptr[p]:path_ptr[p+1]-1]`
- `order`: scattering order of each path (`0, 1, 2, …`)
- `by_order`: `by_order[o+1]` is the contiguous range of path ids of order `o`
"""
struct ScatteringTree{IV<:AbstractVector{<:Integer}, RV<:AbstractVector{<:AbstractUnitRange{<:Integer}}}
    path_data::IV
    path_ptr::IV
    order::IV
    by_order::RV
end

"""
    npaths(tree) -> Int

Total number of paths (including the order-0 root).
"""
npaths(t::ScatteringTree) = length(t.order)

"""
    path_indices(tree, p) -> view

Wavelet indices of path `p` (empty for the order-0 root), as a non-allocating view.
"""
@inline path_indices(t::ScatteringTree, p::Integer) =
    view(t.path_data, t.path_ptr[p]:(t.path_ptr[p + 1] - 1))

"""
    order_range(tree, o) -> UnitRange

Contiguous range of path ids with scattering order `o`.
"""
@inline order_range(t::ScatteringTree, o::Integer) = t.by_order[o + 1]

"""
    build_tree(j_eff, max_order) -> ScatteringTree

Enumerate all admissible paths up to `max_order` from the per-wavelet effective log-scales
`j_eff` (typically `[m.j_eff for m in filter_bank.meta]`). Admissibility: `j_eff` strictly
increasing along the path. Paths are laid out grouped by order (order 0, then 1, then 2, …),
so `by_order` ranges are contiguous.
"""
function build_tree(j_eff::AbstractVector, max_order::Integer)
    n = length(j_eff)
    # levels[o+1] = list of paths (index vectors) of order o
    levels = Vector{Vector{Vector{Int}}}()
    push!(levels, [Int[]])                       # order 0: the root (empty path)
    for _ in 1:max_order
        prev = levels[end]
        cur = Vector{Vector{Int}}()
        for path in prev
            if isempty(path)
                for i in 1:n
                    push!(cur, [i])
                end
            else
                last_jeff = j_eff[path[end]]
                for i in 1:n
                    if j_eff[i] > last_jeff
                        push!(cur, vcat(path, i))
                    end
                end
            end
        end
        push!(levels, cur)
    end

    # Flatten to CSR, grouped by order.
    path_data = Int[]
    path_ptr = Int[1]
    order = Int[]
    by_order = UnitRange{Int}[]
    pid = 0
    for o in 0:max_order
        start = pid + 1
        for path in levels[o + 1]
            append!(path_data, path)
            push!(path_ptr, length(path_data) + 1)
            push!(order, o)
            pid += 1
        end
        push!(by_order, start:pid)
    end

    return ScatteringTree(path_data, path_ptr, order, by_order)
end

"""
    order2_groups(tree, nw) -> Vector{Tuple{Int,Vector{Int},Vector{Int}}}

Every first-order wavelet `j1` with its admissible order-2 children `j2` and the tree path ids of
those `(j1, j2)` pairs, longest group first.

The cascade computes each first-order modulus once per group, and the localized-field output is
indexed by the path ids. Longest-first order balances the parallel backends, since the child count
varies across scales.
"""
function order2_groups(tree::ScatteringTree, nw::Integer)
    children = [Int[] for _ in 1:nw]
    pathids = [Int[] for _ in 1:nw]
    if length(tree.by_order) >= 3
        for p in order_range(tree, 2)
            idx = path_indices(tree, p)
            push!(children[idx[1]], idx[2])
            push!(pathids[idx[1]], p)
        end
    end
    groups = [(j1, children[j1], pathids[j1]) for j1 in 1:nw]
    sort!(groups; by = g -> -length(g[2]))
    return groups
end

end # module PathGraph
