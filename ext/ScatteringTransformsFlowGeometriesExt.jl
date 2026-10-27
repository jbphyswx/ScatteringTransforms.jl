module ScatteringTransformsFlowGeometriesExt

# A FlowGeometries grid as the point set of a scattered-planar transform: the grid fixes the domain the
# points sample, and with it the physical scale of every mode.

using FlowGeometries: FlowGeometries as FG
using ScatteringTransforms: ScatteringTransforms as ST

"""
    scattered_planar_scattering([T,] grid::FlowGeometries.Grids.AbstractGrid, ms, J; kwargs...)

The scattered-planar transform of fields on the active cells of a 2-D Cartesian `grid`, taken in the
grid's linear cell order. The cells' coordinates are the points, `Grids.domain_length(grid, d)` the
period and the lowest cell centre the origin along each axis, and the cell measure the default
`weights`. `T` defaults to the grid's coordinate type; the other keywords are those of the point form.
"""
function ST.scattered_planar_scattering(::Type{T}, grid::FG.Grids.AbstractGrid, ms::NTuple{2, Int},
                                        J::Int; weights = nothing, kwargs...) where {T}
    n = FG.Grids.ncoordinates(grid)
    n == 2 || throw(ArgumentError("a planar scattering transform takes a grid of 2 coordinates; got $n"))
    geo = FG.Grids.grid_geometry(grid)
    geo isa FG.Geometry.AbstractCartesianGeometry || throw(ArgumentError(
        "a planar scattering transform takes a Cartesian grid; for a field on the sphere use " *
        "`spherical_scattering`. Got a $(nameof(typeof(geo)))."))
    x, y = _coordinates(grid)
    period = (FG.Grids.domain_length(grid, 1), FG.Grids.domain_length(grid, 2))
    origin = (first(FG.Grids.bounds(grid, 1)), first(FG.Grids.bounds(grid, 2)))
    w = weights === nothing ? _active(grid, vec(FG.Grids.measure(grid))) : weights
    return ST.ScatteredPlanar.build(T, _active(grid, x), _active(grid, y), ms, J;
                                    period = period, origin = origin, weights = w, kwargs...)
end

ST.scattered_planar_scattering(grid::FG.Grids.AbstractGrid, ms::NTuple{2, Int}, J::Int; kwargs...) =
    ST.scattered_planar_scattering(eltype(grid), grid, ms, J; kwargs...)

# Every cell's coordinates in the grid's linear cell order: a node set's own coordinate vectors, which
# keep their array type, and every other layout's cells written out.
_coordinates(grid::FG.Grids.AbstractUnstructuredGrid) =
    (FG.Grids.coordinates(grid, 1), FG.Grids.coordinates(grid, 2))
_coordinates(grid::FG.Grids.AbstractGrid) = FG.Grids.materialize(grid)

# The entries of a per-cell vector at the grid's active cells.
_active(grid::FG.Grids.AbstractGrid, v::AbstractVector) = _active(FG.Grids.mask(grid), v)
_active(::FG.Grids.AllActive, v::AbstractVector) = v
_active(m::AbstractArray{Bool}, v::AbstractVector) = v[vec(m)]

end # module ScatteringTransformsFlowGeometriesExt
