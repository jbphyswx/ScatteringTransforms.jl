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
                                        J::Int; weights = nothing, period = nothing,
                                        L::Int = 8, kwargs...) where {T}
    n = FG.Grids.ncoordinates(grid)
    n == 2 || throw(ArgumentError("a planar scattering transform takes a grid of 2 coordinates; got $n"))
    geo = FG.Grids.grid_geometry(grid)
    geo isa FG.Geometry.AbstractCartesianGeometry || throw(ArgumentError(
        "a planar scattering transform takes a Cartesian grid; for a field on the sphere use " *
        "`spherical_scattering`. Got a $(nameof(typeof(geo)))."))
    x, y = _coordinates(grid)
    domain = _planar_period(period, grid, ms, J, L, T)
    origin = (first(FG.Grids.bounds(grid, 1)), first(FG.Grids.bounds(grid, 2)))
    w = weights === nothing ? _active(grid, vec(FG.Grids.measure(grid))) : weights
    return ST.ScatteredPlanar.build(T, _active(grid, x), _active(grid, y), ms, J;
                                    period = domain, origin = origin, weights = w, L, kwargs...)
end

_planar_period(period, grid, ms, J, L, ::Type) = period
function _planar_period(::Nothing, grid, ms, J, L, ::Type{T}) where {T}
    wavelet = ST.Filters.Morlet2D{T}(ms, J - 1, zero(T); L)
    width = max(wavelet.sigma_par, wavelet.sigma_perp)
    return map(FG.Grids.topology(grid), (1, 2), ms) do topology, d, n
        _axis_period(topology, grid, d, n, width)
    end
end

_axis_period(::FG.Grids.Periodic, grid, d, n, width) = FG.Grids.period(grid, d)
_axis_period(::FG.Grids.Bounded, grid, d, n, width) = FG.Grids.domain_length(grid, d) * (1 + width / n)

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

function _spherical_points(grid, ::FG.Geometry.AbstractSphericalGeometry, ::Type{T}) where {T}
    FG.Grids.ncoordinates(grid) == 2 || throw(ArgumentError("a spherical transform takes longitude and latitude"))
    λ, φ = _coordinates(grid)
    return T(π) / 2 .- T.(_active(grid, φ)), T.(_active(grid, λ))
end

_spherical_points(grid, geometry, ::Type) = throw(ArgumentError(
    "a spherical scattering transform requires a spherical grid; got $(typeof(geometry))"))

for name in (:spherical_scattering, :spherical_monogenic_scattering)
    @eval begin
        function ST.$name(::Type{T}, grid::FG.Grids.AbstractGrid, lmax::Int, J::Int;
                           weights = nothing, kwargs...) where {T}
            θ, φ = _spherical_points(grid, FG.Grids.grid_geometry(grid), T)
            w = weights === nothing ? _active(grid, vec(FG.Grids.measure(grid))) : weights
            return ST.$name(θ, φ, lmax, J; weights = w, kwargs...)
        end
        ST.$name(grid::FG.Grids.AbstractGrid, lmax::Int, J::Int; kwargs...) =
            ST.$name(eltype(grid), grid, lmax, J; kwargs...)
    end
end

end # module ScatteringTransformsFlowGeometriesExt
