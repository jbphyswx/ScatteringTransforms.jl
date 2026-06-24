module Domains

"""
    Domains.jl — Scattering domain geometry

Tags the geometry a transform acts on, so one generic engine can dispatch across dimensions
and the sphere:

- `Line1D`   — 1D signals (FFT)
- `Plane2D`  — 2D gridded images (FFT)
- `Volume3D` — 3D gridded volumes (FFT)
- `Sphere`   — scalar fields on S² (NUFSHT ext)

`spatial_ndims` reports the number of spatial dimensions the wavelet convolutions act over
(distinct from any trailing batch dimension).
"""

export AbstractDomain, Line1D, Plane2D, Volume3D, Sphere, spatial_ndims

abstract type AbstractDomain end

"1D signal domain."
struct Line1D <: AbstractDomain end

"2D planar (gridded image) domain."
struct Plane2D <: AbstractDomain end

"3D volumetric (gridded) domain."
struct Volume3D <: AbstractDomain end

"Spherical (S²) domain; handled via the NUFSHT extension."
struct Sphere <: AbstractDomain end

"""
    spatial_ndims(domain) -> Int

Number of spatial dimensions the wavelet convolutions act over (the sphere counts as 2).
"""
spatial_ndims(::Line1D) = 1
spatial_ndims(::Plane2D) = 2
spatial_ndims(::Volume3D) = 3
spatial_ndims(::Sphere) = 2

end # module Domains
