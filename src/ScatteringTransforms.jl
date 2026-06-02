"""
    ScatteringTransforms.jl — Native Julia implementation of wavelet scattering transforms

Supports 1D, 2D planar (gridded), and (via extension) 2D spherical scattering.

## Quick Start

```julia
using ScatteringTransforms

# 1D scattering
st = ScatteringTransform1D(; J=8, Q=1, max_order=2)
coeffs = st(signal)

# 2D planar scattering  
st2d = ScatteringTransform2D(; J=4, L=8, max_order=2)
coeffs2d = st2d(image)
```

## Implementation Notes

- FFT-based convolutions for O(N log N) performance
- Frequency-domain Morlet filter banks
- Depth-first tree traversal (memory efficient)
- Modular extensions for spherical (NUFSHT) and GPU support

## References

- Mallat (2012): Group invariant scattering. Comm. Pure Appl. Math.
- Bruna & Mallat (2013): Invariant Scattering Convolution Networks. IEEE PAMI.
- Cheng & Ménard (2021): How to quantify fields or textures? A guide to the scattering transform.
"""
module ScatteringTransforms

using FFTW: FFTW
using LinearAlgebra: LinearAlgebra

# Include core components (creates submodules)
# Order matters: Coefficients must be before Scattering1D/Scattering2D
include("Filters.jl")
include("FilterBanks.jl")
include("ScatteringCore.jl")
include("Coefficients.jl")
include("Scattering1D.jl")
include("Scattering2D.jl")

# Import submodules using X: X pattern
using .Filters: Filters
using .FilterBanks: FilterBanks
using .ScatteringCore: ScatteringCore
using .Scattering1D: Scattering1D
using .Scattering2D: Scattering2D
using .Coefficients: Coefficients

# Re-export key types from submodules (using X: X pattern)
const ScatteringTransform1D = Scattering1D.ScatteringTransform1D
const ScatteringTransform2D = Scattering2D.ScatteringTransform2D
const FilterBank1D = FilterBanks.FilterBank1D
const FilterBank2D = FilterBanks.FilterBank2D
const Morlet1D = Filters.Morlet1D
const Morlet2D = Filters.Morlet2D
const ScatteringCoefficients1D = Coefficients.ScatteringCoefficients1D
const ScatteringCoefficients2D = Coefficients.ScatteringCoefficients2D
const frequency_response = Filters.frequency_response
const build_filter_bank1d = FilterBanks.build_filter_bank1d
const build_filter_bank2d = FilterBanks.build_filter_bank2d
const zeroth_order = Coefficients.zeroth_order
const first_order = Coefficients.first_order
const second_order = Coefficients.second_order
const flatten1d = Coefficients.flatten1d
const flatten2d = Coefficients.flatten2d
const scattering_transform! = Scattering1D.scattering_transform!

export ScatteringTransform1D, ScatteringTransform2D
export FilterBank1D, FilterBank2D
export Morlet1D, Morlet2D
export ScatteringCoefficients1D, ScatteringCoefficients2D
export zeroth_order, first_order, second_order
export flatten1d, flatten2d
export frequency_response
export build_filter_bank1d, build_filter_bank2d
export scattering_transform!

# Precompilation setup for faster first use
# (optional - can be added later)

end # module
