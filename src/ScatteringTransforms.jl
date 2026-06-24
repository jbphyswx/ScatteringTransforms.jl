"""
    ScatteringTransforms.jl — Native Julia implementation of wavelet scattering transforms

Supports 1D, 2D planar (gridded), and (via extension) 2D spherical scattering.

## Quick Start

```julia
using ScatteringTransforms

# 1D scattering (N = signal length, J = number of octaves)
signal = randn(1024)
st = ScatteringTransform1D(1024, 8; Q=1, max_order=2)
coeffs = st(signal)

# 2D planar scattering (N = image size, J = scales, L = orientations)
image = randn(256, 256)
st2d = ScatteringTransform2D((256, 256), 4; L=8, max_order=2)
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

# Include core components (creates submodules)
# Order matters: Coefficients must be before Scattering1D/Scattering2D
include("Backends.jl")
include("Domains.jl")
include("Plans.jl")
include("Filters.jl")
include("FilterBanks.jl")
include("ScatteringCore.jl")
include("Coefficients.jl")
include("PathGraph.jl")
include("ScatteringFields.jl")
include("Scattering1D.jl")
include("Scattering2D.jl")

# Import submodules using X: X pattern
using .Backends: Backends
using .Domains: Domains
using .Plans: Plans
using .Filters: Filters
using .FilterBanks: FilterBanks
using .ScatteringCore: ScatteringCore
using .PathGraph: PathGraph
using .ScatteringFields: ScatteringFields
using .Scattering1D: Scattering1D
using .Scattering2D: Scattering2D
using .Coefficients: Coefficients

# Re-export key types from submodules (using X: X pattern)
const ScatteringTransform1D = Scattering1D.ScatteringTransform1D
const ScatteringTransform2D = Scattering2D.ScatteringTransform2D
const FilterBank1D = FilterBanks.FilterBank1D
const FilterBank2D = FilterBanks.FilterBank2D
const WaveletMeta = FilterBanks.WaveletMeta
const ScatteringTree = PathGraph.ScatteringTree
# Backend taxonomy
const SerialCPU = Backends.SerialCPU
const ThreadedCPU = Backends.ThreadedCPU
const GPUBackend = Backends.GPUBackend
const AutoBackend = Backends.AutoBackend
const Distributed = Backends.Distributed
const MPI = Backends.MPI
# Domain tags
const Line1D = Domains.Line1D
const Plane2D = Domains.Plane2D
const Volume3D = Domains.Volume3D
const Sphere = Domains.Sphere
# Spectral plans
const DirectSumPlan = Plans.DirectSumPlan
const AbstractScatteringPlan = Plans.AbstractScatteringPlan
const forward_transform! = Plans.forward_transform!
const inverse_transform! = Plans.inverse_transform!
const ScatteringField1D = ScatteringFields.ScatteringField1D
const ScatteringField2D = ScatteringFields.ScatteringField2D
const path_field = ScatteringFields.path_field
const scattering_field = ScatteringFields.scattering_field    # 1D + 2D methods added in submodules
const scattering_field! = ScatteringFields.scattering_field!
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
const scattering_transform!    = Scattering1D.scattering_transform!
const scattering_transform2d!  = Scattering2D.scattering_transform2d!
const compute_S1_2d! = Scattering2D.compute_S1_2d!
const compute_S2_2d! = Scattering2D.compute_S2_2d!

# Plotting stubs (implemented in ScatteringTransformsCairoMakieExt)
function plot_filter_bank end
function plot_coefficients end

export ScatteringTransform1D, ScatteringTransform2D
export FilterBank1D, FilterBank2D
export WaveletMeta, ScatteringTree
export SerialCPU, ThreadedCPU, GPUBackend, AutoBackend, Distributed, MPI
export Line1D, Plane2D, Volume3D, Sphere
export DirectSumPlan, AbstractScatteringPlan, forward_transform!, inverse_transform!
export Morlet1D, Morlet2D
export ScatteringCoefficients1D, ScatteringCoefficients2D
export zeroth_order, first_order, second_order
export flatten1d, flatten2d
export frequency_response
export build_filter_bank1d, build_filter_bank2d
export scattering_transform!, scattering_transform2d!
export scattering_field, scattering_field!
export ScatteringField1D, ScatteringField2D, path_field
export compute_S1_2d!, compute_S2_2d!
export plot_filter_bank, plot_coefficients

end # module
