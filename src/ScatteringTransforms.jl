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
include("Scattering3D.jl")
include("Reductions.jl")

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
using .Scattering3D: Scattering3D
using .Coefficients: Coefficients
using .Reductions: Reductions

# Re-export key types from submodules (using X: X pattern)
const ScatteringTransform1D = Scattering1D.ScatteringTransform1D
const ScatteringTransform2D = Scattering2D.ScatteringTransform2D
const ScatteringTransform3D = Scattering3D.ScatteringTransform3D
const FilterBank1D = FilterBanks.FilterBank1D
const FilterBank2D = FilterBanks.FilterBank2D
const FilterBank3D = FilterBanks.FilterBank3D
const Morlet3D = Filters.Morlet3D
const build_filter_bank3d = FilterBanks.build_filter_bank3d
const scattering_transform3d! = Scattering3D.scattering_transform3d!
const WaveletMeta = FilterBanks.WaveletMeta
const ScatteringTree = PathGraph.ScatteringTree
# Backend taxonomy (ecosystem names; DistributedBackend/MPIBackend parametric over inner backend)
const SerialBackend = Backends.SerialBackend
const ThreadedBackend = Backends.ThreadedBackend
const GPUBackend = Backends.GPUBackend
const AutoBackend = Backends.AutoBackend
const DistributedBackend = Backends.DistributedBackend
const MPIBackend = Backends.MPIBackend
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
const flatten1d! = Coefficients.flatten1d!
const flatten2d! = Coefficients.flatten2d!
const flatten_length = Coefficients.flatten_length
const scattering_transform!    = Scattering1D.scattering_transform!
const scattering_transform2d!  = Scattering2D.scattering_transform2d!
const compute_S1_2d! = Scattering2D.compute_S1_2d!
const compute_S2_2d! = Scattering2D.compute_S2_2d!
const compute_shape_sparsity = Scattering2D.compute_shape_sparsity
const normalized_coefficients = Reductions.normalized_coefficients
const log_coefficients = Reductions.log_coefficients

# ============================================================================
# Batched transforms — process a stack of signals/images reusing one plan + workspace
# ============================================================================

"""
    scattering_batch(st::ScatteringTransform1D, X) -> Matrix

Apply a 1D scattering transform to a batch of signals `X` of size `(N, B)` (signals as columns),
returning a `(flatten_length, B)` matrix whose column `b` is `flatten1d(st(X[:, b]))`. The plan
and all workspace buffers are reused across the batch (only the small scalar-S0 wrapper is
re-allocated per column).
"""
function scattering_batch(st::ScatteringTransform1D, X::AbstractMatrix)
    N, B = size(X)
    nw = length(st.filter_bank.wavelets)
    T = eltype(st.filter_bank.averaging) |> real
    coeffs = Coefficients.ScatteringCoefficients1D(nw, T; compute_S2 = st.max_order >= 2)
    out = Matrix{T}(undef, Coefficients.flatten_length(coeffs), B)
    @inbounds for b in 1:B
        c = Scattering1D.scattering_transform!(coeffs, st, view(X, :, b))
        Coefficients.flatten1d!(view(out, :, b), c)
    end
    return out
end

"""
    scattering_batch(st::ScatteringTransform2D, X) -> Matrix

Apply a 2D scattering transform to a batch of images `X` of size `(Ny, Nx, B)`, returning a
`(flatten_length, B)` matrix whose column `b` is `flatten2d(st(X[:, :, b]))`. Plan and workspace
are reused across the batch.
"""
function scattering_batch(st::ScatteringTransform2D, X::AbstractArray{<:Any,3})
    Ny, Nx, B = size(X)
    T = eltype(st.filter_bank.averaging) |> real
    coeffs = Coefficients.ScatteringCoefficients2D(st.filter_bank.J, st.filter_bank.L, T;
                                                   compute_S2 = st.max_order >= 2)
    out = Matrix{T}(undef, Coefficients.flatten_length(coeffs), B)
    @inbounds for b in 1:B
        c = Scattering2D.scattering_transform2d!(coeffs, st, view(X, :, :, b))
        Coefficients.flatten2d!(view(out, :, b), c)
    end
    return out
end

# Backend-dispatched batched transforms. Serial runs in-process; ThreadedBackend / Distributed /
# MPI methods are added by the corresponding extensions (per-task/per-worker workspace).
scattering_batch(::Backends.SerialBackend, st, X) = scattering_batch(st, X)
function scattering_batch(b::Backends.AbstractExecutionBackend, st, X)
    throw(ArgumentError(
        "scattering_batch on backend $(typeof(b)) is not loaded — run `using OhMyThreads` " *
        "(ThreadedBackend), `using Distributed` (DistributedBackend), or `using MPI` (MPIBackend)."))
end

# Plotting stubs (implemented in ScatteringTransformsCairoMakieExt)
function plot_filter_bank end
function plot_coefficients end

export ScatteringTransform1D, ScatteringTransform2D, ScatteringTransform3D
export FilterBank1D, FilterBank2D, FilterBank3D
export build_filter_bank3d, scattering_transform3d!
export WaveletMeta, ScatteringTree
export SerialBackend, ThreadedBackend, GPUBackend, AutoBackend, DistributedBackend, MPIBackend
export Line1D, Plane2D, Volume3D, Sphere
export DirectSumPlan, AbstractScatteringPlan, forward_transform!, inverse_transform!
export Morlet1D, Morlet2D, Morlet3D
export ScatteringCoefficients1D, ScatteringCoefficients2D
export zeroth_order, first_order, second_order
export flatten1d, flatten2d
export frequency_response
export build_filter_bank1d, build_filter_bank2d
export scattering_transform!, scattering_transform2d!
export scattering_batch
export flatten1d!, flatten2d!, flatten_length
export scattering_field, scattering_field!
export ScatteringField1D, ScatteringField2D, path_field
export compute_S1_2d!, compute_S2_2d!
export compute_shape_sparsity, normalized_coefficients, log_coefficients
export plot_filter_bank, plot_coefficients

# Precompile the hot paths (using the dependency-free direct-sum backend, so no weakdep is
# required at precompile time) to cut time-to-first-transform.
using PrecompileTools: @setup_workload, @compile_workload
@setup_workload begin
    @compile_workload begin
        st1 = ScatteringTransform1D(32, 3; Q = 1, max_order = 2, spectral = :direct)
        c1 = st1(zeros(Float64, 32))
        flatten1d(c1)
        scattering_field(st1, zeros(Float64, 32); subsample = 1)
        scattering_batch(st1, zeros(Float64, 32, 2))
        normalized_coefficients(c1)

        st2 = ScatteringTransform2D((16, 16), 2; L = 4, max_order = 2, spectral = :direct)
        c2 = st2(zeros(Float64, 16, 16))
        compute_shape_sparsity(first_order(c2), second_order(c2), st2.filter_bank.meta)

        st3 = ScatteringTransform3D((8, 8, 8), 2; n_orient = 6, max_order = 2, spectral = :direct)
        st3(zeros(Float64, 8, 8, 8))
    end
end

end # module
