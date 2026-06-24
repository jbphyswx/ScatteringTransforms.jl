module Coefficients

"""
    Coefficients.jl — Generic scattering coefficient storage

Parametric types support: Float32/Float64, CPU/GPU arrays, autodiff.
Immutable structs - zero-allocation by reusing S1/S2 buffers.
"""

using LinearAlgebra: LinearAlgebra

export ScatteringCoefficients1D, ScatteringCoefficients2D
export zeroth_order, first_order, second_order
export flatten1d, flatten2d, flatten1d!, flatten2d!, flatten_length
export update_S0

# ============================================================================
# 1D Scattering Coefficients — Fully Generic
# ============================================================================

"""
    ScatteringCoefficients1D{T,V,M,S0}

Immutable container for 1D scattering coefficients.
S0 can be scalar T (return new struct) or mutable container (update in place).
Uses multiple dispatch for optimal S0 handling.

# Type Parameters
- `T`: Element type
- `V`: 1D array type
- `M`: 2D array type  
- `S0`: S0 storage type (T for scalar, AbstractVector{T} for mutable)
"""
struct ScatteringCoefficients1D{T,V<:AbstractVector{T},M<:AbstractMatrix{T},S0}
    S0::S0
    S1::V
    S2::M
    n_wavelets::Int
    
    function ScatteringCoefficients1D(S1::AbstractVector, S2::AbstractMatrix; 
                                      S0=zero(eltype(S1)))
        T = eltype(S1)
        n = length(S1)
        @assert (isempty(S2) || (size(S2, 1) == n && size(S2, 2) == n)) "S2 must be empty or n×n"
        new{T, typeof(S1), typeof(S2), typeof(S0)}(S0, S1, S2, n)
    end
end

# Convenience constructor for pre-allocation
function ScatteringCoefficients1D(n::Int, ::Type{T}=Float64; 
                                    compute_S2::Bool=true) where T
    S1 = Vector{T}(undef, n)
    S2 = compute_S2 ? zeros(T, n, n) : Matrix{T}(undef, 0, 0)
    return ScatteringCoefficients1D(S1, S2; S0=zero(T))
end

n_wavelets(c::ScatteringCoefficients1D) = c.n_wavelets

"""
    zeroth_order(c) -> Real

Extract the zeroth-order (average / S0) scattering coefficient.
"""
@inline zeroth_order(c::ScatteringCoefficients1D) = _extract_S0(c.S0)
@inline _extract_S0(s0) = s0                    # scalar fallback
@inline _extract_S0(s0::AbstractArray) = s0[1]  # mutable container

"""
    first_order(c) -> AbstractVector

Extract the first-order (S1) scattering coefficients.
"""
@inline first_order(c::ScatteringCoefficients1D) = c.S1

"""
    second_order(c) -> AbstractMatrix

Extract the second-order (S2) scattering coefficients.
"""
@inline second_order(c::ScatteringCoefficients1D) = c.S2

"""
    update_S0(c, val)

Update the zeroth-order coefficient storage with `val` and return the coefficients.
"""
function update_S0(c::ScatteringCoefficients1D{T,V,M,S0}, val) where {T,V,M,S0<:AbstractArray}
    # Mutable container S0: update in place, return same struct (true zero alloc)
    c.S0[1] = val
    return c
end

# Scalar S0: return new wrapper (only allocates the small struct, not S1/S2)
function update_S0(c::ScatteringCoefficients1D{T,V,M,S0}, val) where {T,V,M,S0}
    return ScatteringCoefficients1D(c.S1, c.S2; S0=val)
end

"""
    flatten_length(c) -> Int

Length of the flattened coefficient vector `[S0; S1; vec(S2 upper triangle)]`.
"""
flatten_length(c::ScatteringCoefficients1D) = c.n_wavelets * (c.n_wavelets - 1) ÷ 2 + c.n_wavelets + 1

"""
    flatten1d(coeffs::ScatteringCoefficients1D{T}) -> Vector{T}

Flatten to vector: [S0; S1; vec(S2 upper triangular)].
Only includes unique S2 elements where j2 > j1 (saves ~50% space).
"""
function flatten1d(c::ScatteringCoefficients1D)
    return flatten1d!(similar(c.S1, flatten_length(c)), c)
end

"""
    flatten1d!(out, c) -> out

Zero-allocation flatten into a pre-allocated vector of length `flatten_length(c)`.
"""
function flatten1d!(out::AbstractVector, c::ScatteringCoefficients1D)
    n = c.n_wavelets
    length(out) == flatten_length(c) ||
        throw(DimensionMismatch("out length $(length(out)) != flatten_length $(flatten_length(c))"))
    out[1] = zeroth_order(c)
    @inbounds out[2:(1 + n)] .= c.S1
    idx = 2 + n
    S2 = c.S2
    @inbounds for j1 in 1:n, j2 in (j1 + 1):n
        out[idx] = S2[j1, j2]
        idx += 1
    end
    return out
end

# ============================================================================
# 2D Scattering Coefficients — Fully Generic  
# ============================================================================

"""
    ScatteringCoefficients2D{T,V,M,S0}

Immutable container for 2D planar scattering coefficients.
S0 can be scalar T or mutable container - dispatch handles both optimally.
"""
struct ScatteringCoefficients2D{T,V<:AbstractVector{T},M<:AbstractMatrix{T},S0}
    S0::S0
    S1::V
    S2::M
    n_scales::Int
    n_orientations::Int
    n_wavelets::Int  # n_scales * n_orientations
    
    function ScatteringCoefficients2D(S1::AbstractVector, S2::AbstractMatrix;
                                    S0=zero(eltype(S1)), n_scales::Int=0, n_orientations::Int=0)
        T = eltype(S1)
        n = length(S1)
        @assert (isempty(S2) || (size(S2, 1) == n && size(S2, 2) == n)) "S2 must be empty or n×n"
        new{T, typeof(S1), typeof(S2), typeof(S0)}(S0, S1, S2, n_scales, n_orientations, n)
    end
end

# Convenience constructor
function ScatteringCoefficients2D(n_scales::Int, n_orientations::Int, ::Type{T}=Float64;
                                 compute_S2::Bool=true) where T
    n = n_scales * n_orientations
    S1 = Vector{T}(undef, n)
    S2 = compute_S2 ? zeros(T, n, n) : Matrix{T}(undef, 0, 0)
    return ScatteringCoefficients2D(S1, S2; S0=zero(T), n_scales=n_scales, n_orientations=n_orientations)
end

n_scales(c::ScatteringCoefficients2D) = c.n_scales
n_orientations(c::ScatteringCoefficients2D) = c.n_orientations
n_wavelets(c::ScatteringCoefficients2D) = c.n_wavelets

@inline zeroth_order(c::ScatteringCoefficients2D) = _extract_S0(c.S0)

@inline first_order(c::ScatteringCoefficients2D) = c.S1
@inline second_order(c::ScatteringCoefficients2D) = c.S2

function update_S0(c::ScatteringCoefficients2D{T,V,M,S0}, val) where {T,V,M,S0<:AbstractArray}
    c.S0[1] = val
    return c
end

function update_S0(c::ScatteringCoefficients2D{T,V,M,S0}, val) where {T,V,M,S0}
    return ScatteringCoefficients2D(c.S1, c.S2; S0=val, n_scales=c.n_scales, n_orientations=c.n_orientations)
end

flatten_length(c::ScatteringCoefficients2D) = c.n_wavelets * (c.n_wavelets - 1) ÷ 2 + c.n_wavelets + 1

"""
    flatten2d(coeffs::ScatteringCoefficients2D{T}) -> Vector{T}

Flatten to vector: [S0; S1; vec(S2 upper triangular)].
"""
function flatten2d(c::ScatteringCoefficients2D)
    return flatten2d!(similar(c.S1, flatten_length(c)), c)
end

"""
    flatten2d!(out, c) -> out

Zero-allocation flatten into a pre-allocated vector of length `flatten_length(c)`.
"""
function flatten2d!(out::AbstractVector, c::ScatteringCoefficients2D)
    n = c.n_wavelets
    length(out) == flatten_length(c) ||
        throw(DimensionMismatch("out length $(length(out)) != flatten_length $(flatten_length(c))"))
    out[1] = zeroth_order(c)
    @inbounds out[2:(1 + n)] .= c.S1
    idx = 2 + n
    S2 = c.S2
    @inbounds for j1 in 1:n, j2 in (j1 + 1):n
        out[idx] = S2[j1, j2]
        idx += 1
    end
    return out
end

end # module Coefficients
