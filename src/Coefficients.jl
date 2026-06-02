module Coefficients

"""
    Coefficients.jl — Generic scattering coefficient storage

Parametric types support: Float32/Float64, CPU/GPU arrays, autodiff.
Immutable structs - zero-allocation by reusing S1/S2 buffers.
"""

using LinearAlgebra: LinearAlgebra

export ScatteringCoefficients1D, ScatteringCoefficients2D
export zeroth_order, first_order, second_order
export flatten1d, flatten2d
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
    
    function ScatteringCoefficients1D(S1::AbstractVector{T}, S2::AbstractMatrix{T}; 
                                      S0=zero(T)) where T
        n = length(S1)
        @assert (isempty(S2) || (size(S2, 1) == n && size(S2, 2) == n)) "S2 must be empty or n×n"
        new{T, typeof(S1), typeof(S2), typeof(S0)}(S0, S1, S2, n)
    end
end

# Convenience constructor for pre-allocation
function ScatteringCoefficients1D(n::Int, ::Type{T}=Float64; 
                                    compute_S2::Bool=true) where T
    S1 = Vector{T}(undef, n)
    S2 = compute_S2 ? Matrix{T}(undef, n, n) : Matrix{T}(undef, 0, 0)
    return ScatteringCoefficients1D(S1, S2; S0=zero(T))
end

n_wavelets(c::ScatteringCoefficients1D) = c.n_wavelets

# Dispatch on S0 type for zeroth_order
@inline zeroth_order(c::ScatteringCoefficients1D{T,V,M,T}) where {T,V,M} = c.S0
@inline zeroth_order(c::ScatteringCoefficients1D{T,V,M,<:AbstractVector{T}}) where {T,V,M} = c.S0[1]

@inline first_order(c::ScatteringCoefficients1D) = c.S1
@inline second_order(c::ScatteringCoefficients1D) = c.S2

# Dispatch-based S0 update - scalar S0 returns new struct (zero alloc for arrays)
function update_S0(c::ScatteringCoefficients1D{T,V,M,T}, val::T) where {T,V,M}
    return ScatteringCoefficients1D(c.S1, c.S2; S0=val)
end

# Mutable container S0 - update in place, return same struct (true zero alloc)
function update_S0(c::ScatteringCoefficients1D{T,V,M,<:AbstractVector{T}}, val::T) where {T,V,M}
    c.S0[1] = val
    return c
end

"""
    flatten1d(coeffs::ScatteringCoefficients1D{T}) -> Vector{T}

Flatten to vector: [S0; S1; vec(S2 upper triangular)].
Only includes unique S2 elements where j2 > j1 (saves ~50% space).
"""
function flatten1d(c::ScatteringCoefficients1D{T}) where T
    n = c.n_wavelets
    n_s2 = n * (n - 1) ÷ 2
    result = Vector{T}(undef, 1 + n + n_s2)
    
    result[1] = zeroth_order(c)  # Use dispatch to get S0 value
    result[2:1+n] .= c.S1
    
    idx = 2 + n
    S2 = c.S2
    @inbounds for j1 in 1:n, j2 in (j1+1):n
        result[idx] = S2[j1, j2]
        idx += 1
    end
    
    return result
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
    
    function ScatteringCoefficients2D(S1::AbstractVector{T}, S2::AbstractMatrix{T};
                                    S0=zero(T), n_scales::Int=0, n_orientations::Int=0) where T
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
    S2 = compute_S2 ? Matrix{T}(undef, n, n) : Matrix{T}(undef, 0, 0)
    return ScatteringCoefficients2D(S1, S2; S0=zero(T), n_scales=n_scales, n_orientations=n_orientations)
end

n_scales(c::ScatteringCoefficients2D) = c.n_scales
n_orientations(c::ScatteringCoefficients2D) = c.n_orientations
n_wavelets(c::ScatteringCoefficients2D) = c.n_wavelets

# Dispatch on S0 type for zeroth_order
@inline zeroth_order(c::ScatteringCoefficients2D{T,V,M,T}) where {T,V,M} = c.S0
@inline zeroth_order(c::ScatteringCoefficients2D{T,V,M,<:AbstractVector{T}}) where {T,V,M} = c.S0[1]

@inline first_order(c::ScatteringCoefficients2D) = c.S1
@inline second_order(c::ScatteringCoefficients2D) = c.S2

# Dispatch-based S0 update for 2D
function update_S0(c::ScatteringCoefficients2D{T,V,M,T}, val::T) where {T,V,M}
    return ScatteringCoefficients2D(c.S1, c.S2; S0=val, n_scales=c.n_scales, n_orientations=c.n_orientations)
end

function update_S0(c::ScatteringCoefficients2D{T,V,M,<:AbstractVector{T}}, val::T) where {T,V,M}
    c.S0[1] = val
    return c
end

"""
    flatten2d(coeffs::ScatteringCoefficients2D{T}) -> Vector{T}

Flatten to vector: [S0; S1; vec(S2 upper triangular)].
"""
function flatten2d(c::ScatteringCoefficients2D{T}) where T
    n = c.n_wavelets
    n_s2 = n * (n - 1) ÷ 2
    result = Vector{T}(undef, 1 + n + n_s2)
    
    result[1] = zeroth_order(c)  # Use dispatch to get S0 value
    result[2:1+n] .= c.S1
    
    idx = 2 + n
    S2 = c.S2
    @inbounds for j1 in 1:n, j2 in (j1+1):n
        result[idx] = S2[j1, j2]
        idx += 1
    end
    
    return result
end

end # module Coefficients
