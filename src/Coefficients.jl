module Coefficients

using LinearAlgebra: LinearAlgebra

export ScatteringCoefficients1D, ScatteringCoefficients2D
export zeroth_order, first_order, second_order
export flatten1d, flatten2d, flatten1d!, flatten2d!, flatten_length
export flat_length, flat_row_s0, flat_row_s1, flat_row_s2
export update_S0

# ============================================================================
# 1D scattering coefficients
# ============================================================================

"""
    ScatteringCoefficients1D{T,V,M,S0}

Immutable container for 1D scattering coefficients. `S0` is either a scalar, which
[`update_S0`](@ref) replaces by returning a new container, or a 1-element array, which it writes in
place.

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

function ScatteringCoefficients1D(n::Int, ::Type{T}=Float64; 
                                    compute_S2::Bool=true) where T
    S1 = Vector{T}(undef, n)
    S2 = compute_S2 ? zeros(T, n, n) : Matrix{T}(undef, 0, 0)
    return ScatteringCoefficients1D(S1, S2; S0=zero(T))
end

n_wavelets(c::ScatteringCoefficients1D) = c.n_wavelets

"""
    zeroth_order(c) -> Number

The zeroth-order coefficient `S0`, the field mean, which is complex for a complex field.
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
    update_S0(c, val) -> c′

The coefficients `c` with `S0 = val`. A 1-element-array `S0` is written in place and `c` returned; a
scalar `S0` gives a new container sharing `c`'s `S1` and `S2`.
"""
function update_S0(c::ScatteringCoefficients1D{T,V,M,S0}, val) where {T,V,M,S0<:AbstractArray}
    c.S0[1] = val
    return c
end

function update_S0(c::ScatteringCoefficients1D{T,V,M,S0}, val) where {T,V,M,S0}
    return ScatteringCoefficients1D(c.S1, c.S2; S0=val)
end

"""
    flat_length(n) -> Int
    flat_row_s0() -> Int
    flat_row_s1(j, n) -> Int
    flat_row_s2(j1, j2, n) -> Int

Row layout of the flattened coefficient vector `[S0; S1; vec(S2 upper triangle)]` for `n` wavelets,
as produced by [`flatten1d!`](@ref)/[`flatten2d!`](@ref).

The batched paths write flattened columns through these accessors, so they must match the order the
`flatten*!` loops write.
"""
flat_length(n::Integer) = n * (n - 1) ÷ 2 + n + 1
@inline flat_row_s0() = 1
@inline flat_row_s1(j::Integer, n::Integer) = 1 + j
# Rows are laid out `(j1, j2)` for `j2 > j1`, j1-major: skip the complete rows before `j1`, then
# step to `j2` within it.
@inline flat_row_s2(j1::Integer, j2::Integer, n::Integer) =
    1 + n + ((j1 - 1) * n - ((j1 - 1) * j1) ÷ 2) + (j2 - j1)

"""
    flat_eltype(c) -> Type

Element type a flattened column of `c` needs.

`S1` and `S2` are moduli and stay real whatever the input is, but `S0` is the field mean and is
complex for a complex field, so the flat layout is as wide as the two together.
"""
@inline flat_eltype(c) = promote_type(eltype(c.S1), typeof(zeroth_order(c)))

"""
    flatten_length(c) -> Int

Length of the flattened coefficient vector `[S0; S1; vec(S2 upper triangle)]`.
"""
flatten_length(c::ScatteringCoefficients1D) = flat_length(c.n_wavelets)

"""
    flatten1d(coeffs::ScatteringCoefficients1D) -> AbstractVector

Flatten to `[S0; S1; S2 upper triangle]`, keeping only the `S2` entries with `j2 > j1`, in an array
of element type [`flat_eltype`](@ref).
"""
function flatten1d(c::ScatteringCoefficients1D)
    return flatten1d!(similar(c.S1, flat_eltype(c), flatten_length(c)), c)
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
# 2D scattering coefficients
# ============================================================================

"""
    ScatteringCoefficients2D{T,V,M,S0}

Immutable container for scales × orientations scattering coefficients (2D, 3D and scattered planar).
`S0` is a scalar or a 1-element array, as in [`ScatteringCoefficients1D`](@ref).
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

flatten_length(c::ScatteringCoefficients2D) = flat_length(c.n_wavelets)

"""
    flatten2d(coeffs::ScatteringCoefficients2D) -> AbstractVector

Flatten to `[S0; S1; S2 upper triangle]`, as [`flatten1d`](@ref) does.
"""
function flatten2d(c::ScatteringCoefficients2D)
    return flatten2d!(similar(c.S1, flat_eltype(c), flatten_length(c)), c)
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
