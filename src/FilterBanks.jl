module FilterBanks

"""
    FilterBanks.jl — Construct dyadic filter banks for scattering transforms

Creates complete filter bank structures with wavelets at multiple scales
and orientations (for 2D), plus averaging (scaling) filters.
"""

using LinearAlgebra: LinearAlgebra

# Import Filters submodule
using ..Filters: Filters

export FilterBank1D, FilterBank2D, FilterBank3D
export build_filter_bank1d, build_filter_bank2d, build_filter_bank3d
export WaveletMeta
export ComputedFilterBank2D, nwavelets, filter_at, task_bank, batch_views, iscomputed

# Inline fftfreq for a single 0-indexed bin — no allocation.
# Mirrors FFTW.fftfreq(N)[k+1].
@inline _fftfreq(N::Int, k::Int) = k < (N + 1) ÷ 2 ? k / N : (k - N) / N

# The scaling function (low-pass averaging filter) as the *complement* of the wavelet energy:
# `|φ(ω)|² = max(0, 1 − Σⱼ|ψⱼ(ω)|²)`. This makes the Littlewood–Paley sum `Σⱼ|ψⱼ|² + |φ|² ≡ 1` a
# (near) tight frame, so the transform is non-expansive — no frequency is amplified. The DC bin is
# pinned to `φ(0)=1` exactly (the wavelets are zero-mean there), which keeps the localized-field
# spatial mean equal to the globally-averaged coefficient. Works for 1D/2D/3D filter arrays.
function _complement_lowpass(wavelets::AbstractVector{A}) where {T, A<:AbstractArray{T}}
    ϕ = similar(first(wavelets))
    @inbounds for i in eachindex(ϕ)
        s = zero(T)
        for ψ in wavelets
            s += abs2(ψ[i])
        end
        ϕ[i] = sqrt(max(zero(T), one(T) - s))
    end
    ϕ[firstindex(ϕ)] = one(T)            # exact DC = 1 (preserves mean ⇔ averaged-coefficient)
    return ϕ
end

# Globally rescale `wavelets` in place so that `max_ω Σⱼ|ψⱼ(ω)|² = 1` — making the transform
# non-expansive — then return the complement low-pass φ, giving a tight frame with Littlewood–Paley
# sum `Σⱼ|ψⱼ|² + |φ|² ≡ 1`. The rescale is a single global constant, so it does not change the
# *relative* coefficient structure — but it is not 1, so a bank built at another resolution must go
# through here too or its coefficients land on a different scale.
function _tight_frame_lowpass!(wavelets::AbstractVector{A}) where {T, A<:AbstractArray{T}}
    maxs = zero(T)
    @inbounds for i in eachindex(first(wavelets))
        s = zero(T)
        for ψ in wavelets
            s += abs2(ψ[i])
        end
        maxs = max(maxs, s)
    end
    if maxs > zero(T)
        c = inv(sqrt(maxs))
        for ψ in wavelets
            ψ .*= c
        end
    end
    return _complement_lowpass(wavelets)
end

"""
    WaveletMeta{T}

Concrete per-wavelet metadata: a struct rather than a `NamedTuple`, so the container stays
concretely typed.

# Fields
- `scale::Int`: octave index `j`
- `q::Int`: sub-octave index within the octave (1D, `0..Q-1`); `0` for 2D
- `orient::Int`: orientation index `l` (2D, `0..L-1`); `0` for 1D
- `j_eff::T`: effective log-scale used to order paths. `j + q/Q` in 1D, `T(j)` in 2D. The
  second-order admissibility constraint is `j_eff(child) > j_eff(parent)` (frequency strictly
  decreasing) — which for 2D means *scale strictly increasing over all orientation pairs*.
- `center_freq::T`: wavelet center frequency
- `theta::T`: orientation angle in radians (2D); `0` for 1D
"""
struct WaveletMeta{T}
    scale::Int
    q::Int
    orient::Int
    j_eff::T
    center_freq::T
    theta::T
end

"""
    FilterBank1D{T,V,W,MV}

Complete 1D filter bank for the scattering transform. Every container is a type parameter (no
hardcoded `Vector`): `V` the per-filter array type (CPU/GPU/static/…), `W` the wavelet
collection, `MV` the metadata collection.

# Fields
- `wavelets::W`: wavelet filters in the Fourier domain (`W<:AbstractVector{V}`)
- `averaging::V`: low-pass averaging (scaling) filter
- `meta::MV`: per-wavelet `WaveletMeta`
- `J::Int`: number of octaves (scales)
- `Q::Int`: wavelets per octave
"""
struct FilterBank1D{T, V<:AbstractVector{T}, W<:AbstractVector{V}, MV<:AbstractVector{WaveletMeta{T}}}
    wavelets::W
    averaging::V
    meta::MV
    J::Int
    Q::Int
end

"""
    build_filter_bank1d(N::Int, J::Int; Q::Int=1) -> FilterBank1D

Build a 1D Morlet filter bank with dyadic scales.

# Arguments
- `N::Int`: Signal length (FFT size)
- `J::Int`: Maximum scale (number of octaves)
- `Q::Int`: Wavelets per octave (default 1 for dyadic, 8 for high Q)

# Returns
- `FilterBank1D`: Complete filter bank with J scales
"""
function build_filter_bank1d(::Type{T}, N::Int, J::Int; Q::Int=1) where {T<:Real}
    # Create first wavelet to get the array type
    morlet = Filters.Morlet1D{T}(N, 0; Q=Q)
    ψ_sample = Filters.frequency_response(morlet)
    V = typeof(ψ_sample)
    
    wavelets = Vector{V}(undef, 0)
    meta = Vector{WaveletMeta{T}}(undef, 0)

    for j in 0:J-1
        for q in 0:Q-1
            effective_j = j + q / Q

            morlet = Filters.Morlet1D{T}(N, j * Q + q; Q=Q)
            ψ = Filters.frequency_response(morlet)

            push!(wavelets, ψ)
            push!(meta, WaveletMeta{T}(j, q, 0, T(effective_j), morlet.center_freq, zero(T)))
        end
    end
    
    # Low-pass = complement of the wavelet energy (tight-frame Littlewood-Paley ≈ 1)
    ϕ = _tight_frame_lowpass!(wavelets)

    return FilterBank1D(wavelets, ϕ, meta, J, Q)
end
build_filter_bank1d(N::Int, J::Int; kwargs...) = build_filter_bank1d(Float64, N, J; kwargs...)

"""
    FilterBank2D{T,M,W,MV}

Complete 2D filter bank with oriented wavelets. Containers are type parameters (no hardcoded
`Vector`): `M` the per-filter matrix type, `W` the wavelet collection, `MV` the metadata
collection.

# Fields
- `wavelets::W`: oriented wavelet filters (`W<:AbstractVector{M}`)
- `averaging::M`: low-pass averaging filter
- `meta::MV`: per-wavelet `WaveletMeta`
- `J::Int`: number of scales
- `L::Int`: number of orientations
"""
struct FilterBank2D{T, M<:AbstractMatrix{T}, W<:AbstractVector{M}, MV<:AbstractVector{WaveletMeta{T}}}
    wavelets::W
    averaging::M
    meta::MV
    J::Int
    L::Int
end

"""
    build_filter_bank2d(N::NTuple{2,Int}, J::Int; L::Int=8) -> FilterBank2D

Build a 2D oriented Morlet filter bank.

# Arguments
- `N::NTuple{2,Int}`: Image dimensions (Ny, Nx)
- `J::Int`: Number of dyadic scales
- `L::Int`: Number of orientations (default 8, evenly spaced)

# Returns
- `FilterBank2D`: Complete 2D filter bank
"""
function build_filter_bank2d(::Type{T}, N::NTuple{2,Int}, J::Int; L::Int=8) where {T<:Real}
    # Create sample wavelet to get matrix type
    morlet = Filters.Morlet2D{T}(N, 0, 0.0; L=L)
    ψ_sample = Filters.frequency_response(morlet)
    M = typeof(ψ_sample)
    
    wavelets = Vector{M}(undef, 0)
    meta = Vector{WaveletMeta{T}}(undef, 0)

    for j in 0:J-1
        for l in 0:L-1
            theta = T(π) * l / L

            morlet = Filters.Morlet2D{T}(N, j, theta; L=L)
            ψ = Filters.frequency_response(morlet)

            push!(wavelets, ψ)
            # j_eff = T(j): same-scale (different-orientation) pairs share j_eff and are therefore
            # NOT admissible as second-order paths; only strictly coarser scales are.
            push!(meta, WaveletMeta{T}(j, 0, l, T(j), morlet.center_freq, theta))
        end
    end
    
    # Low-pass = complement of the wavelet energy (tight-frame Littlewood-Paley ≈ 1)
    ϕ = _tight_frame_lowpass!(wavelets)

    return FilterBank2D(wavelets, ϕ, meta, J, L)
end
build_filter_bank2d(N::NTuple{2,Int}, J::Int; kwargs...) = build_filter_bank2d(Float64, N, J; kwargs...)

"""
    ComputedFilterBank2D{T,M,MV}

A 2D bank that keeps its wavelets' *parameters* rather than their samples, and evaluates one into a
single scratch array when it is asked for.

A bank of `J·L` wavelets is `J·L+1` arrays of the grid's size, and it is the largest object in a
transform: 1056 MiB at `2048², J=4, L=8, Float64`, against 320 MiB for the whole cascade's working
set. Evaluating instead of storing takes that to two arrays — the low-pass and the scratch — at the
cost of two `exp` per grid point per application, measured 3.2x (N=1024) to 3.8x (N=2048) on the
multiply, which is 9-14% of a cascade's time.

The low-pass has to be stored: it is `sqrt(1 - Σⱼ|ψⱼ|²)`, so recomputing it would mean evaluating
the whole bank. The tight-frame rescale is a single scalar and is applied on materialisation.

The scratch is mutable state, so unlike a stored bank this one cannot be shared between tasks —
[`task_bank`](@ref) gives a task its own, which is one array rather than `J·L`.
"""
struct ComputedFilterBank2D{T, M<:AbstractMatrix{T}, MV<:AbstractVector{WaveletMeta{T}},
                            WV<:AbstractVector{Filters.Morlet2D{T}}}
    morlets::WV
    rescale::T                # the tight-frame constant `_tight_frame_lowpass!` would have baked in
    averaging::M
    scratch::M
    meta::MV
    J::Int
    L::Int
end


"""
    build_filter_bank2d(T, N, J; L=8, cache=true)

`cache=false` returns a [`ComputedFilterBank2D`](@ref), which evaluates each wavelet on demand
instead of holding the whole bank. Slower per multiply, and the only way a large grid fits.
"""
function build_filter_bank2d(::Type{T}, N::NTuple{2, Int}, J::Int, ::Val{false};
                             L::Int = 8) where {T <: Real}
    morlets = [Filters.Morlet2D{T}(N, j, T(π) * l / L; L = L) for j in 0:(J - 1) for l in 0:(L - 1)]
    meta = [WaveletMeta{T}(j, 0, l, T(j), Filters.Morlet2D{T}(N, j, T(π) * l / L; L = L).center_freq,
                           T(π) * l / L) for j in 0:(J - 1) for l in 0:(L - 1)]
    # One pass over the bank to accumulate `Σⱼ|ψⱼ|²`, which fixes both the rescale and the low-pass.
    # Two arrays are live here and one is released; a stored bank would hold `J·L+1`.
    scratch = Matrix{T}(undef, N)
    acc = zeros(T, N)
    for m in morlets
        Filters.frequency_response!(scratch, m)
        @. acc += abs2(scratch)
    end
    mx = maximum(acc)
    c = mx > zero(T) ? inv(sqrt(mx)) : one(T)
    ϕ = Matrix{T}(undef, N)
    @. ϕ = sqrt(max(zero(T), one(T) - acc * c^2))
    ϕ[firstindex(ϕ)] = one(T)
    return ComputedFilterBank2D(morlets, c, ϕ, scratch, meta, J, L)
end
build_filter_bank2d(::Type{T}, N::NTuple{2, Int}, J::Int, ::Val{true}; L::Int = 8) where {T <: Real} =
    build_filter_bank2d(T, N, J; L = L)

"""
    ComputedFilterBank1D{T,V,MV}
    ComputedFilterBank3D{T,A,MV}

The 1D and 3D counterparts of [`ComputedFilterBank2D`](@ref); the saving grows with dimension,
since the bank is `J·n_orient+1` arrays of the grid either way.
"""
struct ComputedFilterBank1D{T, V <: AbstractVector{T}, MV <: AbstractVector{WaveletMeta{T}},
                            WV <: AbstractVector{Filters.Morlet1D{T}}}
    morlets::WV
    rescale::T
    averaging::V
    scratch::V
    meta::MV
    J::Int
    Q::Int
end

struct ComputedFilterBank3D{T, A <: AbstractArray{T, 3}, MV <: AbstractVector{WaveletMeta{T}},
                            WV <: AbstractVector{Filters.Morlet3D{T}}}
    morlets::WV
    rescale::T
    averaging::A
    scratch::A
    meta::MV
    J::Int
    n_orient::Int
end

# `Σⱼ|ψⱼ|²` in one sweep, holding one filter at a time. It fixes both the tight-frame rescale and
# the low-pass, which are the only two things a computed bank has to keep.
function _computed_norm(morlets, scratch, ::Type{T}) where {T}
    acc = zero(scratch)
    for m in morlets
        Filters.frequency_response!(scratch, m)
        @. acc += abs2(scratch)
    end
    mx = maximum(acc)
    c = mx > zero(T) ? inv(sqrt(mx)) : one(T)
    ϕ = similar(scratch)
    @. ϕ = sqrt(max(zero(T), one(T) - acc * c^2))
    ϕ[firstindex(ϕ)] = one(T)
    return c, ϕ
end

"""
    build_filter_bank1d(T, N, J; Q=1, cache=true)
    build_filter_bank3d(T, N, J; n_orient=6, cache=true)

`cache=false` evaluates each wavelet on demand instead of storing the bank.
"""
function build_filter_bank1d(::Type{T}, N::Int, J::Int, ::Val{false}; Q::Int = 1) where {T <: Real}
    morlets = [Filters.Morlet1D{T}(N, j * Q + q; Q = Q) for j in 0:(J - 1) for q in 0:(Q - 1)]
    meta = [WaveletMeta{T}(j, q, 0, T(j + q / Q), Filters.Morlet1D{T}(N, j * Q + q; Q = Q).center_freq,
                           zero(T)) for j in 0:(J - 1) for q in 0:(Q - 1)]
    scratch = Vector{T}(undef, N)
    c, ϕ = _computed_norm(morlets, scratch, T)
    return ComputedFilterBank1D(morlets, c, ϕ, scratch, meta, J, Q)
end
build_filter_bank1d(::Type{T}, N::Int, J::Int, ::Val{true}; Q::Int = 1) where {T <: Real} =
    build_filter_bank1d(T, N, J; Q = Q)

function build_filter_bank3d(::Type{T}, N::NTuple{3, Int}, J::Int, ::Val{false};
                             n_orient::Int = 6) where {T <: Real}
    dirs = Filters.fibonacci_directions(n_orient, T)
    morlets = [Filters.Morlet3D{T}(N, j, d) for j in 0:(J - 1) for d in dirs]
    meta = [WaveletMeta{T}(j, 0, o - 1, T(j), Filters.Morlet3D{T}(N, j, dirs[o]).center_freq, zero(T))
            for j in 0:(J - 1) for o in 1:n_orient]
    scratch = Array{T, 3}(undef, N)
    c, ϕ = _computed_norm(morlets, scratch, T)
    return ComputedFilterBank3D(morlets, c, ϕ, scratch, meta, J, n_orient)
end
build_filter_bank3d(::Type{T}, N::NTuple{3, Int}, J::Int, ::Val{true};
                    n_orient::Int = 6) where {T <: Real} =
    build_filter_bank3d(T, N, J; n_orient = n_orient)

"""
    FilterBank3D{T,A<:AbstractArray{Complex{T},3}}

Complete 3D oriented Morlet filter bank: `J` scales × `n_orient` sphere directions, plus a
low-pass averaging filter.
"""
struct FilterBank3D{T, A<:AbstractArray{T,3}, W<:AbstractVector{A}, MV<:AbstractVector{WaveletMeta{T}}}
    wavelets::W
    averaging::A
    meta::MV
    J::Int
    n_orient::Int
end

"""
    build_filter_bank3d(N::NTuple{3,Int}, J::Int; n_orient::Int=6, T=Float64) -> FilterBank3D

Build a 3D oriented Morlet filter bank with `J` dyadic scales and `n_orient` near-uniform
orientations on the sphere (Fibonacci spiral).
"""
function build_filter_bank3d(::Type{T}, N::NTuple{3,Int}, J::Int; n_orient::Int=6) where {T<:Real}
    dirs = Filters.fibonacci_directions(n_orient, T)
    morlet = Filters.Morlet3D{T}(N, 0, dirs[1])
    ψ_sample = Filters.frequency_response(morlet)
    A = typeof(ψ_sample)

    wavelets = Vector{A}(undef, 0)
    meta = Vector{WaveletMeta{T}}(undef, 0)
    for j in 0:(J - 1)
        for (o, d) in enumerate(dirs)
            morlet = Filters.Morlet3D{T}(N, j, d)
            push!(wavelets, Filters.frequency_response(morlet))
            push!(meta, WaveletMeta{T}(j, 0, o - 1, T(j), morlet.center_freq, zero(T)))
        end
    end
    # Low-pass = complement of the wavelet energy (tight-frame Littlewood-Paley ≈ 1)
    ϕ = _tight_frame_lowpass!(wavelets)
    return FilterBank3D(wavelets, ϕ, meta, J, n_orient)
end
build_filter_bank3d(N::NTuple{3,Int}, J::Int; kwargs...) = build_filter_bank3d(Float64, N, J; kwargs...)

# ---------------------------------------------------------------------------
# Bank accessors, defined here because they dispatch on all three stored banks
# ---------------------------------------------------------------------------

const StoredBank = Union{FilterBank1D, FilterBank2D, FilterBank3D}
const ComputedBank = Union{ComputedFilterBank1D, ComputedFilterBank2D, ComputedFilterBank3D}

"""
    nwavelets(fb) -> Int

How many wavelets the bank holds.
"""
nwavelets(fb::StoredBank) = length(fb.wavelets)
nwavelets(fb::ComputedBank) = length(fb.morlets)

"""
    filter_at(fb, j) -> AbstractArray

The bank's `j`-th wavelet in the Fourier domain.

For a stored bank this is an index. For a computed one it writes into shared scratch, so the result
is valid only until the next call — every cascade here finishes one filter's multiply before asking
for the next.
"""
@inline filter_at(fb::StoredBank, j::Integer) = fb.wavelets[j]
function filter_at(fb::ComputedBank, j::Integer)
    Filters.frequency_response!(fb.scratch, fb.morlets[j])
    fb.scratch .*= fb.rescale
    return fb.scratch
end

"""
    iscomputed(fb) -> Bool

Whether [`filter_at`](@ref) returns shared scratch rather than a stored array. A caller that needs
two filters live at once, or a filter at a resolution the bank does not hold, must materialise.
"""
iscomputed(::StoredBank) = false
iscomputed(::ComputedBank) = true

"""
    task_bank(fb) -> fb′

A bank usable concurrently with `fb`. Stored banks are read-only and are returned as they are; a
computed bank gets its own scratch, sharing everything else.
"""
task_bank(fb::StoredBank) = fb
task_bank(fb::ComputedFilterBank1D) =
    ComputedFilterBank1D(fb.morlets, fb.rescale, fb.averaging, similar(fb.scratch), fb.meta,
                         fb.J, fb.Q)
task_bank(fb::ComputedFilterBank2D) =
    ComputedFilterBank2D(fb.morlets, fb.rescale, fb.averaging, similar(fb.scratch), fb.meta,
                         fb.J, fb.L)
task_bank(fb::ComputedFilterBank3D) =
    ComputedFilterBank3D(fb.morlets, fb.rescale, fb.averaging, similar(fb.scratch), fb.meta,
                         fb.J, fb.n_orient)

"""
    batch_views(fb, spatial) -> Vector

The bank's filters reshaped to `(spatial…, 1)` so they broadcast over a batch axis, built once so
the batched cascade never reshapes in its inner loop.

A computed bank returns the same view `nwavelets` times — one view of the one scratch array. That
is only sound because [`filter_at`](@ref) refills the scratch immediately before each use, which is
what `Batched._filter` does.
"""
batch_views(fb::StoredBank, spatial::Tuple) =
    [reshape(ψ, (spatial..., 1)) for ψ in fb.wavelets]
batch_views(fb::ComputedBank, spatial::Tuple) =
    fill(reshape(fb.scratch, (spatial..., 1)), nwavelets(fb))

end # module FilterBanks
