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

# ---------------------------------------------------------------------------
# Littlewood–Paley normalization
#
# For a real field the wavelet transform `W` obeys `(1−α)‖x‖² ≤ ‖Wx‖² ≤ ‖x‖²` when
#
#     A(ω) = |φ̂(ω)|² + ½ Σ_λ (|ψ̂_λ(ω)|² + |ψ̂_λ(−ω)|²)   lies in   [1−α, 1]
#
# (Andén & Mallat 2014). The wavelets are scaled by one constant putting the maximum of the wavelet
# part of `A` at 1 for the bank continued over every scale. That bank is self-similar in scale, so its
# `A` is periodic in log-frequency (and in angle, for `L` orientations) and the maximum is taken over
# one period: the constant is fixed by the wavelet design (`Q` or `L` or `n_orient`, `sigma0`, `r`).
# ---------------------------------------------------------------------------

# The maximizer of `g` on `[a, b]` by golden-section search.
function _golden_argmax(g::G, a::Float64, b::Float64) where {G}
    r = (sqrt(5.0) - 1) / 2
    c, d = b - r * (b - a), a + r * (b - a)
    gc, gd = g(c), g(d)
    for _ in 1:60
        if gc > gd
            b, d, gd = d, c, gc
            c = b - r * (b - a)
            gc = g(c)
        else
            a, c, gc = c, d, gd
            d = a + r * (b - a)
            gd = g(d)
        end
    end
    return (a + b) / 2
end

# The maximum of `f` over `samples`, refined from the best one by golden-section search along each
# coordinate within `±h`, narrowing `h` each round.
function _refined_max(f::F, samples, h::NTuple{D, Float64}) where {F, D}
    best, fbest = first(samples), f(first(samples))
    for s in samples
        v = f(s)
        v > fbest && ((best, fbest) = (s, v))
    end
    x = best
    for _ in 1:4
        for d in 1:D
            t = _golden_argmax(_along(f, x, d), x[d] - h[d], x[d] + h[d])
            x = Base.setindex(x, t, d)
        end
        h = h ./ 4
    end
    return max(f(x), fbest)
end

# `f` as a function of coordinate `d` of `x` alone.
_along(f::F, x, d::Int) where {F} = u -> f(Base.setindex(x, u, d))

# The 1D Morlet profile at scale 0, `g(ω) = ψ̂₀(ω)`; scale `j` is `g(ω·2^{j/Q})`.
function _morlet1d_profile(Q::Int, r::Real)
    m = Filters.Morlet1D{Float64}(1, 0; Q = Q, r = Float64(r))
    ξ, σ = m.center_freq, m.bandwidth
    κ = exp(-(ξ / σ)^2 / 2)
    return ω -> ω < 0 ? 0.0 : exp(-((ω - ξ) / σ)^2 / 2) - κ * exp(-(ω / σ)^2 / 2), ξ
end

"""
    lp_scale_1d(Q; r=√½) -> c

The wavelet scale of a 1D Morlet bank with `Q` wavelets per octave: `½ Σ_j |c·ψ̂_j(ω)|²` peaks at 1
over the bank continued over every scale.
"""
function lp_scale_1d(Q::Int; r::Real = sqrt(0.5))
    g, ξ = _morlet1d_profile(Q, r)
    A(x) = 0.5 * sum(n -> g(ξ * 2.0^(x[1] + n / Q))^2, (-12Q):(12Q))
    np = 512
    return 1 / sqrt(_refined_max(A, ((k / (np * Q),) for k in 0:(np - 1)), (1 / (np * Q),)))
end

"""
    lp_scale_2d(L; sigma0=0.8) -> c

The wavelet scale of a 2D Morlet bank with `L` orientations: `½ Σ_{j,ℓ} (|c·ψ̂_{j,ℓ}(k)|² +
|c·ψ̂_{j,ℓ}(−k)|²)` peaks at 1 over the bank continued over every scale.
"""
function lp_scale_2d(L::Int; sigma0::Real = 0.8)
    σ, σp, k0 = Float64(sigma0), Float64(sigma0) * L / 4, 3π / 4
    β = exp(-(σ * k0)^2 / 2)
    h(a, b) = a < 0 ? 0.0 : exp(-((a - k0)^2 * σ^2 + b^2 * σp^2) / 2) - β * exp(-(a^2 * σ^2 + b^2 * σp^2) / 2)
    function A(x)   # |k| = k₀·2^s at angle φ
        s, φ = x
        acc = 0.0
        for n in -8:8, l in 0:(L - 1)
            ρ = k0 * 2.0^(s + n)
            a, b = ρ * cos(φ - π * l / L), ρ * sin(φ - π * l / L)
            acc += h(a, b)^2 + h(-a, -b)^2
        end
        return acc / 2
    end
    ns, na = 64, 16
    samples = ((i / ns, π * k / (L * na)) for i in 0:(ns - 1) for k in 0:(na - 1))
    return 1 / sqrt(_refined_max(A, samples, (1 / ns, π / (L * na))))
end

"""
    lp_scale_3d(n_orient; sigma0=0.8) -> c

The wavelet scale of a 3D Morlet bank over `n_orient` Fibonacci directions:
`½ Σ_{j,o} (|c·ψ̂_{j,o}(k)|² + |c·ψ̂_{j,o}(−k)|²)` peaks at 1 over the bank continued over every
scale.
"""
function lp_scale_3d(n_orient::Int; sigma0::Real = 0.8)
    dirs = Filters.fibonacci_directions(n_orient, Float64)
    σ, k0 = Float64(sigma0), 3π / 4
    σp = σ * sqrt(π * n_orient) / 8
    β = exp(-(σ * k0)^2 / 2)
    h(a, p2) = a < 0 ? 0.0 : exp(-((a - k0)^2 * σ^2 + p2 * σp^2) / 2) - β * exp(-(a^2 * σ^2 + p2 * σp^2) / 2)
    function A(x)   # |k| = k₀·2^s along the direction of polar angle θ, azimuth φ
        s, θ, φ = x
        u = (sin(θ) * cos(φ), sin(θ) * sin(φ), cos(θ))
        acc = 0.0
        for n in -8:8
            ρ = k0 * 2.0^(s + n)
            for d in dirs
                a = ρ * (u[1] * d[1] + u[2] * d[2] + u[3] * d[3])
                p2 = max(0.0, ρ^2 - a^2)
                acc += h(a, p2)^2 + h(-a, p2)^2
            end
        end
        return acc / 2
    end
    ns = 24
    pts = Filters.fibonacci_directions(600, Float64)
    samples = ((i / ns, acos(clamp(p[3], -1.0, 1.0)), atan(p[2], p[1])) for i in 0:(ns - 1) for p in pts)
    return 1 / sqrt(_refined_max(A, samples, (1 / ns, 0.1, 0.1)))
end

"""
    littlewood_paley(fb) -> A

The bank's Littlewood–Paley sum for a real field, `A(ω) = |φ̂(ω)|² + ½ Σ_λ (|ψ̂_λ(ω)|² + |ψ̂_λ(−ω)|²)`,
on its frequency grid. The wavelet transform of a real field is bounded by
`minimum(A)·‖x‖² ≤ ‖Wx‖² ≤ maximum(A)·‖x‖²`, and `Inverse.iwavelet!` divides by `A`.
"""
function littlewood_paley(fb)
    S = zero(fb.averaging)
    for j in 1:nwavelets(fb)
        S .+= abs2.(filter_at(fb, j))
    end
    return abs2.(fb.averaging) .+ (S .+ negated(S)) ./ 2
end

"""
    negated(A) -> B

`B[k] = A[−k]` on the DFT grid, with each index taken modulo its axis.
"""
negated(A::AbstractArray) = circshift(reverse(A), ntuple(_ -> 1, ndims(A)))

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
    
    c = T(lp_scale_1d(Q))
    foreach(ψ -> ψ .*= c, wavelets)
    return FilterBank1D(wavelets, Filters.gaussian_lowpass(T, (N,), J), meta, J, Q)
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
    
    c = T(lp_scale_2d(L))
    foreach(ψ -> ψ .*= c, wavelets)
    return FilterBank2D(wavelets, Filters.gaussian_lowpass(T, N, J), meta, J, L)
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

The low-pass is stored, and each wavelet is scaled by the Littlewood–Paley constant as it is
materialised.

The scratch is mutable state, so unlike a stored bank this one cannot be shared between tasks —
[`task_bank`](@ref) gives a task its own, which is one array rather than `J·L`.
"""
struct ComputedFilterBank2D{T, M<:AbstractMatrix{T}, MV<:AbstractVector{WaveletMeta{T}},
                            WV<:AbstractVector{Filters.Morlet2D{T}}}
    morlets::WV
    rescale::T                # the Littlewood–Paley constant, `lp_scale_2d(L)`
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
    meta = [WaveletMeta{T}(j, 0, l, T(j), morlets[j * L + l + 1].center_freq, T(π) * l / L)
            for j in 0:(J - 1) for l in 0:(L - 1)]
    return ComputedFilterBank2D(morlets, T(lp_scale_2d(L)), Filters.gaussian_lowpass(T, N, J),
                                Matrix{T}(undef, N), meta, J, L)
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

"""
    build_filter_bank1d(T, N, J; Q=1, cache=true)
    build_filter_bank3d(T, N, J; n_orient=6, cache=true)

`cache=false` evaluates each wavelet on demand instead of storing the bank.
"""
function build_filter_bank1d(::Type{T}, N::Int, J::Int, ::Val{false}; Q::Int = 1) where {T <: Real}
    morlets = [Filters.Morlet1D{T}(N, j * Q + q; Q = Q) for j in 0:(J - 1) for q in 0:(Q - 1)]
    meta = [WaveletMeta{T}(j, q, 0, T(j + q / Q), morlets[j * Q + q + 1].center_freq, zero(T))
            for j in 0:(J - 1) for q in 0:(Q - 1)]
    return ComputedFilterBank1D(morlets, T(lp_scale_1d(Q)), Filters.gaussian_lowpass(T, (N,), J),
                                Vector{T}(undef, N), meta, J, Q)
end
build_filter_bank1d(::Type{T}, N::Int, J::Int, ::Val{true}; Q::Int = 1) where {T <: Real} =
    build_filter_bank1d(T, N, J; Q = Q)

function build_filter_bank3d(::Type{T}, N::NTuple{3, Int}, J::Int, ::Val{false};
                             n_orient::Int = 6) where {T <: Real}
    dirs = Filters.fibonacci_directions(n_orient, T)
    morlets = [Filters.Morlet3D{T}(N, j, d; n_orient = n_orient) for j in 0:(J - 1) for d in dirs]
    meta = [WaveletMeta{T}(j, 0, o - 1, T(j), morlets[j * n_orient + o].center_freq, zero(T))
            for j in 0:(J - 1) for o in 1:n_orient]
    return ComputedFilterBank3D(morlets, T(lp_scale_3d(n_orient)), Filters.gaussian_lowpass(T, N, J),
                                Array{T, 3}(undef, N), meta, J, n_orient)
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
    morlet = Filters.Morlet3D{T}(N, 0, dirs[1]; n_orient = n_orient)
    ψ_sample = Filters.frequency_response(morlet)
    A = typeof(ψ_sample)

    wavelets = Vector{A}(undef, 0)
    meta = Vector{WaveletMeta{T}}(undef, 0)
    for j in 0:(J - 1)
        for (o, d) in enumerate(dirs)
            morlet = Filters.Morlet3D{T}(N, j, d; n_orient = n_orient)
            push!(wavelets, Filters.frequency_response(morlet))
            push!(meta, WaveletMeta{T}(j, 0, o - 1, T(j), morlet.center_freq, zero(T)))
        end
    end
    c = T(lp_scale_3d(n_orient))
    foreach(ψ -> ψ .*= c, wavelets)
    return FilterBank3D(wavelets, Filters.gaussian_lowpass(T, N, J), meta, J, n_orient)
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
