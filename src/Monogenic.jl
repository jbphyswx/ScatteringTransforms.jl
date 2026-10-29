module Monogenic

# Monogenic wavelet scattering (Felsberg & Sommer 2001; Unser et al. 2009). For a real field `x`, an
# isotropic band-pass `ψ_j` (real, radial, zero-mean) and the Riesz multipliers `R_d(k) = -i k_d/|k|`:
#
#     m₀ = x ⋆ ψ_j,   m_d = x ⋆ (R_d ψ_j),   A_j = √(m₀² + Σ_d m_d²)
#
# The rotation-covariant monogenic amplitude `A_j` takes the place of the analytic modulus
# `|x ⋆ ψ_λ|`, over the same path graph. The bank's Littlewood–Paley sum
# `|φ̂|² + Σ_j |ψ̂_j|² (1 + Σ_d |R_d|²)` is at most 1.

using ..Filters: Filters
using ..FilterBanks: FilterBanks
using ..Plans: Plans
using SpectralBackends: SpectralBackends as SB
using ..PathGraph: PathGraph
using ..Coefficients: Coefficients
using ..ScatteringCore: ScatteringCore
using ..Cascade: Cascade
using ..ScatteringFields: ScatteringFields

export MonogenicFilterBank, ComputedMonogenicFilterBank, MonogenicScatteringTransform
export build_monogenic_bank, riesz_multipliers, monogenic_amplitude, monogenic_components
export monogenic_scattering!

# Radial (isotropic) band-pass response over a D-dim frequency grid: a Gaussian bump on |k| at
# center ξ with width σ, minus a low-pass term pinning the DC bin to zero (admissibility).
_radial_bandpass(dims::NTuple{D,Int}, ξ::T, σ::T) where {D,T} =
    _radial_bandpass!(Array{T,D}(undef, dims), ξ, σ)

function _radial_bandpass!(ψ::AbstractArray{T,D}, ξ::T, σ::T) where {D,T}
    κ = exp(-(ξ / σ)^2 / 2)
    dims = size(ψ)
    function response(I)
        kk = zero(T)
        @inbounds for d in 1:D
            f = T(Filters._frequency(dims[d], I[d]))
            kk += f * f
        end
        kn = sqrt(kk)
        g = exp(-((kn - ξ) / σ)^2 / 2)
        lp = exp(-(kn / σ)^2 / 2)
        return g - κ * lp
    end
    ψ .= response.(CartesianIndices(ψ))
    return ψ
end

"""
    riesz_multipliers(dims, ::Type{T}=Float64) -> NTuple{D, Array{Complex{T},D}}

The `D` Riesz-transform frequency multipliers `R_d(k) = -i k_d/|k|` over a grid of size `dims`, with
`R_d(0) = 0`. On an even axis `d` the Nyquist bin holds `k_d = ±½` at once, and there `R_d` is the
mean of its two values, `0`, so a real field's Riesz components are real. Scale-free, so one set
serves every wavelet scale; `Σ_d |R_d(k)|² = 1` off the DC bin and the Nyquist planes.
"""
function riesz_multipliers(dims::NTuple{D,Int}, ::Type{T}=Float64) where {D,T}
    R = ntuple(_ -> Array{Complex{T},D}(undef, dims), D)
    @inbounds for I in CartesianIndices(dims)
        kk = zero(T)
        ks = ntuple(d -> T(Filters._frequency(dims[d], I[d])), D)
        for d in 1:D
            kk += ks[d]^2
        end
        kn = sqrt(kk)
        @inbounds for d in 1:D
            nyq = Filters._isnyquist(dims[d], I[d])
            R[d][I] = kn == zero(T) || nyq ? zero(Complex{T}) : Complex{T}(zero(T), -ks[d] / kn)
        end
    end
    return R
end

"""
    MonogenicFilterBank{D,T,A,W,R,MV}

Isotropic band-pass wavelets `wavelets` (one per scale/sub-octave), the `D` scale-free Riesz
multipliers `riesz`, and the Gaussian low-pass `averaging`. The wavelets and the low-pass are real
functions of `|k|`; only the Riesz multipliers are complex.
"""
struct MonogenicFilterBank{D, T, A<:AbstractArray{T,D}, W<:AbstractVector{A},
                           R<:NTuple{D,AbstractArray{Complex{T},D}},
                           MV<:AbstractVector{FilterBanks.WaveletMeta{T}}}
    wavelets::W
    riesz::R
    averaging::A
    meta::MV
    J::Int
    Q::Int
end

"""
    build_monogenic_bank([T=Float64,] dims::NTuple{D,Int}, J; Q=1) -> MonogenicFilterBank

Build a `D`-dimensional isotropic Morlet-style monogenic filter bank: `J` octaves × `Q`
sub-octaves of radial band-pass wavelets whose radial profile is [`Filters.Morlet1D`](@ref)'s
(center frequency `ξ_j = ξ₀·2^{-j/Q}`, adjacent wavelets crossing at `√½` of their peak), the Riesz
multipliers, and [`Filters.gaussian_lowpass!`](@ref). The band-pass and its
Riesz components carry `2|ψ̂_j|²` per frequency, so the wavelets are scaled by
`c = lp_scale_1d(Q)/2`, which puts `2c² Σ_j g(|k|·2^{j/Q})²` (`g` the radial profile) at a maximum
of 1 over the bank continued over every scale.
"""
build_monogenic_bank(dims::NTuple{D,Int}, J::Int; kwargs...) where {D} =
    build_monogenic_bank(Float64, dims, J; kwargs...)

function build_monogenic_bank(::Type{T}, dims::NTuple{D,Int}, J::Int; Q::Int=1) where {T<:Real,D}
    A = Array{T,D}
    wavelets = Vector{A}(undef, 0)
    meta = Vector{FilterBanks.WaveletMeta{T}}(undef, 0)
    for j in 0:(J - 1)
        for q in 0:(Q - 1)
            # Reuse the 1D Morlet radial profile (normalized-frequency ξ, σ).
            m = Filters.Morlet1D{T}(1, j * Q + q; Q=Q)
            push!(wavelets, _radial_bandpass(dims, m.center_freq, m.bandwidth))
            push!(meta, FilterBanks.WaveletMeta{T}(j, q, 0, T(j + q / Q), m.center_freq, zero(T)))
        end
    end
    c = _monogenic_scale(T, Q)
    foreach(ψ -> ψ .*= c, wavelets)
    R = riesz_multipliers(dims, T)
    return MonogenicFilterBank{D,T,A,typeof(wavelets),typeof(R),typeof(meta)}(
        wavelets, R, Filters.gaussian_lowpass(T, dims, J), meta, J, Q)
end

_monogenic_scale(::Type{T}, Q::Int) where {T} = T(FilterBanks.lp_scale_1d(Q) / 2)

"""
    ComputedMonogenicFilterBank{D,T,A,R,MV,PV}

The monogenic bank with its radial band-passes evaluated on demand into one scratch array, as in
[`FilterBanks.ComputedFilterBank2D`](@ref). The `D` Riesz multipliers are stored.
"""
struct ComputedMonogenicFilterBank{D, T, A<:AbstractArray{T,D},
                                   R<:NTuple{D,AbstractArray{Complex{T},D}},
                                   MV<:AbstractVector{FilterBanks.WaveletMeta{T}},
                                   PV<:AbstractVector{NTuple{2,T}}}
    profiles::PV   # (ξ, σ) per wavelet — everything `_radial_bandpass` needs
    rescale::T
    riesz::R
    averaging::A
    scratch::A
    meta::MV
    J::Int
    Q::Int
end

const AnyMonogenicBank = Union{MonogenicFilterBank, ComputedMonogenicFilterBank}

FilterBanks.nwavelets(fb::MonogenicFilterBank) = length(fb.wavelets)
FilterBanks.nwavelets(fb::ComputedMonogenicFilterBank) = length(fb.profiles)
FilterBanks.filter_at(fb::MonogenicFilterBank, j::Integer) = fb.wavelets[j]
function FilterBanks.filter_at(fb::ComputedMonogenicFilterBank, j::Integer)
    ξ, σ = fb.profiles[j]
    _radial_bandpass!(fb.scratch, ξ, σ)
    fb.scratch .*= fb.rescale
    return fb.scratch
end
FilterBanks.iscomputed(::MonogenicFilterBank) = false
FilterBanks.iscomputed(::ComputedMonogenicFilterBank) = true
FilterBanks.filter_storage(::MonogenicFilterBank) = FilterBanks.StoredFilters()
FilterBanks.filter_storage(::ComputedMonogenicFilterBank) = FilterBanks.ComputedFilters()
FilterBanks.task_bank(fb::MonogenicFilterBank) = fb
FilterBanks.task_bank(fb::ComputedMonogenicFilterBank) =
    ComputedMonogenicFilterBank(fb.profiles, fb.rescale, fb.riesz, fb.averaging,
                                similar(fb.scratch), fb.meta, fb.J, fb.Q)

build_monogenic_bank(::Type{T}, dims::NTuple{D,Int}, J::Int, ::FilterBanks.StoredFilters;
                     Q::Int=1) where {T<:Real,D} = build_monogenic_bank(T, dims, J; Q=Q)

function build_monogenic_bank(::Type{T}, dims::NTuple{D,Int}, J::Int, ::FilterBanks.ComputedFilters;
                              Q::Int=1) where {T<:Real,D}
    profiles = NTuple{2,T}[]
    meta = FilterBanks.WaveletMeta{T}[]
    for j in 0:(J - 1), q in 0:(Q - 1)
        m = Filters.Morlet1D{T}(1, j * Q + q; Q=Q)
        push!(profiles, (m.center_freq, m.bandwidth))
        push!(meta, FilterBanks.WaveletMeta{T}(j, q, 0, T(j + q / Q), m.center_freq, zero(T)))
    end
    R = riesz_multipliers(dims, T)
    return ComputedMonogenicFilterBank(profiles, _monogenic_scale(T, Q), R,
                                       Filters.gaussian_lowpass(T, dims, J),
                                       Array{T,D}(undef, dims), meta, J, Q)
end

"""
    littlewood_paley(fb::AnyMonogenicBank) -> A

`A(k) = |φ̂(k)|² + Σ_j |ψ̂_j(k)|² (1 + Σ_d |R_d(k)|²)`, the energy the band-pass, Riesz and low-pass
outputs of a real field carry per frequency.
"""
function FilterBanks.littlewood_paley(fb::AnyMonogenicBank)
    S = zero(fb.averaging)
    for j in 1:FilterBanks.nwavelets(fb)
        S .+= abs2.(FilterBanks.filter_at(fb, j))
    end
    Rsum = sum(Rd -> abs2.(Rd), fb.riesz)
    return abs2.(fb.averaging) .+ S .* (1 .+ Rsum)
end

"""
    monogenic_amplitude(m0, riesz_components) -> A

Monogenic amplitude `A = √(m₀² + Σ_d m_d²)` from the band-pass field `m0` and the tuple/vector of
Riesz component fields. Broadcast, so it is CPU/GPU/autodiff-generic.
"""
function monogenic_amplitude(m0::AbstractArray, riesz)
    acc = m0 .^ 2
    for m in riesz
        acc = acc .+ m .^ 2
    end
    return sqrt.(acc)
end

"""
    MonogenicWorkspace{T,D}

The periodized cascade plus what the monogenic amplitude needs on top of it: one real accumulator
per resolution, and the Riesz-weighted band-passes `R_d ψ_j` periodized to each *parent* resolution.

Each `R_d ψ_j` is stored as the periodization of the product, which differs from the product of the
periodized factors. Level 1 stores none: the two full-resolution filters fuse in
[`ScatteringCore.periodize_mul2!`](@ref).
"""
struct MonogenicWorkspace{T, D, PW <: Cascade.PeriodizedWorkspace{T, D},
                          AV <: AbstractVector{<:AbstractArray{T, D}}, FV <: AbstractVector}
    pw::PW
    amp::AV        # per level: the amplitude accumulator at that resolution
    rfilters::FV   # rfilters[l][j][d]; empty at level 1 and wherever `j` is not applied at `l`
end

MonogenicWorkspace(pw::Cascade.PeriodizedWorkspace{T,D}, amp, rfilters) where {T,D} =
    MonogenicWorkspace{T,D,typeof(pw),typeof(amp),typeof(rfilters)}(pw, amp, rfilters)

Base.show(io::IO, mw::MonogenicWorkspace{T, D}) where {T, D} =
    print(io, "MonogenicWorkspace{", T, ", ", D, "}(", mw.pw, ")")

"""
    MonogenicScatteringTransform{D}

Monogenic scattering transform on a `D`-dimensional grid: an isotropic [`MonogenicFilterBank`](@ref),
the scattering path tree (strictly-increasing scale), a spectral `plan`, and the periodized
[`MonogenicWorkspace`](@ref) the cascade runs in.

`D` is the number of spatial axes in `dims`.
"""
struct MonogenicScatteringTransform{D, T, FB<:AnyMonogenicBank, Tree<:PathGraph.ScatteringTree,
                           P<:Plans.AbstractScatteringPlan, G<:AbstractVector,
                           CB<:AbstractArray{Complex{T},D}, MW<:MonogenicWorkspace{T,D}, W} <: ScatteringCore.AbstractScatteringTransform
    filter_bank::FB
    tree::Tree
    groups::G               # (j1, children, path ids), longest-first — the cascade's work list
    max_order::Int
    plan::P
    dims::NTuple{D,Int}
    buffer_signal_fft::CB   # preserved signal FFT, at full resolution
    mw::MW
    workspaces::W
end

MonogenicScatteringTransform(fb, tree, groups, max_order, plan, dims,
                    buffer::AbstractArray{Complex{T},D}, mw) where {T,D} =
    MonogenicScatteringTransform(fb, tree, groups, max_order, plan, dims, buffer, mw,
                         ScatteringCore.Workspaces(MonogenicScatteringTransform))

MonogenicScatteringTransform(fb, tree, groups, max_order, plan, dims,
                    buffer::AbstractArray{Complex{T},D}, mw, workspaces) where {T,D} =
    MonogenicScatteringTransform{D,T,typeof(fb),typeof(tree),typeof(plan),typeof(groups),typeof(buffer),typeof(mw),typeof(workspaces)}(
        fb, tree, groups, max_order, plan, dims, buffer, mw, workspaces)

# The path tree of bank `fb` and the cascade's work list over it.
function tree_groups(fb, max_order::Int)
    tree = PathGraph.build_tree([m.j_eff for m in fb.meta], max_order)
    nw = FilterBanks.nwavelets(fb)
    groups = max_order >= 2 ? PathGraph.order2_groups(tree, nw) : [(j, Int[], Int[]) for j in 1:nw]
    return tree, groups
end

# `R_d ψ_j` periodized onto each parent resolution, for only the pairs the cascade applies, and the
# amplitude accumulators, all on the array type of the bank's filters.
function _monogenic_workspace(fb, pw::Cascade.PeriodizedWorkspace{T, D}) where {T, D}
    nw = FilterBanks.nwavelets(fb)
    R = fb.riesz
    dims = pw.dims
    proto = fb.averaging
    amp = [fill!(similar(proto, T, ntuple(d -> dims[d] ÷ r[d], D)), zero(T)) for r in pw.resolutions]
    empty_r = ntuple(_ -> similar(proto, Complex{T}, ntuple(_ -> 0, D)), D)
    scratch = similar(proto, Complex{T}, dims)
    rfilters = map(eachindex(pw.resolutions)) do l
        v = fill(empty_r, nw)
        l == 1 && return v
        rl = pw.resolutions[l]
        rd = ntuple(d -> dims[d] ÷ rl[d], D)
        for j in 1:nw
            isempty(pw.filters[l][j]) && continue        # `j` is never applied at this level
            ψ = FilterBanks.filter_at(fb, j)
            v[j] = ntuple(D) do d
                @. scratch = R[d] * ψ
                pf = similar(proto, Complex{T}, rd)
                ScatteringCore.periodize_filter!(pf, scratch, rl)
                pf
            end
        end
        v
    end
    return MonogenicWorkspace{T, D, typeof(pw), typeof(amp), typeof(rfilters)}(pw, amp, rfilters)
end

# The transform over bank `fb`, its path tree and work list, the full-grid `plan` and the cascade
# workspace `pw`, every other buffer on the array type of the bank's filters.
function assemble(fb, tree, groups, max_order::Int, plan,
                  pw::Cascade.PeriodizedWorkspace{T, D}) where {T, D}
    mw = _monogenic_workspace(fb, pw)
    buf = fill!(similar(fb.averaging, Complex{T}, pw.dims), zero(Complex{T}))
    return MonogenicScatteringTransform(fb, tree, groups, max_order, plan, pw.dims, buf, mw)
end

"""
    MonogenicScatteringTransform([T=Float64,] dims, J; Q=1, max_order=2, filters=StoredFilters(),
                        oversampling=J, transform=AutoSpectralBackend())

Construct a monogenic scattering transform for fields of size `dims` (`Int` or `NTuple{D,Int}`).
The element type is positional, as for `zeros(T, …)`; omit it for `Float64`.
`filters = FilterBanks.ComputedFilters()` evaluates the band-passes on demand
([`ComputedMonogenicFilterBank`](@ref)); `oversampling` and `transform` are as for the gridded
transforms. The KernelAbstractions extension adds `MonogenicScatteringTransform([T,] dims, J, gpu::GPUBackend; …)`,
a transform on that device.
"""
function MonogenicScatteringTransform(::Type{T}, dims::NTuple{D,Int}, J::Int; Q::Int=1, max_order::Int=2,
                             filters::FilterBanks.AbstractFilterStorage=FilterBanks.StoredFilters(),
                             oversampling::Int=J,
                             transform::SB.AbstractSpectralBackend=SB.AutoSpectralBackend()) where {T,D}
    fb = build_monogenic_bank(T, dims, J, filters; Q=Q)
    plan = Plans.make_plan(transform, T, dims)
    tree, groups = tree_groups(fb, max_order)
    pw = Cascade.build(fb, groups, dims, T, oversampling, plan)
    return assemble(fb, tree, groups, max_order, plan, pw)
end
MonogenicScatteringTransform(::Type{T}, N::Int, J::Int; kwargs...) where {T} =
    MonogenicScatteringTransform(T, (N,), J; kwargs...)
MonogenicScatteringTransform(dims::NTuple{D,Int}, J::Int; kwargs...) where {D} =
    MonogenicScatteringTransform(Float64, dims, J; kwargs...)
MonogenicScatteringTransform(N::Int, J::Int; kwargs...) = MonogenicScatteringTransform(Float64, (N,), J; kwargs...)

Base.show(io::IO, st::MonogenicScatteringTransform{D, T}) where {D, T} =
    print(io, "MonogenicScatteringTransform{", D, ", ", T, "}(dims=", st.dims, ", J=", st.filter_bank.J, ", Q=",
          st.filter_bank.Q, ", max_order=", st.max_order, ", oversampling=", st.mw.pw.oversampling, ", ",
          st.plan, ")")

# `√(|m₀|² + Σ_d |m_d|²)` into `dst`, from the full-resolution spectrum `Xf` band-passed by wavelet `j`
# and produced on level `l`'s grid. Both filters are full resolution, so the Riesz weighting fuses
# into the multiply and nothing is materialised. Leaves `Xf` intact. A real field's `m` are real.
function _amp_from_full!(dst, mw::MonogenicWorkspace{T, D}, fb, Xf, j::Int, l::Int) where {T, D}
    pw = mw.pw
    r = pw.resolutions[l]
    R = fb.riesz
    w, o = pw.work[l], pw.out[l]
    ψ = Cascade._filter(pw, fb, 1, pw.filters[1], j)
    ScatteringCore.periodize_mul!(w, Xf, ψ, r)
    Plans.inverse_transform!(o, pw.plans[l], w)
    @. dst = abs2(o)
    for d in 1:D
        ScatteringCore.periodize_mul2!(w, Xf, R[d], ψ, r)
        Plans.inverse_transform!(o, pw.plans[l], w)
        @. dst += abs2(o)
    end
    @. dst = sqrt(dst)
    return dst
end

# The same, from a first-order amplitude's spectrum `u1f` living on level `l1`, produced on `l2`.
# Here the band-pass has already been periodized to `l1`, so the Riesz-weighted one must be the
# periodized *product* — except at `l1 == 1`, where the factors are still full resolution and fuse.
function _amp_from_parent!(dst, mw::MonogenicWorkspace{T, D}, fb, u1f, j::Int, l1::Int,
                           l2::Int) where {T, D}
    pw = mw.pw
    rr = ntuple(d -> pw.resolutions[l2][d] ÷ pw.resolutions[l1][d], D)
    w, o = pw.work[l2], pw.out[l2]
    ScatteringCore.periodize_mul!(w, u1f, Cascade._filter(pw, fb, l1, pw.filters[l1], j), rr)
    Plans.inverse_transform!(o, pw.plans[l2], w)
    @. dst = abs2(o)
    if l1 == 1
        R = fb.riesz
        ψ = Cascade._filter(pw, fb, 1, pw.filters[1], j)
        for d in 1:D
            ScatteringCore.periodize_mul2!(w, u1f, R[d], ψ, rr)
            Plans.inverse_transform!(o, pw.plans[l2], w)
            @. dst += abs2(o)
        end
    else
        rf = mw.rfilters[l1][j]
        for d in 1:D
            ScatteringCore.periodize_mul!(w, u1f, rf[d], rr)
            Plans.inverse_transform!(o, pw.plans[l2], w)
            @. dst += abs2(o)
        end
    end
    @. dst = sqrt(dst)
    return dst
end

"""
    (st::MonogenicScatteringTransform)(x) -> ScatteringCoefficients1D

Apply the monogenic scattering transform; returns averaged coefficients in a
`ScatteringCoefficients1D` (`S0` mean, `S1` over scales, `S2` over scale pairs). A complex field
`x = a + ib` has amplitude `√(|m₀|² + Σ_d |m_d|²)`, which is `√(A(a)² + A(b)²)` in terms of the
amplitudes of its real and imaginary parts, and a complex `S0`.
"""
function (st::MonogenicScatteringTransform{D})(x::AbstractArray) where {D}
    T = real(eltype(st.buffer_signal_fft))
    n = FilterBanks.nwavelets(st.filter_bank)
    coeffs = Coefficients.ScatteringCoefficients1D(n, T; compute_S2=st.max_order >= 2)
    return monogenic_scattering!(coeffs, st, x)
end

"""
    monogenic_scattering!(coeffs, st, x) -> coeffs

Write the averaged monogenic coefficients of `x` into `coeffs`.
"""
function monogenic_scattering!(coeffs::Coefficients.ScatteringCoefficients1D,
                                st::MonogenicScatteringTransform, x::AbstractArray)
    mw = st.mw
    pw = mw.pw
    fb = st.filter_bank

    c0 = pw.work[1]
    @. c0 = complex(x)
    Plans.forward_transform!(st.buffer_signal_fft, st.plan, c0)
    Xf = st.buffer_signal_fft

    S1 = coeffs.S1
    S2 = coeffs.S2
    isempty(S2) || fill!(S2, zero(eltype(S2)))
    Cascade.begin_means!(pw.means)
    @inbounds for (j1, children, _) in st.groups
        l1 = pw.level[j1]
        a1 = mw.amp[l1]
        _amp_from_full!(a1, mw, fb, Xf, j1, l1)
        Cascade.mean_to!(S1, j1, pw.means, a1)
        isempty(children) && continue
        # `a1` is copied out before any child can reuse its level's accumulator.
        w1 = pw.work[l1]
        @. w1 = complex(a1)
        u1f = pw.spec[l1]
        Plans.forward_transform!(u1f, pw.plans[l1], w1)
        for j2 in children
            l2 = pw.level[j2]
            a2 = mw.amp[l2]
            _amp_from_parent!(a2, mw, fb, u1f, j2, l1, l2)
            Cascade.mean_to!(S2, LinearIndices(S2)[j1, j2], pw.means, a2)
        end
    end
    Cascade.gather_means!(S1, S2, pw.means)

    S0 = ScatteringCore.spatial_average(x)
    return Coefficients.update_S0(coeffs, S0)
end

function ScatteringCore.task_workspace(st::MonogenicScatteringTransform)
    fb = FilterBanks.task_bank(st.filter_bank)
    pw = Cascade.task_copy(st.mw.pw, fb)
    mw = MonogenicWorkspace(pw, map(similar, st.mw.amp), st.mw.rfilters)
    return MonogenicScatteringTransform(fb, st.tree, st.groups, st.max_order, first(pw.plans),
                                st.dims, similar(st.buffer_signal_fft), mw)
end

function ScatteringFields.scattering_field(st::MonogenicScatteringTransform{D}, x::AbstractArray;
        subsample::Int = ScatteringFields.default_subsample(st.dims, st.filter_bank.J)) where {D}
    all(n -> n % subsample == 0, st.dims) ||
        throw(ArgumentError("subsample must divide each field dimension"))
    buf = st.buffer_signal_fft
    T = real(eltype(buf))
    alloc(rd) = similar(buf, Complex{T}, rd)
    fw = Cascade.build_field(st.filter_bank, st.groups, st.dims, T, st.mw.pw.oversampling,
                              subsample, st.filter_bank.averaging; alloc,
                              makeplan = rd -> Plans.plan_like(st.plan, alloc(rd)))
    mw = _monogenic_workspace(st.filter_bank, fw.pw)
    dims = map(n -> n ÷ subsample, st.dims)
    S = promote_type(T, eltype(x))
    data = fill!(similar(buf, S, (dims..., PathGraph.npaths(st.tree))), zero(S))
    field = ScatteringFields.field_container(st.tree, data, subsample, (field = fw, monogenic = mw))
    return ScatteringFields.scattering_field!(field, st, x)
end

function ScatteringFields.scattering_field!(field, st::MonogenicScatteringTransform{D}, x::AbstractArray) where {D}
    fw, mw = field.ws.field, field.ws.monogenic
    pw = mw.pw
    @. pw.work[1] = complex(x)
    Plans.forward_transform!(st.buffer_signal_fft, pw.plans[1], pw.work[1])
    Xf = st.buffer_signal_fft
    p1 = first(PathGraph.order_range(st.tree, 1))
    root = first(PathGraph.order_range(st.tree, 0))
    Cascade.localize!(Cascade._pathview(field.data, Val(D), root), fw, Xf, 1)
    for (j1, children, paths) in st.groups
        l1 = pw.level[j1]
        a1 = _amp_from_full!(mw.amp[l1], mw, st.filter_bank, Xf, j1, l1)
        @. pw.work[l1] = complex(a1)
        u1f = isempty(children) ? pw.out[l1] : pw.spec[l1]
        Plans.forward_transform!(u1f, pw.plans[l1], pw.work[l1])
        Cascade.localize!(Cascade._pathview(field.data, Val(D), p1 + j1 - 1), fw, u1f, l1)
        for (j2, path) in zip(children, paths)
            l2 = pw.level[j2]
            a2 = _amp_from_parent!(mw.amp[l2], mw, st.filter_bank, u1f, j2, l1, l2)
            @. pw.work[l2] = complex(a2)
            Plans.forward_transform!(pw.out[l2], pw.plans[l2], pw.work[l2])
            Cascade.localize!(Cascade._pathview(field.data, Val(D), path), fw, pw.out[l2], l2)
        end
    end
    return field
end

"""
    monogenic_components(st, x, j) -> (; bandpass, riesz, amplitude, phase)

The monogenic decomposition of a real field `x` band-passed by isotropic wavelet `j` (1-based): the
band-pass field `bandpass = x ⋆ ψ_j`, the `D` Riesz component fields `riesz`, the monogenic
`amplitude` `√(bandpass² + Σ|riesz|²)`, and the local monogenic `phase = atan(‖riesz‖, bandpass)`. The
Riesz vector's direction gives the local orientation (e.g. `atan(riesz[2], riesz[1])` in 2D).
"""
function monogenic_components(st::MonogenicScatteringTransform{D}, x::AbstractArray, j::Integer) where {D}
    eltype(x) <: Real || throw(ArgumentError(
        "monogenic_components takes a real field, whose local phase and orientation the monogenic " *
        "signal defines; pass the real and imaginary parts of a complex field separately"))
    plan = st.plan
    fb = st.filter_bank
    ψ = FilterBanks.filter_at(fb, j)
    Xf = Plans.forward_transform(plan, complex.(x))
    bandpass = real.(Plans.inverse_transform(plan, Xf .* ψ))
    riesz = ntuple(d -> real.(Plans.inverse_transform(plan, Xf .* fb.riesz[d] .* ψ)), D)
    amplitude = monogenic_amplitude(bandpass, riesz)
    rnorm = sqrt.(sum(m -> m .^ 2, riesz))
    phase = atan.(rnorm, bandpass)
    return (; bandpass, riesz, amplitude, phase)
end

end # module Monogenic
