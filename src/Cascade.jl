module Cascade

# The periodized scattering cascade shared by 1D, 2D and 3D. A scale-`j` wavelet's output is produced
# directly on a grid decimated by `r = 2^max(j-α, 0)` per axis (`α` is `oversampling`): by
# `ScatteringCore.periodize_mul!`, the decimated samples of a full-size inverse transform equal the
# small inverse transform of the periodized product spectrum. `α ≥ J` decimates nothing and
# reproduces the undecimated cascade bit for bit. Coarse filters are periodizations of the
# full-resolution filter (`ScatteringCore.periodize_filter!`).

using ..Plans: Plans
using ..ScatteringCore: ScatteringCore
using ..FilterBanks: FilterBanks
using ..Filters: Filters
using ..PathGraph: PathGraph

"""
    decimation(N, scale, α) -> NTuple{D,Int}

Per-axis decimation for a wavelet of octave `scale`: `2^max(scale-α, 0)`, reduced on each axis to
the largest power of two that divides `N[d]`. The clamp only reduces decimation, so any `N` is
accepted at no cost in accuracy.
"""
@inline function decimation(N::NTuple{D, Int}, scale::Int, α::Int) where {D}
    want = 1 << max(scale - α, 0)
    return ntuple(D) do d
        r = want
        while r > 1 && N[d] % r != 0
            r >>= 1
        end
        r
    end
end

"""
    DeviceMeans{R,V,M,H}

The per-path sums of a single-field cascade on device arrays, accumulated on the device so that the
cascade synchronises with the host once, after its last path: `red` is a `(1, …, 1)` reduction
target, `s1` holds one sum per wavelet and `s2` one per `(j1, j2)` pair, and `invlen[j]` is one over
the size of the grid wavelet `j` is produced on, which turns a sum into a mean.
"""
struct DeviceMeans{R <: AbstractArray, V <: AbstractVector, M <: AbstractMatrix, H <: AbstractVector}
    red::R
    s1::V
    s2::M
    invlen::H
end

"""
    PeriodizedWorkspace{T,D}

Everything the cascade runs in: one working array and one spectrum array per *live* decimation, the
spectral plan at each of those sizes, and the filters periodized to each resolution they are applied
at.

Everything per-resolution is a vector indexed by level, with `level[j]` the level of wavelet `j` and
level `1` the full grid. `out[l] === work[l]` when the plan inverts in place. `spec[l]` is empty
except at levels that are a parent, where it holds `Û₁` across that wavelet's children. `D` is the
spatial rank of the resolutions; a batched workspace's arrays carry a trailing stack axis on top of
it. `means` is a [`DeviceMeans`](@ref) on a device-resident workspace and `nothing` otherwise.
"""
struct PeriodizedWorkspace{T, D, A <: AbstractArray{Complex{T}}, FA <: AbstractArray{T},
                           RV <: AbstractVector{NTuple{D, Int}}, LV <: AbstractVector{Int},
                           PV <: AbstractVector, AV <: AbstractVector{A},
                           FV <: AbstractVector{<:AbstractVector{FA}}, SF, MS}
    dims::NTuple{D, Int}
    oversampling::Int
    resolutions::RV        # per level; `resolutions[1]` is the full grid
    level::LV              # per wavelet
    plans::PV
    work::AV
    out::AV                # === work[l] when the plan inverts in place
    spec::AV               # Û₁; empty at levels that are never a parent
    # filters[l][j] is ψ_j periodized onto N ./ resolutions[l]. Level 1 holds the originals, which
    # order 1 multiplies against; coarser levels are what order 2 applies at its parent's resolution.
    filters::FV
    # How a filter is shaped for this workspace's buffers (the batched build adds a stack axis), so
    # `task_copy` can rebuild level 1 against a task's own bank.
    shapefilter::SF
    means::MS
end

function PeriodizedWorkspace(dims::NTuple{D, Int}, α::Integer, resolutions, level, plans, work,
                             out, spec, filters, shapefilter,
                             means = nothing) where {D}
    A = eltype(work)
    return PeriodizedWorkspace{real(eltype(A)), D, A, eltype(eltype(filters)), typeof(resolutions),
                               typeof(level), typeof(plans), typeof(work), typeof(filters),
                               typeof(shapefilter), typeof(means)}(
        dims, Int(α), resolutions, level, plans, work, out, spec, filters,
        shapefilter, means)
end

Base.show(io::IO, ws::PeriodizedWorkspace{T, D}) where {T, D} =
    print(io, "PeriodizedWorkspace{", T, ", ", D, "}(dims=", ws.dims, ", oversampling=", ws.oversampling,
          ", resolutions=", ws.resolutions, ")")

_one(::Val{D}) where {D} = ntuple(_ -> 1, D)

"""
    build(fb, groups, dims, T, α, plan) -> PeriodizedWorkspace

`groups` is the cascade work list `(j1, children, pathids)`; it determines which resolutions are
live and which `(wavelet, resolution)` filter pairs are ever applied, so nothing unused is built.
`plan` is the transform's full-grid plan and serves as level 1; the coarser levels are planned with
its spectral backend.
"""
build(fb, groups, dims::NTuple{D, Int}, ::Type{T}, α::Int,
      plan::Plans.AbstractScatteringPlan) where {T, D} =
    build(fb, groups, dims, T, α; alloc = rd -> zeros(Complex{T}, rd),
          makeplan = rd -> Plans.make_plan(Plans.spectral_backend(plan), T, rd), level1 = plan)

# `alloc` and `makeplan` take a spatial resolution and return a buffer and a plan for it, on whatever
# device and with whatever stack axis the caller wants.
#
# `alias_out = false` keeps each `out[l]` distinct from `work[l]`, as the localized-field cascade
# needs: it forward-transforms every modulus, and a leaf's or a child's goes from `out[l]` into
# `work[l]`.
#
# `shapefilter` post-processes each periodized filter; the batched build reshapes to `(spatial…, 1)`
# so filters broadcast over the stack axis.
#
# `level1`, when given, is the full-grid plan.
#
# `device_zeros(S, dims)`, when given, allocates a zeroed device array, and the workspace sums each
# path on the device (`DeviceMeans`).
function build(fb, groups, dims::NTuple{D, Int}, ::Type{T}, α::Int;
               alloc, makeplan, alias_out::Bool = true, shapefilter = identity,
               level1 = nothing, device_zeros = nothing) where {T, D}
    nw = FilterBanks.nwavelets(fb)
    meta = fb.meta
    full = _one(Val(D))

    # Levels: the full grid first, then every distinct resolution some wavelet is produced at.
    decim = [decimation(dims, meta[j].scale, α) for j in 1:nw]
    resolutions = [full]
    for r in decim
        r in resolutions || push!(resolutions, r)
    end
    nl = length(resolutions)
    level = [findfirst(==(r), resolutions)::Int for r in decim]

    isparent = falses(nl)
    for (j1, children, _) in groups
        isempty(children) || (isparent[level[j1]] = true)
    end
    # (wavelet, level) pairs order 2 applies: child j2 at each of its parents' levels.
    needed = [Set{Int}() for _ in 1:nl]
    for (j1, children, _) in groups, j2 in children
        push!(needed[level[j1]], j2)
    end

    rdims(l) = ntuple(d -> dims[d] ÷ resolutions[l][d], D)
    plans = [l > 1 ? makeplan(rdims(l)) : level1 === nothing ? makeplan(dims) : level1 for l in 1:nl]
    work = [alloc(rdims(l)) for l in 1:nl]
    out = [(alias_out && Plans.inplace_inverse(plans[l])) ? work[l] : alloc(rdims(l)) for l in 1:nl]
    empty_buf = alloc(ntuple(_ -> 0, D))
    spec = [isparent[l] ? alloc(rdims(l)) : empty_buf for l in 1:nl]

    # Filters. Level 1 is the bank itself — shared, never copied. Coarser levels hold only the
    # wavelets actually applied there; the rest is one shared empty array.
    proto = FilterBanks.filter_at(fb, 1)
    empty_filt = shapefilter(similar(proto, ntuple(_ -> 0, D)))
    filters = map(1:nl) do l
        l == 1 && return [shapefilter(FilterBanks.filter_at(fb, j)) for j in 1:nw]
        v = fill(empty_filt, nw)
        for j in needed[l]
            pf = similar(proto, rdims(l))
            ScatteringCore.periodize_filter!(pf, FilterBanks.filter_at(fb, j), resolutions[l])
            v[j] = shapefilter(pf)
        end
        v
    end

    means = device_zeros === nothing ? nothing :
        DeviceMeans(device_zeros(T, full), device_zeros(T, (nw,)), device_zeros(T, (nw, nw)),
                    T[one(T) / prod(rdims(level[j])) for j in 1:nw])
    return PeriodizedWorkspace(dims, α, resolutions, level, plans, work, out, spec, filters,
                               shapefilter, means)
end

"""
    cascade!(S1, S2, ws, fb, groups, xfft) -> (S1, S2)

Both scattering orders in one grouped pass, each convolution produced directly on the grid its
wavelet's band needs.

Order 1 multiplies by the full-resolution filter and periodizes the product. Order 2 multiplies `Û₁`
by `ψ_{j₂}` periodized to the parent's grid, then periodizes the product onto the child's; `r₂ ≥ r₁`,
since admissibility makes the child strictly coarser.
"""
function cascade!(S1::AbstractVector, S2::AbstractMatrix, ws::PeriodizedWorkspace,
                  fb, groups, xfft::AbstractArray)
    isempty(S2) || fill!(S2, zero(eltype(S2)))
    begin_means!(ws.means)
    @inbounds for g in groups
        group!(S1, S2, ws, fb, g, xfft)
    end
    gather_means!(S1, S2, ws.means)
    return S1, S2
end

"""
    group!(S1, S2, ws, fb, g, xfft) -> nothing

One first-order wavelet and all of its children. Group `j1` writes only `S1[j1]` and `S2[j1, :]`,
so tasks may run distinct groups concurrently, each with its own [`task_copy`](@ref) of the
workspace. The caller clears `S2`.
"""
function group!(S1::AbstractVector, S2::AbstractMatrix, ws::PeriodizedWorkspace{T, D},
                fb, g, xfft::AbstractArray) where {T, D}
    j1, children = g[1], g[2]
    @inbounds begin
        l1 = ws.level[j1]
        r1 = ws.resolutions[l1]
        w1, o1 = ws.work[l1], ws.out[l1]
        ScatteringCore.periodize_mul!(w1, xfft, _filter(ws, fb, 1, ws.filters[1], j1), r1)
        Plans.inverse_transform!(o1, ws.plans[l1], w1)
        if isempty(children)
            _modulus_mean_to!(S1, j1, ws.means, o1)
            return nothing
        end
        u1f = ws.spec[l1]
        _modulus_mean_in_place_to!(S1, j1, ws.means, o1)
        Plans.forward_transform!(u1f, ws.plans[l1], o1)
        fpar = ws.filters[l1]
        for j2 in children
            l2 = ws.level[j2]
            r2 = ws.resolutions[l2]
            w2, o2 = ws.work[l2], ws.out[l2]
            ScatteringCore.periodize_mul!(w2, u1f, _filter(ws, fb, l1, fpar, j2),
                                          ntuple(d -> r2[d] ÷ r1[d], D))
            Plans.inverse_transform!(o2, ws.plans[l2], w2)
            _modulus_mean_to!(S2, LinearIndices(S2)[j1, j2], ws.means, o2)
        end
    end
    return nothing
end

# Where each path's mean goes. Without `DeviceMeans` it is computed and stored in `S[i]` at once; with
# them its sum is kept in the device array of the same shape as `S` at index `i`, and
# `gather_means!` brings every sum to the host and scales it once the cascade is done.
begin_means!(::Nothing) = nothing
begin_means!(m::DeviceMeans) = (fill!(m.s1, zero(eltype(m.s1))); fill!(m.s2, zero(eltype(m.s2))); nothing)

@inline _modulus_mean_to!(S, i, ::Nothing, o) = (@inbounds S[i] = ScatteringCore.modulus_mean(o); nothing)
@inline _modulus_mean_to!(S, i, m::DeviceMeans, o) =
    _sum_to!(_sink(m, S), i, m.red, ScatteringCore._modulus, o)

# `|o|` written into `o`, which a coarser scale then transforms.
@inline _modulus_mean_in_place_to!(S, i, ::Nothing, o) =
    (@inbounds S[i] = ScatteringCore.modulus_mean!(o, o); nothing)
@inline function _modulus_mean_in_place_to!(S, i, m::DeviceMeans, o)
    @. o = ScatteringCore._modulus(o)
    return _sum_to!(_sink(m, S), i, m.red, real, o)
end

# The mean of a real field `a`, as above.
@inline mean_to!(S, i, ::Nothing, a) = (@inbounds S[i] = ScatteringCore.spatial_average(a); nothing)
@inline mean_to!(S, i, m::DeviceMeans, a) = _sum_to!(_sink(m, S), i, m.red, identity, a)

_sink(m::DeviceMeans, ::AbstractVector) = m.s1
_sink(m::DeviceMeans, ::AbstractMatrix) = m.s2

function _sum_to!(dst::AbstractArray, i::Integer, red::AbstractArray, f::F, A::AbstractArray) where {F}
    fill!(red, zero(eltype(red)))
    Base.mapreducedim!(f, +, red, A)
    copyto!(dst, i, red, 1, 1)
    return nothing
end

"""
    gather_means!(S1, S2, means) -> nothing

The sums a cascade kept on the device, as means in the host arrays `S1` and `S2`; nothing to do
without `means`.
"""
gather_means!(S1, S2, ::Nothing) = nothing
function gather_means!(S1::AbstractVector, S2::AbstractMatrix, m::DeviceMeans)
    copyto!(S1, m.s1)
    S1 .*= m.invlen
    if !isempty(S2)
        copyto!(S2, m.s2)
        S2 .*= transpose(m.invlen)
    end
    return nothing
end

"""
    scattering_values(st, x) -> (S0, S1, S2)

The same periodized cascade as [`cascade!`](@ref), written with allocating transforms and broadcasts
so that reverse-mode backends can differentiate it. Dimension-generic; the per-dimension `scattering`
methods wrap the result in their coefficient container.

It must compute the same function as `st(x)`, decimation included: `synthesize` differentiates this
to match coefficients read from `st(x)`.
"""
function scattering_values(st, x::AbstractArray{<:Any, D}) where {D}
    ws = st.pw
    fb = st.filter_bank
    tree = st.tree
    n = FilterBanks.nwavelets(fb)

    Xf = Plans.forward_transform(st.plan, complex.(x))
    U1 = map(1:n) do j
        l = ws.level[j]
        ScatteringCore._modulus.(Plans.inverse_transform(ws.plans[l],
            ScatteringCore.periodize(Xf .* _filter(ws, fb, 1, ws.filters[1], j),
                                     ws.resolutions[l])))
    end
    S1 = map(u -> sum(u) / length(u), U1)
    S0 = sum(x) / length(x)

    if st.max_order >= 2 && length(tree.by_order) >= 3
        U1f = map(j -> Plans.forward_transform(ws.plans[ws.level[j]], complex.(U1[j])), 1:n)
        paths = collect(PathGraph.order_range(tree, 2))
        vals = map(paths) do p
            idx = PathGraph.path_indices(tree, p)
            j1, j2 = idx[1], idx[2]
            l1, l2 = ws.level[j1], ws.level[j2]
            r1, r2 = ws.resolutions[l1], ws.resolutions[l2]
            m = ScatteringCore._modulus.(Plans.inverse_transform(ws.plans[l2],
                ScatteringCore.periodize(U1f[j1] .* _filter(ws, fb, l1, ws.filters[l1], j2),
                                         ntuple(d -> r2[d] ÷ r1[d], D))))
            sum(m) / length(m)
        end
        # Path-aligned values into the dense (j1,j2) matrix. `pos` is plain-integer bookkeeping
        # outside the differentiable path; the comprehension only reads `vals`.
        pos = zeros(Int, n, n)
        for (k, p) in enumerate(paths)
            idx = PathGraph.path_indices(tree, p)
            pos[idx[1], idx[2]] = k
        end
        Tc = eltype(vals)
        S2 = [pos[j1, j2] == 0 ? zero(Tc) : vals[pos[j1, j2]] for j1 in 1:n, j2 in 1:n]
    else
        S2 = Matrix{eltype(S1)}(undef, 0, 0)
    end
    return S0, S1, S2
end

"""
    task_copy(ws, fb) -> ws′

A workspace usable concurrently with `ws`: fresh buffers and task-local plans, sharing the read-only
periodized filters. Every task that runs [`group!`](@ref) needs its own.

`fb` is the bank the copy runs against. For a computed bank it must be the task's own
`FilterBanks.task_bank`, whose scratch the copy's level-1 filters alias.
"""
function task_copy(ws::PeriodizedWorkspace, fb)
    plans = map(Plans.task_local_plan, ws.plans)
    work = map(similar, ws.work)
    # `out[l]` aliases `work[l]` in the copy wherever it does in `ws`.
    out = map(l -> ws.out[l] === ws.work[l] ? work[l] : similar(ws.out[l]), eachindex(ws.out))
    spec = map(similar, ws.spec)
    filters = ws.filters
    if FilterBanks.iscomputed(fb)
        filters = copy(ws.filters)     # outer vector only — the coarser levels stay shared
        filters[1] = [ws.shapefilter(FilterBanks.filter_at(fb, j)) for j in eachindex(ws.filters[1])]
    end
    return PeriodizedWorkspace(ws.dims, ws.oversampling, ws.resolutions, ws.level, plans, work,
                               out, spec, filters, ws.shapefilter,
                               _copy_means(ws.means))
end

# Tasks running groups of one cascade write disjoint entries of the caller's `S1`/`S2`, so a task's
# copy stores each mean there as it is produced.
_copy_means(::Union{Nothing, DeviceMeans}) = nothing

"""
    with_plans(f, ws) -> ws′

`ws` with every plan replaced by `f(plan)`, sharing all buffers and filters, for instrumenting or
retargeting a cascade's transforms. `f` must preserve [`Plans.inplace_inverse`](@ref), since the
`out`/`work` aliasing follows it.
"""
with_plans(f, ws::PeriodizedWorkspace) =
    PeriodizedWorkspace(ws.dims, ws.oversampling, ws.resolutions, ws.level, map(f, ws.plans),
                        ws.work, ws.out, ws.spec, ws.filters, ws.shapefilter,
                        ws.means)

# A computed bank evaluates its filters on demand into one shared array; `fr[j]` aliases that array
# for every `j`, so refilling it is all that is needed and the entry keeps whatever shape `build`
# gave it. A stored bank hands its own array back and needs no refill.
@inline function _filter(ws::PeriodizedWorkspace, fb, level::Int, fr::AbstractVector, j::Integer)
    (FilterBanks.iscomputed(fb) && level == 1) && FilterBanks.filter_at(fb, j)
    return @inbounds fr[j]
end

# ============================================================================
# Localized (Mallat) fields
# ============================================================================

"""
    FieldWorkspace{T,D}

The periodized cascade plus what the localized transform `S_p = (|U_p| ⋆ φ_J) ↓ s` needs on top of
it: `φ_J` periodized to every level a modulus is produced at, and one buffer and plan at the output
resolution.

Its cascade is built with `alias_out = false` and decimates no wavelet past `s` (see
[`field_oversampling`](@ref)), so every path is produced at or above the output resolution. At the
default `s = 2^(J-1)` the cap changes nothing.
"""
struct FieldWorkspace{T, D, PW <: PeriodizedWorkspace{T, D}, FV <: AbstractVector,
                      A <: AbstractArray{Complex{T}, D}, P}
    pw::PW
    lowpass::FV            # per level
    rout::NTuple{D, Int}
    small::A
    small_out::A           # === small when the output plan inverts in place
    plan::P
end

Base.show(io::IO, fw::FieldWorkspace{T, D}) where {T, D} =
    print(io, "FieldWorkspace{", T, ", ", D, "}(output decimation=", fw.rout, ", ", fw.pw, ")")

"""
    field_oversampling(fb, α, s) -> Int

The smallest `oversampling` at least `α` under which no wavelet decimates past `s`.

`r = 2^max(j-α, 0) ≤ s = 2^q` iff `α ≥ j - q`, so raising `α` to `maxscale - q` caps every
resolution at the output grid without changing the dyadic family any wavelet lands on.
"""
function field_oversampling(fb, α::Int, s::Int)
    meta = fb.meta
    maxscale = maximum(m -> m.scale, view(meta, 1:FilterBanks.nwavelets(fb)))
    return max(α, maxscale - trailing_zeros(s))
end

"""
    build_field(fb, groups, dims, T, α, s, φ; alloc, makeplan) -> FieldWorkspace

Workspace for the localized transform at output subsample factor `s`. `φ` is the full-resolution
low-pass.
"""
function build_field(fb, groups, dims::NTuple{D, Int}, ::Type{T}, α::Int, s::Int,
                     φ::AbstractArray{T, D}; alloc, makeplan) where {T, D}
    pw = build(fb, groups, dims, T, field_oversampling(fb, α, s); alloc = alloc,
               makeplan = makeplan, alias_out = false)
    rout = ntuple(d -> min(s, 1 << trailing_zeros(dims[d])), D)
    rdims(r) = ntuple(d -> dims[d] ÷ r[d], D)

    lowpass = map(pw.resolutions) do r
        all(isone, r) && return φ
        pf = similar(φ, rdims(r))
        ScatteringCore.periodize_filter!(pf, φ, r)
        pf
    end

    plan = makeplan(rdims(rout))
    small = alloc(rdims(rout))
    small_out = Plans.inplace_inverse(plan) ? small : alloc(rdims(rout))
    return FieldWorkspace{T, D, typeof(pw), typeof(lowpass), typeof(small), typeof(plan)}(
        pw, lowpass, rout, small, small_out, plan)
end

"""
    field_workspace(st, subsample) -> FieldWorkspace

The localized-field workspace for transform `st` at output factor `subsample`.

`st` must have fields `filter_bank`, `groups`, `dims`, `plan`, `buffer_input` and `pw`. Buffers take
`buffer_input`'s array type and plans `plan`'s kind, so a device-resident transform gets a
device-resident workspace.
"""
function field_workspace(st, subsample::Int)
    fb = st.filter_bank
    dims = st.dims
    T = real(eltype(st.buffer_input))
    alloc(rd) = similar(st.buffer_input, Complex{T}, rd)
    φ = similar(st.buffer_input, T, dims)
    copyto!(φ, Filters.gaussian_lowpass(T, dims, fb.J))
    return build_field(fb, st.groups, dims, T, st.pw.oversampling, subsample, φ;
                       alloc = alloc, makeplan = rd -> Plans.plan_like(st.plan, alloc(rd)))
end

"""
    localize!(dst, fw, spec, level) -> dst

`dst = real(ifft(spec ⋅ φ̂) ↓ (rout/r))`, where `spec` is the spectrum of a modulus living at `level`.
The product is periodized onto the output grid, so the inverse runs at output size.
"""
function localize!(dst::AbstractArray, fw::FieldWorkspace{T, D}, spec::AbstractArray{<:Any, D},
                   level::Int) where {T, D}
    r = @inbounds fw.pw.resolutions[level]
    ScatteringCore.periodize_mul!(fw.small, spec, (@inbounds fw.lowpass[level]),
                                  ntuple(d -> fw.rout[d] ÷ r[d], D))
    Plans.inverse_transform!(fw.small_out, fw.plan, fw.small)
    # Orders ≥ 1 smooth a modulus and are real; only the order-0 field `x ⋆ φ_J` of a complex input
    # is complex, and `dst`'s element type says which.
    if eltype(dst) <: Real
        @. dst = real(fw.small_out)
    else
        copyto!(dst, fw.small_out)
    end
    return dst
end

"""
    field_group!(data, fw, fb, g, xfft, p1_first) -> nothing

One first-order wavelet and all of its children, written as localized fields into `data`'s trailing
path axis. The order-1 path id is `p1_first + j1 - 1`; the children's come from the group.
"""
function field_group!(data::AbstractArray, fw::FieldWorkspace{T, D}, fb, g,
                      xfft::AbstractArray, p1_first::Int) where {T, D}
    ws = fw.pw
    j1, children, pathids = g[1], g[2], g[3]
    @inbounds begin
        l1 = ws.level[j1]
        r1 = ws.resolutions[l1]
        w1, o1 = ws.work[l1], ws.out[l1]
        ScatteringCore.periodize_mul!(w1, xfft, _filter(ws, fb, 1, ws.filters[1], j1), r1)
        Plans.inverse_transform!(o1, ws.plans[l1], w1)
        ScatteringCore.apply_modulus!(o1, o1)
        # A parent's spectrum must survive its whole child loop, so it goes in `spec`; a leaf's is
        # dead immediately and can use the working array, which the inverse above has freed.
        u1f = isempty(children) ? w1 : ws.spec[l1]
        Plans.forward_transform!(u1f, ws.plans[l1], o1)
        localize!(_pathview(data, Val(D), p1_first + j1 - 1), fw, u1f, l1)
        isempty(children) && return nothing
        fpar = ws.filters[l1]
        for (j2, p) in zip(children, pathids)
            l2 = ws.level[j2]
            r2 = ws.resolutions[l2]
            w2, o2 = ws.work[l2], ws.out[l2]
            ScatteringCore.periodize_mul!(w2, u1f, _filter(ws, fb, l1, fpar, j2),
                                          ntuple(d -> r2[d] ÷ r1[d], D))
            Plans.inverse_transform!(o2, ws.plans[l2], w2)
            ScatteringCore.apply_modulus!(o2, o2)
            # `w2` is free again and is never `u1f`, which is this parent's `spec`.
            Plans.forward_transform!(w2, ws.plans[l2], o2)
            localize!(_pathview(data, Val(D), p), fw, w2, l2)
        end
    end
    return nothing
end

"""
    field_cascade!(data, fw, fb, groups, xfft, root, p1_first) -> data

Every path's localized field. `root` is the order-0 path id, `p1_first` the first order-1 id.
"""
function field_cascade!(data::AbstractArray, fw::FieldWorkspace{T, D}, fb, groups,
                        xfft::AbstractArray, root::Int, p1_first::Int) where {T, D}
    localize!(_pathview(data, Val(D), root), fw, xfft, 1)
    @inbounds for g in groups
        field_group!(data, fw, fb, g, xfft, p1_first)
    end
    return data
end

# `data` is (spatial..., path); `selectdim` on the trailing axis is one path's field, for any D.
@inline _pathview(data::AbstractArray, ::Val{D}, p::Integer) where {D} = selectdim(data, D + 1, p)

end # module Cascade
