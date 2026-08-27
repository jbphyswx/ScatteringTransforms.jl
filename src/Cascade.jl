module Cascade

"""
    Cascade.jl — the periodized scattering cascade, shared by 1D/2D/3D

Every convolution in a scattering transform is followed by a modulus and a mean, and the field it
produces is band-limited by the wavelet that made it. Transforming at full size and then throwing
away all but every `r`-th sample is wasted work: by
[`ScatteringCore.periodize_mul!`](@ref), the decimated samples of a full-size inverse transform
*are* the small inverse transform of the periodized product spectrum, exactly, for any spectrum.

So a scale-`j` wavelet's output is produced directly on a grid decimated by `r = 2^max(j-α, 0)` per
axis. `α` (`oversampling`) is a plain `Int`: `α ≥ J` decimates nothing and reproduces the
undecimated cascade bit for bit.

Filters are **periodizations of the original full-resolution filter**, never Morlets rebuilt on the
coarse grid — see [`ScatteringCore.periodize_filter!`](@ref) for why those are different functions
(they differ by an octave).
"""

using ..Plans: Plans
using ..ScatteringCore: ScatteringCore
using ..FilterBanks: FilterBanks
using ..Filters: Filters
using ..PathGraph: PathGraph
using SpectralBackends: SpectralBackends as SB

"""
    decimation(N, scale, α) -> NTuple{D,Int}

Per-axis decimation for a wavelet of octave `scale`: `2^max(scale-α, 0)`, reduced on each axis to
the largest power of two that divides `N[d]`.

Per axis because one scalar cannot serve every grid — `(128, 100)` admits `(8, 4)` but no scalar
`8`. Clamping rather than throwing keeps every `N` constructible; the clamp only ever *reduces*
decimation, so it costs speed, never accuracy.
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
    PeriodizedWorkspace{T,D}

Everything the cascade runs in: one working array and one spectrum array per *live* decimation, the
spectral plan at each of those sizes, and the filters periodized to each resolution they are applied
at.

Everything per-resolution is a vector indexed by **level**, with `level[j]` the level of wavelet `j`
and level `1` always the full grid. Levels rather than resolution tuples because the cascade looks
these up once per wavelet and once per order-2 path, and an integer index costs nothing where
hashing a tuple would.

`out[l]` is `work[l]` itself when the plan inverts in place, so a resolution costs one array rather
than two. `spec[l]` is empty except at levels that are ever a *parent*, where it holds `Û₁` across
that wavelet's children.

`A`/`FA` are not pinned to rank `D`: `D` is the *spatial* rank, which is what the resolutions are in,
while a batched workspace's arrays carry a trailing stack axis.
"""
struct PeriodizedWorkspace{T, D, A <: AbstractArray{Complex{T}}, FA <: AbstractArray{T},
                           RV <: AbstractVector{NTuple{D, Int}}, LV <: AbstractVector{Int},
                           PV <: AbstractVector, AV <: AbstractVector{A},
                           FV <: AbstractVector{<:AbstractVector{FA}}, SF}
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
    # A computed bank evaluates on demand into one shared array, so level 1 must refill it per
    # wavelet. Coarser sets are always materialised — they are a few percent of a bank, and there is
    # nowhere to periodize into otherwise.
    full_computed::Bool
    # How a filter is shaped for this workspace's buffers (the batched build adds a stack axis).
    # Retained so [`task_copy`](@ref) can rebuild level 1 against a task's own bank.
    shapefilter::SF
end

function PeriodizedWorkspace(dims::NTuple{D, Int}, α::Integer, resolutions, level, plans, work,
                             out, spec, filters, full_computed::Bool, shapefilter) where {D}
    A = eltype(work)
    return PeriodizedWorkspace{real(eltype(A)), D, A, eltype(eltype(filters)), typeof(resolutions),
                               typeof(level), typeof(plans), typeof(work), typeof(filters),
                               typeof(shapefilter)}(
        dims, Int(α), resolutions, level, plans, work, out, spec, filters, full_computed,
        shapefilter)
end

_one(::Val{D}) where {D} = ntuple(_ -> 1, D)

"""
    build(fb, groups, dims, T, α, spectral) -> PeriodizedWorkspace

`groups` is the cascade work list `(j1, children, pathids)`; it determines which resolutions are
live and which `(wavelet, resolution)` filter pairs are ever applied, so nothing unused is built.
"""
build(fb, groups, dims::NTuple{D, Int}, ::Type{T}, α::Int,
      spectral::SB.AbstractSpectralBackend) where {T, D} =
    build(fb, groups, dims, T, α; alloc = rd -> zeros(Complex{T}, rd),
          makeplan = rd -> Plans.make_plan(spectral, T, rd))

# `alloc`/`makeplan` are the only device-aware pieces, so a device-resident transform builds its
# workspace through the same code by supplying device versions — no second implementation, and no
# spectral tag, which a device plan deliberately does not have. Both take the *spatial* resolution;
# a batched build returns arrays and plans that carry a stack axis on top of it.
#
# `alias_out = false` keeps each resolution's inverse destination distinct from its source even when
# the plan could invert in place. The localized-field cascade needs that: it takes a forward
# transform of every modulus, including a leaf's, and a forward transform needs somewhere to write
# that is not its own source.
#
# `shapefilter` post-processes each periodized filter — the batched build reshapes to `(spatial…, 1)`
# so filters broadcast over the stack axis.
function build(fb, groups, dims::NTuple{D, Int}, ::Type{T}, α::Int;
               alloc, makeplan, alias_out::Bool = true, shapefilter = identity) where {T, D}
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
    plans = [l == 1 ? makeplan(dims) : makeplan(rdims(l)) for l in 1:nl]
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

    return PeriodizedWorkspace(dims, α, resolutions, level, plans, work, out, spec, filters,
                               FilterBanks.iscomputed(fb), shapefilter)
end

"""
    cascade!(S1, S2, ws, fb, groups, xfft) -> (S1, S2)

Both scattering orders in one grouped pass, each convolution produced directly on the grid its
wavelet's band actually needs.

Order 1 multiplies at full resolution and periodizes the product, which is why the full-resolution
filter is the one order 1 applies. Order 2 multiplies `Û₁` by `ψ_{j₂}` already periodized to the
*parent's* grid, then periodizes that product onto the *child's*. `r₂ ≥ r₁` always, because
admissibility makes the child strictly coarser.
"""
function cascade!(S1::AbstractVector, S2::AbstractMatrix, ws::PeriodizedWorkspace,
                  fb, groups, xfft::AbstractArray)
    isempty(S2) || fill!(S2, zero(eltype(S2)))
    @inbounds for g in groups
        group!(S1, S2, ws, fb, g, xfft)
    end
    return S1, S2
end

"""
    group!(S1, S2, ws, fb, g, xfft) -> nothing

One first-order wavelet and all of its children. Split out because it is also the unit of
parallelism: group `j1` writes only `S1[j1]` and `S2[j1, :]`, so tasks over groups never collide —
provided each task holds its own [`task_copy`](@ref) of the workspace, whose buffers are mutable.

`S2` is *not* cleared here; the caller does that once.
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
            S1[j1] = ScatteringCore.modulus_mean(o1)
            return nothing
        end
        u1f = ws.spec[l1]
        S1[j1] = ScatteringCore.modulus_mean!(o1, o1)
        Plans.forward_transform!(u1f, ws.plans[l1], o1)
        fpar = ws.filters[l1]
        for j2 in children
            l2 = ws.level[j2]
            r2 = ws.resolutions[l2]
            w2, o2 = ws.work[l2], ws.out[l2]
            ScatteringCore.periodize_mul!(w2, u1f, _filter(ws, fb, l1, fpar, j2),
                                          ntuple(d -> r2[d] ÷ r1[d], D))
            Plans.inverse_transform!(o2, ws.plans[l2], w2)
            S2[j1, j2] = ScatteringCore.modulus_mean(o2)
        end
    end
    return nothing
end

"""
    scattering_values(st, x) -> (S0, S1, S2)

The same periodized cascade as [`cascade!`](@ref), written with allocating transforms and broadcasts
so that reverse-mode backends can differentiate it. Dimension-generic; the per-dimension `scattering`
methods wrap the result in their coefficient container.

Periodizing here is a contract, not an optimisation: `synthesize` differentiates this while the user
reads coefficients from `st(x)`, so at `oversampling < J` an unperiodized version would optimise a
different function than the one being reported.
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

A workspace usable concurrently with `ws`: fresh buffers and task-local plans, sharing the
periodized filters, which are read-only. `fb` is the filter bank the copy will be run against.

The buffers are mutable state, so sharing a `PeriodizedWorkspace` across tasks is a data race —
every task that runs [`group!`](@ref) needs its own.

`fb` is required because a *computed* bank's level-1 entries are all one shared scratch array. A task
given its own bank (`FilterBanks.task_bank`) must also get a level-1 list pointing at that bank's
scratch; sharing the original's would have every task refill its own array and then read another's.
"""
function task_copy(ws::PeriodizedWorkspace, fb)
    plans = map(Plans.task_local_plan, ws.plans)
    work = map(similar, ws.work)
    # An `out` that aliased its `work` must go on aliasing the copy's, or the inverse writes to an
    # array nothing reads.
    out = map(l -> ws.out[l] === ws.work[l] ? work[l] : similar(ws.out[l]), eachindex(ws.out))
    spec = map(similar, ws.spec)
    filters = ws.filters
    if ws.full_computed
        filters = copy(ws.filters)     # outer vector only — the coarser levels stay shared
        filters[1] = [ws.shapefilter(FilterBanks.filter_at(fb, j)) for j in eachindex(ws.filters[1])]
    end
    return PeriodizedWorkspace(ws.dims, ws.oversampling, ws.resolutions, ws.level, plans, work,
                               out, spec, filters, ws.full_computed, ws.shapefilter)
end

"""
    with_plans(f, ws) -> ws′

`ws` with every plan replaced by `f(plan)`, sharing all buffers and filters.

For instrumenting or retargeting the transforms a cascade issues without rebuilding the workspace.
`f` must preserve [`Plans.inplace_inverse`](@ref), which is what the shared `out`/`work` aliasing
was decided from.
"""
with_plans(f, ws::PeriodizedWorkspace) =
    PeriodizedWorkspace(ws.dims, ws.oversampling, ws.resolutions, ws.level, map(f, ws.plans),
                        ws.work, ws.out, ws.spec, ws.filters, ws.full_computed, ws.shapefilter)

# A computed bank evaluates its filters on demand into one shared array; `fr[j]` aliases that array
# for every `j`, so refilling it is all that is needed and the entry keeps whatever shape `build`
# gave it. A stored bank hands its own array back and needs no refill.
@inline function _filter(ws::PeriodizedWorkspace, fb, level::Int, fr::AbstractVector, j::Integer)
    (ws.full_computed && level == 1) && FilterBanks.filter_at(fb, j)
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

Built with `alias_out = false`, because unlike the coefficient cascade this one forward-transforms
every modulus, a leaf's included.

Decimation is capped at `s`: a path produced coarser than the grid it is written to would have to be
upsampled. The cap costs nothing at the default `s = 2^(J-1)`, which already exceeds every wavelet's
own decimation.
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

Duck-typed on `st` (`filter_bank`, `groups`, `dims`, `plan`, `buffer_input`, `pw`), so 1D, 2D and
device-resident transforms all go through it: `buffer_input` supplies the array type and `plan`
the plan kind, which is all that distinguishes a device build from a host one.
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

Fused: periodizing the product onto the output grid *is* the decimated inverse transform, so the
inverse runs at output size.
"""
function localize!(dst::AbstractArray, fw::FieldWorkspace{T, D}, spec::AbstractArray{<:Any, D},
                   level::Int) where {T, D}
    r = @inbounds fw.pw.resolutions[level]
    ScatteringCore.periodize_mul!(fw.small, spec, (@inbounds fw.lowpass[level]),
                                  ntuple(d -> fw.rout[d] ÷ r[d], D))
    Plans.inverse_transform!(fw.small_out, fw.plan, fw.small)
    # Orders ≥ 1 smooth a modulus and are real whatever the input was; only the order-0 field
    # `x ⋆ φ_J` inherits a complex input, which is what the destination's element type records.
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
