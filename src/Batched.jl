module Batched

# One batched cascade for every backend: a `(spatial…, B)` stack runs against a single spectral plan
# over the leading spatial dimensions, with filters reshaped to `(spatial…, 1)` to broadcast over the
# batch axis. The caller supplies the buffers and the plan, so the same code runs on host and device
# arrays.

using ..Plans: Plans
using ..FilterBanks: FilterBanks
using ..ScatteringCore: ScatteringCore
using ..Cascade: Cascade
using ..Coefficients: Coefficients
using ..PathGraph: PathGraph

export BatchWorkspace, batch_cascade!

# ---------------------------------------------------------------------------
# Per-slice ⟨|c|⟩
# ---------------------------------------------------------------------------

# Leaf wavelets: the reduction reads the transform output directly and writes no modulus field.
function slice_modulus_mean!(red, redvec, c)
    fill!(red, zero(eltype(red)))
    Base.mapreducedim!(ScatteringCore._modulus, +, red, c)
    return redvec
end

# Wavelets with children: write the modulus field into `rmod`, which order 2 reads, and reduce it.
# The `Array` method does both in one sweep.
function slice_modulus_mean!(red, redvec, rmod, c)
    @. rmod = ScatteringCore._modulus(c)
    Base.sum!(red, rmod)
    return redvec
end

function slice_modulus_mean!(::Array{T}, redvec::AbstractVector{T}, rmod::Array{T},
                             c::Array{Complex{T}}) where {T <: Real}
    B = length(redvec)
    ns = length(rmod) ÷ B
    @inbounds for b in 1:B
        acc = zero(T)
        off = (b - 1) * ns
        @simd for i in 1:ns
            v = ScatteringCore._modulus(c[off + i])
            rmod[off + i] = v
            acc += v
        end
        redvec[b] = acc
    end
    return redvec
end

"""
    BatchWorkspace{T,D}

Preallocated state for repeated batched transforms at a fixed batch size `B`: a periodized cascade
workspace whose arrays all carry a trailing stack axis, the real staging and reduction buffers, and
the precomputed order-2 groups. Once built, a batched transform does no data-proportional allocation.

`red` is the spatial-reduction target, shaped `(1…, 1, B)` so `sum!` contracts every axis but the
batch; that one shape serves every resolution. `invN[l]` is `1/∏(spatial ÷ r)` at level `l`, since
each mean is over its own decimated grid.
"""
struct BatchWorkspace{T, D, PW <: Cascade.PeriodizedWorkspace{T, D}, RA, CA,
                      MV <: AbstractVector{RA}, RV <: AbstractVector{T},
                      IV <: AbstractVector{T}, G, FB}
    pw::PW
    xr::RA         # real input staging, (spatial…, B)
    xf::CA         # signal spectrum at full resolution, preserved across the cascade
    rmod::MV       # per level: modulus staging; empty where that level is never a parent
    red::RA        # (1…, B) reduction target for `sum!`
    redvec::RV     # the same memory viewed as (B,), so the hot loop reshapes nothing
    invN::IV       # per level: 1/∏(spatial ÷ r)
    n::Int
    groups::G
    bank::FB       # a computed bank refills the shared filter scratch on demand
end

BatchWorkspace(pw::Cascade.PeriodizedWorkspace{T, D}, xr, xf, rmod, red, redvec, invN,
               n::Integer, groups, bank) where {T, D} =
    BatchWorkspace{T, D, typeof(pw), typeof(xr), typeof(xf), typeof(rmod), typeof(redvec),
                   typeof(invN), typeof(groups), typeof(bank)}(
        pw, xr, xf, rmod, red, redvec, invN, Int(n), groups, bank)

Base.show(io::IO, ws::BatchWorkspace{T, D}) where {T, D} =
    print(io, "BatchWorkspace{", T, ", ", D, "}(B=", length(ws.redvec), ", ", ws.pw, ")")

# The batched workspace of transform `st` at batch size `B`. `zeros_like(S, dims)` allocates a zeroed
# buffer and `makeplan(rd)` a plan over the leading axes of an `(rd…, B)` stack, both where the
# workspace is to live; `shapefilter` gives a filter the trailing singleton that broadcasts it across
# the stack.
function build_workspace(st, B::Int, ::Type{T}; zeros_like, makeplan, shapefilter) where {T}
    spatial = st.dims
    D = length(spatial)
    pw = Cascade.build(st.filter_bank, st.groups, spatial, T, st.pw.oversampling;
                       alloc = rd -> zeros_like(Complex{T}, (rd..., B)), makeplan = makeplan,
                       shapefilter = shapefilter)
    rdims(l) = ntuple(d -> spatial[d] ÷ pw.resolutions[l][d], D)
    # A level is a parent exactly where the cascade gave it a `Û₁` buffer, and only a parent's
    # modulus is ever read again.
    rmod = [isempty(pw.spec[l]) ? zeros_like(T, ntuple(_ -> 0, D + 1)) : zeros_like(T, (rdims(l)..., B))
            for l in eachindex(pw.resolutions)]
    invN = [T(1) / prod(rdims(l)) for l in eachindex(pw.resolutions)]
    red = zeros_like(T, (ntuple(_ -> 1, D)..., B))
    return BatchWorkspace(pw, zeros_like(T, (spatial..., B)), zeros_like(Complex{T}, (spatial..., B)),
                          rmod, red, reshape(red, B), invN, FilterBanks.nwavelets(st.filter_bank),
                          st.groups, st.filter_bank)
end

# The stack into the cascade's complex input buffer `c0`. A real stack crosses to the workspace's
# device in its own element type, through `xr`.
_stage!(c0, xr, X::AbstractArray{<:Real}) = (copyto!(xr, X); @. c0 = complex(xr); c0)
_stage!(c0, xr, X::AbstractArray{<:Complex}) = copyto!(c0, X)

# `S0` of every slice, the field's mean: the zero-frequency entry of its spectrum `dc` over its size,
# whose imaginary part is rounding where the field is real.
_put_s0!(row, dc, s, ::Type{<:Real}) = (@. row = real(dc) * s; row)
_put_s0!(row, dc, s, ::Type{<:Complex}) = (@. row = dc * s; row)

"""
    batch_cascade!(out, ws, X) -> out

Write the flattened scattering coefficients of every slice of `X` into the columns of `out`, which
has `Coefficients.flat_length(n)` rows in the layout `flatten1d!`/`flatten2d!` produce. A complex `X`
needs complex columns, for its complex `S0`.

The batched form of [`Cascade.cascade!`](@ref), with each mean replaced by a per-slice reduction and
each decimation tuple extended by a `1` for the stack axis.
"""
function batch_cascade!(out::AbstractMatrix, ws::BatchWorkspace{T, D}, X::AbstractArray) where {T, D}
    pw = ws.pw
    fb = ws.bank
    n = ws.n
    rv = ws.redvec
    Tout = eltype(out)

    c0 = pw.work[1]
    _stage!(c0, ws.xr, X)
    Plans.forward_transform!(ws.xf, pw.plans[1], c0)

    # S0, and clear the S2 block so inadmissible pairs read as zero (matching `flatten*!`).
    _put_s0!(view(out, Coefficients.flat_row_s0(), :), view(ws.xf, ntuple(_ -> 1, Val(D))..., :),
             Tout(ws.invN[1]), Tout)
    @views out[(n + 2):end, :] .= zero(Tout)

    @inbounds for (j1, children, _) in ws.groups
        l1 = pw.level[j1]
        r1 = pw.resolutions[l1]
        w1, o1 = pw.work[l1], pw.out[l1]
        ScatteringCore.periodize_mul!(w1, ws.xf, Cascade._filter(pw, fb, 1, pw.filters[1], j1),
                                      (r1..., 1))
        Plans.inverse_transform!(o1, pw.plans[l1], w1)
        inv1 = Tout(ws.invN[l1])
        if isempty(children)
            slice_modulus_mean!(ws.red, rv, o1)
            @views out[Coefficients.flat_row_s1(j1, n), :] .= rv .* inv1
            continue
        end
        rm1 = ws.rmod[l1]
        slice_modulus_mean!(ws.red, rv, rm1, o1)
        @views out[Coefficients.flat_row_s1(j1, n), :] .= rv .* inv1

        u1f = pw.spec[l1]
        @. w1 = complex(rm1)                    # `w1` was consumed by the inverse above
        Plans.forward_transform!(u1f, pw.plans[l1], w1)
        fpar = pw.filters[l1]
        for j2 in children
            l2 = pw.level[j2]
            r2 = pw.resolutions[l2]
            w2, o2 = pw.work[l2], pw.out[l2]
            ScatteringCore.periodize_mul!(w2, u1f, Cascade._filter(pw, fb, l1, fpar, j2),
                                          (ntuple(d -> r2[d] ÷ r1[d], D)..., 1))
            Plans.inverse_transform!(o2, pw.plans[l2], w2)
            slice_modulus_mean!(ws.red, rv, o2)
            @views out[Coefficients.flat_row_s2(j1, j2, n), :] .= rv .* Tout(ws.invN[l2])
        end
    end
    return out
end

end # module Batched
