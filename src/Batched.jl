module Batched

"""
    Batched.jl — one batched cascade, shared by every backend

Transforms a whole `(spatial…, B)` stack against a single spectral plan built over the leading
spatial dimensions, so each convolution is one plan execution over all `B` slices rather than `B`
executions of a single-slice plan. Filters broadcast over the batch axis by reshaping to
`(spatial…, 1)`.

Nothing here names a device: the buffers and the plan come from the caller, so the same code runs on
CPU arrays with an FFTW (or in-core) plan and on device arrays with a cuFFT/rocFFT plan. The GPU
extension therefore supplies allocation and a plan, not a second copy of the cascade.

The cascade is the batched form of `Scattering*D.cascade!` — grouped by first-order wavelet, so a
first-order field is computed once and reused by all of its children.
"""

using ..Plans: Plans
using ..FilterBanks: FilterBanks
using ..ScatteringCore: ScatteringCore
using ..Cascade: Cascade
using ..Coefficients: Coefficients
using ..PathGraph: PathGraph

export BatchWorkspace, batch_cascade!

# ---------------------------------------------------------------------------
# Per-slice ⟨|c|⟩
#
# The cascade is memory-bandwidth bound, not FFT bound: every wavelet convolution moves roughly
# `6N` words (multiply in, transform out, modulus, reduce) against an `N log N` transform, so a
# redundant pass over the stack costs as much as a sizeable fraction of the FFT itself. These fuse
# the modulus with its reduction so the stack is swept once, not twice.
# ---------------------------------------------------------------------------

# Leaf wavelets: the modulus field is never read again, so it is not written at all — the reduction
# consumes the transform output directly.
function slice_modulus_mean!(red, redvec, c)
    fill!(red, zero(eltype(red)))
    Base.mapreducedim!(ScatteringCore._modulus, +, red, c)
    return redvec
end

# Per-slice sum of a real field — S0's reduction.
slice_sum!(red, redvec, a) = (Base.sum!(red, a); redvec)

# Wavelets with children: the modulus field *is* consumed downstream, so it must be written — but
# the sum can ride along with the write, so the stack is swept once rather than twice. The generic
# form cannot express that; the CPU method is measured at 1.31–1.49× the two-pass version. The leaf
# case above needs no such method — it is already single-pass, and a hand-written loop measures
# within noise of `mapreducedim!`.
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
batch; that one shape serves every resolution.

`invN` is per level. A single `1/∏spatial` would leave every decimated row short by exactly `∏r` —
a clean factor that no batched-versus-batched comparison can see.
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

"""
    batch_cascade!(out, ws, X) -> out

Write the flattened scattering coefficients of every slice of `X` into the columns of `out`.

`out` has `Coefficients.flat_length(n)` rows; row assignment goes through
`Coefficients.flat_row_*`, the same layout `flatten1d!`/`flatten2d!` produce, so batched and
per-slice results are interchangeable.

The batched form of [`Cascade.cascade!`](@ref) — same periodization and the same resolutions, with
the scalar mean replaced by a per-slice reduction over the stack. Each decimation tuple gains a
trailing `1` so the stack axis is never folded, and the filters carry a trailing singleton so they
broadcast across it.
"""
function batch_cascade!(out::AbstractMatrix, ws::BatchWorkspace{T, D}, X::AbstractArray) where {T, D}
    pw = ws.pw
    fb = ws.bank
    n = ws.n
    rv = ws.redvec
    Tout = eltype(out)

    copyto!(ws.xr, X)
    c0 = pw.work[1]
    @. c0 = complex(ws.xr)
    Plans.forward_transform!(ws.xf, pw.plans[1], c0)

    # S0, and clear the S2 block so inadmissible pairs read as zero (matching `flatten*!`).
    slice_sum!(ws.red, rv, ws.xr)
    @views out[Coefficients.flat_row_s0(), :] .= rv .* Tout(ws.invN[1])
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
