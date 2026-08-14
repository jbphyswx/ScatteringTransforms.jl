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
    Base.mapreducedim!(abs, +, red, c)
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
    @. rmod = abs(c)
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
            v = abs(c[off + i])
            rmod[off + i] = v
            acc += v
        end
        redvec[b] = acc
    end
    return redvec
end

"""
    BatchWorkspace{P,RA,CA,WV,RV,G}

Preallocated state for repeated batched transforms at a fixed batch size `B`: the batched plan, the
`(spatial…, B)` scratch, the filters in whatever storage the plan expects, and the precomputed
order-2 groups. Once built, a batched transform does no data-proportional allocation.

`red` is the spatial-reduction target, shaped `(1…, 1, B)` so `sum!` contracts every axis but the
batch.
"""
struct BatchWorkspace{P, RA, CA, WV, RV, G}
    plan::P
    xr::RA         # real input staging,        (spatial…, B)
    xf::CA         # signal spectrum,           preserved across the cascade
    c1::CA         # multiply scratch
    c2::CA         # inverse output, then the first-order spectrum
    c3::CA         # order-2 child inverse, so `c2` stays intact across children
    rmod::RA       # modulus
    red::RA        # (1…, B) reduction target for `sum!`
    redvec::RV     # the same memory viewed as (B,), so the hot loop reshapes nothing
    wavelets::WV   # filters already shaped (spatial…, 1) to broadcast over the batch axis
    n::Int
    groups::G
    invN::Float64  # 1/prod(spatial) — the spatial mean
end

"""
    batch_cascade!(out, ws, X) -> out

Write the flattened scattering coefficients of every slice of `X` into the columns of `out`.

`out` has `Coefficients.flat_length(n)` rows; row assignment goes through
`Coefficients.flat_row_*`, the same layout `flatten1d!`/`flatten2d!` produce, so batched and
per-slice results are interchangeable.
"""
function batch_cascade!(out::AbstractMatrix, ws::BatchWorkspace, X::AbstractArray)
    n = ws.n
    invN = eltype(out)(ws.invN)
    # `redvec` and the filters are pre-shaped on the workspace: `reshape` builds a new array object
    # on every call, and this loop would make one per wavelet and per order-2 path.
    rv = ws.redvec

    copyto!(ws.xr, X)
    @. ws.c1 = complex(ws.xr)
    Plans.forward_transform!(ws.xf, ws.plan, ws.c1)

    # S0, and clear the S2 block so inadmissible pairs read as zero (matching `flatten*!`).
    slice_sum!(ws.red, rv, ws.xr)
    @views out[Coefficients.flat_row_s0(), :] .= rv .* invN
    @views out[(n + 2):end, :] .= zero(eltype(out))

    @inbounds for (j1, children, _) in ws.groups
        psi1 = ws.wavelets[j1]
        @. ws.c1 = ws.xf * psi1
        Plans.inverse_transform!(ws.c2, ws.plan, ws.c1)
        if isempty(children)
            slice_modulus_mean!(ws.red, rv, ws.c2)
            @views out[Coefficients.flat_row_s1(j1, n), :] .= rv .* invN
            continue
        end
        slice_modulus_mean!(ws.red, rv, ws.rmod, ws.c2)
        @views out[Coefficients.flat_row_s1(j1, n), :] .= rv .* invN

        @. ws.c1 = complex(ws.rmod)
        Plans.forward_transform!(ws.c2, ws.plan, ws.c1)      # first-order spectrum, reused below
        for j2 in children
            psi2 = ws.wavelets[j2]
            @. ws.c1 = ws.c2 * psi2
            Plans.inverse_transform!(ws.c3, ws.plan, ws.c1)
            slice_modulus_mean!(ws.red, rv, ws.c3)
            @views out[Coefficients.flat_row_s2(j1, j2, n), :] .= rv .* invN
        end
    end
    return out
end

end # module Batched
