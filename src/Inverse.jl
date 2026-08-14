module Inverse

"""
    Inverse.jl — Reconstruction from scattering / wavelet representations

The scattering transform has **no exact analytic inverse**: the modulus `|·|` discards the
local phase of each wavelet coefficient. But three reconstruction levels exist, two of which are
exact-or-iterative and need no autodiff (the third, gradient-descent synthesis from the
scattering coefficients themselves, lives in the DifferentiationInterface extension):

1. **Exact linear wavelet-frame inverse** — from the *complex* (pre-modulus) wavelet
   coefficients `x ⋆ ψ_λ` plus the low-pass `x ⋆ φ`. Because the bank is a tight frame
   (`Σ_λ |ψ̂_λ|² + |φ̂|² ≡ 1`), the dual frame is the frame itself and reconstruction is the
   conjugate-filter sum
   `x̂ = Σ_λ (x ⋆ ψ_λ) ⋆ ψ_λ^* + (x ⋆ φ) ⋆ φ^*` — exact to machine precision.
   [`wavelet_transform`](@ref) / [`iwavelet`](@ref).

2. **Phase retrieval** from first-order moduli `|x ⋆ ψ_λ|` (Waldspurger & Mallat 2015):
   alternating projections (Gerchberg–Saxton) — reconstruct via the exact inverse, re-impose the
   target magnitudes, repeat. Non-unique up to a global sign/phase. [`reconstruct_phase`](@ref).

All routines here are AD-free. The `!` forms run entirely through the in-place plan primitives
against a [`ReconstructionWorkspace`](@ref), which matters because phase retrieval calls them
`iters` times: the allocating forms are convenience wrappers around them.
"""

using ..Plans: Plans
using ..ScatteringCore: ScatteringCore
using ..Scattering1D: Scattering1D
using ..Scattering2D: Scattering2D
using ..Scattering3D: Scattering3D

export wavelet_transform, wavelet_transform!, iwavelet, iwavelet!, reconstruct_phase
export ReconstructionWorkspace

# Any of the gridded (1D/2D/3D) transforms — all expose `.plan` and `.filter_bank`
# (`.wavelets`, `.averaging`), so the linear inverse is dimension-agnostic by duck typing.
const GriddedScattering = Union{Scattering1D.ScatteringTransform1D,
                                Scattering2D.ScatteringTransform2D,
                                Scattering3D.ScatteringTransform3D}

"""
    ReconstructionWorkspace(st)

Scratch for the linear wavelet inverse and for phase retrieval: the complex wavelet coefficient
fields (`nw` of them — that is the representation's own size), the low-pass field, a spectrum
accumulator, and the conjugated filters.

The conjugates are precomputed because `iwavelet!` would otherwise rebuild `conj.(ψ_λ)` for every
wavelet on every iteration, and `reconstruct_phase` runs `iters` of them.
"""
struct ReconstructionWorkspace{CA, CV, RA}
    wavelet::CV            # (nw) complex coefficient fields
    lowpass::CA
    Xf::CA                 # spectrum of the current estimate
    Xrec::CA               # reconstruction accumulator
    buf::CA                # multiply / transform scratch
    conj_wavelets::CV      # (nw) conj(ψ_λ), built once
    conj_averaging::CA
    field::RA              # real reconstructed field
end

function ReconstructionWorkspace(st::GriddedScattering)
    fb = st.filter_bank
    proto = fb.averaging
    T = real(eltype(proto))
    return ReconstructionWorkspace(
        [similar(proto) for _ in eachindex(fb.wavelets)], similar(proto),
        similar(proto), similar(proto), similar(proto),
        [conj.(ψ) for ψ in fb.wavelets], conj.(proto),
        similar(proto, T))
end

"""
    wavelet_transform(st, x) -> (; wavelet, lowpass)
    wavelet_transform!(ws, st, x) -> (; wavelet, lowpass)

The **linear** (pre-modulus) wavelet layer underlying the scattering transform: the complex
wavelet coefficient fields `wavelet[λ] = x ⋆ ψ_λ` (the continuous wavelet transform) and the
low-pass field `lowpass = x ⋆ φ`. Together these are an *exact, invertible* representation of
`x` (see [`iwavelet`](@ref)); taking `|wavelet[λ]|` and averaging is the first scattering layer.
"""
function wavelet_transform!(ws::ReconstructionWorkspace, st::GriddedScattering, x::AbstractArray)
    fb, plan = st.filter_bank, st.plan
    ws.buf .= complex.(x)
    Plans.forward_transform!(ws.Xf, plan, ws.buf)
    @inbounds for λ in eachindex(fb.wavelets)
        ScatteringCore.wavelet_convolve!(ws.wavelet[λ], ws.Xf, fb.wavelets[λ], plan, ws.buf)
    end
    ScatteringCore.wavelet_convolve!(ws.lowpass, ws.Xf, fb.averaging, plan, ws.buf)
    return (; wavelet = ws.wavelet, lowpass = ws.lowpass)
end

wavelet_transform(st::GriddedScattering, x::AbstractArray) =
    wavelet_transform!(ReconstructionWorkspace(st), st, x)

"""
    iwavelet(st, wavelet, lowpass) -> x
    iwavelet(st, wt) -> x
    iwavelet!(ws, st, wavelet, lowpass) -> x

Exact inverse of [`wavelet_transform`](@ref) via the tight-frame conjugate-filter sum
`x̂ = Σ_λ ψ̂_λ^* · (x̂·ψ̂_λ) + φ̂^* · (x̂·φ̂)`. Returns the reconstructed **real** field. With the
tight-frame bank (`Σ|ψ̂_λ|²+|φ̂|² ≡ 1`) this satisfies `iwavelet(st, wavelet_transform(st, x)...) ≈ x`
to machine precision.

The accumulation is in place: the spectrum sum is built in one buffer rather than rebuilt per
wavelet, which is what makes `iters` rounds of phase retrieval affordable.
"""
function iwavelet!(ws::ReconstructionWorkspace, st::GriddedScattering, wavelet, lowpass::AbstractArray)
    plan = st.plan
    ws.buf .= lowpass
    Plans.forward_transform!(ws.Xrec, plan, ws.buf)
    ws.Xrec .*= ws.conj_averaging
    @inbounds for λ in eachindex(wavelet)
        ws.buf .= wavelet[λ]
        Plans.forward_transform!(ws.Xf, plan, ws.buf)
        ws.Xrec .+= ws.Xf .* ws.conj_wavelets[λ]
    end
    Plans.inverse_transform!(ws.buf, plan, ws.Xrec)
    ws.field .= real.(ws.buf)
    return ws.field
end

iwavelet(st::GriddedScattering, wavelet, lowpass::AbstractArray) =
    copy(iwavelet!(ReconstructionWorkspace(st), st, wavelet, lowpass))

iwavelet(st::GriddedScattering, wt::NamedTuple) = iwavelet(st, wt.wavelet, wt.lowpass)

"""
    reconstruct_phase(st, moduli; iters=200, init=nothing, seed_lowpass=nothing) -> x

Phase retrieval from the first-order **moduli** `moduli[λ] = |x ⋆ ψ_λ|` (the real fields the
scattering transform averages), via Gerchberg–Saxton alternating projections:

1. take the linear wavelet transform of the current estimate;
2. re-impose the target magnitudes on each band-pass channel (keeping the recovered phase);
3. reconstruct with the exact frame inverse [`iwavelet`](@ref); repeat.

The low-pass channel carries no magnitude target, so it is taken from the current estimate each
iteration (or from `seed_lowpass` if supplied). The reconstruction is determined only up to a
global sign/phase. `init` (defaults to a random field matched in energy to the moduli) seeds the
estimate; pass one for reproducibility. Returns the real reconstructed field.

One workspace is built up front and reused across all `iters`, so the loop allocates nothing.
"""
function reconstruct_phase(st::GriddedScattering, moduli;
                           iters::Int=200, init=nothing, seed_lowpass=nothing)
    fb = st.filter_bank
    length(moduli) == length(fb.wavelets) ||
        throw(ArgumentError("expected $(length(fb.wavelets)) moduli, got $(length(moduli))"))
    T = real(eltype(fb.averaging))
    sz = size(first(moduli))

    x = if init === nothing
        # Seed with a random real field scaled to the total band-pass energy.
        e = sqrt(sum(m -> sum(abs2, m), moduli) / length(moduli))
        randn(T, sz) .* (e / sqrt(prod(sz)) + eps(T))
    else
        T.(collect(init))
    end

    ws = ReconstructionWorkspace(st)
    for _ in 1:iters
        wavelet_transform!(ws, st, x)
        # Re-impose the target magnitudes while keeping the recovered phase, in place.
        # `cis(angle(0)) = 1`, so a zero coefficient gracefully takes magnitude `m` (no 0/0).
        @inbounds for λ in eachindex(ws.wavelet)
            w, m = ws.wavelet[λ], moduli[λ]
            @. w = m * cis(angle(w))
        end
        lp = seed_lowpass === nothing ? ws.lowpass : seed_lowpass
        x = iwavelet!(ws, st, ws.wavelet, lp)
    end
    return copy(x)
end

end # module Inverse
