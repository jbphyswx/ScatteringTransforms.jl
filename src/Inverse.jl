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

All routines here are AD-free and reuse the transform's existing tight-frame `filter_bank` and
spectral `plan` (through the non-mutating, element-type-generic `Plans.forward_transform` /
`Plans.inverse_transform`, so `Float32`/`Float64`/GPU arrays all flow through). Load `using FFTW`
for the `O(N log N)` spectral path.
"""

using ..Plans: Plans
using ..Scattering1D: Scattering1D
using ..Scattering2D: Scattering2D
using ..Scattering3D: Scattering3D

export wavelet_transform, iwavelet, reconstruct_phase

# Any of the gridded (1D/2D/3D) transforms — all expose `.plan` and `.filter_bank`
# (`.wavelets`, `.averaging`), so the linear inverse is dimension-agnostic by duck typing.
const GriddedScattering = Union{Scattering1D.ScatteringTransform1D,
                                Scattering2D.ScatteringTransform2D,
                                Scattering3D.ScatteringTransform3D}

"""
    wavelet_transform(st, x) -> (; wavelet, lowpass)

The **linear** (pre-modulus) wavelet layer underlying the scattering transform: the complex
wavelet coefficient fields `wavelet[λ] = x ⋆ ψ_λ` (the continuous wavelet transform) and the
low-pass field `lowpass = x ⋆ φ`. Together these are an *exact, invertible* representation of
`x` (see [`iwavelet`](@ref)); taking `|wavelet[λ]|` and averaging is the first scattering layer.
"""
function wavelet_transform(st::GriddedScattering, x::AbstractArray)
    plan = st.plan
    fb = st.filter_bank
    Xf = Plans.forward_transform(plan, complex.(x))
    wavelet = map(ψ -> Plans.inverse_transform(plan, Xf .* ψ), fb.wavelets)
    lowpass = Plans.inverse_transform(plan, Xf .* fb.averaging)
    return (; wavelet, lowpass)
end

"""
    iwavelet(st, wavelet, lowpass) -> x
    iwavelet(st, wt) -> x

Exact inverse of [`wavelet_transform`](@ref) via the tight-frame conjugate-filter sum
`x̂ = Σ_λ ψ̂_λ^* · (x̂·ψ̂_λ) + φ̂^* · (x̂·φ̂)`. Returns the reconstructed **real** field. With the
tight-frame bank (`Σ|ψ̂_λ|²+|φ̂|² ≡ 1`) this satisfies `iwavelet(st, wavelet_transform(st, x)...) ≈ x`
to machine precision.
"""
function iwavelet(st::GriddedScattering, wavelet, lowpass::AbstractArray)
    plan = st.plan
    fb = st.filter_bank
    Xrec = Plans.forward_transform(plan, lowpass) .* conj.(fb.averaging)
    for λ in eachindex(wavelet)
        Xrec = Xrec .+ Plans.forward_transform(plan, wavelet[λ]) .* conj.(fb.wavelets[λ])
    end
    return real.(Plans.inverse_transform(plan, Xrec))
end

iwavelet(st::GriddedScattering, wt::NamedTuple) = iwavelet(st, wt.wavelet, wt.lowpass)

# Re-impose target magnitudes while keeping the current phase: M .* exp(i·angle(W)).
# `cis(angle(0)) = 1`, so zero coefficients gracefully take magnitude M (no division by zero).
@inline _reimpose_magnitude(W, M) = @. M * cis(angle(W))

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

    local wt
    for _ in 1:iters
        wt = wavelet_transform(st, x)
        W = map(_reimpose_magnitude, wt.wavelet, moduli)
        lp = seed_lowpass === nothing ? wt.lowpass : seed_lowpass
        x = iwavelet(st, W, lp)
    end
    return x
end

end # module Inverse
