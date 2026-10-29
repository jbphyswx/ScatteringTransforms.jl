module Inverse

# Reconstruction without autodiff. The modulus discards each wavelet coefficient's local phase, so
# the scattering transform has no exact inverse; the linear wavelet layer does, by the canonical dual
# frame, and phase retrieval from first-order moduli (Waldspurger & Mallat 2015) is unique only up to
# a global sign.

using ..Plans: Plans
using ..FilterBanks: FilterBanks
using ..ScatteringCore: ScatteringCore

export wavelet_transform, wavelet_transform!, iwavelet, iwavelet!, reconstruct_phase
export ReconstructionWorkspace

"""
    ReconstructionWorkspace(st)

Scratch for the linear wavelet inverse and for phase retrieval: the complex wavelet coefficient
fields (`nw` of them), the low-pass field, a spectrum accumulator, and the bank's
Littlewood–Paley sum `frame`. Every filter's Fourier response is real, so the dual filters are
`ψ̂_λ/A` and `φ̂/A`.
"""
struct ReconstructionWorkspace{CA, CV, RA}
    wavelet::CV            # (nw) complex coefficient fields
    lowpass::CA
    Xf::CA                 # spectrum of the current estimate
    Xrec::CA               # reconstruction accumulator
    buf::CA                # multiply / transform scratch
    field::RA              # real reconstructed field
    frame::RA              # Littlewood–Paley sum `A`
end

function ReconstructionWorkspace(st::ScatteringCore.ScatteringTransform)
    fb = st.filter_bank
    proto = fb.averaging
    T = eltype(proto)
    cplx() = similar(proto, Complex{T})
    frame = copyto!(similar(proto, T), _littlewood_paley_host(fb))
    return ReconstructionWorkspace(
        [cplx() for _ in 1:FilterBanks.nwavelets(fb)], cplx(),
        cplx(), cplx(), cplx(),
        similar(proto, T), frame)
end

Base.show(io::IO, ws::ReconstructionWorkspace) =
    print(io, "ReconstructionWorkspace(", length(ws.wavelet), " wavelets of size ", size(ws.field), ")")

# The bank's Littlewood–Paley sum, formed on the host: `FilterBanks.negated` reverses and shifts
# whole arrays. A computed bank and a stored bank of host arrays are already there.
_littlewood_paley_host(fb) = FilterBanks.littlewood_paley(_on_host(fb))

_on_host(fb) = fb
_on_host(fb::FilterBanks.FilterBank1D{T, <:Array}) where {T} = fb
_on_host(fb::FilterBanks.FilterBank2D{T, <:Array}) where {T} = fb
_on_host(fb::FilterBanks.FilterBank3D{T, <:Array}) where {T} = fb
_on_host(fb::FilterBanks.FilterBank1D) =
    FilterBanks.FilterBank1D(map(Array, fb.wavelets), Array(fb.averaging), fb.meta, fb.J, fb.Q)
_on_host(fb::FilterBanks.FilterBank2D) =
    FilterBanks.FilterBank2D(map(Array, fb.wavelets), Array(fb.averaging), fb.meta, fb.J, fb.L)
_on_host(fb::FilterBanks.FilterBank3D) =
    FilterBanks.FilterBank3D(map(Array, fb.wavelets), Array(fb.averaging), fb.meta, fb.J, fb.n_orient)

"""
    wavelet_transform(st, x) -> (; wavelet, lowpass)
    wavelet_transform!(ws, st, x) -> (; wavelet, lowpass)

The linear (pre-modulus) wavelet layer underlying the scattering transform: the complex wavelet
coefficient fields `wavelet[λ] = x ⋆ ψ_λ` and the low-pass field `lowpass = x ⋆ φ`, which together
determine `x` exactly (see [`iwavelet!`](@ref)). The `!` form returns arrays owned by `ws`.
"""
function wavelet_transform!(ws::ReconstructionWorkspace, st::ScatteringCore.ScatteringTransform, x::AbstractArray)
    fb, plan = st.filter_bank, st.plan
    ws.buf .= complex.(x)
    Plans.forward_transform!(ws.Xf, plan, ws.buf)
    @inbounds for λ in 1:FilterBanks.nwavelets(fb)
        ScatteringCore.wavelet_convolve!(ws.wavelet[λ], ws.Xf, FilterBanks.filter_at(fb, λ),
                                         plan, ws.buf)
    end
    ScatteringCore.wavelet_convolve!(ws.lowpass, ws.Xf, fb.averaging, plan, ws.buf)
    return (; wavelet = ws.wavelet, lowpass = ws.lowpass)
end

wavelet_transform(st::ScatteringCore.ScatteringTransform, x::AbstractArray) =
    wavelet_transform!(ReconstructionWorkspace(st), st, x)

"""
    iwavelet(st, wavelet, lowpass) -> x
    iwavelet(st, wt) -> x
    iwavelet!(ws, st, wavelet, lowpass) -> x

Inverse of [`wavelet_transform!`](@ref) for a real field by the canonical dual frame,
`x = Re F⁻¹[(φ̂·Ŷ + Σ_λ ψ̂_λ·Ŵ_λ) / A]` with `A` the bank's Littlewood–Paley sum: the real part
takes `C(k)` to `(C(k) + conj C(−k))/2`, which turns `C = (φ̂² + Σ_λ ψ̂_λ²)·x̂` into `A·x̂`. Returns
the reconstructed **real** field; `iwavelet(st, wavelet_transform(st, x)) ≈ x` to rounding.
"""
function iwavelet!(ws::ReconstructionWorkspace, st::ScatteringCore.ScatteringTransform, wavelet, lowpass::AbstractArray)
    fb, plan = st.filter_bank, st.plan
    ws.buf .= lowpass
    Plans.forward_transform!(ws.Xrec, plan, ws.buf)
    ws.Xrec .*= fb.averaging
    @inbounds for λ in eachindex(wavelet)
        ws.buf .= wavelet[λ]
        Plans.forward_transform!(ws.Xf, plan, ws.buf)
        ws.Xrec .+= ws.Xf .* FilterBanks.filter_at(fb, λ)
    end
    ws.Xrec ./= ws.frame
    Plans.inverse_transform!(ws.buf, plan, ws.Xrec)
    ws.field .= real.(ws.buf)
    return ws.field
end

iwavelet(st::ScatteringCore.ScatteringTransform, wavelet, lowpass::AbstractArray) =
    copy(iwavelet!(ReconstructionWorkspace(st), st, wavelet, lowpass))

iwavelet(st::ScatteringCore.ScatteringTransform, wt::NamedTuple) = iwavelet(st, wt.wavelet, wt.lowpass)

"""
    reconstruct_phase(st, moduli; iters=200, init=nothing, seed_lowpass=nothing) -> x

Phase retrieval from the first-order **moduli** `moduli[λ] = |x ⋆ ψ_λ|` (the real fields the
scattering transform averages), via Gerchberg–Saxton alternating projections:

1. take the linear wavelet transform of the current estimate;
2. re-impose the target magnitudes on each band-pass channel (keeping the recovered phase);
3. reconstruct with the exact frame inverse [`iwavelet!`](@ref); repeat.

The low-pass channel carries no magnitude target, so it is taken from the current estimate each
iteration (or from `seed_lowpass` if supplied). The reconstruction is determined only up to a
global sign/phase. `init` (defaults to a random field matched in energy to the moduli) seeds the
estimate; pass one for reproducibility. Returns the real reconstructed field, on the array type of
the transform's filters. The loop reuses one workspace.
"""
function reconstruct_phase(st::ScatteringCore.ScatteringTransform, moduli;
                           iters::Int=200, init=nothing, seed_lowpass=nothing)
    fb = st.filter_bank
    length(moduli) == FilterBanks.nwavelets(fb) ||
        throw(ArgumentError("expected $(FilterBanks.nwavelets(fb)) moduli, got $(length(moduli))"))
    T = real(eltype(fb.averaging))
    sz = size(first(moduli))

    # The estimate lives where the filters do; a random seed is drawn on the host and copied over.
    x = similar(fb.averaging, T, sz)
    if init === nothing
        # Seed with a random real field scaled to the total band-pass energy.
        e = sqrt(sum(m -> sum(abs2, m), moduli) / length(moduli))
        copyto!(x, randn(T, sz) .* (e / sqrt(prod(sz)) + eps(T)))
    else
        copyto!(x, init)
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
