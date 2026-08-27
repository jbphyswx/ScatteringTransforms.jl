module Scattering1D

"""
    Scattering1D.jl — 1D Scattering Transform

Implements first- and second-order 1D scattering transforms.
"""

# Import sibling modules
using ..Plans: Plans
using ComputationalBackends: ComputationalBackends as CB
using SpectralBackends: SpectralBackends as SB
using ..FilterBanks: FilterBanks
using ..ScatteringCore: ScatteringCore
using ..Coefficients: Coefficients
using ..PathGraph: PathGraph
using ..Cascade: Cascade
using ..ScatteringFields: ScatteringFields

export ScatteringTransform1D
export scattering_transform!, cascade!

"""
    ScatteringTransform1D{T,V,M,P,Tree,FB,G}

1D scattering transform: a filter bank, the admissible path tree, a spectral plan, and the
workspace the cascade runs in. Every array field is a type parameter, so the same struct holds CPU,
GPU or static storage.

# Fields
- `filter_bank`: pre-computed 1D filter bank
- `tree`: admissible scattering paths
- `groups`: `(j1, children)` from `tree`, longest-first — the order the cascade walks
- `max_order`: maximum scattering order (1 or 2)
- `plan`: spectral transform plan (in-core direct sum by default; FFTW fast path if loaded)
- `buffer_input`: complex buffer for real→complex promotion, and multiply scratch
- `buffer_signal_fft`: the input spectrum, read-only for the whole cascade
- `buffer_conv`: inverse-transform output
- `dims`: the signal's length, as a 1-tuple
- `buffer_u1_fft`: the spectrum of the current first-order modulus, reused across that wavelet's
  children. The modulus itself needs no buffer of its own — the cascade writes it straight into
  `buffer_input`, which is what its transform reads.
"""
struct ScatteringTransform1D{T, V<:AbstractVector{Complex{T}},
                             P<:Plans.AbstractScatteringPlan, Tree<:PathGraph.ScatteringTree,
                             FB, G<:AbstractVector, PW}
    filter_bank::FB         # any FilterBank1D (CPU/GPU/static/…), kept as a type param
    tree::Tree              # admissible scattering paths (source of truth for second-order)
    groups::G               # (j1, children) from the tree, longest-first — the cascade's work list
    max_order::Int
    plan::P                 # spectral plan (direct-sum default, FFTW fast path); concrete type param
    buffer_input::V         # complex buffer for real→complex promotion of input
    buffer_signal_fft::V    # preserves signal FFT across the whole cascade
    buffer_conv::V          # convolution / inverse-transform output
    dims::NTuple{1,Int}
    buffer_u1_fft::V        # its spectrum, reused across that j1's children
    pw::PW                  # periodized cascade workspace — see `Cascade`
end

"""
    ScatteringTransform1D([T=Float64,] N, J; Q=1, max_order=2, spectral=AutoSpectralBackend())

Build a 1D scattering transform for length-`N` signals over `J` octaves. The element type is
positional, as for `zeros(T, …)`; omit it for `Float64`.
"""
function ScatteringTransform1D(::Type{T}, N::Int, J::Int;
                               Q::Int=1,
                               max_order::Int=2,
                               cache::Bool=true,
                               oversampling::Int=J,
                               spectral::SB.AbstractSpectralBackend=SB.AutoSpectralBackend()) where {T}
    filter_bank = FilterBanks.build_filter_bank1d(T, N, J, Val(cache); Q=Q)
    tree = PathGraph.build_tree([m.j_eff for m in filter_bank.meta], max_order)
    nw = FilterBanks.nwavelets(filter_bank)
    groups = max_order >= 2 ? PathGraph.order2_groups(tree, nw) :
             [(j, Int[], Int[]) for j in 1:nw]
    plan = Plans.make_plan(spectral, T, (N,))

    dummy = zeros(Complex{T}, N)
    # Workspace is O(N), not O(nw·N): the cascade holds one first-order modulus and its spectrum at
    # a time, because it finishes every child of a `j1` before moving to the next.
    # `buffer_conv` is `buffer_input` itself when the plan inverts in place.
    input = similar(dummy)
    conv = Plans.inplace_inverse(plan) ? input : similar(dummy)
    pw = Cascade.build(filter_bank, groups, (N,), T, oversampling, Plans.spectral_backend(plan))
    return ScatteringTransform1D(filter_bank, tree, groups, max_order, plan,
                                 input, similar(dummy), conv, (N,), similar(dummy), pw)
end
ScatteringTransform1D(N::Int, J::Int; kwargs...) = ScatteringTransform1D(Float64, N, J; kwargs...)

# Shares filter bank / tree / groups; copies only the buffers and the plan's scratch.
function ScatteringCore.task_workspace(st::ScatteringTransform1D)
    input = similar(st.buffer_input)
    conv = st.buffer_conv === st.buffer_input ? input : similar(st.buffer_conv)
    fb = FilterBanks.task_bank(st.filter_bank)
    return ScatteringTransform1D(fb, st.tree, st.groups,
                                 st.max_order, Plans.task_local_plan(st.plan),
                                 input, similar(st.buffer_signal_fft), conv,
                                 st.dims, similar(st.buffer_u1_fft), Cascade.task_copy(st.pw, fb))
end

"""
    (st::ScatteringTransform1D)(signal) -> ScatteringCoefficients1D

Apply scattering transform to 1D signal.
Returns type-stable ScatteringCoefficients1D with element type matching input.
"""
function (st::ScatteringTransform1D)(signal::AbstractVector)
    num_w = FilterBanks.nwavelets(st.filter_bank)
    T = real(eltype(st.buffer_input))
    
    # Pre-allocate coefficient storage
    coeffs = Coefficients.ScatteringCoefficients1D(num_w, T; compute_S2=st.max_order >= 2)
    
    # Apply in-place transform, get result with updated S0 (zero alloc for S1/S2)
    return scattering_transform!(coeffs, st, signal)
end

"""
    scattering_transform!(coeffs, st, signal)

In-place scattering transform. Fills pre-allocated S1/S2, returns the coefficients with S0 updated.
Allocation-free; only allocates a new wrapper struct when S0 is a scalar (immutable).
"""
function scattering_transform!(coeffs::Coefficients.ScatteringCoefficients1D,
                              st::ScatteringTransform1D,
                              signal::AbstractVector)
    st.buffer_input .= complex.(signal)
    Plans.forward_transform!(st.buffer_signal_fft, st.plan, st.buffer_input)
    cascade!(coeffs.S1, coeffs.S2, st, st.buffer_signal_fft)
    return Coefficients.update_S0(coeffs, ScatteringCore.spatial_average(signal))
end

"""
    scattering_transform!(coeffs, backend, st, signal)

Transform one signal on an explicit execution backend. `SerialBackend` runs the cascade in this
task; `ThreadedBackend` (OhMyThreads extension) spreads the first-order wavelet groups across
tasks, which is the only parallel axis available when there is a single field rather than a batch.
The input transform is done once up front, so only the group loop is parallel.
"""
function scattering_transform!(coeffs::Coefficients.ScatteringCoefficients1D,
                               backend::CB.AbstractExecutionBackend,
                               st::ScatteringTransform1D, signal::AbstractVector)
    st.buffer_input .= complex.(signal)
    Plans.forward_transform!(st.buffer_signal_fft, st.plan, st.buffer_input)
    cascade!(coeffs.S1, coeffs.S2, backend, st, st.buffer_signal_fft)
    return Coefficients.update_S0(coeffs, ScatteringCore.spatial_average(signal))
end

cascade!(S1::AbstractVector, S2::AbstractMatrix, ::CB.AbstractSerialBackend,
         st::ScatteringTransform1D, signal_fft::AbstractVector) = cascade!(S1, S2, st, signal_fft)

"""
    cascade!(S1, S2, st, signal_fft) -> (S1, S2)

Both scattering orders in one pass over the tree, grouped by first-order wavelet:

    for (j1, children):  U₁ = |x ⋆ ψ_j1| ;  S1[j1] = ⟨U₁⟩
                         Û₁ = fft(U₁)     ;  S2[j1,j2] = ⟨|U₁ ⋆ ψ_j2|⟩  for each child

The first-order convolution is therefore evaluated once, not once for `S1` and again for `S2`, and
only one `U₁`/`Û₁` pair is live at a time rather than one per wavelet. Wavelets with no admissible
child skip the modulus buffer entirely, reducing to a single fused `⟨|·|⟩`.

`signal_fft` is read only, so the caller's preserved signal spectrum survives the call.
"""
cascade!(S1::AbstractVector, S2::AbstractMatrix, st::ScatteringTransform1D,
         signal_fft::AbstractVector) =
    Cascade.cascade!(S1, S2, st.pw, st.filter_bank, st.groups, signal_fft)

# The undecimated form, kept as the oracle the periodized cascade is pinned against.
function _cascade_full!(S1::AbstractVector, S2::AbstractMatrix, st::ScatteringTransform1D,
                        signal_fft::AbstractVector)
    isempty(S2) || fill!(S2, zero(eltype(S2)))
    fb = st.filter_bank
    @inbounds for (j1, children, _) in st.groups
        ScatteringCore.wavelet_convolve!(st.buffer_conv, signal_fft, FilterBanks.filter_at(fb, j1),
                                         st.plan, st.buffer_input)
        if isempty(children)
            S1[j1] = ScatteringCore.modulus_mean(st.buffer_conv)
            continue
        end
        # Modulus written straight into the transform's input, not via a real buffer and a widen.
        S1[j1] = ScatteringCore.modulus_mean!(st.buffer_input, st.buffer_conv)
        Plans.forward_transform!(st.buffer_u1_fft, st.plan, st.buffer_input)
        for j2 in children
            ScatteringCore.wavelet_convolve!(st.buffer_conv, st.buffer_u1_fft, FilterBanks.filter_at(fb, j2),
                                             st.plan, st.buffer_input)
            S2[j1, j2] = ScatteringCore.modulus_mean(st.buffer_conv)
        end
    end
    return S1, S2
end

# ============================================================================
# Localized (Mallat) scattering field: S_p x = (|U_p x| ⋆ φ_J) ↓ s
# ============================================================================

# Default decimation factor for the localized field: `2^(J-1)`, reduced to the largest such power of
# two that divides `N` (so `N % s == 0`).
function _default_subsample(N::Int, J::Int)
    ds = 1 << max(0, J - 1)
    while ds > 1 && N % ds != 0
        ds >>= 1
    end
    return ds
end

function ScatteringFields.scattering_field(st::ScatteringTransform1D, signal::AbstractVector;
        subsample::Int = _default_subsample(st.dims[1], st.filter_bank.J))
    N = st.dims[1]
    N % subsample == 0 ||
        throw(ArgumentError("subsample factor $subsample must divide signal length $N"))
    # Real unless the input is complex, in which case the order-0 field is too.
    T = promote_type(real(eltype(st.buffer_input)), eltype(signal))
    M = N ÷ subsample
    npath = PathGraph.npaths(st.tree)
    data = zeros(T, M, npath)   # zero-filled: unsupported higher-order paths stay 0, not garbage
    field = ScatteringFields.ScatteringField1D(st.tree, data, subsample,
                                               Cascade.field_workspace(st, subsample))
    return ScatteringFields.scattering_field!(field, st, signal)
end

function ScatteringFields.scattering_field!(field::ScatteringFields.ScatteringField1D,
        st::ScatteringTransform1D, signal::AbstractVector)
    st.buffer_input .= complex.(signal)
    Plans.forward_transform!(st.buffer_signal_fft, st.plan, st.buffer_input)
    Cascade.field_cascade!(field.data, field.ws, st.filter_bank, st.groups, st.buffer_signal_fft,
                           first(PathGraph.order_range(st.tree, 0)),
                           first(PathGraph.order_range(st.tree, 1)))
    return field
end

# ============================================================================
# Non-mutating, autodiff-friendly forward: composes the non-mutating spectral transforms with
# broadcast modulus + mean. No preallocated workspace, no in-place writes — so it differentiates
# cleanly through DifferentiationInterface and accepts Dual/Float32 inputs. Numerically matches the
# in-place `st(signal)`.
# ============================================================================

function ScatteringCore.scattering(st::ScatteringTransform1D, signal::AbstractVector)
    S0, S1, S2 = Cascade.scattering_values(st, signal)
    return Coefficients.ScatteringCoefficients1D(S1, S2; S0=S0)
end

end # module Scattering1D
