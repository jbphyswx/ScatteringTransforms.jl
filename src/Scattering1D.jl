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
- `buffer_mod`: real modulus buffer for the localized-field path
- `buffer_u1`, `buffer_u1_fft`: the current first-order modulus and its spectrum, reused across
  that wavelet's children — one pair, not one per wavelet
"""
struct ScatteringTransform1D{T, V<:AbstractVector{Complex{T}}, M<:AbstractVector{T},
                             P<:Plans.AbstractScatteringPlan, Tree<:PathGraph.ScatteringTree,
                             FB<:FilterBanks.FilterBank1D, G<:AbstractVector}
    filter_bank::FB         # any FilterBank1D (CPU/GPU/static/…), kept as a type param
    tree::Tree              # admissible scattering paths (source of truth for second-order)
    groups::G               # (j1, children) from the tree, longest-first — the cascade's work list
    max_order::Int
    plan::P                 # spectral plan (direct-sum default, FFTW fast path); concrete type param
    buffer_input::V         # complex buffer for real→complex promotion of input
    buffer_signal_fft::V    # preserves signal FFT across the whole cascade
    buffer_conv::V          # convolution / inverse-transform output
    buffer_mod::M           # real modulus buffer (localized-field path)
    buffer_u1::M            # first-order modulus of the current j1
    buffer_u1_fft::V        # its spectrum, reused across that j1's children
end

"""
    ScatteringTransform1D([T=Float64,] N, J; Q=1, max_order=2, spectral=AutoSpectralBackend())

Build a 1D scattering transform for length-`N` signals over `J` octaves. The element type is
positional, as for `zeros(T, …)`; omit it for `Float64`.
"""
function ScatteringTransform1D(::Type{T}, N::Int, J::Int;
                               Q::Int=1,
                               max_order::Int=2,
                               spectral::SB.AbstractSpectralBackend=SB.AutoSpectralBackend()) where {T}
    filter_bank = FilterBanks.build_filter_bank1d(T, N, J; Q=Q)
    tree = PathGraph.build_tree([m.j_eff for m in filter_bank.meta], max_order)
    groups = max_order >= 2 ? PathGraph.order2_groups(tree, length(filter_bank.wavelets)) :
             [(j, Int[], Int[]) for j in 1:length(filter_bank.wavelets)]
    plan = Plans.make_plan(spectral, T, (N,))

    dummy = zeros(Complex{T}, N)
    # Workspace is O(N), not O(nw·N): the cascade holds one first-order modulus and its spectrum at
    # a time, because it finishes every child of a `j1` before moving to the next.
    return ScatteringTransform1D(filter_bank, tree, groups, max_order, plan,
                                 similar(dummy), similar(dummy), similar(dummy),
                                 Vector{T}(undef, N), Vector{T}(undef, N), similar(dummy))
end
ScatteringTransform1D(N::Int, J::Int; kwargs...) = ScatteringTransform1D(Float64, N, J; kwargs...)

# Shares filter bank / tree / groups; copies only the buffers and the plan's scratch.
ScatteringCore.task_workspace(st::ScatteringTransform1D) =
    ScatteringTransform1D(st.filter_bank, st.tree, st.groups, st.max_order,
                          Plans.task_local_plan(st.plan),
                          similar(st.buffer_input), similar(st.buffer_signal_fft),
                          similar(st.buffer_conv), similar(st.buffer_mod),
                          similar(st.buffer_u1), similar(st.buffer_u1_fft))

"""
    (st::ScatteringTransform1D)(signal) -> ScatteringCoefficients1D

Apply scattering transform to 1D signal.
Returns type-stable ScatteringCoefficients1D with element type matching input.
"""
function (st::ScatteringTransform1D)(signal::AbstractVector)
    num_w = length(st.filter_bank.wavelets)
    T = eltype(st.buffer_mod)
    
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
function cascade!(S1::AbstractVector, S2::AbstractMatrix, st::ScatteringTransform1D,
                  signal_fft::AbstractVector)
    isempty(S2) || fill!(S2, zero(eltype(S2)))
    wavelets = st.filter_bank.wavelets
    @inbounds for (j1, children, _) in st.groups
        ScatteringCore.wavelet_convolve!(st.buffer_conv, signal_fft, wavelets[j1],
                                         st.plan, st.buffer_input)
        if isempty(children)
            S1[j1] = ScatteringCore.modulus_mean(st.buffer_conv)
            continue
        end
        S1[j1] = ScatteringCore.modulus_mean!(st.buffer_u1, st.buffer_conv)
        st.buffer_input .= complex.(st.buffer_u1)
        Plans.forward_transform!(st.buffer_u1_fft, st.plan, st.buffer_input)
        for j2 in children
            ScatteringCore.wavelet_convolve!(st.buffer_conv, st.buffer_u1_fft, wavelets[j2],
                                             st.plan, st.buffer_input)
            S2[j1, j2] = ScatteringCore.modulus_mean(st.buffer_conv)
        end
    end
    return S1, S2
end

# ============================================================================
# Localized (Mallat) scattering field: S_p x = (|U_p x| ⋆ φ_J) ↓ s
# ============================================================================

"""
    _default_subsample(N, J) -> Int

Default decimation factor for the localized field: `2^(J-1)`, reduced to the largest such
power of two that divides `N` (so `N % s == 0`).
"""
function _default_subsample(N::Int, J::Int)
    ds = 1 << max(0, J - 1)
    while ds > 1 && N % ds != 0
        ds >>= 1
    end
    return ds
end

# Low-pass a real field `U` (length N) by φ_J and decimate by `ds` into `dst` (length N÷ds).
# Reuses `buffer_input`/`buffer_conv`; does NOT touch `buffer_signal_fft` (the preserved x FFT).
@inline function _lowpass_downsample!(dst, st::ScatteringTransform1D, U::AbstractVector, ds::Int)
    φ = st.filter_bank.averaging
    st.buffer_input .= complex.(U)
    Plans.forward_transform!(st.buffer_conv, st.plan, st.buffer_input)     # U_fft -> buffer_conv
    st.buffer_conv .*= φ
    Plans.inverse_transform!(st.buffer_input, st.plan, st.buffer_conv)     # (U ⋆ φ) -> buffer_input
    # Decimate by ds via a strided view + broadcast (CPU + GPU compatible).
    @views dst .= real.(st.buffer_input[1:ds:(1 + (length(dst) - 1) * ds)])
    return dst
end

function ScatteringFields.scattering_field(st::ScatteringTransform1D, signal::AbstractVector;
        subsample::Int = _default_subsample(length(st.buffer_mod), st.filter_bank.J))
    N = length(st.buffer_mod)
    N % subsample == 0 ||
        throw(ArgumentError("subsample factor $subsample must divide signal length $N"))
    T = eltype(st.buffer_mod)
    M = N ÷ subsample
    npath = PathGraph.npaths(st.tree)
    data = zeros(T, M, npath)   # zero-filled: unsupported higher-order paths stay 0, not garbage
    field = ScatteringFields.ScatteringField1D(st.tree, data, subsample)
    return ScatteringFields.scattering_field!(field, st, signal)
end

function ScatteringFields.scattering_field!(field::ScatteringFields.ScatteringField1D,
        st::ScatteringTransform1D, signal::AbstractVector)
    tree = st.tree
    ds = field.subsample
    data = field.data

    # x FFT (preserved across passes in buffer_signal_fft)
    st.buffer_input .= complex.(signal)
    Plans.forward_transform!(st.buffer_signal_fft, st.plan, st.buffer_input)

    # order 0 (root): (x ⋆ φ_J) ↓
    copyto!(st.buffer_mod, signal)
    root = first(PathGraph.order_range(tree, 0))
    _lowpass_downsample!(view(data, :, root), st, st.buffer_mod, ds)

    # Orders 1 and 2 in one grouped pass, as in `cascade!`: the first-order field of each `j1` is
    # computed once, low-passed into its own column, then transformed once and reused by every child.
    p1_first = first(PathGraph.order_range(tree, 1))
    wavelets = st.filter_bank.wavelets
    @inbounds for (j1, children, pathids) in st.groups
        ScatteringCore.wavelet_convolve!(st.buffer_conv, st.buffer_signal_fft, wavelets[j1],
            st.plan, st.buffer_input)
        ScatteringCore.apply_modulus!(st.buffer_u1, st.buffer_conv)
        _lowpass_downsample!(view(data, :, p1_first + j1 - 1), st, st.buffer_u1, ds)
        isempty(children) && continue
        st.buffer_input .= complex.(st.buffer_u1)
        Plans.forward_transform!(st.buffer_u1_fft, st.plan, st.buffer_input)
        for (j2, p) in zip(children, pathids)
            ScatteringCore.wavelet_convolve!(st.buffer_conv, st.buffer_u1_fft, wavelets[j2],
                st.plan, st.buffer_input)
            ScatteringCore.apply_modulus!(st.buffer_mod, st.buffer_conv)
            _lowpass_downsample!(view(data, :, p), st, st.buffer_mod, ds)
        end
    end
    return field
end

# ============================================================================
# Non-mutating, autodiff-friendly forward: composes the non-mutating spectral transforms with
# broadcast modulus + mean. No preallocated workspace, no in-place writes — so it differentiates
# cleanly through DifferentiationInterface and accepts Dual/Float32 inputs. Numerically matches the
# in-place `st(signal)`.
# ============================================================================

function ScatteringCore.scattering(st::ScatteringTransform1D, signal::AbstractVector)
    plan = st.plan
    fb = st.filter_bank
    tree = st.tree
    n = length(fb.wavelets)

    Xf = Plans.forward_transform(plan, complex.(signal))
    # First-order moduli |x ⋆ ψ_j| and their means S1[j].
    U1 = map(ψ -> abs.(Plans.inverse_transform(plan, Xf .* ψ)), fb.wavelets)
    S1 = map(u -> sum(u) / length(u), U1)
    S0 = sum(signal) / length(signal)

    if st.max_order >= 2 && length(tree.by_order) >= 3
        U1f = map(u -> Plans.forward_transform(plan, complex.(u)), U1)
        r2 = PathGraph.order_range(tree, 2)
        s2vals = map(collect(r2)) do p
            idx = PathGraph.path_indices(tree, p)
            m = abs.(Plans.inverse_transform(plan, U1f[idx[1]] .* fb.wavelets[idx[2]]))
            sum(m) / length(m)
        end
        # Route path-aligned values into the dense (j1,j2) S2 matrix. `pos` is plain-integer
        # bookkeeping (outside the differentiable path); the comprehension just reads s2vals.
        pos = zeros(Int, n, n)
        for (k, p) in enumerate(r2)
            idx = PathGraph.path_indices(tree, p)
            pos[idx[1], idx[2]] = k
        end
        Tc = eltype(s2vals)
        S2 = [pos[j1, j2] == 0 ? zero(Tc) : s2vals[pos[j1, j2]] for j1 in 1:n, j2 in 1:n]
    else
        S2 = Matrix{eltype(S1)}(undef, 0, 0)
    end
    return Coefficients.ScatteringCoefficients1D(S1, S2; S0=S0)
end

end # module Scattering1D
