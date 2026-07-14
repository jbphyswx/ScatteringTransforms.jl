module Scattering2D

"""
    Scattering2D.jl — 2D Planar Scattering Transform

Implements 2D scattering with oriented Morlet wavelets.

Design: All `!` functions are zero-allocation. The callable wrapper allocates
coefficient storage once and delegates to `scattering_transform2d!`.
Workspace buffers (buffer_input, buffer_conv, buffer_mod, U1_buffers,
U1_fft_buffers) are pre-allocated in the struct constructor.
"""

using ..Plans: Plans
using ..FilterBanks: FilterBanks
using ..ScatteringCore: ScatteringCore
using ..Coefficients: Coefficients
using ..PathGraph: PathGraph
using ..ScatteringFields: ScatteringFields

export ScatteringTransform2D, scattering_transform2d!
export compute_S1_2d!, compute_S2_2d!
export compute_shape_sparsity

"""
    ScatteringTransform2D{T,M,R}

2D planar scattering transform with oriented wavelets and pre-allocated workspace.

# Type Parameters
- `T`: Real element type (Float32, Float64, ...)
- `M`: Complex matrix type for buffers (Matrix{Complex{T}}, CuMatrix{Complex{T}}, ...)
- `R`: Real matrix type for modulus buffers (Matrix{T}, ...)

# Fields
- `filter_bank`: Pre-computed 2D filter bank
- `max_order::Int`: Maximum scattering order (1 or 2)
- `plan`: spectral transform plan (in-core direct sum by default; FFTW fast path if loaded)
- `buffer_input`: Complex matrix for real→complex promotion (zero alloc)
- `buffer_signal_fft`: Preserved copy of signal FFT (buffer_conv gets overwritten)
- `buffer_conv`: Complex matrix for IFFT output
- `buffer_mod`: Real matrix for modulus output
- `U1_buffers`: Real matrices for first-order moduli (one per wavelet)
- `U1_fft_buffers`: Complex matrices for FFT of U1 (one per wavelet)
"""
struct ScatteringTransform2D{T, M<:AbstractMatrix{Complex{T}}, R<:AbstractMatrix{T},
                             P<:Plans.AbstractScatteringPlan, Tree<:PathGraph.ScatteringTree,
                             FB<:FilterBanks.FilterBank2D, UB<:AbstractVector, UF<:AbstractVector}
    filter_bank::FB
    tree::Tree
    max_order::Int
    plan::P
    buffer_input::M
    buffer_signal_fft::M
    buffer_conv::M
    buffer_mod::R
    U1_buffers::UB
    U1_fft_buffers::UF
end

function ScatteringTransform2D(N::NTuple{2,Int}, J::Int;
                               L::Int=8,
                               max_order::Int=2,
                               T::Type=Float64,
                               spectral::Plans.AbstractSpectralBackend=Plans.AutoSpectral())
    filter_bank = FilterBanks.build_filter_bank2d(N, J; L=L, T=T)
    tree = PathGraph.build_tree([m.j_eff for m in filter_bank.meta], max_order)
    plan = Plans.make_plan(spectral, T, N)

    dummy = zeros(Complex{T}, N)
    num_w = length(filter_bank.wavelets)
    buffer_input      = similar(dummy)
    buffer_signal_fft = similar(dummy)
    buffer_conv       = similar(dummy)
    buffer_mod        = zeros(T, N)
    if max_order >= 2
        U1_buffers     = [zeros(T, N) for _ in 1:num_w]
        U1_fft_buffers = [similar(dummy) for _ in 1:num_w]
    else
        U1_buffers     = Matrix{T}[]
        U1_fft_buffers = Matrix{Complex{T}}[]
    end
    return ScatteringTransform2D(filter_bank, tree, max_order, plan,
                                 buffer_input, buffer_signal_fft, buffer_conv, buffer_mod,
                                 U1_buffers, U1_fft_buffers)
end

"""
    (st::ScatteringTransform2D)(image) -> ScatteringCoefficients2D

Apply 2D scattering transform to image. Allocates coefficient storage once,
then delegates to `scattering_transform2d!`.
"""
function (st::ScatteringTransform2D)(image::AbstractMatrix)
    J = st.filter_bank.J
    L = st.filter_bank.L
    T = eltype(st.buffer_mod)
    coeffs = Coefficients.ScatteringCoefficients2D(J, L, T; compute_S2=st.max_order >= 2)
    return scattering_transform2d!(coeffs, st, image)
end

"""
    scattering_transform2d!(coeffs, st, image)

In-place 2D scattering transform. Zero allocations for S1/S2 (buffers reused).
"""
function scattering_transform2d!(coeffs::Coefficients.ScatteringCoefficients2D,
                                  st::ScatteringTransform2D,
                                  image::AbstractMatrix)
    # Zero-alloc real→complex: write into buffer_input
    st.buffer_input .= complex.(image)
    
    # Zero-alloc FFT via mul! — store into buffer_signal_fft (preserved across passes)
    Plans.forward_transform!(st.buffer_signal_fft, st.plan, st.buffer_input)
    
    # S1: first order — buffer_signal_fft intact; compute_S1_2d! writes into buffer_conv
    compute_S1_2d!(coeffs.S1, st, st.buffer_signal_fft)
    
    # S2: second order — buffer_signal_fft still intact
    if st.max_order >= 2
        compute_S2_2d!(coeffs.S2, st, st.buffer_signal_fft)
    end
    
    S0_val = ScatteringCore.spatial_average(image)
    return Coefficients.update_S0(coeffs, S0_val)
end

"""
    compute_S1_2d!(S1, st, image_fft)

First-order 2D scattering coefficients. Zero allocations: uses `buffer_input`
as multiply scratch and `buffer_conv` as IFFT output (non-aliased).
"""
function compute_S1_2d!(S1::AbstractVector, st::ScatteringTransform2D,
                         image_fft::AbstractMatrix)
    @inbounds for (j, ψ_fft) in enumerate(st.filter_bank.wavelets)
        ScatteringCore.wavelet_convolve!(st.buffer_conv, image_fft, ψ_fft,
                                          st.plan, st.buffer_input)
        ScatteringCore.apply_modulus!(st.buffer_mod, st.buffer_conv)
        S1[j] = ScatteringCore.spatial_average(st.buffer_mod)
    end
    return S1
end

"""
    compute_S2_2d!(S2, st, image_fft)

Second-order 2D scattering coefficients. Zero allocations: U1_buffers and
U1_fft_buffers are pre-allocated; mul! used for all FFTs.
"""
function compute_S2_2d!(S2::AbstractMatrix, st::ScatteringTransform2D,
                         image_fft::AbstractMatrix)
    num_w = length(st.filter_bank.wavelets)
    
    # Pass 1: first-order moduli into U1_buffers
    @inbounds for (j1, ψ1_fft) in enumerate(st.filter_bank.wavelets)
        ScatteringCore.wavelet_convolve!(st.buffer_conv, image_fft, ψ1_fft,
                                          st.plan, st.buffer_input)
        ScatteringCore.apply_modulus!(st.U1_buffers[j1], st.buffer_conv)
    end
    
    # Pass 2: FFT each U1 into U1_fft_buffers (zero alloc via mul!)
    @inbounds for j1 in 1:num_w
        st.buffer_input .= complex.(st.U1_buffers[j1])
        Plans.forward_transform!(st.U1_fft_buffers[j1], st.plan, st.buffer_input)
    end
    
    # Pass 3: second-order scattering over the ADMISSIBLE paths only. The constraint is
    # j_eff strictly increasing, i.e. *scale strictly increasing over all orientation pairs*.
    # The previous flat-index loop (`j2 in (j1+1):num_w`) wrongly included same-scale,
    # different-orientation pairs (where scale₂ == scale₁); the tree excludes them.
    tree = st.tree
    @inbounds for p in PathGraph.order_range(tree, 2)
        idx = PathGraph.path_indices(tree, p)
        j1, j2 = idx[1], idx[2]
        ψ2_fft = st.filter_bank.wavelets[j2]
        ScatteringCore.wavelet_convolve!(st.buffer_conv, st.U1_fft_buffers[j1], ψ2_fft,
                                          st.plan, st.buffer_input)
        ScatteringCore.apply_modulus!(st.buffer_mod, st.buffer_conv)
        S2[j1, j2] = ScatteringCore.spatial_average(st.buffer_mod)
    end
    return S2
end

"""
    compute_shape_sparsity(S1, S2, meta) -> (; sparsity, shape)

Reduced second-order descriptors (in the spirit of the reduced wavelet scattering transform,
Allys et al. 2019; Cheng & Ménard 2021), as `J × J` matrices over scale pairs `(j1, j2)` with
`j2 > j1`:

- `sparsity` (`s₂₁`): the orientation-averaged ratio `⟨S₂ / S₁⟩` — how much energy cascades
  from scale `j1` to the coarser scale `j2` (large for sparse/intermittent fields).
- `shape` (`s₂₂`): the **anisotropy** of the cascade — the normalized second angular harmonic
  `⟨S₂ · cos(2 Δθ)⟩ / ⟨S₂⟩` over orientation pairs, where `Δθ = θ₂ − θ₁`. It is `≈ 0` for
  statistically isotropic fields and departs from zero when the field has oriented structure.
"""
function compute_shape_sparsity(S1::AbstractVector{T},
                                S2::AbstractMatrix{T},
                                meta::AbstractVector{<:FilterBanks.WaveletMeta}) where T<:Real
    J = maximum(m.scale for m in meta) + 1

    sparsity = zeros(T, J, J)
    shape = zeros(T, J, J)

    for j1 in 0:(J - 1), j2 in 0:(J - 1)
        j2 > j1 || continue
        idx1 = [i for (i, m) in enumerate(meta) if m.scale == j1]
        idx2 = [i for (i, m) in enumerate(meta) if m.scale == j2]
        (isempty(idx1) || isempty(idx2)) && continue

        s21_sum = zero(T)
        s21_count = 0
        harm_num = zero(T)     # Σ S₂ cos(2Δθ)
        harm_den = zero(T)     # Σ S₂
        for i1 in idx1, i2 in idx2
            s2 = S2[i1, i2]
            if S1[i1] > 0
                s21_sum += s2 / S1[i1]
                s21_count += 1
            end
            dθ = meta[i2].theta - meta[i1].theta
            harm_num += s2 * cos(2 * dθ)
            harm_den += s2
        end
        s21_count > 0 && (sparsity[j1 + 1, j2 + 1] = s21_sum / s21_count)
        harm_den > 0 && (shape[j1 + 1, j2 + 1] = harm_num / harm_den)
    end

    return (sparsity = sparsity, shape = shape)
end

# ============================================================================
# Localized (Mallat) 2D scattering field: S_p x = (|U_p x| ⋆ φ_J) ↓ s (per dim)
# ============================================================================

function _default_subsample(Ny::Int, Nx::Int, J::Int)
    ds = 1 << max(0, J - 1)
    while ds > 1 && (Ny % ds != 0 || Nx % ds != 0)
        ds >>= 1
    end
    return ds
end

# Low-pass a real field `U` (Ny×Nx) by φ_J and decimate by `ds` per dim into `dst`.
# Reuses buffer_input/buffer_conv; leaves buffer_signal_fft (the preserved image FFT) intact.
@inline function _lowpass_downsample!(dst, st::ScatteringTransform2D, U::AbstractMatrix, ds::Int)
    φ = st.filter_bank.averaging
    st.buffer_input .= complex.(U)
    Plans.forward_transform!(st.buffer_conv, st.plan, st.buffer_input)     # U_fft -> buffer_conv
    st.buffer_conv .*= φ
    Plans.inverse_transform!(st.buffer_input, st.plan, st.buffer_conv)     # (U ⋆ φ) -> buffer_input
    # Decimate by ds per dim via a strided view + broadcast (CPU + GPU compatible).
    My, Mx = size(dst)
    @views dst .= real.(st.buffer_input[1:ds:(1 + (My - 1) * ds), 1:ds:(1 + (Mx - 1) * ds)])
    return dst
end

function ScatteringFields.scattering_field(st::ScatteringTransform2D, image::AbstractMatrix;
        subsample::Int = _default_subsample(size(st.buffer_mod, 1), size(st.buffer_mod, 2),
                                            st.filter_bank.J))
    Ny, Nx = size(st.buffer_mod)
    (Ny % subsample == 0 && Nx % subsample == 0) ||
        throw(ArgumentError("subsample factor $subsample must divide both image dims ($Ny, $Nx)"))
    T = eltype(st.buffer_mod)
    My, Mx = Ny ÷ subsample, Nx ÷ subsample
    npath = PathGraph.npaths(st.tree)
    data = zeros(T, My, Mx, npath)
    field = ScatteringFields.ScatteringField2D(st.tree, data, subsample)
    return ScatteringFields.scattering_field!(field, st, image)
end

function ScatteringFields.scattering_field!(field::ScatteringFields.ScatteringField2D,
        st::ScatteringTransform2D, image::AbstractMatrix)
    tree = st.tree
    ds = field.subsample
    data = field.data

    st.buffer_input .= complex.(image)
    Plans.forward_transform!(st.buffer_signal_fft, st.plan, st.buffer_input)

    # order 0 (root): (x ⋆ φ_J) ↓
    copyto!(st.buffer_mod, image)
    root = first(PathGraph.order_range(tree, 0))
    _lowpass_downsample!(view(data, :, :, root), st, st.buffer_mod, ds)

    # order 1
    @inbounds for p in PathGraph.order_range(tree, 1)
        j = PathGraph.path_indices(tree, p)[1]
        ScatteringCore.wavelet_convolve!(st.buffer_conv, st.buffer_signal_fft,
            st.filter_bank.wavelets[j], st.plan, st.buffer_input)
        ScatteringCore.apply_modulus!(st.buffer_mod, st.buffer_conv)
        _lowpass_downsample!(view(data, :, :, p), st, st.buffer_mod, ds)
    end

    # order 2
    if st.max_order >= 2 && length(tree.by_order) >= 3
        num_w = length(st.filter_bank.wavelets)
        @inbounds for (j1, ψ1_fft) in enumerate(st.filter_bank.wavelets)
            ScatteringCore.wavelet_convolve!(st.buffer_conv, st.buffer_signal_fft, ψ1_fft,
                st.plan, st.buffer_input)
            ScatteringCore.apply_modulus!(st.U1_buffers[j1], st.buffer_conv)
        end
        @inbounds for j1 in 1:num_w
            # Broadcast real→complex (no scalar indexing): CPU + GPU compatible.
            st.buffer_input .= complex.(st.U1_buffers[j1])
            Plans.forward_transform!(st.U1_fft_buffers[j1], st.plan, st.buffer_input)
        end
        @inbounds for p in PathGraph.order_range(tree, 2)
            idx = PathGraph.path_indices(tree, p)
            j1, j2 = idx[1], idx[2]
            ScatteringCore.wavelet_convolve!(st.buffer_conv, st.U1_fft_buffers[j1],
                st.filter_bank.wavelets[j2], st.plan, st.buffer_input)
            ScatteringCore.apply_modulus!(st.buffer_mod, st.buffer_conv)
            _lowpass_downsample!(view(data, :, :, p), st, st.buffer_mod, ds)
        end
    end
    return field
end

# ============================================================================
# Non-mutating, autodiff-friendly forward (Part A) — see Scattering1D for the rationale.
# ============================================================================

function ScatteringCore.scattering(st::ScatteringTransform2D, image::AbstractMatrix)
    plan = st.plan
    fb = st.filter_bank
    tree = st.tree
    n = length(fb.wavelets)

    Xf = Plans.forward_transform(plan, complex.(image))
    U1 = map(ψ -> abs.(Plans.inverse_transform(plan, Xf .* ψ)), fb.wavelets)
    S1 = map(u -> sum(u) / length(u), U1)
    S0 = sum(image) / length(image)

    if st.max_order >= 2 && length(tree.by_order) >= 3
        U1f = map(u -> Plans.forward_transform(plan, complex.(u)), U1)
        r2 = PathGraph.order_range(tree, 2)
        s2vals = map(collect(r2)) do p
            idx = PathGraph.path_indices(tree, p)
            m = abs.(Plans.inverse_transform(plan, U1f[idx[1]] .* fb.wavelets[idx[2]]))
            sum(m) / length(m)
        end
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
    return Coefficients.ScatteringCoefficients2D(S1, S2; S0=S0,
        n_scales=fb.J, n_orientations=fb.L)
end

end # module Scattering2D
