module Scattering2D

"""
    Scattering2D.jl — 2D Planar Scattering Transform

Implements 2D scattering with oriented Morlet wavelets.

"""

using ..Plans: Plans
using ComputationalBackends: ComputationalBackends as CB
using SpectralBackends: SpectralBackends as SB
using ..FilterBanks: FilterBanks
using ..ScatteringCore: ScatteringCore
using ..Coefficients: Coefficients
using ..PathGraph: PathGraph
using ..ScatteringFields: ScatteringFields

export ScatteringTransform2D, scattering_transform2d!, cascade!
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
- `buffer_u1`, `buffer_u1_fft`: the current first-order modulus and its spectrum, reused across
  that wavelet's children — one pair, not one per wavelet
"""
struct ScatteringTransform2D{T, M<:AbstractMatrix{Complex{T}}, R<:AbstractMatrix{T},
                             P<:Plans.AbstractScatteringPlan, Tree<:PathGraph.ScatteringTree,
                             FB<:FilterBanks.FilterBank2D, G<:AbstractVector}
    filter_bank::FB
    tree::Tree
    groups::G               # (j1, children, path ids) from the tree, longest-first
    max_order::Int
    plan::P
    buffer_input::M
    buffer_signal_fft::M
    buffer_conv::M
    buffer_mod::R
    buffer_u1::R            # first-order modulus of the current j1
    buffer_u1_fft::M        # its spectrum, reused across that j1's children
end

"""
    ScatteringTransform2D([T=Float64,] N, J; L=8, max_order=2, spectral=AutoSpectralBackend())

Build a 2D planar scattering transform for `N = (Ny, Nx)` images over `J` scales and `L`
orientations. The element type is positional, as for `zeros(T, …)`; omit it for `Float64`.
"""
function ScatteringTransform2D(::Type{T}, N::NTuple{2,Int}, J::Int;
                               L::Int=8,
                               max_order::Int=2,
                               spectral::SB.AbstractSpectralBackend=SB.AutoSpectralBackend()) where {T}
    filter_bank = FilterBanks.build_filter_bank2d(T, N, J; L=L)
    tree = PathGraph.build_tree([m.j_eff for m in filter_bank.meta], max_order)
    groups = max_order >= 2 ? PathGraph.order2_groups(tree, length(filter_bank.wavelets)) :
             [(j, Int[], Int[]) for j in 1:length(filter_bank.wavelets)]
    plan = Plans.make_plan(spectral, T, N)

    dummy = zeros(Complex{T}, N)
    # O(prod(N)) workspace, not O(nw·prod(N)): one first-order modulus and its spectrum are live at
    # a time, because the cascade finishes every child of a `j1` before moving to the next.
    return ScatteringTransform2D(filter_bank, tree, groups, max_order, plan,
                                 similar(dummy), similar(dummy), similar(dummy),
                                 zeros(T, N), zeros(T, N), similar(dummy))
end
ScatteringTransform2D(N::NTuple{2,Int}, J::Int; kwargs...) =
    ScatteringTransform2D(Float64, N, J; kwargs...)

# Shares filter bank / tree / groups; copies only the buffers and the plan's scratch.
ScatteringCore.task_workspace(st::ScatteringTransform2D) =
    ScatteringTransform2D(st.filter_bank, st.tree, st.groups, st.max_order,
                          Plans.task_local_plan(st.plan),
                          similar(st.buffer_input), similar(st.buffer_signal_fft),
                          similar(st.buffer_conv), similar(st.buffer_mod),
                          similar(st.buffer_u1), similar(st.buffer_u1_fft))

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
    st.buffer_input .= complex.(image)
    Plans.forward_transform!(st.buffer_signal_fft, st.plan, st.buffer_input)
    cascade!(coeffs.S1, coeffs.S2, st, st.buffer_signal_fft)
    return Coefficients.update_S0(coeffs, ScatteringCore.spatial_average(image))
end

"""
    scattering_transform2d!(coeffs, backend, st, image)

Transform one image on an explicit execution backend. `SerialBackend` runs the cascade in this
task; `ThreadedBackend` (OhMyThreads extension) spreads the first-order wavelet groups across
tasks, which is the only parallel axis available for a single image.
"""
function scattering_transform2d!(coeffs::Coefficients.ScatteringCoefficients2D,
                                 backend::CB.AbstractExecutionBackend,
                                 st::ScatteringTransform2D, image::AbstractMatrix)
    st.buffer_input .= complex.(image)
    Plans.forward_transform!(st.buffer_signal_fft, st.plan, st.buffer_input)
    cascade!(coeffs.S1, coeffs.S2, backend, st, st.buffer_signal_fft)
    return Coefficients.update_S0(coeffs, ScatteringCore.spatial_average(image))
end

cascade!(S1::AbstractVector, S2::AbstractMatrix, ::CB.AbstractSerialBackend,
         st::ScatteringTransform2D, image_fft::AbstractMatrix) = cascade!(S1, S2, st, image_fft)

"""
    cascade!(S1, S2, st, image_fft) -> (S1, S2)

Both scattering orders in one grouped pass — see the 1D `cascade!` for the scheme. Admissibility is
`j_eff` strictly increasing, i.e. *scale* strictly increasing across all orientation pairs, which is
what the tree encodes; same-scale different-orientation pairs are not order-2 paths.
"""
function cascade!(S1::AbstractVector, S2::AbstractMatrix, st::ScatteringTransform2D,
                  image_fft::AbstractMatrix)
    isempty(S2) || fill!(S2, zero(eltype(S2)))
    wavelets = st.filter_bank.wavelets
    @inbounds for (j1, children, _) in st.groups
        ScatteringCore.wavelet_convolve!(st.buffer_conv, image_fft, wavelets[j1],
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

    # Bucket the wavelets by scale once, rather than rescanning `meta` for every scale pair: the
    # scan is O(nw) and the pair loop is O(J²), so the original form was O(J²·nw) with two vector
    # allocations per pair.
    by_scale = [Int[] for _ in 0:(J - 1)]
    for (i, m) in enumerate(meta)
        push!(by_scale[m.scale + 1], i)
    end

    for j1 in 0:(J - 1), j2 in 0:(J - 1)
        j2 > j1 || continue
        idx1 = by_scale[j1 + 1]
        idx2 = by_scale[j2 + 1]
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

    # Orders 1 and 2 in one grouped pass, as in `cascade!`.
    p1_first = first(PathGraph.order_range(tree, 1))
    wavelets = st.filter_bank.wavelets
    @inbounds for (j1, children, pathids) in st.groups
        ScatteringCore.wavelet_convolve!(st.buffer_conv, st.buffer_signal_fft, wavelets[j1],
            st.plan, st.buffer_input)
        ScatteringCore.apply_modulus!(st.buffer_u1, st.buffer_conv)
        _lowpass_downsample!(view(data, :, :, p1_first + j1 - 1), st, st.buffer_u1, ds)
        isempty(children) && continue
        st.buffer_input .= complex.(st.buffer_u1)
        Plans.forward_transform!(st.buffer_u1_fft, st.plan, st.buffer_input)
        for (j2, p) in zip(children, pathids)
            ScatteringCore.wavelet_convolve!(st.buffer_conv, st.buffer_u1_fft, wavelets[j2],
                st.plan, st.buffer_input)
            ScatteringCore.apply_modulus!(st.buffer_mod, st.buffer_conv)
            _lowpass_downsample!(view(data, :, :, p), st, st.buffer_mod, ds)
        end
    end
    return field
end

# ============================================================================
# Non-mutating, autodiff-friendly forward — see Scattering1D for the rationale.
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
