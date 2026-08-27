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
using ..Cascade: Cascade

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
- `dims`: the field's spatial size
- `buffer_u1_fft`: the spectrum of the current first-order modulus, reused across that wavelet's
  children. The modulus itself needs no buffer of its own — the cascade writes it straight into
  `buffer_input`, which is what its transform reads.
"""
struct ScatteringTransform2D{T, M<:AbstractMatrix{Complex{T}},
                             P<:Plans.AbstractScatteringPlan, Tree<:PathGraph.ScatteringTree,
                             FB, G<:AbstractVector, PW}
    filter_bank::FB
    tree::Tree
    groups::G               # (j1, children, path ids) from the tree, longest-first
    max_order::Int
    plan::P
    buffer_input::M
    buffer_signal_fft::M
    buffer_conv::M
    dims::NTuple{2,Int}
    buffer_u1_fft::M        # its spectrum, reused across that j1's children
    pw::PW                  # periodized cascade workspace — see `Cascade`
end

"""
    ScatteringTransform2D([T=Float64,] N, J; L=8, max_order=2, spectral=AutoSpectralBackend())

Build a 2D planar scattering transform for `N = (Ny, Nx)` images over `J` scales and `L`
orientations. The element type is positional, as for `zeros(T, …)`; omit it for `Float64`.
"""
function ScatteringTransform2D(::Type{T}, N::NTuple{2,Int}, J::Int;
                               L::Int=8,
                               max_order::Int=2,
                               cache::Bool=true,
                               oversampling::Int=J,
                               spectral::SB.AbstractSpectralBackend=SB.AutoSpectralBackend()) where {T}
    filter_bank = FilterBanks.build_filter_bank2d(T, N, J, Val(cache); L=L)
    tree = PathGraph.build_tree([m.j_eff for m in filter_bank.meta], max_order)
    nw = FilterBanks.nwavelets(filter_bank)
    groups = max_order >= 2 ? PathGraph.order2_groups(tree, nw) :
             [(j, Int[], Int[]) for j in 1:nw]
    plan = Plans.make_plan(spectral, T, N)

    dummy = zeros(Complex{T}, N)
    # O(prod(N)) workspace, not O(nw·prod(N)): one first-order modulus and its spectrum are live at
    # a time, because the cascade finishes every child of a `j1` before moving to the next.
    # `buffer_conv` is `buffer_input` itself when the plan inverts in place — the convolution's input
    # is scratch and its output replaces it, so the two need not be separate arrays.
    input = similar(dummy)
    conv = Plans.inplace_inverse(plan) ? input : similar(dummy)
    pw = Cascade.build(filter_bank, groups, N, T, oversampling, Plans.spectral_backend(plan))
    return ScatteringTransform2D(filter_bank, tree, groups, max_order, plan,
                                 input, similar(dummy), conv, N, similar(dummy), pw)
end
ScatteringTransform2D(N::NTuple{2,Int}, J::Int; kwargs...) =
    ScatteringTransform2D(Float64, N, J; kwargs...)

# Shares filter bank / tree / groups; copies only the buffers and the plan's scratch. Where the
# original aliases `buffer_conv` to `buffer_input`, the copy has to as well.
function ScatteringCore.task_workspace(st::ScatteringTransform2D)
    input = similar(st.buffer_input)
    conv = st.buffer_conv === st.buffer_input ? input : similar(st.buffer_conv)
    fb = FilterBanks.task_bank(st.filter_bank)
    return ScatteringTransform2D(fb, st.tree, st.groups,
                                 st.max_order, Plans.task_local_plan(st.plan),
                                 input, similar(st.buffer_signal_fft), conv,
                                 st.dims, similar(st.buffer_u1_fft), Cascade.task_copy(st.pw, fb))
end

"""
    (st::ScatteringTransform2D)(image) -> ScatteringCoefficients2D

Apply 2D scattering transform to image. Allocates coefficient storage once,
then delegates to `scattering_transform2d!`.
"""
function (st::ScatteringTransform2D)(image::AbstractMatrix)
    J = st.filter_bank.J
    L = st.filter_bank.L
    T = real(eltype(st.buffer_input))
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
cascade!(S1::AbstractVector, S2::AbstractMatrix, st::ScatteringTransform2D,
         image_fft::AbstractMatrix) =
    Cascade.cascade!(S1, S2, st.pw, st.filter_bank, st.groups, image_fft)

# The undecimated form, kept as the oracle the periodized cascade is pinned against.
function _cascade_full!(S1::AbstractVector, S2::AbstractMatrix, st::ScatteringTransform2D,
                        image_fft::AbstractMatrix)
    isempty(S2) || fill!(S2, zero(eltype(S2)))
    fb = st.filter_bank
    @inbounds for (j1, children, _) in st.groups
        ScatteringCore.wavelet_convolve!(st.buffer_conv, image_fft, FilterBanks.filter_at(fb, j1),
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

function ScatteringFields.scattering_field(st::ScatteringTransform2D, image::AbstractMatrix;
        subsample::Int = _default_subsample(st.dims[1], st.dims[2],
                                            st.filter_bank.J))
    Ny, Nx = st.dims
    (Ny % subsample == 0 && Nx % subsample == 0) ||
        throw(ArgumentError("subsample factor $subsample must divide both image dims ($Ny, $Nx)"))
    # Real unless the input is complex, in which case the order-0 field is too.
    T = promote_type(real(eltype(st.buffer_input)), eltype(image))
    My, Mx = Ny ÷ subsample, Nx ÷ subsample
    npath = PathGraph.npaths(st.tree)
    data = zeros(T, My, Mx, npath)
    field = ScatteringFields.ScatteringField2D(st.tree, data, subsample,
                                               Cascade.field_workspace(st, subsample))
    return ScatteringFields.scattering_field!(field, st, image)
end

function ScatteringFields.scattering_field!(field::ScatteringFields.ScatteringField2D,
        st::ScatteringTransform2D, image::AbstractMatrix)
    st.buffer_input .= complex.(image)
    Plans.forward_transform!(st.buffer_signal_fft, st.plan, st.buffer_input)
    Cascade.field_cascade!(field.data, field.ws, st.filter_bank, st.groups, st.buffer_signal_fft,
                           first(PathGraph.order_range(st.tree, 0)),
                           first(PathGraph.order_range(st.tree, 1)))
    return field
end

# ============================================================================
# Non-mutating, autodiff-friendly forward — see Scattering1D for the rationale.
# ============================================================================

function ScatteringCore.scattering(st::ScatteringTransform2D, image::AbstractMatrix)
    S0, S1, S2 = Cascade.scattering_values(st, image)
    fb = st.filter_bank
    return Coefficients.ScatteringCoefficients2D(S1, S2; S0=S0,
        n_scales=fb.J, n_orientations=fb.L)
end

end # module Scattering2D
