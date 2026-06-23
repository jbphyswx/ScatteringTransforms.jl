module Scattering2D

"""
    Scattering2D.jl — 2D Planar Scattering Transform

Implements 2D scattering with oriented Morlet wavelets.

Design: All `!` functions are zero-allocation. The callable wrapper allocates
coefficient storage once and delegates to `scattering_transform2d!`.
Workspace buffers (buffer_input, buffer_conv, buffer_mod, U1_buffers,
U1_fft_buffers) are pre-allocated in the struct constructor.
"""

using FFTW: FFTW
using LinearAlgebra: LinearAlgebra

using ..FilterBanks: FilterBanks
using ..ScatteringCore: ScatteringCore
using ..Coefficients: Coefficients
using ..PathGraph: PathGraph

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
- `fft_plan`, `ifft_plan`: Pre-planned 2D FFT/IFFT (used via `mul!`)
- `buffer_input`: Complex matrix for real→complex promotion (zero alloc)
- `buffer_signal_fft`: Preserved copy of signal FFT (buffer_conv gets overwritten)
- `buffer_conv`: Complex matrix for IFFT output
- `buffer_mod`: Real matrix for modulus output
- `U1_buffers`: Real matrices for first-order moduli (one per wavelet)
- `U1_fft_buffers`: Complex matrices for FFT of U1 (one per wavelet)
"""
struct ScatteringTransform2D{T,M<:AbstractMatrix{Complex{T}},R<:AbstractMatrix{T},FP,IP,Tree<:PathGraph.ScatteringTree}
    filter_bank::FilterBanks.FilterBank2D{T,M}   # carry the matrix-type param M (was abstract {T})
    tree::Tree      # admissible scattering paths (source of truth for second-order)
    max_order::Int
    fft_plan::FP    # concrete plan type param — no dynamic dispatch on mul!
    ifft_plan::IP

    buffer_input::M
    buffer_signal_fft::M
    buffer_conv::M
    buffer_mod::R
    U1_buffers::Vector{R}
    U1_fft_buffers::Vector{M}
    
    function ScatteringTransform2D(N::NTuple{2,Int}, J::Int;
                                   L::Int=8,
                                   max_order::Int=2,
                                   T::Type=Float64)
        filter_bank = FilterBanks.build_filter_bank2d(N, J; L=L, T=T)
        tree = PathGraph.build_tree([m.j_eff for m in filter_bank.meta], max_order)

        # Pre-plan 2D FFTs; mul!(out, plan, src) is zero-allocation
        dummy = zeros(Complex{T}, N)
        fft_plan  = FFTW.plan_fft(dummy)
        ifft_plan = FFTW.plan_ifft(dummy)
        
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
        
        new{T, typeof(buffer_conv), typeof(buffer_mod), typeof(fft_plan), typeof(ifft_plan), typeof(tree)}(
            filter_bank, tree, max_order, fft_plan, ifft_plan,
            buffer_input, buffer_signal_fft, buffer_conv, buffer_mod,
            U1_buffers, U1_fft_buffers
        )
    end
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
    @inbounds @simd for i in eachindex(image)
        st.buffer_input[i] = complex(image[i])
    end
    
    # Zero-alloc FFT via mul! — store into buffer_signal_fft (preserved across passes)
    LinearAlgebra.mul!(st.buffer_signal_fft, st.fft_plan, st.buffer_input)
    
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
                                          st.ifft_plan, st.buffer_input)
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
                                          st.ifft_plan, st.buffer_input)
        ScatteringCore.apply_modulus!(st.U1_buffers[j1], st.buffer_conv)
    end
    
    # Pass 2: FFT each U1 into U1_fft_buffers (zero alloc via mul!)
    @inbounds for j1 in 1:num_w
        @simd for i in eachindex(st.U1_buffers[j1])
            st.buffer_input[i] = complex(st.U1_buffers[j1][i])
        end
        LinearAlgebra.mul!(st.U1_fft_buffers[j1], st.fft_plan, st.buffer_input)
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
                                          st.ifft_plan, st.buffer_input)
        ScatteringCore.apply_modulus!(st.buffer_mod, st.buffer_conv)
        S2[j1, j2] = ScatteringCore.spatial_average(st.buffer_mod)
    end
    return S2
end

"""
    compute_shape_sparsity(S1, S2, meta)

Compute reduced shape and sparsity statistics from scattering coefficients.

Following Skinner et al. (2025), these are:
- s₂₁ (sparsity): ⟨S₂/S₁⟩ over orientations
- s₂₂ (shape): ⟨S₂^∥ / S₂^⊥⟩ over orientations
"""
function compute_shape_sparsity(S1::AbstractVector{T},
                                S2::AbstractMatrix{T},
                                meta::AbstractVector{<:FilterBanks.WaveletMeta}) where T<:Real
    # Group by scales
    J = maximum(m.scale for m in meta) + 1
    L = length(meta) ÷ J  # orientations per scale
    
    sparsity = zeros(T, J, J)
    shape = zeros(T, J, J)
    
    for j1 in 0:J-1, j2 in 0:J-1
        # Indices for this scale pair
        idx1 = [i for (i, m) in enumerate(meta) if m.scale == j1]
        idx2 = [i for (i, m) in enumerate(meta) if m.scale == j2]
        
        # Sparsity: average of S2/S1 over orientations
        if j2 > j1 && !isempty(idx1) && !isempty(idx2)
            s21_sum = zero(T)
            s21_count = 0
            for i1 in idx1, i2 in idx2
                if S1[i1] > 0
                    s21_sum += S2[i1, i2] / S1[i1]
                    s21_count += 1
                end
            end
            if s21_count > 0
                sparsity[j1+1, j2+1] = s21_sum / s21_count
            end
        end
    end
    
    return (sparsity=sparsity, shape=shape)
end

end # module Scattering2D
