module Scattering2D

"""
    Scattering2D.jl — 2D Planar Scattering Transform

Implements 2D scattering with oriented Morlet wavelets.
"""

using FFTW: FFTW
using LinearAlgebra: LinearAlgebra

using ..FilterBanks: FilterBanks
using ..ScatteringCore: ScatteringCore
using ..Coefficients: Coefficients

export ScatteringTransform2D
export compute_shape_sparsity

"""
    ScatteringTransform2D

2D planar scattering transform with oriented wavelets.

# Fields
- `filter_bank::FilterBanks.FilterBank2D`: Pre-computed 2D filter bank
- `max_order::Int`: Maximum scattering order (1 or 2)
- `fft_plan`: Pre-planned 2D FFT
- `ifft_plan`: Pre-planned 2D IFFT
"""
struct ScatteringTransform2D
    filter_bank::FilterBanks.FilterBank2D
    max_order::Int
    fft_plan
    ifft_plan
    
    function ScatteringTransform2D(N::NTuple{2,Int}, J::Int;
                                 L::Int=8,
                                 max_order::Int=2)
        filter_bank = FilterBanks.build_filter_bank2d(N, J; L=L)
        
        # Pre-plan 2D FFTs
        dummy = zeros(ComplexF64, N)
        fft_plan = FFTW.plan_fft(dummy)
        ifft_plan = FFTW.plan_ifft(dummy)
        
        new(filter_bank, max_order, fft_plan, ifft_plan)
    end
end

"""
    (st::ScatteringTransform2D)(image)

Apply 2D scattering transform to image.

# Returns
- `Dict{String, Array}`: Scattering coefficients
  - "S0": 0th order
  - "S1": 1st order [scale_index]
  - "S2": 2nd order [scale1, scale2] (if max_order >= 2)
"""
function (st::ScatteringTransform2D)(image::AbstractMatrix{T}) where T<:Real
    # Use input element type for coefficients
    num_w = length(st.filter_bank.wavelets)
    J = st.filter_bank.J
    L = st.filter_bank.L
    
    # Pre-allocate coefficient storage
    coeffs = Coefficients.ScatteringCoefficients2D(J, L, T; compute_S2=st.max_order >= 2)
    
    # FFT of input (element type matches image)
    image_complex = complex.(image)
    image_fft = st.fft_plan * image_complex
    
    # S1: first order (fills in place)
    compute_S1_2d!(coeffs.S1, st, image_fft, st.ifft_plan)
    
    # S2: second order (fills in place)
    if st.max_order >= 2
        compute_S2_2d!(coeffs.S2, st, image_fft)
    end
    
    # S0: averaging - use dispatch-based update (zero alloc for arrays, true zero alloc if S0 is mutable container)
    S0_val = ScatteringCore.spatial_average(image)
    return Coefficients.update_S0(coeffs, S0_val)
end

"""
    compute_S1_2d!(S1, st, image_fft, ifft_plan)

Compute first-order 2D scattering coefficients in-place.
"""
function compute_S1_2d!(S1::AbstractVector{T}, st::ScatteringTransform2D, 
                         image_fft::AbstractMatrix{Complex{T}}, ifft_plan) where T<:Real
    @inbounds for (j, ψ_fft) in enumerate(st.filter_bank.wavelets)
        convolved = ScatteringCore.wavelet_convolve(image_fft, ψ_fft, ifft_plan)
        modulus = ScatteringCore.apply_modulus(convolved)
        S1[j] = real(ScatteringCore.spatial_average(modulus))
    end
    return S1
end

"""
    compute_S2_2d!(S2, st, image_fft)

Compute second-order 2D scattering coefficients in-place.
"""
function compute_S2_2d!(S2::AbstractMatrix{T}, st::ScatteringTransform2D,
                       image_fft::AbstractMatrix{Complex{T}}) where T<:Real
    num_w = length(st.filter_bank.wavelets)
    
    # First-order moduli
    U1_temp = Vector{Matrix{T}}(undef, num_w)
    @inbounds for (j1, ψ1_fft) in enumerate(st.filter_bank.wavelets)
        convolved = ScatteringCore.wavelet_convolve(image_fft, ψ1_fft, st.ifft_plan)
        U1_temp[j1] = ScatteringCore.apply_modulus(convolved)
    end
    
    # Second order: fills pre-allocated S2
    @inbounds for j1 in 1:num_w
        U1_fft = st.fft_plan * complex.(U1_temp[j1])
        
        for j2 in (j1+1):num_w
            ψ2_fft = st.filter_bank.wavelets[j2]
            convolved = ScatteringCore.wavelet_convolve(U1_fft, ψ2_fft, st.ifft_plan)
            
            modulus = ScatteringCore.apply_modulus(convolved)
            S2[j1, j2] = real(ScatteringCore.spatial_average(modulus))
        end
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
                                meta::Vector{NamedTuple}) where T<:Real
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
