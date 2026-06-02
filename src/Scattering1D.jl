module Scattering1D

"""
    Scattering1D.jl — 1D Scattering Transform

Implements first- and second-order 1D scattering transforms.
"""

using FFTW: FFTW
using LinearAlgebra: LinearAlgebra

# Import sibling modules
using ..FilterBanks: FilterBanks
using ..ScatteringCore: ScatteringCore
using ..Coefficients: Coefficients

export ScatteringTransform1D
export scattering_transform!, compute_S1!, compute_S2!

"""
    ScatteringTransform1D{T}

1D scattering transform with configurable parameters and workspace buffers.

# Fields
- `filter_bank::FilterBanks.FilterBank1D{T,V}`: Pre-computed filter bank
- `max_order::Int`: Maximum scattering order (1 or 2)
- `fft_plan`: Pre-planned FFT
- `ifft_plan`: Pre-planned inverse FFT
- `buffer_conv`: Complex buffer for convolution (size N)
- `buffer_mod`: Real buffer for modulus (size N)
- `U1_buffers`: Vector of real buffers for S2 computation (one per wavelet)
"""
struct ScatteringTransform1D{T,V<:AbstractVector{Complex{T}},M<:AbstractVector{T}}
    filter_bank::FilterBanks.FilterBank1D{T,V}
    max_order::Int
    fft_plan
    ifft_plan
    
    # Workspace buffers for zero-allocation transforms
    buffer_conv::V       # Complex buffer for convolution output
    buffer_mod::M        # Real buffer for modulus output
    U1_buffers::Vector{M}  # Real buffers for S2 first-order moduli
    
    function ScatteringTransform1D(N::Int, J::Int; 
                                   Q::Int=1, 
                                   max_order::Int=2,
                                   T::Type=Float64)
        filter_bank = FilterBanks.build_filter_bank1d(N, J; Q=Q, T=T)
        
        # Pre-plan FFTs
        dummy = zeros(Complex{T}, N)
        fft_plan = FFTW.plan_fft(dummy)
        ifft_plan = FFTW.plan_ifft(dummy)
        
        # Pre-allocate workspace buffers
        num_w = length(filter_bank.wavelets)
        buffer_conv = similar(dummy)
        buffer_mod = Vector{T}(undef, N)
        
        # Only allocate U1 buffers if S2 is needed
        U1_buffers = max_order >= 2 ? [Vector{T}(undef, N) for _ in 1:num_w] : Vector{T}[]
        
        new{T, typeof(buffer_conv), typeof(buffer_mod)}(
            filter_bank, max_order, fft_plan, ifft_plan, 
            buffer_conv, buffer_mod, U1_buffers
        )
    end
end

"""
    (st::ScatteringTransform1D)(signal) -> ScatteringCoefficients1D

Apply scattering transform to 1D signal.
Returns type-stable ScatteringCoefficients1D with element type matching input.
"""
function (st::ScatteringTransform1D)(signal::AbstractVector{T}) where T<:Real
    num_w = length(st.filter_bank.wavelets)
    
    # Pre-allocate coefficient storage
    coeffs = Coefficients.ScatteringCoefficients1D(num_w, T; compute_S2=st.max_order >= 2)
    
    # Apply in-place transform, get result with updated S0 (zero alloc for S1/S2)
    return scattering_transform!(coeffs, st, signal)
end

"""
    scattering_transform!(coeffs, st, signal)

In-place scattering transform. Fills pre-allocated S1/S2, returns new struct with updated S0.
Zero allocations for S1/S2 (buffers reused), only allocates new wrapper struct.
"""
function scattering_transform!(coeffs::Coefficients.ScatteringCoefficients1D{T,V,M},
                              st::ScatteringTransform1D{T},
                              signal::AbstractVector{T}) where {T,V,M}
    N = length(signal)
    
    # FFT of input (FFTW allocates, but we reuse buffer_conv for subsequent ops)
    signal_fft = st.fft_plan * complex.(signal)
    
    # S1: First order (fills coeffs.S1 in place)
    compute_S1!(coeffs.S1, st, signal_fft)
    
    # S2: Second order (if requested, fills coeffs.S2 in place)
    if st.max_order >= 2
        compute_S2!(coeffs.S2, st, signal_fft)
    end
    
    # S0: 0th order - use dispatch-based update (zero alloc for arrays, true zero alloc if S0 is mutable container)
    S0_val = ScatteringCore.spatial_average(signal)
    return Coefficients.update_S0(coeffs, S0_val)
end

"""
    compute_S1!(S1, st, signal_fft)

Compute first-order scattering coefficients in-place using workspace buffers.
Zero allocations if st buffers are pre-allocated.
"""
function compute_S1!(S1::AbstractVector{T}, st::ScatteringTransform1D{T,V,M}, 
                     signal_fft::AbstractVector{Complex{T}}) where {T,V,M}
    @inbounds for (j, ψ_fft) in enumerate(st.filter_bank.wavelets)
        # In-place convolution into buffer_conv
        ScatteringCore.wavelet_convolve!(st.buffer_conv, signal_fft, ψ_fft, 
                                          st.ifft_plan, st.buffer_conv)
        
        # In-place modulus into buffer_mod
        ScatteringCore.apply_modulus!(st.buffer_mod, st.buffer_conv)
        
        # Average and store
        S1[j] = real(ScatteringCore.spatial_average(st.buffer_mod))
    end
    return S1
end

"""
    compute_S2!(S2, st, signal_fft)

Compute second-order scattering coefficients in-place using workspace buffers.
Uses pre-allocated U1_buffers in st to avoid allocations.
"""
function compute_S2!(S2::AbstractMatrix{T}, st::ScatteringTransform1D{T,V,M}, 
                     signal_fft::AbstractVector{Complex{T}}) where {T,V,M}
    num_w = length(st.filter_bank.wavelets)
    
    # First-order moduli into pre-allocated U1_buffers
    @inbounds for (j1, ψ1_fft) in enumerate(st.filter_bank.wavelets)
        # In-place convolution
        ScatteringCore.wavelet_convolve!(st.buffer_conv, signal_fft, ψ1_fft, 
                                          st.ifft_plan, st.buffer_conv)
        # In-place modulus into U1_buffers[j1]
        ScatteringCore.apply_modulus!(st.U1_buffers[j1], st.buffer_conv)
    end
    
    # Second order: fills pre-allocated S2 array
    @inbounds for j1 in 1:num_w
        # FFT of modulus (allocates, then copy to buffer_conv for reuse)
        U1_fft = st.fft_plan * complex.(st.U1_buffers[j1])
        
        for j2 in (j1+1):num_w
            ψ2_fft = st.filter_bank.wavelets[j2]
            # In-place convolution
            ScatteringCore.wavelet_convolve!(st.buffer_conv, U1_fft, ψ2_fft, 
                                              st.ifft_plan, st.buffer_conv)
            # In-place modulus
            ScatteringCore.apply_modulus!(st.buffer_mod, st.buffer_conv)
            # Average and store
            S2[j1, j2] = real(ScatteringCore.spatial_average(st.buffer_mod))
        end
    end
    return S2
end

end # module Scattering1D
