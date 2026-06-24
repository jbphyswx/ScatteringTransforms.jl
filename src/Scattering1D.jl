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
using ..PathGraph: PathGraph
using ..ScatteringFields: ScatteringFields

export ScatteringTransform1D
export scattering_transform!, compute_S1!, compute_S2!

"""
    ScatteringTransform1D{T}

1D scattering transform with configurable parameters and workspace buffers.

# Fields
- `filter_bank::FilterBanks.FilterBank1D{T,V}`: Pre-computed filter bank
- `max_order::Int`: Maximum scattering order (1 or 2)
- `fft_plan`: Pre-planned FFT (via `mul!`)
- `ifft_plan`: Pre-planned IFFT (via `mul!`)
- `buffer_input`: Complex buffer for real→complex cast of input (zero alloc)
- `buffer_signal_fft`: Complex buffer holding the FFT of the input signal (preserved across S1/S2 passes)
- `buffer_conv`: Complex buffer for convolution output (IFFT result)
- `buffer_mod`: Real buffer for modulus output
- `U1_buffers`: Vector of real buffers for S2 computation (one per wavelet)
- `U1_fft_buffers`: Vector of complex buffers for FFT of U1 (one per wavelet)
"""
struct ScatteringTransform1D{T,V<:AbstractVector{Complex{T}},M<:AbstractVector{T},FP,IP,Tree<:PathGraph.ScatteringTree}
    filter_bank::FilterBanks.FilterBank1D{T,V}
    tree::Tree      # admissible scattering paths (source of truth for second-order)
    max_order::Int
    fft_plan::FP    # concrete plan type param — no dynamic dispatch on mul!
    ifft_plan::IP

    # Workspace buffers for zero-allocation transforms
    buffer_input::V       # Complex buffer for real→complex promotion of input
    buffer_signal_fft::V  # Preserves signal FFT across S1/S2 passes (buffer_conv gets overwritten)
    buffer_conv::V        # Complex buffer for convolution output (IFFT result)
    buffer_mod::M         # Real buffer for modulus output
    U1_buffers::Vector{M}    # Real buffers for S2 first-order moduli
    U1_fft_buffers::Vector{V}  # Complex buffers for FFT of each U1
    
    function ScatteringTransform1D(N::Int, J::Int; 
                                   Q::Int=1, 
                                   max_order::Int=2,
                                   T::Type=Float64)
        filter_bank = FilterBanks.build_filter_bank1d(N, J; Q=Q, T=T)
        tree = PathGraph.build_tree([m.j_eff for m in filter_bank.meta], max_order)

        # Pre-plan FFTs using plan_fft / plan_ifft
        # mul!(out, plan, src) writes result directly into out — zero allocation
        dummy = zeros(Complex{T}, N)
        fft_plan  = FFTW.plan_fft(dummy)
        ifft_plan = FFTW.plan_ifft(dummy)
        
        # Pre-allocate workspace buffers
        num_w = length(filter_bank.wavelets)
        buffer_input      = similar(dummy)   # for real→complex of input
        buffer_signal_fft = similar(dummy)   # preserved copy of signal FFT
        buffer_conv       = similar(dummy)   # convolution / ifft output
        buffer_mod        = Vector{T}(undef, N)
        
        # Only allocate U1 / U1_fft buffers if S2 is needed
        if max_order >= 2
            U1_buffers     = [Vector{T}(undef, N) for _ in 1:num_w]
            U1_fft_buffers = [similar(dummy) for _ in 1:num_w]
        else
            U1_buffers     = Vector{T}[]
            U1_fft_buffers = Vector{Complex{T}}[]
        end
        
        new{T, typeof(buffer_conv), typeof(buffer_mod), typeof(fft_plan), typeof(ifft_plan), typeof(tree)}(
            filter_bank, tree, max_order, fft_plan, ifft_plan,
            buffer_input, buffer_signal_fft, buffer_conv, buffer_mod,
            U1_buffers, U1_fft_buffers
        )
    end
end

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

In-place scattering transform. Fills pre-allocated S1/S2, returns new struct with updated S0.
Zero allocations: uses `buffer_input` for real→complex promotion, `mul!` for FFT.
Only allocates a new wrapper struct when S0 is a scalar (immutable).
"""
function scattering_transform!(coeffs::Coefficients.ScatteringCoefficients1D,
                              st::ScatteringTransform1D,
                              signal::AbstractVector)
    # Zero-alloc real→complex: write into pre-allocated buffer_input
    @inbounds @simd for i in eachindex(signal)
        st.buffer_input[i] = complex(signal[i])
    end
    
    # Zero-alloc FFT: mul!(out, plan, src) writes FFT(buffer_input) into buffer_signal_fft
    LinearAlgebra.mul!(st.buffer_signal_fft, st.fft_plan, st.buffer_input)
    
    # S1: First order — passes buffer_signal_fft as signal_fft.
    # NOTE: compute_S1! will overwrite buffer_conv but NOT buffer_signal_fft.
    compute_S1!(coeffs.S1, st, st.buffer_signal_fft)
    
    # S2: Second order — also uses buffer_signal_fft (still intact)
    if st.max_order >= 2
        compute_S2!(coeffs.S2, st, st.buffer_signal_fft)
    end
    
    # S0: 0th order - use dispatch-based update
    S0_val = ScatteringCore.spatial_average(signal)
    return Coefficients.update_S0(coeffs, S0_val)
end

"""
    compute_S1!(S1, st, signal_fft)

Compute first-order scattering coefficients in-place using workspace buffers.
Zero allocations: `buffer_input` is used as the pointwise-multiply scratch,
`buffer_conv` receives the IFFT output (no aliasing).
"""
function compute_S1!(S1::AbstractVector, st::ScatteringTransform1D, 
                     signal_fft::AbstractVector)
    @inbounds for (j, ψ_fft) in enumerate(st.filter_bank.wavelets)
        # buffer_input = scratch for multiply; buffer_conv = IFFT output (non-aliased)
        ScatteringCore.wavelet_convolve!(st.buffer_conv, signal_fft, ψ_fft, 
                                          st.ifft_plan, st.buffer_input)
        
        # In-place modulus into buffer_mod
        ScatteringCore.apply_modulus!(st.buffer_mod, st.buffer_conv)
        
        # Average and store
        S1[j] = ScatteringCore.spatial_average(st.buffer_mod)
    end
    return S1
end

"""
    compute_S2!(S2, st, signal_fft)

Compute second-order scattering coefficients in-place using workspace buffers.
Zero allocations: uses `U1_fft_buffers` (pre-allocated per-wavelet FFT scratch)
to avoid `complex.()` and `plan * x` allocations in the hot loop.
"""
function compute_S2!(S2::AbstractMatrix, st::ScatteringTransform1D, 
                     signal_fft::AbstractVector)
    num_w = length(st.filter_bank.wavelets)
    
    # Pass 1: compute first-order moduli |x ★ ψ_j1| into U1_buffers
    @inbounds for (j1, ψ1_fft) in enumerate(st.filter_bank.wavelets)
        ScatteringCore.wavelet_convolve!(st.buffer_conv, signal_fft, ψ1_fft, 
                                          st.ifft_plan, st.buffer_input)
        ScatteringCore.apply_modulus!(st.U1_buffers[j1], st.buffer_conv)
    end
    
    # Pass 2: FFT each U1 into U1_fft_buffers (zero alloc via mul!)
    @inbounds for j1 in 1:num_w
        # Zero-alloc real→complex promotion into buffer_input, then FFT
        @simd for i in eachindex(st.U1_buffers[j1])
            st.buffer_input[i] = complex(st.U1_buffers[j1][i])
        end
        LinearAlgebra.mul!(st.U1_fft_buffers[j1], st.fft_plan, st.buffer_input)
    end

    # Pass 3: second-order scattering over the ADMISSIBLE paths (j_eff strictly increasing),
    # taken from the scattering tree rather than a flat upper-triangle loop.
    tree = st.tree
    @inbounds for p in PathGraph.order_range(tree, 2)
        idx = PathGraph.path_indices(tree, p)
        j1, j2 = idx[1], idx[2]
        ψ2_fft = st.filter_bank.wavelets[j2]
        # buffer_input = multiply scratch; buffer_conv = IFFT output
        ScatteringCore.wavelet_convolve!(st.buffer_conv, st.U1_fft_buffers[j1], ψ2_fft,
                                          st.ifft_plan, st.buffer_input)
        ScatteringCore.apply_modulus!(st.buffer_mod, st.buffer_conv)
        S2[j1, j2] = ScatteringCore.spatial_average(st.buffer_mod)
    end
    return S2
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
    @inbounds @simd for i in eachindex(U)
        st.buffer_input[i] = complex(U[i])
    end
    LinearAlgebra.mul!(st.buffer_conv, st.fft_plan, st.buffer_input)        # U_fft -> buffer_conv
    @inbounds @simd for i in eachindex(st.buffer_conv)
        st.buffer_conv[i] *= φ[i]
    end
    LinearAlgebra.mul!(st.buffer_input, st.ifft_plan, st.buffer_conv)       # (U ⋆ φ) -> buffer_input
    @inbounds for k in 1:length(dst)
        dst[k] = real(st.buffer_input[(k - 1) * ds + 1])
    end
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
    @inbounds @simd for i in eachindex(signal)
        st.buffer_input[i] = complex(signal[i])
    end
    LinearAlgebra.mul!(st.buffer_signal_fft, st.fft_plan, st.buffer_input)

    # order 0 (root): (x ⋆ φ_J) ↓
    @inbounds @simd for i in eachindex(signal)
        st.buffer_mod[i] = signal[i]
    end
    root = first(PathGraph.order_range(tree, 0))
    _lowpass_downsample!(view(data, :, root), st, st.buffer_mod, ds)

    # order 1: (|x ⋆ ψ_j| ⋆ φ_J) ↓
    @inbounds for p in PathGraph.order_range(tree, 1)
        j = PathGraph.path_indices(tree, p)[1]
        ScatteringCore.wavelet_convolve!(st.buffer_conv, st.buffer_signal_fft,
            st.filter_bank.wavelets[j], st.ifft_plan, st.buffer_input)
        ScatteringCore.apply_modulus!(st.buffer_mod, st.buffer_conv)
        _lowpass_downsample!(view(data, :, p), st, st.buffer_mod, ds)
    end

    # order 2: (||x ⋆ ψ_j1| ⋆ ψ_j2| ⋆ φ_J) ↓
    if st.max_order >= 2 && length(tree.by_order) >= 3
        num_w = length(st.filter_bank.wavelets)
        @inbounds for (j1, ψ1_fft) in enumerate(st.filter_bank.wavelets)
            ScatteringCore.wavelet_convolve!(st.buffer_conv, st.buffer_signal_fft, ψ1_fft,
                st.ifft_plan, st.buffer_input)
            ScatteringCore.apply_modulus!(st.U1_buffers[j1], st.buffer_conv)
        end
        @inbounds for j1 in 1:num_w
            @simd for i in eachindex(st.U1_buffers[j1])
                st.buffer_input[i] = complex(st.U1_buffers[j1][i])
            end
            LinearAlgebra.mul!(st.U1_fft_buffers[j1], st.fft_plan, st.buffer_input)
        end
        @inbounds for p in PathGraph.order_range(tree, 2)
            idx = PathGraph.path_indices(tree, p)
            j1, j2 = idx[1], idx[2]
            ScatteringCore.wavelet_convolve!(st.buffer_conv, st.U1_fft_buffers[j1],
                st.filter_bank.wavelets[j2], st.ifft_plan, st.buffer_input)
            ScatteringCore.apply_modulus!(st.buffer_mod, st.buffer_conv)
            _lowpass_downsample!(view(data, :, p), st, st.buffer_mod, ds)
        end
    end
    return field
end

end # module Scattering1D
