module Scattering1D

using ..Plans: Plans
using ComputationalBackends: ComputationalBackends as CB
using SpectralBackends: SpectralBackends as SB
using ..FilterBanks: FilterBanks
using ..ScatteringCore: ScatteringCore
using ..Coefficients: Coefficients
using ..PathGraph: PathGraph
using ..Cascade: Cascade
using ..ScatteringFields: ScatteringFields

export scattering_transform!, cascade!

"""
    ScatteringCore.ScatteringTransform{1}([T=Float64,] N, J; Q=1, max_order=2, filters=StoredFilters(),
                          oversampling=J, transform=AutoSpectralBackend())

Build a 1D scattering transform for length-`N` signals over `J` octaves. The element type is
positional, as for `zeros(T, …)`; omit it for `Float64`.

`filters = FilterBanks.ComputedFilters()` evaluates each wavelet on demand (see
`FilterBanks.ComputedFilterBank1D`).
`oversampling = α` produces a scale-`j` wavelet's output decimated by `2^max(j-α, 0)`; the default
`J` decimates nothing. `transform` is the `SpectralBackends` tag of the discrete Fourier transform
the cascade runs on (see `Plans.make_plan`).
"""
function ScatteringCore.ScatteringTransform{1}(::Type{T}, N::Int, J::Int;
                               Q::Int=1,
                               max_order::Int=2,
                               filters::FilterBanks.AbstractFilterStorage=FilterBanks.StoredFilters(),
                               oversampling::Int=J,
                               transform::SB.AbstractSpectralBackend=SB.AutoSpectralBackend()) where {T}
    filter_bank = FilterBanks.build_filter_bank1d(T, N, J, filters; Q=Q)
    tree = PathGraph.build_tree([m.j_eff for m in filter_bank.meta], max_order)
    nw = FilterBanks.nwavelets(filter_bank)
    groups = max_order >= 2 ? PathGraph.order2_groups(tree, nw) :
             [(j, Int[], Int[]) for j in 1:nw]
    plan = Plans.make_plan(transform, T, (N,))

    pw = Cascade.build(filter_bank, groups, (N,), T, oversampling, plan)
    return ScatteringCore.ScatteringTransform{1}(filter_bank, tree, groups, max_order, plan,
                                 zeros(Complex{T}, N), zeros(Complex{T}, N), (N,), pw)
end
"""
    (st::ScatteringCore.ScatteringTransform{1})(signal) -> ScatteringCoefficients1D

The scattering coefficients of `signal`, in a new container of the transform's precision.
"""
function (st::ScatteringCore.ScatteringTransform{1})(signal::AbstractVector)
    num_w = FilterBanks.nwavelets(st.filter_bank)
    T = real(eltype(st.buffer_input))

    coeffs = Coefficients.ScatteringCoefficients1D(num_w, T; compute_S2=st.max_order >= 2)

    return scattering_transform!(coeffs, st, signal)
end

"""
    scattering_transform!(coeffs, st, signal; execution = SerialBackend()) -> coeffs′

Fill `coeffs.S1`/`coeffs.S2` and return the coefficients with `S0` set by
[`Coefficients.update_S0`](@ref), which allocates a new container only when `S0` is a scalar.

`execution` is where the cascade runs: `SerialBackend` in this task; `ThreadedBackend` (OhMyThreads
extension) runs the first-order wavelet groups as parallel tasks after one serial input transform.
"""
function scattering_transform!(coeffs::Coefficients.ScatteringCoefficients1D,
                               st::ScatteringCore.ScatteringTransform{1}, signal::AbstractVector;
                               execution::CB.AbstractExecutionBackend = CB.SerialBackend())
    st.buffer_input .= complex.(signal)
    Plans.forward_transform!(st.buffer_signal_fft, st.plan, st.buffer_input)
    cascade!(coeffs.S1, coeffs.S2, execution, st, st.buffer_signal_fft)
    return Coefficients.update_S0(coeffs, ScatteringCore.spatial_average(signal))
end

cascade!(S1::AbstractVector, S2::AbstractMatrix, ::CB.AbstractSerialBackend,
         st::ScatteringCore.ScatteringTransform{1}, signal_fft::AbstractVector) = cascade!(S1, S2, st, signal_fft)

"""
    cascade!(S1, S2, st, signal_fft) -> (S1, S2)

Both scattering orders in one pass over the tree, grouped by first-order wavelet:

    for (j1, children):  U₁ = |x ⋆ ψ_j1| ;  S1[j1] = ⟨U₁⟩
                         Û₁ = fft(U₁)     ;  S2[j1,j2] = ⟨|U₁ ⋆ ψ_j2|⟩  for each child

One `U₁`/`Û₁` pair is live at a time, and a wavelet with no admissible child reduces to a single
`⟨|·|⟩`. `signal_fft` is read only.
"""
cascade!(S1::AbstractVector, S2::AbstractMatrix, st::ScatteringCore.ScatteringTransform{1},
         signal_fft::AbstractVector) =
    Cascade.cascade!(S1, S2, st.pw, st.filter_bank, st.groups, signal_fft)

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

function ScatteringFields.scattering_field(st::ScatteringCore.ScatteringTransform{1}, signal::AbstractVector;
        subsample::Int = _default_subsample(st.dims[1], st.filter_bank.J))
    N = st.dims[1]
    N % subsample == 0 ||
        throw(ArgumentError("subsample factor $subsample must divide signal length $N"))
    # Real unless the input is complex, in which case the order-0 field is too.
    T = promote_type(real(eltype(st.buffer_input)), eltype(signal))
    M = N ÷ subsample
    npath = PathGraph.npaths(st.tree)
    # On the transform's own array type; zero-filled, since paths of order 3 and above are not computed.
    data = fill!(similar(st.buffer_input, T, (M, npath)), zero(T))
    field = ScatteringFields.ScatteringField1D(st.tree, data, subsample,
                                               Cascade.field_workspace(st, subsample))
    return ScatteringFields.scattering_field!(field, st, signal)
end

function ScatteringFields.scattering_field!(field::ScatteringFields.ScatteringField1D,
        st::ScatteringCore.ScatteringTransform{1}, signal::AbstractVector)
    st.buffer_input .= complex.(signal)
    Plans.forward_transform!(st.buffer_signal_fft, st.plan, st.buffer_input)
    Cascade.field_cascade!(field.data, field.ws, st.filter_bank, st.groups, st.buffer_signal_fft,
                           first(PathGraph.order_range(st.tree, 0)),
                           first(PathGraph.order_range(st.tree, 1)))
    return field
end

# ============================================================================
# Non-mutating forward for autodiff; see `ScatteringCore.scattering`.
# ============================================================================

function ScatteringCore.scattering(st::ScatteringCore.ScatteringTransform{1}, signal::AbstractVector)
    S0, S1, S2 = Cascade.scattering_values(st, signal)
    return Coefficients.ScatteringCoefficients1D(S1, S2; S0=S0)
end

end # module Scattering1D
