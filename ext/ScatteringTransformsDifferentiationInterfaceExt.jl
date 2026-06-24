module ScatteringTransformsDifferentiationInterfaceExt

"""
    ScatteringTransformsDifferentiationInterfaceExt — gradient-descent synthesis

Provides [`ScatteringTransforms.synthesize`](@ref): reconstruct a field whose scattering
coefficients match a target by minimizing `loss(scattering(st, x̂), target)` with Adam, the
gradient supplied through DifferentiationInterface (DI). DI is backend-agnostic, so any `ADTypes`
backend works — `AutoMooncake()`, `AutoEnzyme()`, `AutoZygote()`, … — the user picks one and
loads the corresponding AD package.

The differentiable forward is the non-mutating `ScatteringTransforms.scattering(st, x)`: with the
in-core direct-sum spectral backend it is plain matmul/broadcast (differentiable by every
backend); with the FFTW fast path it differentiates via `AbstractFFTs` ChainRules (Mooncake /
Zygote). For reverse-mode synthesis the direct-sum backend is the most portable default.
"""

using DifferentiationInterface: DifferentiationInterface as DI
using ScatteringTransforms: ScatteringTransforms

# The gridded transforms expose the same non-mutating `scattering` + `buffer_mod` shape.
const _Gridded = Union{ScatteringTransforms.ScatteringTransform1D,
                       ScatteringTransforms.ScatteringTransform2D,
                       ScatteringTransforms.ScatteringTransform3D}

# A target given as a field array -> its coefficients; otherwise it already is a coeff container.
_target_coeffs(st, target::AbstractArray) = ScatteringTransforms.scattering(st, target)
_target_coeffs(::Any, target) = target

# Spatial shape of the transform input (the real modulus workspace carries it).
_field_shape(st) = size(st.buffer_mod)

function ScatteringTransforms.synthesize(st::_Gridded, target;
                                         backend,
                                         init = nothing,
                                         iters::Int = 500,
                                         lr::Real = 0.05,
                                         loss = ScatteringTransforms.scattering_loss)
    tc = _target_coeffs(st, target)
    T = real(eltype(st.filter_bank.averaging))
    x = init === nothing ? randn(T, _field_shape(st)) : T.(collect(init))

    objective(z) = loss(ScatteringTransforms.scattering(st, z), tc)
    prep = DI.prepare_gradient(objective, backend, x)

    # Adam — first-order, no extra optimizer dependency.
    losses = Vector{T}(undef, iters)
    m = zero(x)
    v = zero(x)
    β1, β2, ϵ, η = T(0.9), T(0.999), T(1e-8), T(lr)
    for t in 1:iters
        val, g = DI.value_and_gradient(objective, prep, backend, x)
        losses[t] = T(val)
        @. m = β1 * m + (1 - β1) * g
        @. v = β2 * v + (1 - β2) * g * g
        bc1 = 1 - β1^t
        bc2 = 1 - β2^t
        @. x = x - η * (m / bc1) / (sqrt(v / bc2) + ϵ)
    end
    return (; field = x, losses)
end

end # module ScatteringTransformsDifferentiationInterfaceExt
