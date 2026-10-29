module ScatteringTransformsDifferentiationInterfaceExt

using DifferentiationInterface: DifferentiationInterface as DI
using ScatteringTransforms: ScatteringTransforms

# A target given as a field array -> its coefficients; otherwise it already is a coeff container.
_target_coeffs(st, target::AbstractArray) = ScatteringTransforms.ScatteringCore.scattering(st, target)
_target_coeffs(::Any, target) = target

# Spatial shape of the transform input.
_field_shape(st) = st.dims

# The transform, target and loss are arguments, declared `DI.Constant` below, so the differentiated
# function captures nothing and only `z` is active; Enzyme rejects a closure over the transform with
# `EnzymeMutabilityException`.
_objective(z, st, tc, loss) = loss(ScatteringTransforms.ScatteringCore.scattering(st, z), tc)

function ScatteringTransforms.synthesize(st::ScatteringTransforms.ScatteringCore.ScatteringTransform, target;
                                         backend,
                                         init = nothing,
                                         iters::Int = 500,
                                         lr::Real = 0.05,
                                         loss = ScatteringTransforms.scattering_loss)
    tc = _target_coeffs(st, target)
    T = real(eltype(st.filter_bank.averaging))
    x = init === nothing ? randn(T, _field_shape(st)) : T.(collect(init))

    ctx = (DI.Constant(st), DI.Constant(tc), DI.Constant(loss))
    prep = DI.prepare_gradient(_objective, backend, x, ctx...)

    # Adam — first-order, no extra optimizer dependency.
    losses = Vector{T}(undef, iters)
    m = zero(x)
    v = zero(x)
    β1, β2, ϵ, η = T(0.9), T(0.999), T(1e-8), T(lr)
    for t in 1:iters
        val, g = DI.value_and_gradient(_objective, prep, backend, x, ctx...)
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
