module Reductions

# Normalized and log scattering descriptors (Cheng & Ménard 2021; Allys et al. 2019). The 2D
# orientation reductions are `Scattering2D.compute_shape_sparsity`.

using ..Coefficients: Coefficients

export normalized_coefficients, log_coefficients

const _Coeffs = Union{Coefficients.ScatteringCoefficients1D, Coefficients.ScatteringCoefficients2D}

"""
    normalized_coefficients(c) -> (; S0, s1, s2)

Amplitude-normalized coefficients: `s1[j] = S1[j]/S0` and `s2[j1,j2] = S2[j1,j2]/S1[j1]`
(zero where `S1[j1] == 0`).
"""
function normalized_coefficients(c::_Coeffs)
    S0 = Coefficients.zeroth_order(c)
    S1 = Coefficients.first_order(c)
    S2 = Coefficients.second_order(c)
    s1 = S1 ./ ifelse(iszero(S0), one(S0), S0)
    positive = reshape(S1 .> zero(eltype(S1)), :, 1)
    denominator = reshape(ifelse.(S1 .> zero(eltype(S1)), S1, one(eltype(S1))), :, 1)
    s2 = isempty(S2) ? copy(S2) : ifelse.(positive, S2 ./ denominator, zero(eltype(S2)))
    return (S0 = S0, s1 = s1, s2 = s2)
end

"""
    log_coefficients(c; pad=eps) -> (; S0, logS1, logS2)

Log-coefficients `log(S1 + pad)`, `log(S2 + pad)` (the small `pad` keeps structural zeros
finite). Useful to gaussianize the heavy-tailed coefficients of intermittent fields.
"""
function log_coefficients(c::_Coeffs; pad::Real = eps(real(eltype(Coefficients.first_order(c)))))
    S1 = Coefficients.first_order(c)
    S2 = Coefficients.second_order(c)
    return (S0 = Coefficients.zeroth_order(c),
            logS1 = log.(S1 .+ pad),
            logS2 = log.(S2 .+ pad))
end

end # module Reductions
