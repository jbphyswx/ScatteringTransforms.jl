module Reductions

"""
    Reductions.jl — Reduced / normalized scattering descriptors

Post-processing of the raw `(S0, S1, S2)` coefficients into the descriptors commonly used in
practice (Cheng & Ménard 2021; Allys et al. 2019):

- `normalized_coefficients` — `s1 = S1/S0`, `s2 = S2/S1` (removes dependence on the field's
  overall amplitude / first-order power), the standard inputs for classification & regression.
- `log_coefficients` — `log S1`, `log S2`, which gaussianize heavy-tailed scattering
  coefficients of intermittent fields.

Orientation-reduced second-order descriptors (sparsity `s₂₁`, anisotropy/shape `s₂₂`) for 2D
are produced by `compute_shape_sparsity` (in the 2D transform module).
"""

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
    s1 = S0 != zero(S0) ? S1 ./ S0 : copy(S1)
    s2 = zero(S2)
    @inbounds for j1 in axes(S2, 1), j2 in axes(S2, 2)
        if S1[j1] > zero(eltype(S1))
            s2[j1, j2] = S2[j1, j2] / S1[j1]
        end
    end
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
