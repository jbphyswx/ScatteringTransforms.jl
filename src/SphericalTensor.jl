struct _SHTTensorGrid{V}
    theta::V
    nphi::Int
end

struct _TensorSHTScratch{M,C,P}
    latitude::M
    spectrum::C
    values::C
    longitude::P
end

function _sht_scratch(::Type{T}, lmax, B, grid::_SHTTensorGrid) where {T}
    nθ = length(grid.theta)
    spectrum = Matrix{Complex{T}}(undef, grid.nphi, nθ * B)
    longitude = Plans.make_plan(SB.AutoSpectralBackend(), T, (grid.nphi,); batch = nθ * B)
    return _TensorSHTScratch(Matrix{T}(undef, lmax + 1, lmax + 1), spectrum, similar(spectrum), longitude)
end

_direct_plan_points(p, ::_SHTTensorGrid) = nothing

function _tensor_synthesize!(f, grid::_SHTTensorGrid, tab, scr::_TensorSHTScratch, a, nb, derivative)
    T = real(eltype(scr.spectrum))
    nθ, nφ, K = length(grid.theta), grid.nphi, (tab.lmax + 1)^2
    P = scr.latitude
    fill!(scr.spectrum, zero(eltype(scr.spectrum)))
    @inbounds for r in 1:nθ
        _latitude_values!(P, tab, grid.theta[r])
        for b in 1:nb, m in 0:tab.lmax
            z = zero(Complex{T})
            for ℓ in m:tab.lmax
                p = derivative == 1 ? -_dtheta_pbar(P, ℓ, m) : P[ℓ + 1, m + 1]
                ap = a[(b - 1) * K + _lm(ℓ, m)]
                am = a[(b - 1) * K + _lm(ℓ, -m)]
                z += p * (m == 0 ? complex(ap) : complex(ap, -am) / sqrt(T(2)))
            end
            derivative == 2 && (z *= im * m)
            col = (b - 1) * nθ + r
            scr.spectrum[m + 1, col] = nφ * z
            m == 0 || (scr.spectrum[nφ - m + 1, col] = nφ * conj(z))
        end
    end
    Plans.inverse_transform!(scr.values, scr.longitude, scr.spectrum)
    @inbounds for b in 1:nb, k in 1:nφ, r in 1:nθ
        value = real(scr.values[k, (b - 1) * nθ + r])
        f[(b - 1) * nθ * nφ + (k - 1) * nθ + r] =
            derivative == 2 ? value / sin(grid.theta[r]) : value
    end
    return f
end

_sht_synthesize!(f, grid::_SHTTensorGrid, tab, scr::_TensorSHTScratch, a, nb::Int) =
    _tensor_synthesize!(f, grid, tab, scr, a, nb, 0)

function _sht_adjoint!(a, grid::_SHTTensorGrid, tab, scr::_TensorSHTScratch, f, nb::Int)
    T = real(eltype(scr.spectrum))
    nθ, nφ, K = length(grid.theta), grid.nphi, (tab.lmax + 1)^2
    fill!(scr.values, zero(eltype(scr.values)))
    @inbounds for b in 1:nb, k in 1:nφ, r in 1:nθ
        scr.values[k, (b - 1) * nθ + r] = f[(b - 1) * nθ * nφ + (k - 1) * nθ + r]
    end
    Plans.forward_transform!(scr.spectrum, scr.longitude, scr.values)
    fill!(a, zero(eltype(a)))
    P = scr.latitude
    @inbounds for r in 1:nθ
        _latitude_values!(P, tab, grid.theta[r])
        for b in 1:nb, m in 0:tab.lmax
            z = scr.spectrum[m + 1, (b - 1) * nθ + r]
            for ℓ in m:tab.lmax
                p = P[ℓ + 1, m + 1]
                if m == 0
                    a[(b - 1) * K + _lm(ℓ, 0)] += p * real(z)
                else
                    a[(b - 1) * K + _lm(ℓ, m)] += sqrt(T(2)) * p * real(z)
                    a[(b - 1) * K + _lm(ℓ, -m)] -= sqrt(T(2)) * p * imag(z)
                end
            end
        end
    end
    return a
end

_direct_riesz_scratch(plan, field, ::_SHTTensorGrid) = similar(field)

function _direct_riesz_energy!(out, plan, C, h, scratch, grid::_SHTTensorGrid)
    _scale_direct_coeffs!(plan.ha, plan, C, h)
    _tensor_synthesize!(scratch, grid, plan.tab, plan.op.scr, plan.ha, plan.B, 1)
    @. out = scratch^2
    _tensor_synthesize!(scratch, grid, plan.tab, plan.op.scr, plan.ha, plan.B, 2)
    @. out += scratch^2
    return out
end

function _direct_riesz_components!(uθ, uφ, plan, C, h, grid::_SHTTensorGrid)
    _scale_direct_coeffs!(plan.ha, plan, C, h)
    _tensor_synthesize!(uθ, grid, plan.tab, plan.op.scr, plan.ha, plan.B, 1)
    _tensor_synthesize!(uφ, grid, plan.tab, plan.op.scr, plan.ha, plan.B, 2)
    return nothing
end

Plans.close_plan!(p::DirectSHTSphericalPlan) = _close_sht_scratch!(p.op.scr)
_close_sht_scratch!(::_SHTScratch) = nothing
_close_sht_scratch!(scratch::_TensorSHTScratch) = Plans.close_plan!(scratch.longitude)
