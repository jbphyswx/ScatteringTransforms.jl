module ScatteringTransformsOhMyThreadsExt

"""
    ScatteringTransformsOhMyThreadsExt — multithreaded batched transforms (ThreadedBackend)

Parallelizes `scattering_batch` over the batch dimension with OhMyThreads. Each task gets its
own transform workspace (buffers + any plan scratch are mutated during a transform and are not
safe to share), obtained as a per-task `deepcopy` of the transform; the filter bank and path
tree are read-only and shared. Results are written to disjoint output columns, so there is no
contention.
"""

using OhMyThreads: OhMyThreads as OMT
using ScatteringTransforms: ScatteringTransforms

const ST = ScatteringTransforms
const ThreadedBackend = ST.Backends.ThreadedBackend

function ST.scattering_batch(::ThreadedBackend, st::ST.Scattering1D.ScatteringTransform1D, X::AbstractMatrix)
    N, B = size(X)
    nw = length(st.filter_bank.wavelets)
    T = real(eltype(st.filter_bank.averaging))
    out = Matrix{T}(undef, ST.Coefficients.flatten_length(
        ST.Coefficients.ScatteringCoefficients1D(nw, T; compute_S2 = st.max_order >= 2)), B)
    OMT.@tasks for b in 1:B
        @local begin
            st_local = deepcopy(st)
            coeffs_local = ST.Coefficients.ScatteringCoefficients1D(nw, T; compute_S2 = st.max_order >= 2)
        end
        c = ST.Scattering1D.scattering_transform!(coeffs_local, st_local, view(X, :, b))
        ST.Coefficients.flatten1d!(view(out, :, b), c)
    end
    return out
end

function ST.scattering_batch(::ThreadedBackend, st::ST.Scattering2D.ScatteringTransform2D, X::AbstractArray{<:Any,3})
    Ny, Nx, B = size(X)
    T = real(eltype(st.filter_bank.averaging))
    coeff_proto = ST.Coefficients.ScatteringCoefficients2D(st.filter_bank.J, st.filter_bank.L, T;
                                                           compute_S2 = st.max_order >= 2)
    out = Matrix{T}(undef, ST.Coefficients.flatten_length(coeff_proto), B)
    OMT.@tasks for b in 1:B
        @local begin
            st_local = deepcopy(st)
            coeffs_local = ST.Coefficients.ScatteringCoefficients2D(st.filter_bank.J, st.filter_bank.L, T;
                                                                    compute_S2 = st.max_order >= 2)
        end
        c = ST.Scattering2D.scattering_transform2d!(coeffs_local, st_local, view(X, :, :, b))
        ST.Coefficients.flatten2d!(view(out, :, b), c)
    end
    return out
end

end # module ScatteringTransformsOhMyThreadsExt
