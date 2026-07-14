module ScatteringTransformsDistributedExt

"""
    ScatteringTransformsDistributedExt — multi-process batched transforms (DistributedBackend)

Distributes `scattering_batch` over the batch dimension across worker processes with
`Distributed.pmap`. Because FFTW/CUFFT plans are not serializable, each worker **rebuilds** the
transform from a small serializable spec (`transform_spec`) and runs the *inner* local backend
(`local_backend(b)` — serial by default, or e.g. `ThreadedBackend`) on its column chunk; the
column blocks are then concatenated. Load `using Distributed` (and `@everywhere using
ScatteringTransforms`) to enable.
"""

using Distributed: Distributed
using ScatteringTransforms: ScatteringTransforms

const ST = ScatteringTransforms
const DistributedBackend = ST.Backends.DistributedBackend

# Partition 1:n into ≤ k contiguous column ranges.
function _column_chunks(n::Int, k::Int)
    k = max(1, min(k, n))
    return [(div((c - 1) * n, k) + 1):(div(c * n, k)) for c in 1:k]
end

function _distributed_batch(b::DistributedBackend, st, X, slicer)
    spec = ST.transform_spec(st)
    inner = ST.Backends.local_backend(b)
    ncols = size(X)[end]
    chunks = _column_chunks(ncols, Distributed.nworkers())
    parts = Distributed.pmap(chunks) do cols
        st_local = ST.rebuild_transform(spec)
        ST.scattering_batch(inner, st_local, slicer(X, cols))
    end
    return reduce(hcat, parts)
end

ST.scattering_batch(b::DistributedBackend, st::ST.Scattering1D.ScatteringTransform1D, X::AbstractMatrix) =
    _distributed_batch(b, st, X, (A, cols) -> A[:, cols])

ST.scattering_batch(b::DistributedBackend, st::ST.Scattering2D.ScatteringTransform2D, X::AbstractArray{<:Any,3}) =
    _distributed_batch(b, st, X, (A, cols) -> A[:, :, cols])

end # module ScatteringTransformsDistributedExt
