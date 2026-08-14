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
using ComputationalBackends: ComputationalBackends as CB
using ScatteringTransforms: ScatteringTransforms as ST

# Partition 1:n into ≤ k contiguous column ranges.
function _column_chunks(n::Int, k::Int)
    k = max(1, min(k, n))
    return [(div((c - 1) * n, k) + 1):(div(c * n, k)) for c in 1:k]
end

# One chunk per worker, deliberately: the closure below rebuilds the transform — filter bank and
# spectral plan — on the worker for every chunk it receives. Splitting finer would let `pmap`
# rebalance around a straggler, but would pay that rebuild once per chunk instead of once per
# worker. Finer chunking is only worth it alongside a per-worker transform cache.
function _distributed_batch(b::CB.AbstractDistributedBackend, st, X, slicer)
    spec = ST.transform_spec(st)
    inner = CB.local_backend(b)
    ncols = size(X)[end]
    chunks = _column_chunks(ncols, Distributed.nworkers())
    parts = Distributed.pmap(chunks) do cols
        st_local = ST.rebuild_transform(spec)
        ST.scattering_batch(inner, st_local, slicer(X, cols))
    end
    # Write the blocks into one preallocated output rather than growing through `reduce(hcat, …)`,
    # which reallocates and copies the whole result once per chunk.
    flen = size(first(parts), 1)
    out = Matrix{eltype(first(parts))}(undef, flen, ncols)
    @inbounds for (cols, part) in zip(chunks, parts)
        copyto!(view(out, :, cols), part)
    end
    return out
end

ST.scattering_batch(b::CB.AbstractDistributedBackend, st::ST.Scattering1D.ScatteringTransform1D, X::AbstractMatrix) =
    _distributed_batch(b, st, X, (A, cols) -> view(A, :, cols))

ST.scattering_batch(b::CB.AbstractDistributedBackend, st::ST.Scattering2D.ScatteringTransform2D, X::AbstractArray{<:Any,3}) =
    _distributed_batch(b, st, X, (A, cols) -> view(A, :, :, cols))

ST.scattering_batch(b::CB.AbstractDistributedBackend, st::ST.Scattering3D.ScatteringTransform3D, X::AbstractArray{<:Any,4}) =
    _distributed_batch(b, st, X, (A, cols) -> view(A, :, :, :, cols))

end # module ScatteringTransformsDistributedExt
