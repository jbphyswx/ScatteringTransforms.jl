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

# The work `pmap` sends to a worker, as a named function taking only plain data. It must not be a
# closure over `_distributed_batch`'s scope: `pmap` serialises whatever the closure captures, closure
# conversion can capture more of the enclosing scope than the body textually uses, and `st` is in that
# scope holding FFTW/FastTransforms plans — pointers into native memory that do not survive a
# serialisation round trip. Capturing it corrupted the caller's transform in place.
_chunk(spec, inner, slicer, X, cols) =
    ST.scattering_batch(inner, ST.rebuild_transform(spec), slicer(X, cols))

# One chunk per worker, deliberately: `_chunk` rebuilds the transform — filter bank and
# spectral plan — on the worker for every chunk it receives. Splitting finer would let `pmap`
# rebalance around a straggler, but would pay that rebuild once per chunk instead of once per
# worker. Finer chunking is only worth it alongside a per-worker transform cache.
function _distributed_batch(b::CB.AbstractDistributedBackend, st, X, slicer)
    spec = ST.transform_spec(st)
    inner = CB.local_backend(b)
    ncols = size(X)[end]
    chunks = _column_chunks(ncols, Distributed.nworkers())
    # With no separate worker processes there is nothing to distribute: `pmap` would hand the work
    # back to this process, paying a serialise and a full transform rebuild to do so.
    parts = Distributed.nworkers() == 1 ?
        [_chunk(spec, inner, slicer, X, cols) for cols in chunks] :
        Distributed.pmap(cols -> _chunk(spec, inner, slicer, X, cols), chunks)
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

# The nonuniform and spherical surfaces rebuild from a spec that carries their sample locations, so
# they distribute the same way; only the batch axis differs.
ST.scattering_batch(b::CB.AbstractDistributedBackend,
                    st::ST.SubsampledScattering.MultiResolutionScattering, X::AbstractArray) =
    _distributed_batch(b, st, X, (A, cols) -> selectdim(A, ndims(A), cols))

ST.scattering_batch(b::CB.AbstractDistributedBackend,
                    st::ST.ScatteredPlanar.ScatteredPlanarScattering, X::AbstractMatrix) =
    _distributed_batch(b, st, X, (A, cols) -> view(A, :, cols))

ST.scattering_batch(b::CB.AbstractDistributedBackend, st::ST.SphericalCore.SphericalScattering,
                    X::AbstractArray) =
    _distributed_batch(b, st, X, (A, cols) -> selectdim(A, ndims(A), cols))

ST.scattering_batch(b::CB.AbstractDistributedBackend,
                    st::ST.SphericalCore.SphericalMonogenicScattering, X::AbstractArray) =
    _distributed_batch(b, st, X, (A, cols) -> selectdim(A, ndims(A), cols))

end # module ScatteringTransformsDistributedExt
