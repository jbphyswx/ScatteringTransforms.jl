module ScatteringTransformsDistributedExt

# `scattering_batch` on `DistributedBackend`: each worker rebuilds the transform from
# `transform_spec(st)`, since spectral plans are not serializable, transforms its column chunk with
# `local_backend(b)`, and the chunks are concatenated in order. Workers need
# `@everywhere using ScatteringTransforms`.

using Distributed: Distributed
using ComputationalBackends: ComputationalBackends as CB
using FlowTransformBindings: FlowTransformBindings as FTB
using ScatteringTransforms: ScatteringTransforms as ST

# Partition 1:n into ≤ k contiguous column ranges.
function _column_chunks(n::Int, k::Int)
    k = max(1, min(k, n))
    return [(div((c - 1) * n, k) + 1):(div(c * n, k)) for c in 1:k]
end

# The work `pmap` sends to a worker: a named function of plain data. It must not close over
# `_distributed_batch`'s scope, where `st` holds native plan pointers that do not survive
# serialisation; closure conversion can capture more of a scope than the body uses. The rebuilt
# transform is closed on the way out, freeing its library plans.
function _chunk(spec, inner, slicer, X, cols)
    st = ST.rebuild_transform(spec)
    try
        return ST.scattering_batch(st, slicer(X, cols); execution = inner)
    finally
        ST.close_transform!(st)
    end
end

# On a worker the processes carry the parallelism, so FastTransforms runs on one OpenMP thread there.
_remote_chunk(spec, inner, slicer, X, cols) =
    Base.ScopedValues.with(() -> _chunk(spec, inner, slicer, X, cols),
                           FTB.FASTTRANSFORMS_THREADS => 1)

# One chunk per worker: `_chunk` rebuilds the transform for every chunk it receives.
function _distributed_batch(b::CB.AbstractDistributedBackend, st, X, slicer)
    spec = ST.transform_spec(st)
    inner = CB.local_backend(b)
    ncols = size(X)[end]
    chunks = _column_chunks(ncols, Distributed.nworkers())
    # With no separate worker processes the chunks run in this process, without serialisation.
    parts = Distributed.nworkers() == 1 ?
        [_chunk(spec, inner, slicer, X, cols) for cols in chunks] :
        Distributed.pmap(cols -> _remote_chunk(spec, inner, slicer, X, cols), chunks)
    flen = size(first(parts), 1)
    out = Matrix{eltype(first(parts))}(undef, flen, ncols)
    @inbounds for (cols, part) in zip(chunks, parts)
        copyto!(view(out, :, cols), part)
    end
    return out
end

ST._scattering_batch(b::CB.AbstractDistributedBackend, st::ST.Scattering1D.ScatteringTransform1D, X::AbstractMatrix) =
    _distributed_batch(b, st, X, (A, cols) -> view(A, :, cols))

ST._scattering_batch(b::CB.AbstractDistributedBackend, st::ST.Scattering2D.ScatteringTransform2D, X::AbstractArray{<:Any,3}) =
    _distributed_batch(b, st, X, (A, cols) -> view(A, :, :, cols))

ST._scattering_batch(b::CB.AbstractDistributedBackend, st::ST.Scattering3D.ScatteringTransform3D, X::AbstractArray{<:Any,4}) =
    _distributed_batch(b, st, X, (A, cols) -> view(A, :, :, :, cols))

# The nonuniform and spherical surfaces rebuild from a spec that carries their sample locations, so
# they distribute the same way; only the batch axis differs.
ST._scattering_batch(b::CB.AbstractDistributedBackend,
                     st::ST.ScatteredPlanar.ScatteredPlanarScattering, X::AbstractMatrix) =
    _distributed_batch(b, st, X, (A, cols) -> view(A, :, cols))

ST._scattering_batch(b::CB.AbstractDistributedBackend, st::ST.SphericalCore.SphericalScattering,
                     X::AbstractArray) =
    _distributed_batch(b, st, X, (A, cols) -> selectdim(A, ndims(A), cols))

ST._scattering_batch(b::CB.AbstractDistributedBackend,
                     st::ST.SphericalCore.SphericalMonogenicScattering, X::AbstractArray) =
    _distributed_batch(b, st, X, (A, cols) -> selectdim(A, ndims(A), cols))

end # module ScatteringTransformsDistributedExt
