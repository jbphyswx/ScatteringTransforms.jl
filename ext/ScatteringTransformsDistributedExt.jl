module ScatteringTransformsDistributedExt

using Distributed: Distributed
using ComputationalBackends: ComputationalBackends as CB
using FlowTransformBindings: FlowTransformBindings as FTB
using ScatteringTransforms: ScatteringTransforms as ST

ST.Execution.available(::CB.AbstractDistributedBackend) = true

struct WorkerTransform{H} <: ST.ScatteringCore.AbstractWorker
    pid::Int
    handle::H
end

function _install_transform(spec)
    st = ST.rebuild_transform(spec)
    return Distributed.RemoteChannel(Distributed.myid()) do
        channel = Channel{typeof(st)}(1)
        put!(channel, st)
        channel
    end
end

function _remote_batch(handle, inner, X)
    st = take!(handle)
    try
        return Base.ScopedValues.with(FTB.FASTTRANSFORMS_THREADS => 1) do
            ST.scattering_batch(st, X; execution = inner)
        end
    finally
        put!(handle, st)
    end
end

function _release_transform(handle)
    ST.close_transform!(take!(handle))
    close(handle)
    return nothing
end

function ST.close_worker!(worker::WorkerTransform)
    worker.pid in Distributed.procs() || return nothing
    Distributed.remotecall_fetch(_release_transform, worker.pid, worker.handle)
    return nothing
end

function _worker_for!(st, pid)
    workers = st.workspaces.workers
    index = findfirst(w -> w isa WorkerTransform && w.pid == pid, workers)
    index === nothing || return workers[index]
    handle = Distributed.remotecall_fetch(_install_transform, pid, ST.transform_spec(st))
    worker = WorkerTransform(pid, handle)
    push!(workers, worker)
    return worker
end

function _send_columns!(out, worker::WorkerTransform, inner, block, cols)
    part = Distributed.remotecall_fetch(_remote_batch, worker.pid, worker.handle, inner, block)
    copyto!(view(out, :, cols), part)
    return nothing
end

function _distributed_batch(b::CB.AbstractDistributedBackend, st, X)
    ncols = size(X)[end]
    T = ST._flat_eltype(ST._coefficient_eltype(st), X)
    out = Matrix{T}(undef, ST.flat_rows(st), ncols)
    ncols == 0 && return out
    pids = Distributed.workers()
    n = min(ncols, length(pids))
    inner = CB.local_backend(b)
    lock(st.workspaces.lock) do
        @sync for t in 1:n
            cols = (div((t - 1) * ncols, n) + 1):div(t * ncols, n)
            worker = _worker_for!(st, pids[t])
            block = copy(selectdim(X, ndims(X), cols))
            @async _send_columns!(out, worker, inner, block, cols)
        end
    end
    return out
end

ST._scattering_batch(b::CB.AbstractDistributedBackend,
                     st::ST.ScatteringCore.AbstractScatteringTransform, X::AbstractArray) =
    _distributed_batch(b, st, X)

end # module ScatteringTransformsDistributedExt
