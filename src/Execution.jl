module Execution

# What `AutoBackend` resolves to, and which execution extensions are loaded. The backends come from
# `ComputationalBackends`, which leaves resolving `AbstractAutoBackend` to each consumer; this
# package's policy is its own `resolve_backend`.

using ComputationalBackends: ComputationalBackends as CB

"""
    available(execution) -> Bool

Whether the extension for `execution` is loaded.
"""
available(::CB.AbstractExecutionBackend) = false
available(::CB.AbstractSerialBackend) = true

"""
    resolve_backend(backend) -> AbstractLocalBackend

Concrete backends pass through unchanged — a request is honoured exactly or refused, never
silently downgraded. `AutoBackend` resolves on real capability: `ThreadedBackend` when there is
more than one thread *and* the OhMyThreads extension is loaded, otherwise `SerialBackend`.
"""
resolve_backend(backend::CB.AbstractExecutionBackend) = backend
resolve_backend(::CB.AbstractAutoBackend) =
    (Threads.nthreads() > 1 && available(CB.ThreadedBackend())) ? CB.ThreadedBackend() : CB.SerialBackend()

"""
    check_available(backend) -> backend

Return `backend` if its extension is loaded, and otherwise throw an `ArgumentError` naming the
package to load. An `AutoBackend` must go through [`resolve_backend`](@ref) first.
"""
check_available(b::CB.AbstractSerialBackend) = b
function check_available(b::CB.AbstractThreadedBackend)
    available(b) || throw(ArgumentError(
        "ThreadedBackend requires the OhMyThreads extension. Run `using OhMyThreads`."))
    return b
end
function check_available(b::CB.AbstractGPUBackend)
    available(b) || throw(ArgumentError(
        "GPUBackend requires the KernelAbstractions extension. Run `using KernelAbstractions, " *
        "AbstractFFTs` plus a device package such as `using CUDA`."))
    return b
end
function check_available(b::CB.AbstractDistributedBackend)
    available(b) || throw(ArgumentError(
        "DistributedBackend requires the Distributed extension. Run `using Distributed`."))
    check_available(CB.local_backend(b))
    return b
end
function check_available(b::CB.AbstractMPIBackend)
    available(b) || throw(ArgumentError("MPIBackend requires the MPI extension. Run `using MPI`."))
    check_available(CB.local_backend(b))
    return b
end
check_available(::CB.AbstractAutoBackend) = throw(ArgumentError(
    "AutoBackend must be resolved before use — call `Execution.resolve_backend(b)` first."))

end # module Execution
