module Execution

# What `AutoBackend` resolves to, and which execution extensions are loaded. The backends come from
# `ComputationalBackends`, which leaves resolving `AbstractAutoBackend` to each consumer; this
# package's policy is its own `resolve_backend`.

using ComputationalBackends: ComputationalBackends as CB

_ext_loaded(name::Symbol) = Base.get_extension(parentmodule(@__MODULE__), name) !== nothing

"`true` when the OhMyThreads extension is loaded, i.e. `ThreadedBackend` can actually run."
have_threads() = _ext_loaded(:ScatteringTransformsOhMyThreadsExt)

"`true` when the Distributed extension is loaded."
have_distributed() = _ext_loaded(:ScatteringTransformsDistributedExt)

"`true` when the MPI extension is loaded."
have_mpi() = _ext_loaded(:ScatteringTransformsMPIExt)

"`true` when the KernelAbstractions extension is loaded, i.e. `GPUBackend` can actually run."
have_gpu() = _ext_loaded(:ScatteringTransformsKernelAbstractionsExt)

"""
    resolve_backend(backend) -> AbstractLocalBackend

Concrete backends pass through unchanged — a request is honoured exactly or refused, never
silently downgraded. `AutoBackend` resolves on real capability: `ThreadedBackend` when there is
more than one thread *and* the OhMyThreads extension is loaded, otherwise `SerialBackend`.
"""
resolve_backend(backend::CB.AbstractExecutionBackend) = backend
resolve_backend(::CB.AbstractAutoBackend) =
    (Threads.nthreads() > 1 && have_threads()) ? CB.ThreadedBackend() : CB.SerialBackend()

"""
    check_available(backend) -> backend

Return `backend` if its extension is loaded, and otherwise throw an `ArgumentError` naming the
package to load. An `AutoBackend` must go through [`resolve_backend`](@ref) first.
"""
check_available(b::CB.AbstractSerialBackend) = b
function check_available(b::CB.AbstractThreadedBackend)
    have_threads() || throw(ArgumentError(
        "ThreadedBackend requires the OhMyThreads extension. Run `using OhMyThreads`."))
    return b
end
function check_available(b::CB.AbstractGPUBackend)
    have_gpu() || throw(ArgumentError(
        "GPUBackend requires the KernelAbstractions extension. Run `using KernelAbstractions, " *
        "AbstractFFTs` plus a device package such as `using CUDA`."))
    return b
end
function check_available(b::CB.AbstractDistributedBackend)
    have_distributed() || throw(ArgumentError(
        "DistributedBackend requires the Distributed extension. Run `using Distributed`."))
    check_available(CB.local_backend(b))
    return b
end
function check_available(b::CB.AbstractMPIBackend)
    have_mpi() || throw(ArgumentError("MPIBackend requires the MPI extension. Run `using MPI`."))
    check_available(CB.local_backend(b))
    return b
end
check_available(::CB.AbstractAutoBackend) = throw(ArgumentError(
    "AutoBackend must be resolved before use — call `Execution.resolve_backend(b)` first."))

end # module Execution
