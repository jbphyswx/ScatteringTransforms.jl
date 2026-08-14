module Execution

"""
    Execution.jl — execution-backend resolution

Execution backends themselves come from `ComputationalBackends` (`SerialBackend`,
`ThreadedBackend`, `GPUBackend{B}`, `DistributedBackend{Inner}`, `MPIBackend{Inner,C}`,
`AutoBackend`); this module only decides what `AutoBackend` resolves to and answers which
execution extensions are loaded.

`ComputationalBackends.resolve_backend(::AbstractAutoBackend)` deliberately errors so that each
consumer states its own policy, and defining a method on it here would be type piracy — so
[`resolve_backend`](@ref) below is a distinct function owned by this package.
"""

using ComputationalBackends: ComputationalBackends as CB

# Which execution extensions are loaded. Reached only from [`resolve_backend`](@ref) on an
# `AutoBackend` and from [`check_available`](@ref), which the batch entry points call only after
# dispatch has already failed to find a backend method — so an honoured request never asks.
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

Throw unless `backend` can actually execute, naming the package that would enable it. Called by the
batch entry points so an unloadable request fails immediately instead of at a `MethodError`.
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
