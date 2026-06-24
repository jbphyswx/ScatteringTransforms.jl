module Backends

"""
    Backends.jl — Compute-backend taxonomy

Names mirror the jbphyswx ecosystem (`SerialBackend`, `ThreadedBackend`, `GPUBackend{B}`,
`AutoBackend`), with two orthogonal concerns:

- **Local compute backend** — what one process/rank computes on: `SerialBackend`,
  `ThreadedBackend` (OhMyThreads ext), `GPUBackend{B}` (CUDA / KernelAbstractions ext).
- **Distribution wrapper** — how work is split across processes, **parametric over the inner
  local backend**: `DistributedBackend{Inner}` (Distributed ext), `MPIBackend{Inner}` (MPI ext).
  A flat (non-parametric) distributed backend cannot say what each worker runs on; the
  parametric form makes `DistributedBackend{GPUBackend{CUDABackend}}` (multi-node multi-GPU) and
  `MPIBackend{ThreadedBackend}` (hybrid) expressible. The wrapper owns only
  "partition / gather"; `inner` owns the compute. (CUDA-aware MPI ⇒ MPI is not CPU-only.)

`AutoBackend` resolves to the best available local backend at plan time. Heavy backend
implementations live in extensions; this module only defines dispatch types + a few helpers.
"""

export AbstractExecutionBackend, SerialBackend, ThreadedBackend, GPUBackend, AutoBackend
export DistributedBackend, MPIBackend, local_backend, is_distributed

abstract type AbstractExecutionBackend end

"Serial single-threaded CPU compute (always available, no extension needed)."
struct SerialBackend <: AbstractExecutionBackend end

"Multithreaded CPU compute (requires `using OhMyThreads`)."
struct ThreadedBackend <: AbstractExecutionBackend end

"""
    GPUBackend{B}

GPU compute on backend object `B` (e.g. a `CUDABackend` / KernelAbstractions backend). Requires
the corresponding extension (`using CUDA`).
"""
struct GPUBackend{B} <: AbstractExecutionBackend
    backend::B
end

"Resolve to the best available local backend at plan time."
struct AutoBackend <: AbstractExecutionBackend end

"""
    DistributedBackend{Inner}

Distribute work across worker processes, each running `inner` locally. Parametric over the inner
local backend. Requires `using Distributed`.
"""
struct DistributedBackend{Inner<:AbstractExecutionBackend} <: AbstractExecutionBackend
    inner::Inner
end
DistributedBackend() = DistributedBackend(SerialBackend())   # 1-arg form is auto-generated

"""
    MPIBackend{Inner}

Distribute work across MPI ranks, each running `inner` locally. Parametric over the inner local
backend. Requires `using MPI`. Not CPU-only — `MPIBackend{GPUBackend{...}}` targets multi-GPU.
"""
struct MPIBackend{Inner<:AbstractExecutionBackend} <: AbstractExecutionBackend
    inner::Inner
end
MPIBackend() = MPIBackend(SerialBackend())   # 1-arg form is auto-generated

"""
    local_backend(backend) -> AbstractExecutionBackend

The per-process compute backend — `inner` for distribution wrappers, the backend itself otherwise.
"""
local_backend(b::AbstractExecutionBackend) = b
local_backend(b::DistributedBackend) = b.inner
local_backend(b::MPIBackend) = b.inner

"`true` if `backend` distributes work across processes."
is_distributed(::AbstractExecutionBackend) = false
is_distributed(::DistributedBackend) = true
is_distributed(::MPIBackend) = true

end # module Backends
