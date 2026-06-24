module Backends

"""
    Backends.jl — Compute-backend taxonomy

Two orthogonal concerns:

- **Local compute backend** — what a single process/rank computes on:
  `SerialCPU`, `ThreadedCPU` (OhMyThreads ext), `GPUBackend{B}` (CUDA / KernelAbstractions ext).
- **Distribution wrapper** — how work is split across processes, *parametric over the inner
  local backend*: `Distributed{Inner}` (Distributed ext), `MPI{Inner}` (MPI ext). MPI is not
  CPU-only: with CUDA-aware MPI, `MPI{GPUBackend{CUDABackend}}` is a multi-GPU cluster. The
  wrapper owns only "partition the batch / gather results"; `inner` owns the scatter compute.

`AutoBackend` resolves to the best available local backend at plan time.

Heavy backend implementations live in extensions; this module only defines the dispatch types
and a couple of helpers, so the core has no parallel/GPU dependencies.
"""

export AbstractBackend, SerialCPU, ThreadedCPU, GPUBackend, AutoBackend
export Distributed, MPI, local_backend, is_distributed

abstract type AbstractBackend end

"Serial single-threaded CPU compute (always available, no extension needed)."
struct SerialCPU <: AbstractBackend end

"Multithreaded CPU compute (requires `using OhMyThreads`)."
struct ThreadedCPU <: AbstractBackend end

"""
    GPUBackend{B}

GPU compute on backend object `B` (e.g. a `CUDABackend` / KernelAbstractions backend). Requires
the corresponding extension (`using CUDA`).
"""
struct GPUBackend{B} <: AbstractBackend
    backend::B
end

"Resolve to the best available local backend at plan time."
struct AutoBackend <: AbstractBackend end

"""
    Distributed{Inner}

Distribute the batch dimension across worker processes, each running `inner` locally. Requires
`using Distributed`.
"""
struct Distributed{Inner<:AbstractBackend} <: AbstractBackend
    inner::Inner
end
Distributed() = Distributed(SerialCPU())   # 1-arg form is the auto-generated constructor

"""
    MPI{Inner}

Distribute the batch dimension across MPI ranks, each running `inner` locally. Requires
`using MPI`. Not CPU-only — `MPI{GPUBackend{...}}` targets multi-GPU clusters.
"""
struct MPI{Inner<:AbstractBackend} <: AbstractBackend
    inner::Inner
end
MPI() = MPI(SerialCPU())   # 1-arg form is the auto-generated constructor

"""
    local_backend(backend) -> AbstractBackend

The per-process compute backend. For distribution wrappers this is the wrapped `inner`; for a
plain local backend it is the backend itself.
"""
local_backend(b::AbstractBackend) = b
local_backend(b::Distributed) = b.inner
local_backend(b::MPI) = b.inner

"`true` if `backend` distributes work across processes."
is_distributed(::AbstractBackend) = false
is_distributed(::Distributed) = true
is_distributed(::MPI) = true

end # module Backends
