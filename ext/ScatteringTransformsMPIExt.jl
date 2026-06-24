module ScatteringTransformsMPIExt

"""
    ScatteringTransformsMPIExt — MPI batched transforms (MPIBackend)

SPMD `scattering_batch`: every rank holds its own transform `st` (built locally, so no plan
serialization is needed) and the batch `X`, computes the column block assigned to its rank with
the inner local backend (`local_backend(b)`), and the per-rank column blocks are combined with
`MPI.Allgatherv!` so every rank returns the full `(flatten_length, B)` matrix.

Load `using MPI` and `MPI.Init()` before use; run under `mpiexec`. Not CPU-only — the inner
backend may be `GPUBackend{…}` for multi-GPU with CUDA-aware MPI.
"""

using MPI: MPI
using ScatteringTransforms: ScatteringTransforms

const ST = ScatteringTransforms
const MPIBackend = ST.Backends.MPIBackend

# Contiguous, in-order block of columns assigned to `rank` (0-based) of `nranks` (may be empty).
_rank_block(ncols::Int, nranks::Int, rank::Int) =
    (div(rank * ncols, nranks) + 1):(div((rank + 1) * ncols, nranks))

function _mpi_batch(b::MPIBackend, st, X, slicer)
    MPI.Initialized() ||
        throw(ArgumentError("MPI is not initialized — call `MPI.Init()` before scattering_batch(MPIBackend(), …)."))
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)
    ncols = size(X)[end]
    inner = ST.Backends.local_backend(b)

    # Each rank computes its own contiguous column block (blocks tile 1:ncols in rank order).
    localmat = ST.scattering_batch(inner, st, slicer(X, _rank_block(ncols, nranks, rank)))
    flen = size(localmat, 1)

    # Per-rank element counts are known to all ranks (same flen, deterministic blocks).
    counts = Cint[flen * length(_rank_block(ncols, nranks, r)) for r in 0:(nranks - 1)]
    recv = Vector{eltype(localmat)}(undef, sum(counts))
    MPI.Allgatherv!(vec(localmat), MPI.VBuffer(recv, counts), comm)
    # Column-major concatenation in rank order == columns 1:ncols in order.
    return reshape(recv, flen, ncols)
end

ST.scattering_batch(b::MPIBackend, st::ST.ScatteringTransform1D, X::AbstractMatrix) =
    _mpi_batch(b, st, X, (A, cols) -> A[:, cols])

ST.scattering_batch(b::MPIBackend, st::ST.ScatteringTransform2D, X::AbstractArray{<:Any,3}) =
    _mpi_batch(b, st, X, (A, cols) -> A[:, :, cols])

end # module ScatteringTransformsMPIExt
