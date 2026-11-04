module ScatteringTransformsMPIExt

# SPMD `scattering_batch` on `MPIBackend`: every rank holds its own `st` and the whole batch `X`,
# transforms its contiguous column block with `local_backend(b)`, and `MPI.Allgatherv!` gives every
# rank the full `(flatten_length, B)` matrix. Requires `MPI.Init()`; the inner backend may be a
# `GPUBackend` with CUDA-aware MPI.

using MPI: MPI
using ComputationalBackends: ComputationalBackends as CB
using FlowTransformBindings: FlowTransformBindings as FTB
using ScatteringTransforms: ScatteringTransforms as ST

ST.Execution.available(::CB.AbstractMPIBackend) = true

# Contiguous, in-order block of columns assigned to `rank` (0-based) of `nranks` (may be empty).
_rank_block(ncols::Int, nranks::Int, rank::Int) =
    (div(rank * ncols, nranks) + 1):(div((rank + 1) * ncols, nranks))

function _mpi_batch(b::CB.AbstractMPIBackend, st, X)
    MPI.Initialized() ||
        throw(ArgumentError("MPI is not initialized — call `MPI.Init()` before scattering_batch(…; execution = MPIBackend())."))
    comm = b.comm === nothing ? MPI.COMM_WORLD : b.comm
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)
    ncols = size(X)[end]
    inner = CB.local_backend(b)

    # Each rank computes its own contiguous column block (blocks tile 1:ncols in rank order).
    part = ST.scattering_batch(st, ST._batch_column(X, _rank_block(ncols, nranks, rank)); execution = inner)
    localmat = ST._coefficients(part)
    flen = size(localmat, 1)

    # Per-rank element counts are known to all ranks (same flen, deterministic blocks).
    counts = Cint[flen * length(_rank_block(ncols, nranks, r)) for r in 0:(nranks - 1)]
    recv = Vector{eltype(localmat)}(undef, sum(counts))
    MPI.Allgatherv!(vec(localmat), MPI.VBuffer(recv, counts), comm)
    # Column-major concatenation in rank order == columns 1:ncols in order.
    return ST.Plans._with_fit(reshape(recv, flen, ncols), _gather_fits(_fit_part(part), ncols, nranks, comm))
end

_fit_part(::AbstractMatrix) = nothing
_fit_part(part::NamedTuple) = part.fit

# The fit outcomes of every rank's block, in column order: each travels as `converged`, `iterations` and
# `residual` in the residual's type.
_gather_fits(::Nothing, ncols, nranks, comm) = nothing
function _gather_fits(fit::AbstractVector{FTB.SolverResult{T}}, ncols::Int, nranks::Int, comm) where {T}
    send = Vector{T}(undef, 3 * length(fit))
    for (i, s) in enumerate(fit)
        send[3i - 2], send[3i - 1], send[3i] = T(s.converged), T(s.iterations), s.residual
    end
    counts = Cint[3 * length(_rank_block(ncols, nranks, r)) for r in 0:(nranks - 1)]
    recv = Vector{T}(undef, sum(counts))
    MPI.Allgatherv!(send, MPI.VBuffer(recv, counts), comm)
    return [FTB.SolverResult{T}(!iszero(recv[3i - 2]), Int(recv[3i - 1]), recv[3i]) for i in 1:ncols]
end

ST._scattering_batch(b::CB.AbstractMPIBackend,
                     st::ST.ScatteringCore.AbstractScatteringTransform, X::AbstractArray) =
    _mpi_batch(b, st, X)

end # module ScatteringTransformsMPIExt
