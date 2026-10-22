module ScatteringTransformsMPIExt

# SPMD `scattering_batch` on `MPIBackend`: every rank holds its own `st` and the whole batch `X`,
# transforms its contiguous column block with `local_backend(b)`, and `MPI.Allgatherv!` gives every
# rank the full `(flatten_length, B)` matrix. Requires `MPI.Init()`; the inner backend may be a
# `GPUBackend` with CUDA-aware MPI.

using MPI: MPI
using ComputationalBackends: ComputationalBackends as CB
using ScatteringTransforms: ScatteringTransforms as ST

# Contiguous, in-order block of columns assigned to `rank` (0-based) of `nranks` (may be empty).
_rank_block(ncols::Int, nranks::Int, rank::Int) =
    (div(rank * ncols, nranks) + 1):(div((rank + 1) * ncols, nranks))

function _mpi_batch(b::CB.AbstractMPIBackend, st, X, slicer)
    MPI.Initialized() ||
        throw(ArgumentError("MPI is not initialized — call `MPI.Init()` before scattering_batch(…; execution = MPIBackend())."))
    comm = b.comm === nothing ? MPI.COMM_WORLD : b.comm
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)
    ncols = size(X)[end]
    inner = CB.local_backend(b)

    # Each rank computes its own contiguous column block (blocks tile 1:ncols in rank order).
    localmat = ST.scattering_batch(st, slicer(X, _rank_block(ncols, nranks, rank)); execution = inner)
    flen = size(localmat, 1)

    # Per-rank element counts are known to all ranks (same flen, deterministic blocks).
    counts = Cint[flen * length(_rank_block(ncols, nranks, r)) for r in 0:(nranks - 1)]
    recv = Vector{eltype(localmat)}(undef, sum(counts))
    MPI.Allgatherv!(vec(localmat), MPI.VBuffer(recv, counts), comm)
    # Column-major concatenation in rank order == columns 1:ncols in order.
    return reshape(recv, flen, ncols)
end

ST._scattering_batch(b::CB.AbstractMPIBackend, st::ST.Scattering1D.ScatteringTransform1D, X::AbstractMatrix) =
    _mpi_batch(b, st, X, (A, cols) -> view(A, :, cols))

ST._scattering_batch(b::CB.AbstractMPIBackend, st::ST.Scattering2D.ScatteringTransform2D, X::AbstractArray{<:Any,3}) =
    _mpi_batch(b, st, X, (A, cols) -> view(A, :, :, cols))

ST._scattering_batch(b::CB.AbstractMPIBackend, st::ST.Scattering3D.ScatteringTransform3D, X::AbstractArray{<:Any,4}) =
    _mpi_batch(b, st, X, (A, cols) -> view(A, :, :, :, cols))

# Every rank holds its own transform, so the nonuniform and spherical surfaces differ only in their
# batch axis.
ST._scattering_batch(b::CB.AbstractMPIBackend,
                     st::ST.ScatteredPlanar.ScatteredPlanarScattering, X::AbstractMatrix) =
    _mpi_batch(b, st, X, (A, cols) -> view(A, :, cols))

ST._scattering_batch(b::CB.AbstractMPIBackend, st::ST.SphericalCore.SphericalScattering,
                     X::AbstractArray) =
    _mpi_batch(b, st, X, (A, cols) -> selectdim(A, ndims(A), cols))

ST._scattering_batch(b::CB.AbstractMPIBackend, st::ST.SphericalCore.SphericalMonogenicScattering,
                     X::AbstractArray) =
    _mpi_batch(b, st, X, (A, cols) -> selectdim(A, ndims(A), cols))

end # module ScatteringTransformsMPIExt
