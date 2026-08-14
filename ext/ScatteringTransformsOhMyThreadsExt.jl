module ScatteringTransformsOhMyThreadsExt

"""
    ScatteringTransformsOhMyThreadsExt — multithreaded transforms (ThreadedBackend)

Parallelism is taken on the outermost race-free axis that has enough independent work:

  * **batch** when there is more than one slice — task `b` reads `X[…, b]` and writes `out[:, b]`,
    both exclusive;
  * **first-order wavelet** for a single field — group `j1` writes only `S1[j1]` and `S2[j1, :]`,
    also exclusive, so one large image still uses every thread.

Each task takes one workspace from `ScatteringCore.task_workspace`, which shares the read-only
filter bank, path tree and work list and copies only the buffers. The filter bank is the bulk of a
transform (33 MiB of 38 MiB for a 256×256, J=4, L=8 transform), so per-task cost is `O(N)`, not
`O(nw·N)`.

The two levels never nest: a batch task runs the serial `cascade!`, so a slice is never subdivided
again while `nthreads` tasks are already in flight.
"""

using OhMyThreads: OhMyThreads as OMT
using LinearAlgebra: LinearAlgebra
using ComputationalBackends: ComputationalBackends as CB
using ScatteringTransforms: ScatteringTransforms as ST

"""
    with_serial_blas(f) -> f()

Run `f` with BLAS pinned to one thread, restoring the previous count after.

The nonuniform and spherical plans call BLAS, whose thread count is process-global. These tasks
already occupy every core, so leaving it alone oversubscribes: a scattered-planar batch on 8 threads
runs at `0.34x` of serial with BLAS at 4, and `2.1x` with it pinned. Serial paths deliberately do
*not* pin — there BLAS threading is worth up to `1.3x` at large point counts.
"""
function with_serial_blas(f)
    prev = LinearAlgebra.BLAS.get_num_threads()
    try
        LinearAlgebra.BLAS.set_num_threads(1)
        return f()
    finally
        LinearAlgebra.BLAS.set_num_threads(prev)
    end
end

# ---------------------------------------------------------------------------
# Batch axis
# ---------------------------------------------------------------------------

function ST.scattering_batch!(out::AbstractMatrix, ::CB.AbstractThreadedBackend,
                              st::ST.Scattering1D.ScatteringTransform1D, X::AbstractMatrix)
    T = real(eltype(st.filter_bank.averaging))
    OMT.@tasks for b in 1:size(X, 2)
        @local begin
            ws = ST.ScatteringCore.task_workspace(st)
            coeffs = ST.batch_coeffs(st, T)
        end
        c = ST.Scattering1D.scattering_transform!(coeffs, ws, view(X, :, b))
        ST.Coefficients.flatten1d!(view(out, :, b), c)
    end
    return out
end

# ---------------------------------------------------------------------------
# Chunked batching: threads own chunks of the batch, each chunk transformed by one batched plan
# ---------------------------------------------------------------------------

"""
    chunked_batch!(out, st, X; chunk, fft_nthreads = 1) -> out

Thread over contiguous chunks of the batch, transforming each chunk with a single batched plan.

`chunk = 1` degenerates to the per-slice threaded path; `chunk = B` to one fully batched plan on one
task. In between, the chunk sets the working-set size each task sweeps per operation — which is the
quantity that decides whether the cascade's `O(nw + paths)` operations reuse cache or stream memory.
Sweep it for a given transform rather than assuming; `benchmark/batch_topology.jl` does that.

`fft_nthreads` stays 1 by default: the tasks already occupy every core, so letting the FFT library
thread underneath them would oversubscribe.
"""
function chunked_batch!(out::AbstractMatrix, st, X::AbstractArray; chunk::Int,
                        fft_nthreads::Int = 1)
    B = size(X)[end]
    chunk >= 1 || throw(ArgumentError("chunk must be positive, got $chunk"))
    chunk = min(chunk, B)
    nfull = div(B, chunk)
    slicer(A, c) = ndims(A) == 2 ? view(A, :, c) :
                   ndims(A) == 3 ? view(A, :, :, c) : view(A, :, :, :, c)
    OMT.@tasks for k in 1:nfull
        @local ws = ST.batch_workspace(st, chunk; fft_nthreads = fft_nthreads)
        cols = ((k - 1) * chunk + 1):(k * chunk)
        ST.Batched.batch_cascade!(view(out, :, cols), ws, slicer(X, cols))
    end
    # Whatever does not divide evenly goes through one last plan of the remainder's size.
    rest = (nfull * chunk + 1):B
    if !isempty(rest)
        ws = ST.batch_workspace(st, length(rest); fft_nthreads = fft_nthreads)
        ST.Batched.batch_cascade!(view(out, :, rest), ws, slicer(X, rest))
    end
    return out
end

function ST.scattering_batch(b::CB.AbstractThreadedBackend,
                             st::ST.Scattering1D.ScatteringTransform1D, X::AbstractMatrix)
    T = real(eltype(st.filter_bank.averaging))
    nw = length(st.filter_bank.wavelets)
    flen = ST.Coefficients.flatten_length(
        ST.Coefficients.ScatteringCoefficients1D(nw, T; compute_S2 = st.max_order >= 2))
    return ST.scattering_batch!(Matrix{T}(undef, flen, size(X, 2)), b, st, X)
end

function ST.scattering_batch!(out::AbstractMatrix, ::CB.AbstractThreadedBackend,
                              st::ST.Scattering2D.ScatteringTransform2D, X::AbstractArray{<:Any,3})
    T = real(eltype(st.filter_bank.averaging))
    OMT.@tasks for b in 1:size(X, 3)
        @local begin
            ws = ST.ScatteringCore.task_workspace(st)
            coeffs = ST.batch_coeffs(st, T)
        end
        c = ST.Scattering2D.scattering_transform2d!(coeffs, ws, view(X, :, :, b))
        ST.Coefficients.flatten2d!(view(out, :, b), c)
    end
    return out
end

function ST.scattering_batch(b::CB.AbstractThreadedBackend,
                             st::ST.Scattering2D.ScatteringTransform2D, X::AbstractArray{<:Any,3})
    T = real(eltype(st.filter_bank.averaging))
    flen = ST.Coefficients.flatten_length(
        ST.Coefficients.ScatteringCoefficients2D(st.filter_bank.J, st.filter_bank.L, T;
                                                 compute_S2 = st.max_order >= 2))
    return ST.scattering_batch!(Matrix{T}(undef, flen, size(X, 3)), b, st, X)
end

function ST.scattering_batch!(out::AbstractMatrix, ::CB.AbstractThreadedBackend,
                              st::ST.Scattering3D.ScatteringTransform3D, X::AbstractArray{<:Any,4})
    T = real(eltype(st.filter_bank.averaging))
    OMT.@tasks for b in 1:size(X, 4)
        @local begin
            ws = ST.ScatteringCore.task_workspace(st)
            coeffs = ST.batch_coeffs(st, T)
        end
        c = ST.Scattering3D.scattering_transform3d!(coeffs, ws, view(X, :, :, :, b))
        ST.Coefficients.flatten2d!(view(out, :, b), c)
    end
    return out
end

function ST.scattering_batch(b::CB.AbstractThreadedBackend,
                             st::ST.Scattering3D.ScatteringTransform3D, X::AbstractArray{<:Any,4})
    T = real(eltype(st.filter_bank.averaging))
    flen = ST.Coefficients.flatten_length(
        ST.Coefficients.ScatteringCoefficients2D(st.filter_bank.J, st.filter_bank.n_orient, T;
                                                 compute_S2 = st.max_order >= 2))
    return ST.scattering_batch!(Matrix{T}(undef, flen, size(X, 4)), b, st, X)
end

# ---------------------------------------------------------------------------
# The nonuniform and spherical surfaces
#
# Same batch axis, different per-task state: these transforms keep their scratch on the transform
# (nonuniform) or in a separate workspace object (spherical), so each task takes its own and the
# shared plan, filter bank and point set stay read-only.
# ---------------------------------------------------------------------------

function ST.scattering_batch!(out::AbstractMatrix, ::CB.AbstractThreadedBackend,
                              st::ST.SubsampledScattering.MultiResolutionScattering, X::AbstractArray)
    T = eltype(out)
    D = ndims(X)
    OMT.@tasks for b in 1:size(X, D)
        @local begin
            ws = ST.ScatteringCore.task_workspace(st)
            coeffs = ST.batch_coeffs(st, T)
        end
        c = ST.SubsampledScattering.subsampled_scattering!(coeffs, ws, selectdim(X, D, b))
        ST._flatten_into!(view(out, :, b), c)
    end
    return out
end

function ST.scattering_batch!(out::AbstractMatrix, ::CB.AbstractThreadedBackend,
                              st::ST.ScatteredPlanar.ScatteredPlanarScattering, X::AbstractMatrix)
    T = eltype(out)
    with_serial_blas() do
        OMT.@tasks for b in 1:size(X, 2)
            @local begin
                ws = ST.ScatteringCore.task_workspace(st)
                coeffs = ST.batch_coeffs(st, T)
            end
            c = ST.ScatteredPlanar.scattered_planar_scattering!(coeffs, ws, view(X, :, b))
            ST.Coefficients.flatten2d!(view(out, :, b), c)
        end
    end
    return out
end

for (TT, WS, fun) in (
        (:SphericalScattering, :SphericalWorkspace, :spherical_scattering!),
        (:SphericalMonogenicScattering, :SphericalMonogenicWorkspace, :spherical_monogenic_scattering!))
    @eval function ST.scattering_batch!(out::AbstractMatrix, ::CB.AbstractThreadedBackend,
                                        st::ST.SphericalCore.$TT{T}, X::AbstractArray) where {T}
        D = ndims(X)
        with_serial_blas() do
            OMT.@tasks for b in 1:size(X, D)
                # One task-local bundle, not four `@local` bindings: the workspace is built *from*
                # the task's own transform, and separate bindings cannot refer to one another. The
                # plan carries the analysis scratch `sphere_coeffs!` writes through, so it is copied
                # too.
                @local task = let stl = ST.SphericalCore.task_local(st)
                    (st = stl, ws = ST.SphericalCore.$WS(stl, selectdim(X, D, 1)),
                     S1 = zeros(T, st.J),
                     S2 = st.max_order >= 2 ? zeros(T, st.J, st.J) : Matrix{T}(undef, 0, 0))
                end
                r = ST.SphericalCore.$fun(task.S1, task.S2, task.st, task.ws, selectdim(X, D, b))
                ST._flatten_spherical!(view(out, :, b), r.S0, r.S1, r.S2, st.J)
            end
        end
        return out
    end
end

ST.scattering_batch(b::CB.AbstractThreadedBackend,
                    st::ST.SubsampledScattering.MultiResolutionScattering{T}, X::AbstractArray) where {T} =
    ST.scattering_batch!(Matrix{T}(undef, ST.flat_rows(st), size(X)[end]), b, st, X)

ST.scattering_batch(b::CB.AbstractThreadedBackend,
                    st::ST.ScatteredPlanar.ScatteredPlanarScattering, X::AbstractMatrix) =
    ST.scattering_batch!(Matrix{real(eltype(st.filter_bank.averaging))}(undef, ST.flat_rows(st),
                                                                       size(X, 2)), b, st, X)

for TT in (:SphericalScattering, :SphericalMonogenicScattering)
    @eval ST.scattering_batch(b::CB.AbstractThreadedBackend, st::ST.SphericalCore.$TT{T},
                              X::AbstractArray) where {T} =
        ST.scattering_batch!(Matrix{T}(undef, ST.flat_rows(st), size(X)[end]), b, st, X)
end

# ---------------------------------------------------------------------------
# First-order-wavelet axis — for a single field, where there is no batch to spread over
# ---------------------------------------------------------------------------

# Group `j1` writes only `S1[j1]` and the row `S2[j1, :]`, so groups are independent and the result
# is identical to the serial cascade. `PathGraph.order2_groups` already orders them longest-first,
# which is what keeps the dynamic scheduler balanced: the child count varies ~3× across scales.
for (Mod, TT, FT) in ((:Scattering1D, :ScatteringTransform1D, :AbstractVector),
                      (:Scattering2D, :ScatteringTransform2D, :AbstractMatrix),
                      (:Scattering3D, :ScatteringTransform3D, :(AbstractArray{<:Any,3})))
    @eval function ST.$Mod.cascade!(S1::AbstractVector, S2::AbstractMatrix,
                                    ::CB.AbstractThreadedBackend, st::ST.$Mod.$TT,
                                    signal_fft::$FT)
        isempty(S2) || fill!(S2, zero(eltype(S2)))
        wavelets = st.filter_bank.wavelets
        OMT.@tasks for g in st.groups
            @set scheduler = OMT.DynamicScheduler()
            @local ws = ST.ScatteringCore.task_workspace(st)
            j1, children = g[1], g[2]
            ST.ScatteringCore.wavelet_convolve!(ws.buffer_conv, signal_fft, wavelets[j1],
                                                ws.plan, ws.buffer_input)
            if isempty(children)
                S1[j1] = ST.ScatteringCore.modulus_mean(ws.buffer_conv)
            else
                S1[j1] = ST.ScatteringCore.modulus_mean!(ws.buffer_u1, ws.buffer_conv)
                ws.buffer_input .= complex.(ws.buffer_u1)
                ST.Plans.forward_transform!(ws.buffer_u1_fft, ws.plan, ws.buffer_input)
                for j2 in children
                    ST.ScatteringCore.wavelet_convolve!(ws.buffer_conv, ws.buffer_u1_fft,
                                                        wavelets[j2], ws.plan, ws.buffer_input)
                    S2[j1, j2] = ST.ScatteringCore.modulus_mean(ws.buffer_conv)
                end
            end
        end
        return S1, S2
    end
end

end # module ScatteringTransformsOhMyThreadsExt
