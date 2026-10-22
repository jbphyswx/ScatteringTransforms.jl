module ScatteringTransformsOhMyThreadsExt

# `ThreadedBackend`: tasks over the batch when there is more than one slice (task `b` reads
# `X[…, b]` and writes `out[:, b]`), and over first-order wavelet groups for a single field (group
# `j1` writes only `S1[j1]` and `S2[j1, :]`). Each task takes a `ScatteringCore.task_workspace`,
# which shares the filter bank, path tree and work list. A batch task runs the serial cascade, so the
# two levels never nest.

using OhMyThreads: OhMyThreads as OMT
using LinearAlgebra: LinearAlgebra
using ComputationalBackends: ComputationalBackends as CB
using FlowTransformBindings: FlowTransformBindings as FTB
using ScatteringTransforms: ScatteringTransforms as ST

"""
    with_serial_blas(f) -> f()

Run `f` with BLAS pinned to one thread, restoring the previous count after.

The nonuniform and spherical plans call BLAS, whose thread count is process-global. The threaded
paths already occupy every core, so they pin it; serial paths leave it to the caller.
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

# The spherical farms: BLAS on one thread, and FastTransforms on one OpenMP thread inside the tasks.
_spherical_farm(f) = Base.ScopedValues.with(() -> with_serial_blas(f), FTB.FASTTRANSFORMS_THREADS => 1)

# ---------------------------------------------------------------------------
# Batch axis
# ---------------------------------------------------------------------------

# `out` decides each column's `S0` slot: real for a real stack, complex for a complex one.
function ST._scattering_batch!(out::AbstractMatrix, ::CB.AbstractThreadedBackend,
                               st::ST.Scattering1D.ScatteringTransform1D, X::AbstractMatrix)
    TO = eltype(out)
    OMT.@tasks for b in 1:size(X, 2)
        @local begin
            ws = ST.ScatteringCore.task_workspace(st)
            coeffs = ST.batch_coeffs(st, real(TO), TO)
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

`chunk` sets the working set each task sweeps per cascade operation, from one slice (`chunk = 1`)
to the whole batch on one task (`chunk = B`); `benchmark/batch_topology.jl` sweeps it.
`fft_nthreads` defaults to `1`, since the tasks already occupy every core.
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

function ST._scattering_batch(b::CB.AbstractThreadedBackend,
                              st::ST.Scattering1D.ScatteringTransform1D, X::AbstractMatrix)
    T = real(eltype(st.filter_bank.averaging))
    nw = ST.FilterBanks.nwavelets(st.filter_bank)
    flen = ST.Coefficients.flatten_length(
        ST.Coefficients.ScatteringCoefficients1D(nw, T; compute_S2 = st.max_order >= 2))
    return ST._scattering_batch!(Matrix{ST._flat_eltype(T, X)}(undef, flen, size(X, 2)), b, st, X)
end

function ST._scattering_batch!(out::AbstractMatrix, ::CB.AbstractThreadedBackend,
                               st::ST.Scattering2D.ScatteringTransform2D, X::AbstractArray{<:Any,3})
    TO = eltype(out)
    OMT.@tasks for b in 1:size(X, 3)
        @local begin
            ws = ST.ScatteringCore.task_workspace(st)
            coeffs = ST.batch_coeffs(st, real(TO), TO)
        end
        c = ST.Scattering2D.scattering_transform2d!(coeffs, ws, view(X, :, :, b))
        ST.Coefficients.flatten2d!(view(out, :, b), c)
    end
    return out
end

function ST._scattering_batch(b::CB.AbstractThreadedBackend,
                              st::ST.Scattering2D.ScatteringTransform2D, X::AbstractArray{<:Any,3})
    T = real(eltype(st.filter_bank.averaging))
    flen = ST.Coefficients.flatten_length(
        ST.Coefficients.ScatteringCoefficients2D(st.filter_bank.J, st.filter_bank.L, T;
                                                 compute_S2 = st.max_order >= 2))
    return ST._scattering_batch!(Matrix{ST._flat_eltype(T, X)}(undef, flen, size(X, 3)), b, st, X)
end

function ST._scattering_batch!(out::AbstractMatrix, ::CB.AbstractThreadedBackend,
                               st::ST.Scattering3D.ScatteringTransform3D, X::AbstractArray{<:Any,4})
    TO = eltype(out)
    OMT.@tasks for b in 1:size(X, 4)
        @local begin
            ws = ST.ScatteringCore.task_workspace(st)
            coeffs = ST.batch_coeffs(st, real(TO), TO)
        end
        c = ST.Scattering3D.scattering_transform3d!(coeffs, ws, view(X, :, :, :, b))
        ST.Coefficients.flatten2d!(view(out, :, b), c)
    end
    return out
end

function ST._scattering_batch(b::CB.AbstractThreadedBackend,
                              st::ST.Scattering3D.ScatteringTransform3D, X::AbstractArray{<:Any,4})
    T = real(eltype(st.filter_bank.averaging))
    flen = ST.Coefficients.flatten_length(
        ST.Coefficients.ScatteringCoefficients2D(st.filter_bank.J, st.filter_bank.n_orient, T;
                                                 compute_S2 = st.max_order >= 2))
    return ST._scattering_batch!(Matrix{ST._flat_eltype(T, X)}(undef, flen, size(X, 4)), b, st, X)
end

# ---------------------------------------------------------------------------
# The nonuniform and spherical surfaces
#
# Same batch axis, different per-task state: these transforms keep their scratch on the transform
# (nonuniform) or in a separate workspace object (spherical), so each task takes its own and the
# shared plan, filter bank and point set stay read-only.
#
# The per-task plans own C library plans. They are built before the tasks start, one per column
# chunk, and closed in `finally`: an unconverged solve throws an exception a caller can recover from.
# ---------------------------------------------------------------------------

# Partition 1:n into ≤ k contiguous column ranges.
_column_chunks(n::Int, k::Int) =
    (k = max(1, min(k, n)); [(div((c - 1) * n, k) + 1):(div(c * n, k)) for c in 1:k])

# `f(items)` for `items = [make(1), …, make(n)]`, after which `close!` runs on every item built,
# whether `f` or a later `make` returned or threw.
function _with_task_items(f, make, close!, n::Int)
    items = [make(1)]
    try
        for i in 2:n
            push!(items, make(i))
        end
        return f(items)
    finally
        foreach(close!, items)
    end
end

_close_task_plan!(ws, st) = (ws.plan === st.plan || ST.Plans.close_plan!(ws.plan); nothing)

function ST._scattering_batch!(out::AbstractMatrix, ::CB.AbstractThreadedBackend,
                               st::ST.ScatteredPlanar.ScatteredPlanarScattering, X::AbstractMatrix)
    TO = eltype(out)
    W = ST.Plans.batch_width(st.plan)
    W == 1 && return _planar_threaded_perfield!(out, st, X)

    # A batched transform's buffers and plan are `W` wide, so each task transforms its groups of `W`
    # columns, one execution per cascade step per group, the last group padded with zero columns.
    B = size(X, 2)
    B == 0 && return out
    ngroups = cld(B, W)
    chunks = _column_chunks(ngroups, Threads.nthreads())
    _with_task_items(_ -> ST.ScatteringCore.task_workspace(st), ws -> _close_task_plan!(ws, st),
                     length(chunks)) do wss
        with_serial_blas() do
            OMT.@tasks for c in eachindex(chunks)
                ws = wss[c]
                coeffs = ST.batch_coeffs(ws, real(TO), TO)
                stage = ST._PlanarStage(ws, TO, eltype(X))
                for g in chunks[c]
                    ST._planar_group!(out, ws, X, ((g - 1) * W + 1):min(g * W, B), stage, coeffs)
                end
            end
        end
    end
    return out
end

function _planar_threaded_perfield!(out::AbstractMatrix,
                                    st::ST.ScatteredPlanar.ScatteredPlanarScattering,
                                    X::AbstractMatrix)
    TO = eltype(out)
    size(X, 2) == 0 && return out
    chunks = _column_chunks(size(X, 2), Threads.nthreads())
    _with_task_items(_ -> ST.ScatteringCore.task_workspace(st), ws -> _close_task_plan!(ws, st),
                     length(chunks)) do wss
        with_serial_blas() do
            OMT.@tasks for c in eachindex(chunks)
                ws = wss[c]
                coeffs = ST.batch_coeffs(st, real(TO), TO)
                for b in chunks[c]
                    r = ST.ScatteredPlanar.scattered_planar_scattering!(coeffs, ws, view(X, :, b))
                    ST.Coefficients.flatten2d!(view(out, :, b), r)
                end
            end
        end
    end
    return out
end

for (TT, WS, fun) in (
        (:SphericalScattering, :SphericalWorkspace, :spherical_scattering!),
        (:SphericalMonogenicScattering, :SphericalMonogenicWorkspace, :spherical_monogenic_scattering!))
    @eval function _spherical_threaded_perfield!(out::AbstractMatrix,
                                                 st::ST.SphericalCore.$TT{T},
                                                 X::AbstractArray) where {T}
        D = ndims(X)
        size(X, D) == 0 && return out
        chunks = _column_chunks(size(X, D), Threads.nthreads())
        # The plan carries the analysis scratch `sphere_coeffs!` writes through, so each task takes
        # its own copy of the transform.
        _with_task_items(_ -> ST.SphericalCore.task_local(st),
                         stl -> ST.SphericalCore.close_task_local!(stl, st), length(chunks)) do stls
            _spherical_farm() do
                OMT.@tasks for c in eachindex(chunks)
                    stl = stls[c]
                    ws = ST.SphericalCore.$WS(stl, selectdim(X, D, 1))
                    S1 = zeros(T, st.J)
                    S2 = st.max_order >= 2 ? zeros(T, st.J, st.J) : Matrix{T}(undef, 0, 0)
                    for b in chunks[c]
                        r = ST.SphericalCore.$fun(S1, S2, stl, ws, selectdim(X, D, b))
                        ST._flatten_spherical!(view(out, :, b), r.S0, r.S1, r.S2, st.J)
                    end
                end
            end
        end
        return out
    end
end

# The monogenic amplitude is built from per-field spin-0 transforms, so its only parallel axis is the
# batch, one field per task.
ST._scattering_batch!(out::AbstractMatrix, ::CB.AbstractThreadedBackend,
                      st::ST.SphericalCore.SphericalMonogenicScattering, X::AbstractArray) =
    _spherical_threaded_perfield!(out, st, X)

# One task per column chunk, each running one transform per cascade step over its whole chunk. The
# cascade solves through its plan's scratch, so every chunk takes a task-local plan, including a
# chunk at the shared plan's own width.
function ST._scattering_batch!(out::AbstractMatrix, ::CB.AbstractThreadedBackend,
                               st::ST.SphericalCore.SphericalScattering{T},
                               X::AbstractArray) where {T}
    ST._spherical_batches(st.plan, X) || return _spherical_threaded_perfield!(out, st, X)
    B = size(X, 2)
    chunks = _column_chunks(B, Threads.nthreads())
    _with_task_items(i -> ST.SphericalCore.task_local_batch_plan(st.plan, length(chunks[i])),
                     p -> (p === st.plan || ST.Plans.close_plan!(p)), length(chunks)) do plans
        _spherical_farm() do
            OMT.@tasks for i in eachindex(chunks)
                cols = chunks[i]
                k = length(cols)
                Xc = view(X, :, cols)
                stb = ST.SphericalCore.SphericalScattering(st.lmax, st.J, st.max_order, plans[i],
                                                           st.sigma2)
                ws = ST.SphericalCore.SphericalWorkspace(stb, Xc)
                S1 = zeros(T, st.J, k)
                S2 = st.max_order >= 2 ? zeros(T, st.J, st.J, k) : Array{T, 3}(undef, 0, 0, 0)
                r = ST.SphericalCore.spherical_scattering_batch!(S1, S2, stb, ws, Xc)
                empty2 = Matrix{T}(undef, 0, 0)
                for (j, b) in enumerate(cols)
                    S2b = isempty(r.S2) ? empty2 : view(r.S2, :, :, j)
                    ST._flatten_spherical!(view(out, :, b), r.S0[j], view(r.S1, :, j), S2b, st.J)
                end
            end
        end
    end
    return out
end

ST._scattering_batch(b::CB.AbstractThreadedBackend,
                     st::ST.ScatteredPlanar.ScatteredPlanarScattering, X::AbstractMatrix) =
    ST._scattering_batch!(Matrix{ST._flat_eltype(real(eltype(st.filter_bank.averaging)), X)}(
                              undef, ST.flat_rows(st), size(X, 2)), b, st, X)

for TT in (:SphericalScattering, :SphericalMonogenicScattering)
    @eval ST._scattering_batch(b::CB.AbstractThreadedBackend, st::ST.SphericalCore.$TT{T},
                               X::AbstractArray) where {T} =
        ST._scattering_batch!(Matrix{T}(undef, ST.flat_rows(st), size(X)[end]), b, st, X)
end

# ---------------------------------------------------------------------------
# First-order-wavelet axis — for a single field, where there is no batch to spread over
# ---------------------------------------------------------------------------

# Group `j1` writes only `S1[j1]` and the row `S2[j1, :]`, so groups are independent and the result
# is identical to the serial cascade. `PathGraph.order2_groups` orders them longest-first, which
# balances the dynamic scheduler.
for (Mod, TT, FT) in ((:Scattering1D, :ScatteringTransform1D, :AbstractVector),
                      (:Scattering2D, :ScatteringTransform2D, :AbstractMatrix),
                      (:Scattering3D, :ScatteringTransform3D, :(AbstractArray{<:Any,3})))
    @eval function ST.$Mod.cascade!(S1::AbstractVector, S2::AbstractMatrix,
                                    ::CB.AbstractThreadedBackend, st::ST.$Mod.$TT,
                                    signal_fft::$FT)
        isempty(S2) || fill!(S2, zero(eltype(S2)))
        OMT.@tasks for g in st.groups
            @set scheduler = OMT.DynamicScheduler()
            # Each task takes its own workspace: the periodized cascade's buffers are mutable, and
            # a computed bank evaluates its filters into shared scratch.
            @local ws = ST.ScatteringCore.task_workspace(st)
            ST.Cascade.group!(S1, S2, ws.pw, ws.filter_bank, g, signal_fft)
        end
        return S1, S2
    end
end

end # module ScatteringTransformsOhMyThreadsExt
