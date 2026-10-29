module ScatteringTransformsOhMyThreadsExt

using OhMyThreads: OhMyThreads as OMT
using LinearAlgebra: LinearAlgebra
using ComputationalBackends: ComputationalBackends as CB
using FlowTransformBindings: FlowTransformBindings as FTB
using ScatteringTransforms: ScatteringTransforms as ST

ST.Execution.available(::CB.AbstractThreadedBackend) = true

function with_serial_blas(f)
    prev = LinearAlgebra.BLAS.get_num_threads()
    try
        LinearAlgebra.BLAS.set_num_threads(1)
        return f()
    finally
        LinearAlgebra.BLAS.set_num_threads(prev)
    end
end

_spherical_farm(f) = Base.ScopedValues.with(() -> with_serial_blas(f), FTB.FASTTRANSFORMS_THREADS => 1)
_farm(f, st) = with_serial_blas(f)
_farm(f, st::Union{ST.SphericalCore.SphericalScatteringTransform,ST.SphericalCore.SphericalMonogenicScatteringTransform}) =
    _spherical_farm(f)

_columns(X, cols) = selectdim(X, ndims(X), cols)

function ST._scattering_batch(b::CB.AbstractThreadedBackend,
                               st::ST.ScatteringCore.AbstractScatteringTransform, X::AbstractArray)
    T = ST._flat_eltype(ST._coefficient_eltype(st), X)
    out = Matrix{T}(undef, ST.flat_rows(st), size(X)[end])
    return ST._scattering_batch!(out, b, st, X)
end

function _run_columns!(out, ws, X, cols)
    coeffs = ST.batch_coeffs(ws, real(eltype(out)), eltype(out))
    for b in cols
        c = ST._single_scattering!(coeffs, ws, selectdim(X, ndims(X), b))
        ST._flatten_into!(view(out, :, b), c)
    end
    return nothing
end

function ST._scattering_batch!(out::AbstractMatrix, ::CB.AbstractThreadedBackend,
                               st::ST.ScatteringCore.AbstractScatteringTransform, X::AbstractArray)
    B = size(X)[end]
    B == 0 && return out
    n = min(B, Threads.nthreads())
    ST.ScatteringCore.with_task_workspaces(st, n) do workspaces
        _farm(st) do
            OMT.@tasks for t in 1:n
                cols = (div((t - 1) * B, n) + 1):div(t * B, n)
                _run_columns!(out, workspaces[t], X, cols)
            end
        end
    end
    return out
end

function _run_planar_groups!(out, ws, X, groups, W, B)
    for g in groups
        ST._planar_group!(out, ws, X, ((g - 1) * W + 1):min(g * W, B))
    end
    return nothing
end

function ST._scattering_batch!(out::AbstractMatrix, ::CB.AbstractThreadedBackend,
                               st::ST.ScatteredPlanar.ScatteredPlanarScatteringTransform, X::AbstractMatrix)
    B, W = size(X, 2), ST.Plans.batch_width(st.plan)
    ngroups = cld(B, W)
    ngroups == 0 && return out
    n = min(ngroups, Threads.nthreads())
    ST.ScatteringCore.with_task_workspaces(st, n) do workspaces
        with_serial_blas() do
            OMT.@tasks for t in 1:n
                groups = (div((t - 1) * ngroups, n) + 1):div(t * ngroups, n)
                _run_planar_groups!(out, workspaces[t], X, groups, W, B)
            end
        end
    end
    return out
end

function ST._scattering_batch!(out::AbstractMatrix, b::CB.AbstractThreadedBackend,
                               st::ST.SphericalCore.SphericalScatteringTransform, X::AbstractArray)
    size(X)[end] == 0 && return out
    ST._spherical_batches(st.plan, X) || return _spherical_perfield!(out, b, st, X)
    B = size(X)[end]
    n = min(B, Threads.nthreads())
    width = cld(B, n)
    ST.ScatteringCore.with_batch_workspaces(() -> ST.SphericalBatchWorkspace(st, width),
                                           st, width, b, n) do entries
        _spherical_farm() do
            OMT.@tasks for t in 1:n
                cols = ((t - 1) * width + 1):min(t * width, B)
                isempty(cols) || ST._apply_cached_batch!(view(out, :, cols), entries[t], _columns(X, cols))
            end
        end
    end
    return out
end

function _spherical_perfield!(out, b, st, X)
    B = size(X)[end]
    B == 0 && return out
    n = min(B, Threads.nthreads())
    ST.ScatteringCore.with_task_workspaces(st, n) do workspaces
        _spherical_farm() do
            OMT.@tasks for t in 1:n
                cols = (div((t - 1) * B, n) + 1):div(t * B, n)
                _run_columns!(out, workspaces[t], X, cols)
            end
        end
    end
    return out
end

"""
    chunked_batch!(out, st, X; chunk, fft_nthreads = 1) -> out

Transform contiguous column chunks with retained batched plans, one workspace per task.
"""
function chunked_batch!(out::AbstractMatrix, st, X::AbstractArray; chunk::Int, fft_nthreads::Int = 1)
    B = size(X)[end]
    B == 0 && return out
    chunk >= 1 || throw(ArgumentError("chunk must be positive; got $chunk"))
    width = min(chunk, B)
    nfull = div(B, width)
    n = min(nfull, Threads.nthreads())
    build() = ST.batch_workspace(st, width; fft_nthreads)
    ST.ScatteringCore.with_batch_workspaces(build, st, width, CB.ThreadedBackend(), n, fft_nthreads) do entries
        OMT.@tasks for t in 1:n
            _run_batch_chunks!(out, entries[t], X, t:n:nfull, width)
        end
    end
    cols = (nfull * width + 1):B
    if !isempty(cols)
        ST.ScatteringCore.with_batch_workspaces(() -> ST.batch_workspace(st, length(cols); fft_nthreads),
                                               st, length(cols), CB.ThreadedBackend(), 1, fft_nthreads, :tail) do entries
            ST._apply_cached_batch!(view(out, :, cols), first(entries), _columns(X, cols))
        end
    end
    return out
end

function _run_batch_chunks!(out, entry::ST.ScatteringCore.CachedBatch, X, groups, width)
    for g in groups
        cols = ((g - 1) * width + 1):(g * width)
        ST._apply_cached_batch!(view(out, :, cols), entry, _columns(X, cols))
    end
    return nothing
end

function _run_groups!(S1, S2, ws, groups, spectrum, t, n)
    for i in t:n:length(groups)
        ST.Cascade.group!(S1, S2, ws.pw, ws.filter_bank, groups[i], spectrum)
    end
    return nothing
end

for (Mod, D, FT) in ((:Scattering1D, 1, :AbstractVector),
                     (:Scattering2D, 2, :AbstractMatrix),
                     (:Scattering3D, 3, :(AbstractArray{<:Any,3})))
    @eval function ST.$Mod.cascade!(S1::AbstractVector, S2::AbstractMatrix,
                                    ::CB.AbstractThreadedBackend,
                                    st::ST.ScatteringCore.ScatteringTransform{$D}, spectrum::$FT)
        isempty(S2) || fill!(S2, zero(eltype(S2)))
        n = min(length(st.groups), Threads.nthreads())
        ST.ScatteringCore.with_task_workspaces(st, n) do workspaces
            OMT.@tasks for t in 1:n
                _run_groups!(S1, S2, workspaces[t], st.groups, spectrum, t, n)
            end
        end
        return S1, S2
    end
end

end # module ScatteringTransformsOhMyThreadsExt
