module ScatteringTransformsNonuniformFFTsExt

"""
    ScatteringTransformsNonuniformFFTsExt — NonuniformFFTs.jl fast path for scattered planar scattering

The second fast spectral plan for the scattered-planar cascade (`ScatteredPlanar`), alongside
FINUFFT. Analysis is a Type-1 NUFFT (points → modes) or an LSMR least-squares solve;
synthesis is a Type-2 NUFFT (modes → points) scaled by `1/prod(ms)`. It implements the same
`ST.Plans.AbstractScatteringPlan` interface as the in-core `ST.Plans.DirectNUFFTPlan`, so the
cascade is identical — only the transform underneath changes.

NonuniformFFTs is pure Julia: no binary dependency, a threaded CPU path, and a
KernelAbstractions GPU path. It lays its modes out in `AbstractFFTs.fftfreq` order, which is the
lattice the wavelet bank is built on (and the order FINUFFT is asked for with `modeord = 1`), so
the two fast backends are interchangeable.

Selected by `spectral = ST.Plans.NonuniformFFTsBackend()`, or by
`SpectralBackends.NUFFTSpectralBackend()` / `AutoSpectralBackend()` when this is the loaded NUFFT.

The foreign plan is held behind the type parameter `P` and every `NonuniformFFTs` call is inside a
function body: an extension that names a runtime-only symbol in a *signature* fails to precompile,
and the tests would not catch it because the extension only loads with its trigger package.
"""

using NonuniformFFTs: NonuniformFFTs
using ScatteringTransforms: ScatteringTransforms as ST

"""
    NonuniformFFTsScatteringPlan{T,B,…}

Scattered-planar spectral plan backed by `NonuniformFFTs.PlanNUFFT` over fixed points `(x, y)` and a
uniform mode grid `ms`. `solve` selects the least-squares inversion over the plain Type-1 adjoint; the
solver's workspace lives on the plan, so a solve adds nothing per call beyond the transforms it issues.
It is also held only for the path the plan takes — a plan that does not solve carries none of it, and
the full-grid set is built on the first complex solve, which a cascade over real fields never reaches.

Those transforms do allocate, which makes this the one backend where a solve allocates in steady
state. Measured at `M = 500`, `ms = (16, 16)`, `nufft_nthreads = 1`, single-threaded Julia: 768 B for
one execution, and 108 kB for a solve that takes ~70 iterations — about 1.4 kB per iteration, which is
its two transforms and nothing else. An allocation profile attributes all of it to `Threads.@threads`
regions in NonuniformFFTs' deconvolution and spreading steps, which build their task scaffolding — a
`Task`, a task list, a lock and a condition — on every call, including at one thread where they
parallelise nothing. It therefore scales with `Threads.nthreads()` rather than with problem size. The
FINUFFT plan measures exactly zero on the same case because its execution enters no Julia threaded
region; its library threads are pooled rather than created per call.

`B`, the number of co-located fields transformed per execution, is a type parameter rather than a
field. NonuniformFFTs takes one array per transform instead of a trailing batch axis, so a `(…, B)`
stack has to be presented as `B` views of its last dimension; carrying `B` in the type keeps that
tuple's length statically known, so building it costs nothing per call.

Two NUFFT plans are held, because the two directions of the cascade have different data types.
Everything this backend *analyses* is real — the sampled field, and the modulus `|U|` at every later
step — so its spectrum is Hermitian and analysis runs on a real-data plan, whose uniform side is the
half spectrum `(ms[1] ÷ 2 + 1, ms[2])` in `rfftfreq × fftfreq` order. Measured against the complex
plan on the same points, that is 1.14x at `ms = 16²` rising to 1.76x at `200²` for type-1 and 1.87x
for type-2, and it halves the mode count the least-squares solve carries. Synthesis cannot use it: the
wavelet-filtered field is complex by construction and its modulus is the coefficient, so a real-valued
type-2 would be the wrong quantity. The complex plan serves that direction.
"""
struct NonuniformFFTsScatteringPlan{T, B, P, PR, CV <: AbstractArray{Complex{T}},
                                    MM <: AbstractArray{Complex{T}},
                                    HX <: AbstractArray{Complex{T}}, PT <: AbstractArray{T},
                                    RV <: AbstractVector{T}, IV <: AbstractVector{Int}, US, GK,
                                    BW} <: ST.Plans.AbstractScatteringPlan
    plan::P                 # complex-data plan: synthesis (Type-2, modes → complex points)
    rplan::PR               # real-data plan: analysis and the solve, over the half spectrum
    ms::NTuple{2, Int}
    M::Int
    invN::T                 # 1/prod(ms); makes synthesis the ifft-convention inverse
    solve::Bool
    maxiter::Int
    rtol::T
    damp::T                 # Tikhonov λ; 0 unless the mode grid is over-specified
    sx::RV                  # (M) points already scaled to the 2π-periodic domain, retained so a
    sy::RV                  #     task can build its own plan
    eps::T                  # requested relative tolerance, under the constructor's own keyword name.
                            #     Kept instead of the half-support it maps to, so a rebuild re-applies
                            #     `_half_support` rather than inverting a step function.
    nthreads::Int           # the FFT thread count (0 = whatever the process planner is set to)
    # One plan owns both transforms because one cascade needs both: synthesis is always complex, and
    # every analysis after the first is a modulus and so always real. `forward_transform!` selects by
    # the element type of the field it is handed — a real field's spectrum is Hermitian and is analysed
    # and solved on the half grid, a complex field's is not and needs the full grid.
    #
    # Scratch is held only for the path a plan actually takes. `task_local_plan` builds one plan per
    # task, so every buffer here is multiplied by the thread count. Point-space scratch dominates once
    # `M` is large: at `ms = 512²`, `M = 4·prod(ms)`, `ntrans = 8` one `(M, B)` complex array is
    # 134 MiB against 16.8 MiB for a half spectrum. None is held — the caller's own array serves
    # wherever it already has the plan's array type, which is the cascade's case since its buffers are
    # `similar` to the points these plans were built from.
    hx::HX                  # (ms[1]÷2+1, ms[2][, B]) the half spectrum: the analysis output, and the
                            #      half-grid solve's iterate before expansion onto the full grid.
                            #      Unconditional: every step after the first analyses a modulus.
    # `(u, t, v, w, h, hbar)` for `Plans.lsmr_solve!` — two point-space, four mode-space.
    rsolve::Union{Nothing, Tuple{PT, PT, HX, HX, HX, HX}}    # `nothing` unless this plan solves
    csolve::Base.RefValue{Union{Nothing, Tuple{CV, MM, MM, MM, MM}}}  # `(t, v, w, h, hbar)`, built on
                            #      the first complex solve; a cascade over real fields performs none.
                            #      `u` is `cbuf`, which no synthesis is using while a solve runs.
    # One point-space buffer per element type, allocated the first time something needs to land in the
    # plan's own array type rather than the caller's. The cascade's buffers already have it.
    rbuf::Base.RefValue{Union{Nothing, PT}}
    cbuf::Base.RefValue{Union{Nothing, CV}}
    mir::IV                 # (ms[2]) index of `-f₂` on the second axis, for the conjugate fill
    us_ovs::US              # the real plan's *oversampled* spectrum, refilled by every execution. Its
                            #      extent is `Ñ = σ·ms`, so `+ms[2]/2` — absent from `fftfreq(ms[2])`
                            #      — is an ordinary interior frequency here. That is the one entry the
                            #      conjugate fill cannot reach; see `_expand_hermitian!`.
    novs::NTuple{2, Int}    # `Ñ` per axis, axis 1 as its full length rather than the rfft one
    gk::GK                  # per-axis kernel Fourier coefficients, to undo the deconvolution by hand
    normfactor::T           # `∏ 2π/Ñ_d`, the normalisation the deconvolution step applies
    ls_batch::BW            # per-column solver bookkeeping, or `nothing` for a single field
end

ST.Plans.batch_width(::NonuniformFFTsScatteringPlan{T, B}) where {T, B} = B

# One array per transform is what `exec_type1!`/`exec_type2!` take, so a `(…, B)` stack is handed over
# as `B` views of its last dimension. Each view is contiguous, and `Val(B)` keeps the tuple's length in
# the type, so this compiles to no work.
@inline _fields(A::AbstractArray, ::Val{B}) where {B} =
    ntuple(i -> selectdim(A, ndims(A), i), Val(B))

# `PlanNUFFT` holds the scratch every execution writes through, so tasks cannot share one. The
# points are retained above precisely so a task can build its own.
#
# The rebuild carries the batch width and the thread count. The thread count defaults to one because
# the tasks already hold the cores, and a multi-threaded FFT plan spawns a Julia task per thread on
# every execution.
function ST.Plans.task_local_plan(p::NonuniformFFTsScatteringPlan{T, B}) where {T, B}
    return _plan_at(p.ms, p.M, p.sx, p.sy, p.eps, T, p.solve, p.maxiter, p.rtol, B,
                    ST.Plans.per_task_nthreads(p.nthreads), p.damp)
end

# Where the plan lives follows the points, so there is nothing to pass: `get_backend` reads the
# KernelAbstractions backend off `sx` and every buffer is `similar` to it. Host points give a host
# plan, device points a device-resident one, through the same code — and a task's rebuilt plan lands
# on the same device, because it rebuilds from these same points.
function _plan_at(ms::NTuple{2, Int}, M::Int, sx, sy, eps::T, ::Type{T}, solve, maxiter,
                  rtol::T, B::Int = 1, nthreads::Int = 0, damp::T = zero(T)) where {T}
    halfsupport = _half_support(eps)
    # Serialised on the package-wide planner lock: a host plan's smooth-grid FFT is planned through
    # the same libfftw3 every other backend here plans through, and this builder runs inside spawned
    # tasks. `nthreads > 0` additionally pins that planner's thread count, which the FFT plan bakes in
    # — a count of 0 leaves it at whatever the process has, which is NonuniformFFTs' own default.
    function build()
        # `sort_points` permutes the points once, at `set_points!`, so every later execution reads them
        # in block order. A plan here is built once over fixed points and then transformed through
        # `(1 + nparents) + (nw + npaths)` times per field — many more if it solves — so paying that
        # once is the right trade at any size. Swept over `ms` from `8²` to `512²` and point counts
        # from `0.1·prod(ms)` to `10·prod(ms)`: neutral below `64²`, 1.4-1.6x at `256²`-`512²` with
        # dense point sets, worst case 0.97x.
        # A pinned thread count has to reach the spreading, not only the FFT. With block partitioning
        # on, NonuniformFFTs spreads over `Threads.nthreads()` and holds one block buffer per thread
        # per transform — sized `prod(block_dims .+ 2m)` — no matter what thread count it is given.
        # A plan built per task would then carry a full set of them each, and `task_local_plan` builds
        # one plan per task, over two transforms, `ntrans` wide: the buffers multiply by all four.
        # `block_size = nothing` is the documented way off that path, and also drops FFTW to one
        # thread. There is no way to cap the spreading to an arbitrary count, so a pin above one keeps
        # the library's own threading.
        blocking = nthreads == 1 ? (; block_size = nothing) : (;)
        common = (m = NonuniformFFTs.HalfSupport(halfsupport), ntransforms = Val(B),
                  sort_points = NonuniformFFTs.Static.True(),
                  backend = NonuniformFFTs.KA.get_backend(sx), blocking...)
        pl = NonuniformFFTs.PlanNUFFT(Complex{T}, ms; common...)
        rpl = NonuniformFFTs.PlanNUFFT(T, ms; common...)
        NonuniformFFTs.set_points!(pl, (sx, sy))
        NonuniformFFTs.set_points!(rpl, (sx, sy))
        return (pl, rpl)
    end
    plan, rplan = Base.@lock ST.Plans.PLANNER_LOCK begin
        nthreads > 0 ? ST.Plans.with_fft_nthreads(build, nthreads) : build()
    end
    m1h = ms[1] ÷ 2 + 1
    rpts() = B == 1 ? similar(sx, T, M) : similar(sx, T, M, B)
    half() = B == 1 ? similar(sx, Complex{T}, m1h, ms[2]) :
                      similar(sx, Complex{T}, m1h, ms[2], B)
    # `mir[j]` is the index of `-f₂` on the second axis. Under `fftfreq` that is `1` for `j = 1` and
    # `ms[2] - j + 2` otherwise; for even `ms[2]` the entry at `-ms[2]/2` maps to itself, and that one
    # is filled from the oversampled spectrum instead, so the value is unused there.
    mir = similar(sx, Int, ms[2])
    copyto!(mir, [j == 1 ? 1 : ms[2] - j + 2 for j in 1:ms[2]])
    # The oversampled spectrum and the pieces needed to deconvolve a value read straight out of it.
    # One oversampled array per transform, so `B` of them when batched — the expansion reads the one
    # belonging to the column it is filling.
    us_ovs = rplan.data.ûs
    novs = (2 * (size(us_ovs[1], 1) - 1), size(us_ovs[1], 2))
    gk = ntuple(d -> NonuniformFFTs.fourier_coefficients(rplan.kernels[d]), 2)
    normfactor = prod(2 * T(π) / novs[d] for d in 1:2)
    # Only a batched solve needs the per-column machinery. The two norms are each held at both ranks
    # over one allocation, since the point stack is rank 2 and the mode stack rank 3.
    batch = if solve && B > 1
        nrm_p = similar(sx, T, 1, B)
        nrm_m = similar(sx, T, 1, 1, B)
        coef() = similar(sx, T, 1, 1, B)
        host() = Vector{T}(undef, B)
        ST.Plans.BatchedLSMRWork(nrm_p, reshape(nrm_p, 1, 1, B), nrm_m, reshape(nrm_m, 1, B),
                                 coef(), coef(), coef(),
                                 host(), host(), host(), host(), host(),
                                 [ST.Plans.lsmr_init(zero(T), zero(T)) for _ in 1:B])
    else
        nothing
    end
    hx = half()
    # The half-grid solver's vectors, only for a plan that solves. Own memory rather than views of a
    # full-grid array: a half grid is a *strided sub-block* of a full one, not a contiguous prefix, and
    # a NUFFT writes its output through a dense buffer.
    rsolve = solve ? (rpts(), rpts(), half(), half(), half(), half()) : nothing
    # Types for the arrays that are not built here, from zero-length arrays so a device plan names its
    # own array type without this file referring to any device package.
    PTT = typeof(B == 1 ? similar(sx, T, 0) : similar(sx, T, 0, 0))
    CVT = typeof(B == 1 ? similar(sx, Complex{T}, 0) : similar(sx, Complex{T}, 0, 0))
    MMT = typeof(B == 1 ? similar(sx, Complex{T}, 0, 0) : similar(sx, Complex{T}, 0, 0, 0))
    return NonuniformFFTsScatteringPlan{T, B, typeof(plan), typeof(rplan), CVT, MMT, typeof(hx),
                                        PTT, typeof(sx), typeof(mir), typeof(us_ovs), typeof(gk),
                                        typeof(batch)}(
        plan, rplan, ms, M, one(T) / prod(ms), solve, maxiter, rtol, damp, sx, sy, eps, nthreads,
        hx, rsolve,
        Base.RefValue{Union{Nothing, Tuple{CVT, MMT, MMT, MMT, MMT}}}(nothing),
        Base.RefValue{Union{Nothing, PTT}}(nothing),
        Base.RefValue{Union{Nothing, CVT}}(nothing),
        mir, us_ovs, novs, gk, normfactor, batch)
end

# ---------------------------------------------------------------------------
# Reaching the transform without a copy
# ---------------------------------------------------------------------------
#
# `exec_type1!`/`exec_type2!` take an array of the plan's own element type. Every array the cascade
# hands in already has one — `ScatteredPlanar.build` makes its buffers `similar` to the same points —
# so these return it unchanged and the plan carries no point-space scratch at all. Any other caller
# gets one copy through a buffer allocated on its first such call.

@inline _real_in(plan::NonuniformFFTsScatteringPlan{T, B, P, PR, CV, MM, HX, PT},
                 x::AbstractArray{<:Real}) where {T, B, P, PR, CV, MM, HX, PT} =
    x isa PT ? x : _rbuf!(plan, x)

@noinline function _rbuf!(plan::NonuniformFFTsScatteringPlan{T, B, P, PR, CV, MM, HX, PT},
                          x) where {T, B, P, PR, CV, MM, HX, PT}
    buf = plan.rbuf[]
    if buf === nothing
        buf = B == 1 ? similar(plan.sx, T, plan.M) : similar(plan.sx, T, plan.M, B)
        plan.rbuf[] = buf
    end
    copyto!(buf, x)
    return buf::PT
end

# Synthesis destination. When it is `out` itself, the scaling that follows is an in-place multiply.
@inline _pts_out(plan::NonuniformFFTsScatteringPlan{T, B, P, PR, CV},
                 out::AbstractArray) where {T, B, P, PR, CV} =
    out isa CV ? out : _cbuf(plan)

@inline _cplx_in(plan::NonuniformFFTsScatteringPlan{T, B, P, PR, CV},
                 x::AbstractArray) where {T, B, P, PR, CV} =
    x isa CV ? x : copyto!(_cbuf(plan), x)

@noinline function _cbuf(plan::NonuniformFFTsScatteringPlan{T, B, P, PR, CV}) where {T, B, P, PR, CV}
    buf = plan.cbuf[]
    if buf === nothing
        buf = B == 1 ? similar(plan.sx, Complex{T}, plan.M) :
                       similar(plan.sx, Complex{T}, plan.M, B)
        plan.cbuf[] = buf
    end
    return buf::CV
end

# The adjoint of the real-data Type-2, in the inner product LSMR actually uses.
#
# Writing `A` for the half spectrum → real samples map, `A(h)_j = Σ_{k₁=0} hₖ e^{ik·xⱼ} +
# 2 Σ_{k₁>0} Re(hₖ e^{ik·xⱼ})`: every row above DC enters twice, once as `k` and once as `−k`. So `A`
# is only ℝ-linear — `A(i·e_k) ≠ i·A(e_k)` — and its adjoint is with respect to the real inner product
# `⟨h, g⟩ = Re Σ conj(hₖ) gₖ`, which treats real and imaginary parts as separate coordinates. That is
# exactly the inner product LSMR works in: every scalar in the recurrence is real and every inner
# product reaches the vectors through `norm`, which for a complex array is the real Euclidean norm.
#
# `exec_type1!` supplies half of that adjoint on every row above DC, so those rows are doubled here.
# Verified against `⟨A h, v⟩ = ⟨h, A† v⟩` to 4e-15 for even and odd `ms[1]`.
function _real_adjoint!(dst::AbstractArray, plan::NonuniformFFTsScatteringPlan{T, 1},
                        src::AbstractArray) where {T}
    NonuniformFFTs.exec_type1!(dst, plan.rplan, src)
    @views dst[2:end, :] .*= 2
    return _sym_dc_row!(dst, plan)
end

function _real_adjoint!(dst::AbstractArray, plan::NonuniformFFTsScatteringPlan{T, B},
                        src::AbstractArray) where {T, B}
    NonuniformFFTs.exec_type1!(_fields(dst, Val(B)), plan.rplan, _fields(src, Val(B)))
    @views dst[2:end, :, :] .*= 2
    return _sym_dc_row!(dst, plan)
end

# Replace the DC row by its Hermitian-symmetric part, `h[1,j] ← (h[1,j] + conj(h[1,-j]))/2`.
#
# An r2c transform halves one axis, so the stored half keeps the full range on the others — which
# means the DC row holds both `(0,k₂)` and `(0,−k₂)`. Those are negatives of each other, so they
# contribute the same `cos` and opposite `sin`: four real parameters over a two-dimensional space. As
# a transform output that is harmless (the data fixes both), but as free unknowns in a fit it is a
# null space, and `A` cannot see the antisymmetric part at all.
#
# So the fit is restricted to the symmetric part. The projection is self-adjoint and `A∘P = A`, so the
# operator pair stays consistent and applying it here keeps every iterate in the range of `P`. Without
# it the bidiagonalisation terminates exactly on that null space, which divides by zero in the update
# coefficients.
function _sym_dc_row!(dst::AbstractArray, plan::NonuniformFFTsScatteringPlan)
    m1, m2 = plan.ms
    m1h = m1 ÷ 2 + 1
    nyq = iseven(m2) ? m2 ÷ 2 + 1 : 0            # column holding `-m2/2`, which has no partner
    nb = ndims(dst) == 2 ? 1 : size(dst, 3)
    @inbounds for c in 1:nb, j in 1:m2
        # A mode `(a, c)` puts `h` at `(a, c)` and `conj(h)` at `(-a, -c)`. For even `ms`, one of those
        # two frequencies is off the `fftfreq` grid — `+ms[1]/2` for the top row, `+ms[2]/2` for this
        # column — so the expansion onto the grid can only carry half of the mode. Left free, the fit
        # uses it and the dropped half makes the answer stop reproducing the samples. Excluded, the
        # solve runs over exactly the real fields the grid can hold.
        if j == nyq
            for i in 1:m1h
                dst[i, j, c] = 0
            end
            continue
        end
        iseven(m1) && (dst[m1h, j, c] = 0)
        jj = plan.mir[j]
        jj < j && continue                       # each pair handled once, at its lower index
        a, b = dst[1, j, c], dst[1, jj, c]
        s = (a + conj(b)) / 2                    # `j == jj` at `k₂ = 0`, where this is `real(a)`
        dst[1, j, c] = s
        dst[1, jj, c] = conj(s)
    end
    return dst
end

# Expand a half spectrum onto the full `fftfreq × fftfreq` grid the wavelet bank is built on. Rows up
# to `m1h` are the transform's own output; the rest are the conjugates of their frequency partners,
# which is exact because the analysed field is real. Trailing `Colon`s carry the batch axis when there
# is one, so this serves both ranks.

# Oversampled `fftfreq(Ñ)` index (1-based) of integer frequency `f`.
@inline _ovs_index(f::Int, Ñ::Int) = f >= 0 ? f + 1 : Ñ + f + 1

# Kernel-Fourier-coefficient index of frequency `f`. These plans are built unshifted, so `ĝ` is laid
# out in `fftfreq` order like the output and the index is the plain one — `f + 1` forward, wrapping for
# negative `f`. `ĝ` is even, so the out-of-range `+N/2` lands on `-N/2`'s slot, which is the same value.
@inline _gk_index(f::Int, N::Int) = f >= 0 ? f + 1 : N + f + 1

# Expanding a *solution* is not the same operation as expanding a transform's output, and the
# difference is exactly the unpaired frequency. There, `H` holds a coefficient the model cannot
# represent — the mask keeps it at zero — so the full grid gets zero. The analysis expansion instead
# recovers a genuine transform value from the oversampled buffer. Using that branch here injects the
# last transform's oversampled spectrum into the answer, which for even `ms` is most of the error.
function _expand_solution!(F::AbstractArray, plan::NonuniformFFTsScatteringPlan, H::AbstractArray)
    ms = plan.ms
    hi1, hi2 = (ms[1] - 1) ÷ 2, (ms[2] - 1) ÷ 2
    nb = ndims(F) == 2 ? 1 : size(F, 3)
    @inbounds for c in 1:nb, j in 1:ms[2], i in 1:ms[1]
        f1 = i - 1 <= hi1 ? i - 1 : i - 1 - ms[1]
        f2 = j - 1 <= hi2 ? j - 1 : j - 1 - ms[2]
        F[i, j, c] = if f1 >= 0
            H[f1 + 1, j, c]
        elseif !(iseven(ms[2]) && f2 == -(ms[2] ÷ 2))
            conj(H[-f1 + 1, plan.mir[j], c])
        else
            zero(eltype(F))
        end
    end
    return F
end

function _expand_hermitian!(F::AbstractArray, plan::NonuniformFFTsScatteringPlan, H::AbstractArray)
    ms = plan.ms
    hi1, hi2 = (ms[1] - 1) ÷ 2, (ms[2] - 1) ÷ 2
    nb = ndims(F) == 2 ? 1 : size(F, 3)
    @inbounds for c in 1:nb, j in 1:ms[2], i in 1:ms[1]
        f1 = i - 1 <= hi1 ? i - 1 : i - 1 - ms[1]
        f2 = j - 1 <= hi2 ? j - 1 : j - 1 - ms[2]
        if f1 >= 0
            # Straight out of the transform's own output.
            F[i, j, c] = H[f1 + 1, j, c]
        elseif !(iseven(ms[2]) && f2 == -(ms[2] ÷ 2))
            # `X[-k] = conj(X[k])` for real samples, and `-k` is on the grid.
            F[i, j, c] = conj(H[-f1 + 1, plan.mir[j], c])
        else
            # `-k` has `f₂ = +ms[2]/2`, which `fftfreq(ms[2])` does not carry — but the oversampled
            # spectrum runs to `Ñ₂ > ms[2]`, so there it is an ordinary interior frequency. Read it
            # from there and apply the deconvolution and normalisation the truncating step would have.
            n1, n2 = -f1, -f2
            β = plan.normfactor / plan.gk[1][n1 + 1] / plan.gk[2][_gk_index(n2, ms[2])]
            F[i, j, c] = conj(β * plan.us_ovs[c][_ovs_index(n1, plan.novs[1]),
                                                 _ovs_index(n2, plan.novs[2])])
        end
    end
    return F
end

# The plan holds device/threading state and scratch, so it prints as one line rather than dumping
# its internals, and a concurrent task takes its own.
Base.show(io::IO, p::NonuniformFFTsScatteringPlan{T, B}) where {T, B} =
    print(io, "NonuniformFFTsScatteringPlan{", T, "}(ms=", p.ms, ", M=", p.M, ", ntrans=", B,
          ", solve=", p.solve, ")")
Base.show(io::IO, ::MIME"text/plain", p::NonuniformFFTsScatteringPlan) = show(io, p)

ST.Plans.spectral_backend(::NonuniformFFTsScatteringPlan) = ST.Plans.NonuniformFFTsBackend()
ST.Plans.plan_points(p::NonuniformFFTsScatteringPlan) = (p.sx, p.sy)
ST.Plans.plan_analysis(p::NonuniformFFTsScatteringPlan) =
    (solve = p.solve, maxiter = p.maxiter, rtol = p.rtol, damp = p.damp, eps = p.eps,
     nufft_nthreads = p.nthreads)

# NonuniformFFTs expresses accuracy as the convolution kernel's half-support, not as a tolerance, so
# the `eps` this interface takes — a FINUFFT-style tolerance — is mapped to the smallest half-support
# that meets it, and the two fast backends then honour the same request.
#
# Kaiser–Bessel aliasing error goes as `ε ≈ exp(-2πm√(1 - 1/σ))` for half-support `m` and oversampling
# `σ`, so that inverted is the mapping. At `σ = 2` — NonuniformFFTs' default, and what these plans use
# — it agrees with the library's documented `m = 4` giving ~1e-7, and returns the same widths FINUFFT
# picks for the same tolerance.
_half_support(tol::Real, σ::Real = 2) =
    max(2, ceil(Int, log(1 / tol) / (2π * sqrt(1 - 1 / σ))))

function ST.Plans.nonuniformffts_scattered_plan(x, y, ms::NTuple{2, Int}, ::Type{T};
                                                period = nothing, solve::Bool = false,
                                                maxiter::Int = 100, eps = nothing,
                                                rtol::Real = ST.Plans.default_solver_rtol(T, ST.Plans.NonuniformFFTsBackend(), eps),
                                                damp::Real = 0, ntrans::Int = 1,
                                                nufft_nthreads::Int = 0) where {T}
    # `ntrans` is honoured, not swallowed: the plan is built `ntrans` wide and the cascade issues one
    # execution per step for the whole stack. Whether that pays depends on the mode grid, and it is
    # the caller's choice to make — batching amortises per-call overhead but multiplies the
    # oversampled working set by `ntrans`, so it wins on small grids and loses once `ntrans` copies of
    # that grid stop fitting cache.
    M = length(x)
    length(y) == M || throw(DimensionMismatch("x and y must have equal length"))
    ST.Plans.warn_underdetermined(M, ms, solve, damp)
    xmin, ymin = T(minimum(x)), T(minimum(y))
    # Same default period as the in-core plan: a uniform 0:m-1 grid maps to the exact DFT nodes.
    px = period === nothing ? (T(maximum(x)) - xmin) * ms[1] / (ms[1] - 1) : T(period[1])
    py = period === nothing ? (T(maximum(y)) - ymin) * ms[2] / (ms[2] - 1) : T(period[2])
    sx = T(2π) .* (T.(x) .- xmin) ./ px
    sy = T(2π) .* (T.(y) .- ymin) ./ py

    tol = eps === nothing ? ST.Plans.default_nufft_eps(T) : eps
    return _plan_at(ms, M, sx, sy, T(tol), T, solve, maxiter, T(rtol), ntrans, nufft_nthreads,
                    T(damp))
end

# Synthesis: modes → points (Type-2), scaled by 1/prod(ms) so it is the ifft-convention inverse.
function ST.Plans.inverse_transform!(out_pts::AbstractVector, plan::NonuniformFFTsScatteringPlan,
                                     Xmodes::AbstractMatrix)
    dst = _pts_out(plan, out_pts)
    NonuniformFFTs.exec_type2!(dst, plan.plan, Xmodes)
    @. out_pts = dst * plan.invN
    return out_pts
end

# Analysis: points → modes. Type-1 adjoint (fft-equivalent on a uniform grid) unless `solve`.
#
# Real samples: their spectrum is Hermitian, so the transform runs on the real-data plan over the half
# grid and the result is expanded onto the full grid the wavelet bank is built on. Half the FFT work
# and half the grid memory of the complex form.
function ST.Plans.forward_transform!(Xmodes::AbstractMatrix, plan::NonuniformFFTsScatteringPlan,
                                     x_pts::AbstractVector{<:Real})
    b = _real_in(plan, x_pts)
    if plan.solve
        _lsmr_solve_real!(Xmodes, plan, b)
    else
        NonuniformFFTs.exec_type1!(plan.hx, plan.rplan, b)
        _expand_hermitian!(Xmodes, plan, plan.hx)
    end
    return Xmodes
end

# Complex samples: nothing is redundant, so this is the full-grid transform on the complex plan.
function ST.Plans.forward_transform!(Xmodes::AbstractMatrix, plan::NonuniformFFTsScatteringPlan,
                                     x_pts::AbstractVector{<:Complex})
    if plan.solve
        _lsmr_solve!(Xmodes, plan, x_pts)
    else
        NonuniformFFTs.exec_type1!(Xmodes, plan.plan, _cplx_in(plan, x_pts))
    end
    return Xmodes
end

# Batched forms. A plan built `B` wide executes exactly `B` transforms per call, so these are the only
# valid shapes for it, just as the shapes above are the only valid ones for a `B = 1` plan.
function ST.Plans.inverse_transform!(out_pts::AbstractMatrix,
                                     plan::NonuniformFFTsScatteringPlan{T, B},
                                     Xmodes::AbstractArray{<:Any, 3}) where {T, B}
    dst = _pts_out(plan, out_pts)
    NonuniformFFTs.exec_type2!(_fields(dst, Val(B)), plan.plan, _fields(Xmodes, Val(B)))
    @. out_pts = dst * plan.invN
    return out_pts
end

function ST.Plans.forward_transform!(Xmodes::AbstractArray{<:Any, 3},
                                     plan::NonuniformFFTsScatteringPlan{T, B},
                                     x_pts::AbstractMatrix{<:Real}) where {T, B}
    b = _real_in(plan, x_pts)
    if plan.solve
        _lsmr_solve_batched_real!(Xmodes, plan, b)
    else
        NonuniformFFTs.exec_type1!(_fields(plan.hx, Val(B)), plan.rplan, _fields(b, Val(B)))
        _expand_hermitian!(Xmodes, plan, plan.hx)
    end
    return Xmodes
end

function ST.Plans.forward_transform!(Xmodes::AbstractArray{<:Any, 3},
                                     plan::NonuniformFFTsScatteringPlan{T, B},
                                     x_pts::AbstractMatrix{<:Complex}) where {T, B}
    if plan.solve
        _lsmr_solve_batched!(Xmodes, plan, x_pts)
    else
        NonuniformFFTs.exec_type1!(_fields(Xmodes, Val(B)), plan.plan,
                                   _fields(_cplx_in(plan, x_pts), Val(B)))
    end
    return Xmodes
end

# The half-grid solver's vectors. Present whenever the plan was built to solve.
@inline function _rsolve(plan::NonuniformFFTsScatteringPlan)
    rs = plan.rsolve
    rs === nothing && throw(ArgumentError("plan was not built with `solve = true`"))
    return rs
end

# `(t, v, w, h, hbar)` for a full-grid solve, built on first use. A cascade over real fields never
# reaches this, and it is the larger of the two sets — the full mode grid is twice the half grid.
@noinline function _csolve(plan::NonuniformFFTsScatteringPlan{T, B, P, PR, CV, MM}) where {T, B, P,
                                                                                           PR, CV, MM}
    got = plan.csolve[]
    got === nothing || return got
    pts() = B == 1 ? similar(plan.sx, Complex{T}, plan.M) :
                     similar(plan.sx, Complex{T}, plan.M, B)
    modes() = B == 1 ? similar(plan.sx, Complex{T}, plan.ms) :
                       similar(plan.sx, Complex{T}, (plan.ms..., B))
    made = (pts(), modes(), modes(), modes(), modes())
    plan.csolve[] = made
    return made::Tuple{CV, MM, MM, MM, MM}
end

# Least-squares inversion. `A f̃ = x` is solved and scaled by `N` afterwards rather than inflating the
# right-hand side by `prod(ms)`, which keeps intermediates at the field's own magnitude.
#
# Real samples: the fit runs over the half grid, which is the space a real field's coefficients
# actually occupy. `A` is the real Type-2 and `A†` the weighted Type-1 — see `_real_adjoint!`.
function _lsmr_solve_real!(f::AbstractMatrix, plan::NonuniformFFTsScatteringPlan{T},
                           b::AbstractVector) where {T}
    u, t, v, w, h, hbar = _rsolve(plan)
    info = ST.Plans.lsmr_solve!(plan.hx,
        (dst, src) -> NonuniformFFTs.exec_type2!(dst, plan.rplan, src),  # A : half modes → real pts
        (dst, src) -> _real_adjoint!(dst, plan, src),                    # A†: real pts → half modes
        b, u, t, v, w, h, hbar;
        damp = plan.damp, atol = plan.rtol, btol = plan.rtol,
        conlim = inv(Base.eps(T)), maxiter = plan.maxiter)
    ST.Plans._check_solve(info, plan.M, (plan.ms[1] ÷ 2 + 1, plan.ms[2]), plan.rtol, plan.maxiter)
    plan.hx .*= one(T) / plan.invN
    return _expand_solution!(f, plan, plan.hx)
end

# Complex samples: no redundancy to exploit, so the fit is over the full grid on the complex plan.
function _lsmr_solve!(f::AbstractMatrix, plan::NonuniformFFTsScatteringPlan{T},
                      x_pts::AbstractVector) where {T}
    t, v, w, h, hbar = _csolve(plan)
    info = ST.Plans.lsmr_solve!(f,
        (dst, src) -> NonuniformFFTs.exec_type2!(dst, plan.plan, src),   # A : modes → points
        (dst, src) -> NonuniformFFTs.exec_type1!(dst, plan.plan, src),   # A†: points → modes
        x_pts, _cbuf(plan), t, v, w, h, hbar;
        damp = plan.damp, atol = plan.rtol, btol = plan.rtol,
        conlim = inv(Base.eps(T)), maxiter = plan.maxiter)
    ST.Plans._check_solve(info, plan.M, plan.ms, plan.rtol, plan.maxiter)
    f .*= one(T) / plan.invN
    return f
end

# The batched forms of both. A plan's width is fixed, so the whole stack advances together and every
# scalar in the recurrence becomes one per column — see `Plans.lsmr_solve_batched!`.
function _lsmr_solve_batched_real!(f::AbstractArray{<:Any, 3},
                                   plan::NonuniformFFTsScatteringPlan{T, B},
                                   b::AbstractMatrix) where {T, B}
    u, t, v, w, h, hbar = _rsolve(plan)
    info = ST.Plans.lsmr_solve_batched!(plan.hx,
        (dst, src) -> NonuniformFFTs.exec_type2!(_fields(dst, Val(B)), plan.rplan,
                                                 _fields(src, Val(B))),
        (dst, src) -> _real_adjoint!(dst, plan, src),
        b, u, t, v, w, h, hbar, plan.ls_batch;
        damp = plan.damp, atol = plan.rtol, btol = plan.rtol,
        conlim = inv(Base.eps(T)), maxiter = plan.maxiter)
    ST.Plans._check_solve(info, plan.M, (plan.ms[1] ÷ 2 + 1, plan.ms[2]), plan.rtol, plan.maxiter)
    plan.hx .*= one(T) / plan.invN
    return _expand_solution!(f, plan, plan.hx)
end

function _lsmr_solve_batched!(f::AbstractArray{<:Any, 3},
                              plan::NonuniformFFTsScatteringPlan{T, B},
                              x_pts::AbstractMatrix) where {T, B}
    t, v, w, h, hbar = _csolve(plan)
    info = ST.Plans.lsmr_solve_batched!(f,
        (dst, src) -> NonuniformFFTs.exec_type2!(_fields(dst, Val(B)), plan.plan,
                                                 _fields(src, Val(B))),
        (dst, src) -> NonuniformFFTs.exec_type1!(_fields(dst, Val(B)), plan.plan,
                                                 _fields(src, Val(B))),
        x_pts, _cbuf(plan), t, v, w, h, hbar, plan.ls_batch;
        damp = plan.damp, atol = plan.rtol, btol = plan.rtol,
        conlim = inv(Base.eps(T)), maxiter = plan.maxiter)
    ST.Plans._check_solve(info, plan.M, plan.ms, plan.rtol, plan.maxiter)
    f .*= one(T) / plan.invN
    return f
end

end # module ScatteringTransformsNonuniformFFTsExt