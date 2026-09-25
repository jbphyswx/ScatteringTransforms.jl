# ---------------------------------------------------------------------------
# Scattered planar transform by nonuniform FFT, through FlowTransformBindings
#
# The numeric contract of `DirectNUFFTPlan`: `forward_transform!` is the Type-1 adjoint onto the
# `fftfreq` mode grid, or an LSMR least-squares inversion when `solve`; `inverse_transform!` is the
# Type-2 synthesis scaled by `1/prod(ms)`.
#
# Real samples have a Hermitian spectrum, so they are analysed on a real-data plan over the half
# `k₁ ≥ 0`. Its second axis holds `ms[2] + 1` modes when `ms[2]` is even, so both `±ms[2]/2` are
# modes, and every entry of the full grid is a mode of the half or the conjugate of one. A real solve
# fits the real trigonometric polynomials of the half with each Nyquist frequency of an even axis split
# evenly between `±ms[d]/2`, the real form of the DFT, `prod(ms)` real unknowns; the solution goes to
# the `fftfreq` grid with `+ms[d]/2` aliased onto `-ms[d]/2`, the grid's only mode at that frequency.
# ---------------------------------------------------------------------------

"""
    NUFFTScatteringPlan{T,B}

Scattered-planar plan over FlowTransformBindings NUFFT plans on fixed points: a complex plan for
synthesis and for complex samples, and a real plan for real samples and the real solve. `B` is the
number of co-located fields each execution transforms.
"""
struct NUFFTScatteringPlan{T, B, L, PC, PR, CV <: AbstractArray{Complex{T}}, MM <: AbstractArray{Complex{T}},
                           HX <: AbstractArray{Complex{T}}, PT <: AbstractArray{T}, RV <: AbstractVector{T},
                           IV <: AbstractVector{Int}, RS, BW} <: AbstractScatteringPlan
    backend::L
    cplan::PC
    rplan::PR
    ms::NTuple{2, Int}
    M::Int
    invN::T                 # 1/prod(ms): synthesis is the ifft-convention inverse
    solve::Bool
    maxiter::Int
    rtol::T
    damp::T                 # Tikhonov λ; 0 unless the mode grid is over-specified
    sx::RV                  # (M) points on the 2π-periodic domain, kept so a task can build its own
    sy::RV
    eps::T
    nthreads::Int
    hx::HX                  # the real plan's half spectrum, (ms[1] ÷ 2 + 1, n₂[, B])
    dc::HX                  # (1, n₂[, B]) scratch for the DC-row projection
    rsolve::RS              # `(u, t, v, w, h, hbar)` over the half, or `nothing` unless this plan solves
    csolve::Base.RefValue{Union{Nothing, Tuple{CV, MM, MM, MM, MM}}}   # full-grid solve, built on first use
    rbuf::Base.RefValue{Union{Nothing, PT}}
    cbuf::Base.RefValue{Union{Nothing, CV}}
    col::IV                 # (ms[2]) column of the half holding each `fftfreq(ms[2])` frequency
    negcol::IV              # (ms[2]) column holding its negative
    negcol_half::IV         # (n₂) column holding the negative of each column's frequency
    ls_batch::BW
end

batch_width(::NUFFTScatteringPlan{T, B}) where {T, B} = B

_fft_index(f::Int, n::Int) = f >= 0 ? f + 1 : n + f + 1
_fft_frequency(i::Int, n::Int) = (k = i - 1; k <= (n - 1) ÷ 2 ? k : k - n)

function _like_ints(ref::AbstractArray, v::Vector{Int})
    d = similar(ref, Int, length(v))
    copyto!(d, v)
    return d
end

"""
    nufft_scattered_plan(backend, x, y, ms, T; period, solve, maxiter, rtol, damp, eps, ntrans,
                         nufft_nthreads) -> NUFFTScatteringPlan

The scattered-planar plan over points `(x, y)` and mode grid `ms` on a FlowTransformBindings NUFFT
`backend`. `nufft_nthreads = 0` takes the session's `Threads.nthreads()`.
"""
function nufft_scattered_plan(backend::SB.AbstractNUFFTSpectralBackend, x, y, ms::NTuple{2, Int}, ::Type{T};
                              period = nothing, solve::Bool = false, maxiter::Int = 100, eps = nothing,
                              rtol::Real = default_solver_rtol(T, backend, eps), damp::Real = 0,
                              ntrans::Int = 1, nufft_nthreads::Int = 0) where {T}
    M = length(x)
    length(y) == M || throw(DimensionMismatch("x and y must have equal length"))
    warn_underdetermined(M, ms, solve, damp)
    xmin, ymin = T(minimum(x)), T(minimum(y))
    # Default period so a uniform 0:m-1 grid (span m-1) maps to the exact DFT nodes 2π·(0:m-1)/m.
    px = period === nothing ? (T(maximum(x)) - xmin) * ms[1] / (ms[1] - 1) : T(period[1])
    py = period === nothing ? (T(maximum(y)) - ymin) * ms[2] / (ms[2] - 1) : T(period[2])
    sx = T(2π) .* (T.(x) .- xmin) ./ px
    sy = T(2π) .* (T.(y) .- ymin) ./ py
    tol = eps === nothing ? FTB.default_tolerance(T) : T(eps)
    return _nufft_plan_at(backend, ms, M, sx, sy, T(tol), T, solve, maxiter, T(rtol), ntrans, nufft_nthreads, T(damp))
end

function _nufft_plan_at(backend, ms::NTuple{2, Int}, M::Int, sx, sy, eps::T, ::Type{T}, solve, maxiter,
                        rtol::T, B::Int, nthreads::Int, damp::T) where {T}
    n2 = ms[2] + iseven(ms[2])
    kw = (; ntrans = B, tol = eps, order = FTB.FFTModes(), nthreads = nthreads > 0 ? nthreads : Threads.nthreads())
    cplan, rplan = Base.@lock PLANNER_LOCK begin
        (FTB.plan_nufft(backend, Complex{T}, (sx, sy), ms; kw...),
         FTB.plan_nufft(backend, T, (sx, sy), (ms[1], n2); kw...))
    end
    m1h = ms[1] ÷ 2 + 1
    rpts() = B == 1 ? similar(sx, T, M) : similar(sx, T, M, B)
    half() = B == 1 ? similar(sx, Complex{T}, m1h, n2) : similar(sx, Complex{T}, m1h, n2, B)
    f2 = [_fft_frequency(j, ms[2]) for j in 1:ms[2]]
    col = _like_ints(sx, [_fft_index(f, n2) for f in f2])
    negcol = _like_ints(sx, [_fft_index(-f, n2) for f in f2])
    negcol_half = _like_ints(sx, [_fft_index(-_fft_frequency(k, n2), n2) for k in 1:n2])
    batch = if solve && B > 1
        nrm_p = similar(sx, T, 1, B)
        nrm_m = similar(sx, T, 1, 1, B)
        coef() = similar(sx, T, 1, 1, B)
        host() = Vector{T}(undef, B)
        BatchedLSMRWork(nrm_p, reshape(nrm_p, 1, 1, B), nrm_m, reshape(nrm_m, 1, B),
                        coef(), coef(), coef(), host(), host(), host(), host(), host(),
                        [lsmr_init(zero(T), zero(T)) for _ in 1:B])
    else
        nothing
    end
    hx = half()
    dc = B == 1 ? similar(sx, Complex{T}, 1, n2) : similar(sx, Complex{T}, 1, n2, B)
    rsolve = solve ? (rpts(), rpts(), half(), half(), half(), half()) : nothing
    # Types of the buffers built on first use, from zero-length arrays `similar` to the points, so a
    # device plan's types follow its points.
    PTT = typeof(B == 1 ? similar(sx, T, 0) : similar(sx, T, 0, 0))
    CVT = typeof(B == 1 ? similar(sx, Complex{T}, 0) : similar(sx, Complex{T}, 0, 0))
    MMT = typeof(B == 1 ? similar(sx, Complex{T}, 0, 0) : similar(sx, Complex{T}, 0, 0, 0))
    return NUFFTScatteringPlan{T, B, typeof(backend), typeof(cplan), typeof(rplan), CVT, MMT, typeof(hx),
                               PTT, typeof(sx), typeof(col), typeof(rsolve), typeof(batch)}(
        backend, cplan, rplan, ms, M, one(T) / prod(ms), solve, maxiter, rtol, damp, sx, sy, eps, nthreads,
        hx, dc, rsolve,
        Base.RefValue{Union{Nothing, Tuple{CVT, MMT, MMT, MMT, MMT}}}(nothing),
        Base.RefValue{Union{Nothing, PTT}}(nothing), Base.RefValue{Union{Nothing, CVT}}(nothing),
        col, negcol, negcol_half, batch)
end

Base.show(io::IO, p::NUFFTScatteringPlan{T, B}) where {T, B} =
    print(io, "NUFFTScatteringPlan{", T, "}(", nameof(typeof(p.backend)), ", ms=", p.ms, ", M=", p.M,
          ", ntrans=", B, ", solve=", p.solve, ")")
Base.show(io::IO, ::MIME"text/plain", p::NUFFTScatteringPlan) = show(io, p)

spectral_backend(p::NUFFTScatteringPlan) = p.backend
plan_points(p::NUFFTScatteringPlan) = (p.sx, p.sy)
plan_analysis(p::NUFFTScatteringPlan) =
    (solve = p.solve, maxiter = p.maxiter, rtol = p.rtol, damp = p.damp, eps = p.eps, nufft_nthreads = p.nthreads)

# The FlowTransformBindings plans hold the scratch every execution writes through, so a task builds its
# own over the same points, at one thread unless one was asked for.
task_local_plan(p::NUFFTScatteringPlan{T, B}) where {T, B} =
    _nufft_plan_at(p.backend, p.ms, p.M, p.sx, p.sy, p.eps, T, p.solve, p.maxiter, p.rtol, B,
                   per_task_nthreads(p.nthreads), p.damp)

close_plan!(p::NUFFTScatteringPlan) =
    Base.@lock PLANNER_LOCK begin
        FTB.close!(p.cplan)
        FTB.close!(p.rplan)
        nothing
    end

# ---------------------------------------------------------------------------
# Reaching the transform without a copy. The cascade's buffers are `similar` to the plan's points, so
# they already have the plan's array types and pass straight through; any other array is copied into a
# buffer made on its first use.
# ---------------------------------------------------------------------------

@inline _real_in(p::NUFFTScatteringPlan{T, B, L, PC, PR, CV, MM, HX, PT}, x::AbstractArray{<:Real}) where {T, B, L, PC, PR, CV, MM, HX, PT} =
    x isa PT ? x : copyto!(_rbuf(p), x)

@noinline function _rbuf(p::NUFFTScatteringPlan{T, B, L, PC, PR, CV, MM, HX, PT}) where {T, B, L, PC, PR, CV, MM, HX, PT}
    buf = p.rbuf[]
    buf === nothing && (buf = B == 1 ? similar(p.sx, T, p.M) : similar(p.sx, T, p.M, B); p.rbuf[] = buf)
    return buf::PT
end

@inline _pts_out(p::NUFFTScatteringPlan{T, B, L, PC, PR, CV}, out::AbstractArray) where {T, B, L, PC, PR, CV} =
    out isa CV ? out : _cbuf(p)

@inline _cplx_in(p::NUFFTScatteringPlan{T, B, L, PC, PR, CV}, x::AbstractArray) where {T, B, L, PC, PR, CV} =
    x isa CV ? x : copyto!(_cbuf(p), x)

@noinline function _cbuf(p::NUFFTScatteringPlan{T, B, L, PC, PR, CV}) where {T, B, L, PC, PR, CV}
    buf = p.cbuf[]
    buf === nothing && (buf = B == 1 ? similar(p.sx, Complex{T}, p.M) : similar(p.sx, Complex{T}, p.M, B); p.cbuf[] = buf)
    return buf::CV
end

# ---------------------------------------------------------------------------
# The half spectrum and the full `fftfreq` grid
# ---------------------------------------------------------------------------

# Rows `1:np` of the full grid hold `k₁ = 0 … np - 1`, read from the half; the rest hold `k₁ < 0`, the
# conjugates of the half at `(-k₁, -k₂)`.
function _expand_hermitian!(F::AbstractArray, p::NUFFTScatteringPlan, H::AbstractArray)
    m1 = p.ms[1]
    np = (m1 - 1) ÷ 2 + 1
    tail = ntuple(_ -> Colon(), ndims(F) - 2)
    view(F, 1:np, :, tail...) .= view(H, 1:np, p.col, tail...)
    np < m1 && (view(F, (np + 1):m1, :, tail...) .= conj.(view(H, (m1 + 1 - np):-1:2, p.negcol, tail...)))
    return F
end

# A solution `G` of the real series on the `fftfreq` grid. `_expand_hermitian!` writes `G` at every
# grid frequency; along an even axis the grid's `-m/2` also receives `G` at `+m/2`, which the series
# holds and the grid does not, and the corner `(-m₁/2, -m₂/2)` receives `G(+m₁/2, +m₂/2)` as well. On
# the uniform nodes `2πj/m` the two frequencies coincide, so the grid coefficients are the DFT there.
function _expand_solution!(F::AbstractArray, p::NUFFTScatteringPlan, H::AbstractArray)
    _expand_hermitian!(F, p, H)
    m1, m2 = p.ms
    np = (m1 - 1) ÷ 2 + 1
    tail = ntuple(_ -> Colon(), ndims(F) - 2)
    # Row k₁ = -m₁/2 (grid row np + 1) takes G(+m₁/2, k₂), the half's last row.
    iseven(m1) && (view(F, (np + 1):(np + 1), :, tail...) .+= view(H, (np + 1):(np + 1), p.col, tail...))
    if iseven(m2)
        n2 = m2 + 1
        c = m2 ÷ 2 + 1                         # grid column of k₂ = -m₂/2
        hp = _fft_index(m2 ÷ 2, n2)            # half column of +m₂/2
        hm = _fft_index(-(m2 ÷ 2), n2)         # half column of -m₂/2
        # Column k₂ = -m₂/2 takes G(k₁, +m₂/2): the half itself for k₁ ≥ 0, the conjugate of
        # G(-k₁, -m₂/2) for k₁ < 0.
        view(F, 1:np, c:c, tail...) .+= view(H, 1:np, hp:hp, tail...)
        np < m1 && (view(F, (np + 1):m1, c:c, tail...) .+= conj.(view(H, (m1 + 1 - np):-1:2, hm:hm, tail...)))
        iseven(m1) && (view(F, (np + 1):(np + 1), c:c, tail...) .+= view(H, (np + 1):(np + 1), hp:hp, tail...))
    end
    return F
end

# The subspace a real solve runs in, an orthogonal projection the iterates stay inside. The DC row holds
# both `(0, k₂)` and `(0, -k₂)`, so it is replaced by its conjugate-symmetric part, whose complement the
# real synthesis maps to zero. On an even axis the Nyquist frequency is split evenly between `±m/2`,
# which keeps its cosine: the row `k₁ = m₁/2` is made conjugate-symmetric in the same way, and the
# columns `±m₂/2` are replaced by their mean.
function _project_half!(H::AbstractArray, p::NUFFTScatteringPlan)
    m1, m2 = p.ms
    tail = ntuple(_ -> Colon(), ndims(H) - 2)
    if iseven(m2)
        n2 = m2 + 1
        hp, hm = _fft_index(m2 ÷ 2, n2), _fft_index(-(m2 ÷ 2), n2)
        a, b = view(H, :, hp:hp, tail...), view(H, :, hm:hm, tail...)
        a .= (a .+ b) ./ 2
        b .= a
    end
    _symmetric_row!(H, 1, p, tail)
    iseven(m1) && _symmetric_row!(H, m1 ÷ 2 + 1, p, tail)
    return H
end

# Row `r` of the half replaced by its conjugate-symmetric part `(h[r, k₂] + conj(h[r, -k₂]))/2`.
function _symmetric_row!(H::AbstractArray, r::Int, p::NUFFTScatteringPlan, tail)
    row = view(H, r:r, :, tail...)
    p.dc .= (row .+ conj.(view(H, r:r, p.negcol_half, tail...))) ./ 2
    row .= p.dc
    return H
end

# The adjoint of the real Type-2 in the real inner product LSMR works in: FlowTransformBindings' Type-1
# with each row `k₁ > 0` counted for itself and its conjugate.
function _real_adjoint!(dst::AbstractArray, p::NUFFTScatteringPlan, src::AbstractArray)
    FTB.nufft_type1!(dst, p.rplan, src)
    view(dst, 2:size(dst, 1), ntuple(_ -> Colon(), ndims(dst) - 1)...) .*= 2
    return _project_half!(dst, p)
end

# ---------------------------------------------------------------------------
# The transforms
# ---------------------------------------------------------------------------

function inverse_transform!(out_pts::AbstractArray, plan::NUFFTScatteringPlan, Xmodes::AbstractArray)
    dst = _pts_out(plan, out_pts)
    FTB.nufft_type2!(dst, plan.cplan, Xmodes)
    @. out_pts = dst * plan.invN
    return out_pts
end

function forward_transform!(Xmodes::AbstractArray, plan::NUFFTScatteringPlan, x_pts::AbstractArray{<:Real})
    b = _real_in(plan, x_pts)
    if plan.solve
        _lsmr_solve_real!(Xmodes, plan, b)
    else
        FTB.nufft_type1!(plan.hx, plan.rplan, b)
        _expand_hermitian!(Xmodes, plan, plan.hx)
    end
    return Xmodes
end

function forward_transform!(Xmodes::AbstractArray, plan::NUFFTScatteringPlan, x_pts::AbstractArray{<:Complex})
    if plan.solve
        _lsmr_solve_complex!(Xmodes, plan, _cplx_in(plan, x_pts))
    else
        FTB.nufft_type1!(Xmodes, plan.cplan, _cplx_in(plan, x_pts))
    end
    return Xmodes
end

@inline function _rsolve(plan::NUFFTScatteringPlan)
    rs = plan.rsolve
    rs === nothing && throw(ArgumentError("plan was not built with `solve = true`"))
    return rs
end

@noinline function _csolve(p::NUFFTScatteringPlan{T, B, L, PC, PR, CV, MM}) where {T, B, L, PC, PR, CV, MM}
    got = p.csolve[]
    got === nothing || return got
    pts() = B == 1 ? similar(p.sx, Complex{T}, p.M) : similar(p.sx, Complex{T}, p.M, B)
    modes() = B == 1 ? similar(p.sx, Complex{T}, p.ms) : similar(p.sx, Complex{T}, (p.ms..., B))
    made = (pts(), modes(), modes(), modes(), modes())
    p.csolve[] = made
    return made::Tuple{CV, MM, MM, MM, MM}
end

# `A f̃ = x` is solved and scaled by `N` afterwards, which keeps intermediates at the field's own
# magnitude.
function _lsmr_solve_real!(f::AbstractArray, plan::NUFFTScatteringPlan{T, B}, b::AbstractArray) where {T, B}
    u, t, v, w, h, hbar = _rsolve(plan)
    applyA!(dst, src) = FTB.nufft_type2!(dst, plan.rplan, src)
    applyAt!(dst, src) = _real_adjoint!(dst, plan, src)
    info = B == 1 ?
        lsmr_solve!(plan.hx, applyA!, applyAt!, b, u, t, v, w, h, hbar;
                    damp = plan.damp, atol = plan.rtol, btol = plan.rtol,
                    conlim = inv(Base.eps(T)), maxiter = plan.maxiter) :
        lsmr_solve_batched!(plan.hx, applyA!, applyAt!, b, u, t, v, w, h, hbar, plan.ls_batch;
                            damp = plan.damp, atol = plan.rtol, btol = plan.rtol,
                            conlim = inv(Base.eps(T)), maxiter = plan.maxiter)
    _check_solve(info, plan.M, size(plan.hx)[1:2], plan.rtol, plan.maxiter)
    plan.hx .*= one(T) / plan.invN
    return _expand_solution!(f, plan, plan.hx)
end

function _lsmr_solve_complex!(f::AbstractArray, plan::NUFFTScatteringPlan{T, B}, x_pts::AbstractArray) where {T, B}
    t, v, w, h, hbar = _csolve(plan)
    applyA!(dst, src) = FTB.nufft_type2!(dst, plan.cplan, src)
    applyAt!(dst, src) = FTB.nufft_type1!(dst, plan.cplan, src)
    info = B == 1 ?
        lsmr_solve!(f, applyA!, applyAt!, x_pts, _cbuf(plan), t, v, w, h, hbar;
                    damp = plan.damp, atol = plan.rtol, btol = plan.rtol,
                    conlim = inv(Base.eps(T)), maxiter = plan.maxiter) :
        lsmr_solve_batched!(f, applyA!, applyAt!, x_pts, _cbuf(plan), t, v, w, h, hbar, plan.ls_batch;
                            damp = plan.damp, atol = plan.rtol, btol = plan.rtol,
                            conlim = inv(Base.eps(T)), maxiter = plan.maxiter)
    _check_solve(info, plan.M, plan.ms, plan.rtol, plan.maxiter)
    f .*= one(T) / plan.invN
    return f
end
