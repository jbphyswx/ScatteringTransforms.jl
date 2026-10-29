# ---------------------------------------------------------------------------
# Scattered planar transform by nonuniform FFT, through FlowTransformBindings
#
# The numeric contract of `DirectNUFFTPlan`: `forward_transform!` is the Type-1 adjoint onto the
# `fftfreq` mode grid, or an LSMR inversion under `LeastSquares()`; `inverse_transform!` is the
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
    NUFFTScatteringPlan

Scattered-planar plan over FlowTransformBindings NUFFT plans on fixed points: a complex plan for
synthesis and for complex samples, and a real plan for real samples and the real solve. `B` is the
number of co-located fields each execution transforms, and `transform` the NUFFT library tag the
plans run on.
"""
struct NUFFTScatteringPlan{T, L, PC, PR, CV <: AbstractMatrix{Complex{T}}, HX <: AbstractArray{Complex{T}},
                           PT <: AbstractMatrix{T}, RV <: AbstractVector{T}, IV <: AbstractVector{Int},
                           DC <: AbstractArray{Complex{T}}, A, RS, CS, RB, CB} <: AbstractScatteringPlan
    transform::L
    cplan::PC
    rplan::PR
    ms::NTuple{2, Int}
    M::Int
    B::Int
    invN::T                 # 1/prod(ms): synthesis is the ifft-convention inverse
    analysis::A
    maxiter::Int
    rtol::T
    damp::T                 # Tikhonov λ
    sx::RV                  # (M) points on the 2π-periodic domain, kept so a task can build its own
    sy::RV
    tol::T
    nthreads::Int
    hx::HX                  # the real plan's half spectrum, (ms[1] ÷ 2 + 1, n₂[, B])
    dc::DC                  # (1, n₂, B) scratch for the conjugate-symmetric rows
    rsolve::RS              # the half-grid solve's operator and `FTB.LSMRWorkspace`, or `nothing`
    csolve::CS
    rbuf::RB
    cbuf::CB
    col::IV                 # (ms[2]) column of the half holding each `fftfreq(ms[2])` frequency
    negcol::IV              # (ms[2]) column holding its negative
    negcol_half::IV         # (n₂) column holding the negative of each column's frequency
end

batch_width(p::NUFFTScatteringPlan) = p.B

_fft_index(f::Int, n::Int) = f >= 0 ? f + 1 : n + f + 1

function _like_ints(ref::AbstractArray, v::Vector{Int})
    d = similar(ref, Int, length(v))
    copyto!(d, v)
    return d
end

"""
    nufft_scattered_plan(backend, x, y, ms, T; period, origin = (0, 0), analysis, maxiter, rtol, damp,
                         tol, ntrans, nthreads) -> NUFFTScatteringPlan

The scattered-planar plan over points `(x, y)` and mode grid `ms` on a FlowTransformBindings NUFFT
`backend`, on a domain of length `period` with lower corner `origin` (see
[`make_scattered_plan`](@ref)). `nthreads = 0` takes the session's `Threads.nthreads()`.
"""
function nufft_scattered_plan(backend::SB.AbstractNUFFTSpectralBackend, x, y, ms::NTuple{2, Int}, ::Type{T};
                              period, origin = (0, 0), analysis::AbstractAnalysis = WeightedAdjoint(),
                              maxiter::Int = 100, tol = nothing,
                              rtol::Real = default_solver_rtol(T, backend, tol),
                              damp::Real = 0, ntrans::Int = 1, nthreads::Int = 0) where {T}
    M = length(x)
    sx, sy = _folded_points(x, y, period, origin, T)
    warn_underdetermined(M, ms, analysis, damp)
    tolerance = tol === nothing ? FTB.default_tolerance(T) : T(tol)
    return _nufft_plan_at(backend, ms, M, sx, sy, T(tolerance), T, analysis, maxiter,
                          T(rtol), ntrans, nthreads, T(damp))
end

function _nufft_plan_at(backend, ms::NTuple{2, Int}, M::Int, sx, sy, tol::T, ::Type{T},
                        analysis::AbstractAnalysis, maxiter,
                        rtol::T, B::Int, nthreads::Int, damp::T) where {T}
    n2 = ms[2] + iseven(ms[2])
    kw = (; ntrans = B, tol, order = FTB.FFTModes(), nthreads = nthreads > 0 ? nthreads : Threads.nthreads())
    cplan = FTB.plan_nufft(backend, Complex{T}, (sx, sy), ms; kw...)
    rplan = FTB.plan_nufft(backend, T, (sx, sy), (ms[1], n2); kw...)
    m1h = ms[1] ÷ 2 + 1
    f2 = [FTB.mode_frequency(FTB.FFTModes(), j, ms[2]) for j in 1:ms[2]]
    col = _like_ints(sx, [_fft_index(f, n2) for f in f2])
    negcol = _like_ints(sx, [_fft_index(-f, n2) for f in f2])
    negcol_half = _like_ints(sx, [_fft_index(-FTB.mode_frequency(FTB.FFTModes(), k, n2), n2) for k in 1:n2])
    hx = similar(sx, Complex{T}, m1h, n2, B)
    dc = similar(sx, Complex{T}, 1, n2, B)
    rsolve, csolve = _nufft_solvers(analysis, cplan, rplan, ms, dc, negcol_half)
    PT = typeof(similar(sx, T, 0, 0))
    CV = typeof(similar(sx, Complex{T}, 0, 0))
    rbuf = Base.RefValue{Union{Nothing,PT}}(nothing)
    cbuf = Base.RefValue{Union{Nothing,CV}}(nothing)
    return NUFFTScatteringPlan{T, typeof(backend), typeof(cplan), typeof(rplan), CV,
                               typeof(hx), PT, typeof(sx), typeof(col), typeof(dc),
                               typeof(analysis), typeof(rsolve), typeof(csolve), typeof(rbuf), typeof(cbuf)}(
        backend, cplan, rplan, ms, M, B, one(T) / prod(ms), analysis, maxiter, rtol, damp, sx, sy,
        tol, nthreads, hx, dc, rsolve, csolve, rbuf, cbuf, col, negcol, negcol_half)
end

# An operator and the workspace of its solve.
_solver(op) = (op, FTB.LSMRWorkspace(op))
_complex_solver(cplan) = _solver(FTB.NUFFTOperator(cplan))

_nufft_solvers(::WeightedAdjoint, cplan, rplan, ms, dc, negcol) = (nothing, nothing)
function _nufft_solvers(::LeastSquares, cplan, rplan, ms, dc, negcol)
    rsolve = _solver(_HalfSolve(FTB.NUFFTOperator(rplan), ms, dc, negcol))
    CS = Base.promote_op(_complex_solver, typeof(cplan))
    return rsolve, Base.RefValue{Union{Nothing,CS}}(nothing)
end

# The half-grid solve's operator: the real plan's type 2 over the half spectrum, with type 1 projected
# onto the subspace the solve runs in (`_project_half!`). The projection acts within rows, so it is
# orthogonal in the half's inner product, which weights each row `k₁ > 0` by two.
struct _HalfSolve{O, D, I}
    op::O
    ms::NTuple{2, Int}
    dc::D
    negcol_half::I
end

FTB.lsmr_ncolumns(s::_HalfSolve) = FTB.lsmr_ncolumns(s.op)
FTB.lsmr_allocate_domain(s::_HalfSolve) = FTB.lsmr_allocate_domain(s.op)
FTB.lsmr_allocate_range(s::_HalfSolve) = FTB.lsmr_allocate_range(s.op)
FTB.lsmr_domain_norm2!(out, s::_HalfSolve, v, n) = FTB.lsmr_domain_norm2!(out, s.op, v, n)
FTB.lsmr_forward!(u, s::_HalfSolve, v, c, n) = FTB.lsmr_forward!(u, s.op, v, c, n)

function FTB.lsmr_adjoint!(v, s::_HalfSolve, u, c, n)
    FTB.lsmr_adjoint!(v, s.op, u, c, n)
    return _project_half!(v, s.ms, s.dc, s.negcol_half)
end

Base.show(io::IO, p::NUFFTScatteringPlan{T}) where {T} =
    print(io, "NUFFTScatteringPlan{", T, "}(", nameof(typeof(p.transform)), ", ms=", p.ms, ", M=", p.M,
          ", ntrans=", p.B, ", analysis=", p.analysis, ")")
Base.show(io::IO, ::MIME"text/plain", p::NUFFTScatteringPlan) = show(io, p)

spectral_backend(p::NUFFTScatteringPlan) = p.transform
plan_points(p::NUFFTScatteringPlan) = (p.sx, p.sy)
plan_analysis(p::NUFFTScatteringPlan) =
    (analysis = p.analysis, maxiter = p.maxiter, rtol = p.rtol, damp = p.damp,
     tol = p.tol, nthreads = p.nthreads)

# The FlowTransformBindings plans hold the scratch every execution writes through, so a task builds its
# own over the same points, at one thread unless one was asked for.
task_local_plan(p::NUFFTScatteringPlan{T}) where {T} =
    _nufft_plan_at(p.transform, p.ms, p.M, p.sx, p.sy, p.tol, T, p.analysis, p.maxiter, p.rtol, p.B,
                   per_task_nthreads(p.nthreads), p.damp)

close_plan!(p::NUFFTScatteringPlan) = (FTB.close!(p.cplan); FTB.close!(p.rplan); nothing)

# ---------------------------------------------------------------------------
# Reaching the transform without a copy. The cascade's buffers are `similar` to the plan's points, so
# they already have the plan's array types and pass straight through; any other array is copied into a
# buffer made on its first use.
# ---------------------------------------------------------------------------

_point_matrix(x::AbstractVector) = reshape(x, :, 1)
_point_matrix(x::AbstractMatrix) = x

@inline function _real_in(p::NUFFTScatteringPlan{T,L,PC,PR,CV,HX,PT,RV},
                          x::AbstractArray{<:Real}) where {T,L,PC,PR,CV,HX,PT,RV}
    return x isa Union{PT,RV} ? x : copyto!(_point_buffer!(p.rbuf, p.sx, p.M, p.B, PT), x)
end

@inline function _pts_out(p::NUFFTScatteringPlan{T,L,PC,PR,CV},
                          out::AbstractArray) where {T,L,PC,PR,CV}
    values = _point_matrix(out)
    return values isa CV ? values : _point_buffer!(p.cbuf, p.sx, p.M, p.B, CV)
end

@inline function _cplx_in(p::NUFFTScatteringPlan, x::AbstractArray)
    values = _point_matrix(x)
    destination = _pts_out(p, values)
    return destination === values ? values : copyto!(destination, values)
end

@inline _cplx_in(::NUFFTScatteringPlan{T,L,PC,PR,Matrix{Complex{T}}},
                 x::Vector{Complex{T}}) where {T,L,PC,PR} = x
@inline _pts_out(::NUFFTScatteringPlan{T,L,PC,PR,Matrix{Complex{T}}},
                 out::Vector{Complex{T}}) where {T,L,PC,PR} = out

@noinline function _point_buffer!(cache, ref, M, B, ::Type{A}) where {A}
    buffer = cache[]
    if buffer === nothing
        buffer = similar(ref, eltype(A), M, B)
        cache[] = buffer
    end
    return buffer::A
end

@noinline function _complex_solver!(cache, cplan)
    solver = cache[]
    if solver === nothing
        solver = _complex_solver(cplan)
        cache[] = solver
    end
    return solver
end

# ---------------------------------------------------------------------------
# The half spectrum and the full `fftfreq` grid
# ---------------------------------------------------------------------------

# Rows `1:np` of the full grid hold `k₁ = 0 … np - 1`, read from the half; the rest hold `k₁ < 0`, the
# conjugates of the half at `(-k₁, -k₂)`.
_half_view(H, rows, cols, ::Val{2}) = view(H, rows, cols, 1)
_half_view(H, rows, cols, ::Val{3}) = view(H, rows, cols, :)

function _expand_hermitian!(F::AbstractArray{T,N}, p::NUFFTScatteringPlan, H::AbstractArray{<:Any,3}) where {T,N}
    m1 = p.ms[1]
    np = (m1 - 1) ÷ 2 + 1
    tail = ntuple(_ -> Colon(), Val(N - 2))
    view(F, 1:np, :, tail...) .= _half_view(H, 1:np, p.col, Val(N))
    np < m1 && (view(F, (np + 1):m1, :, tail...) .=
                conj.(_half_view(H, (m1 + 1 - np):-1:2, p.negcol, Val(N))))
    return F
end

# A solution `G` of the real series on the `fftfreq` grid. `_expand_hermitian!` writes `G` at every
# grid frequency; along an even axis the grid's `-m/2` also receives `G` at `+m/2`, which the series
# holds and the grid does not, and the corner `(-m₁/2, -m₂/2)` receives `G(+m₁/2, +m₂/2)` as well. On
# the uniform nodes `2πj/m` the two frequencies coincide, so the grid coefficients are the DFT there.
function _expand_solution!(F::AbstractArray{T,N}, p::NUFFTScatteringPlan, H::AbstractArray{<:Any,3}) where {T,N}
    _expand_hermitian!(F, p, H)
    m1, m2 = p.ms
    np = (m1 - 1) ÷ 2 + 1
    tail = ntuple(_ -> Colon(), Val(N - 2))
    # Row k₁ = -m₁/2 (grid row np + 1) takes G(+m₁/2, k₂), the half's last row.
    iseven(m1) && (view(F, (np + 1):(np + 1), :, tail...) .+=
                   _half_view(H, (np + 1):(np + 1), p.col, Val(N)))
    if iseven(m2)
        n2 = m2 + 1
        c = m2 ÷ 2 + 1                         # grid column of k₂ = -m₂/2
        hp = _fft_index(m2 ÷ 2, n2)            # half column of +m₂/2
        hm = _fft_index(-(m2 ÷ 2), n2)         # half column of -m₂/2
        # Column k₂ = -m₂/2 takes G(k₁, +m₂/2): the half itself for k₁ ≥ 0, the conjugate of
        # G(-k₁, -m₂/2) for k₁ < 0.
        view(F, 1:np, c:c, tail...) .+= _half_view(H, 1:np, hp:hp, Val(N))
        np < m1 && (view(F, (np + 1):m1, c:c, tail...) .+=
                    conj.(_half_view(H, (m1 + 1 - np):-1:2, hm:hm, Val(N))))
        iseven(m1) && (view(F, (np + 1):(np + 1), c:c, tail...) .+=
                       _half_view(H, (np + 1):(np + 1), hp:hp, Val(N)))
    end
    return F
end

# The subspace a real solve runs in, an orthogonal projection the iterates stay inside. The DC row holds
# both `(0, k₂)` and `(0, -k₂)`, so it is replaced by its conjugate-symmetric part, whose complement the
# real synthesis maps to zero. On an even axis the Nyquist frequency is split evenly between `±m/2`,
# which keeps its cosine: the row `k₁ = m₁/2` is made conjugate-symmetric in the same way, and the
# columns `±m₂/2` are replaced by their mean. `dc` is `(1, n₂, B)` scratch and `negcol_half` the column
# of each column's negative frequency.
function _project_half!(H::AbstractArray{<:Any,3}, ms::NTuple{2, Int}, dc::AbstractArray, negcol_half::AbstractVector)
    m1, m2 = ms
    if iseven(m2)
        n2 = m2 + 1
        hp, hm = _fft_index(m2 ÷ 2, n2), _fft_index(-(m2 ÷ 2), n2)
        a, b = view(H, :, hp:hp, :), view(H, :, hm:hm, :)
        a .= (a .+ b) ./ 2
        b .= a
    end
    _symmetric_row!(H, 1, dc, negcol_half)
    iseven(m1) && _symmetric_row!(H, m1 ÷ 2 + 1, dc, negcol_half)
    return H
end

# Row `r` of the half replaced by its conjugate-symmetric part `(h[r, k₂] + conj(h[r, -k₂]))/2`.
function _symmetric_row!(H::AbstractArray{<:Any,3}, r::Int, dc::AbstractArray, negcol_half::AbstractVector)
    row = view(H, r:r, :, :)
    dc .= (row .+ conj.(view(H, r:r, negcol_half, :))) ./ 2
    row .= dc
    return H
end

# ---------------------------------------------------------------------------
# The transforms
# ---------------------------------------------------------------------------

function inverse_transform!(out_pts::AbstractArray, plan::NUFFTScatteringPlan, Xmodes::AbstractArray)
    dst = _pts_out(plan, out_pts)
    FTB.nufft_type2!(dst, plan.cplan, Xmodes)
    dst === out_pts || copyto!(out_pts, dst)
    out_pts .*= plan.invN
    return out_pts
end

function forward_transform!(Xmodes::AbstractArray, plan::NUFFTScatteringPlan, x_pts::AbstractArray{<:Real})
    b = _real_in(plan, x_pts)
    _forward_real!(Xmodes, plan, b, plan.analysis)
    return Xmodes
end

function forward_transform!(Xmodes::AbstractArray, plan::NUFFTScatteringPlan, x_pts::AbstractArray{<:Complex})
    _forward_complex!(Xmodes, plan, _cplx_in(plan, x_pts), plan.analysis)
    return Xmodes
end

_forward_real!(f, plan, b, ::LeastSquares) = _lsmr_solve_real!(f, plan, b)
_forward_complex!(f, plan, b, ::LeastSquares) = _lsmr_solve_complex!(f, plan, b)

function _forward_real!(f, plan, b, ::WeightedAdjoint)
    FTB.nufft_type1!(plan.hx, plan.rplan, b)
    return _expand_hermitian!(f, plan, plan.hx)
end

_forward_complex!(f, plan, b, ::WeightedAdjoint) = FTB.nufft_type1!(f, plan.cplan, b)

# `A f̃ = x` is solved and scaled by `N` afterwards, which keeps intermediates at the field's own
# magnitude. At `cond(A) = 1/eps` the smallest singular direction carries nothing the precision can
# represent, so that is `conlim`.
function _lsmr_solve_real!(f::AbstractArray, plan::NUFFTScatteringPlan{T}, b::AbstractArray) where {T}
    op, ws = plan.rsolve
    FTB.lsmr!(plan.hx, op, b, ws; rtol = plan.rtol, damp = plan.damp,
        conlim = inv(Base.eps(T)), maxiter = plan.maxiter)
    plan.hx .*= one(T) / plan.invN
    return _expand_solution!(f, plan, plan.hx)
end

function _lsmr_solve_complex!(f::AbstractArray, plan::NUFFTScatteringPlan{T}, x_pts::AbstractArray) where {T}
    op, ws = _complex_solver!(plan.csolve, plan.cplan)
    FTB.lsmr!(f, op, x_pts, ws; rtol = plan.rtol, damp = plan.damp,
        conlim = inv(Base.eps(T)), maxiter = plan.maxiter)
    f .*= one(T) / plan.invN
    return f
end
