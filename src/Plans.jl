module Plans

"""
    Plans.jl — Spectral transform plans

The scattering transform needs a forward/inverse spectral transform to convolve with the
frequency-domain wavelets. The core ships a dependency-free default — a **direct-summation DFT** —
and fast paths live in extensions (FFTW for uniform sampling, `AbstractFFTs` on device, FINUFFT /
NonuniformFFTs for scattered points).

A plan implements two in-place primitives:

    forward_transform!(out, plan, x)   # x -> X̂   (fft convention)
    inverse_transform!(out, plan, x)   # X̂ -> x   (ifft convention, 1/N scaled)

so the transform engine never references FFTW/CUDA directly. Plans transform the leading `D`
dimensions of their argument and leave any trailing dimension as an untouched batch axis, matching
`AbstractFFTs.plan_fft(A, 1:D)`.

Which transform a plan performs is selected by a `SpectralBackends` tag, not by a `Symbol`.
"""

using LinearAlgebra: LinearAlgebra
using SpectralBackends: SpectralBackends as SB

"""
    AbstractScatteringPlan

Supertype for spectral transform plans. A plan implements [`forward_transform!`](@ref) and
[`inverse_transform!`](@ref); the in-core default is [`DirectSumPlan`](@ref), with fast paths
(FFTW, `AbstractFFTs`/device, FINUFFT, NonuniformFFTs) provided by extensions.
"""
abstract type AbstractScatteringPlan end

"""
    forward_transform!(out, plan, x) -> out

In-place forward (fft-convention) spectral transform. Methods provided by concrete plans.
"""
function forward_transform! end

"""
    inverse_transform!(out, plan, x) -> out

In-place inverse (ifft-convention, `1/N`-scaled) spectral transform.
"""
function inverse_transform! end

"""
    inplace_inverse(plan) -> Bool

Whether `inverse_transform!(x, plan, x)` is valid. When it is, a transform points its convolution
output at its multiply scratch and owns one fewer field-sized array; when it is not, the two stay
distinct. Default `false`: the direct-sum plan alternates through its own scratch and aliasing would
corrupt an odd number of axes.
"""
inplace_inverse(::AbstractScatteringPlan) = false

"""
    forward_transform(plan, x) -> X̂
    inverse_transform(plan, x) -> x

Non-mutating, allocating, element-type-generic spectral transforms — the autodiff-friendly
counterparts of the in-place `forward_transform!`/`inverse_transform!`. They never touch the plan's
preallocated scratch and never mutate their inputs, so they accept `ForwardDiff.Dual`/`Float32`
inputs and are differentiable by reverse-mode backends. Used by the non-mutating `scattering(st, x)`
path; the in-place `!` versions remain the production hot path.

[`DirectSumPlan`](@ref) implements these as dense per-axis DFT matrix-multiplies (`W*x`) —
differentiable by every AD backend with no special rules. The matrices are built once on first use
and cached on the plan, so the `O(N²)` build is not repeated per call.
"""
function forward_transform end

"""
    inverse_transform(plan, x) -> x

See [`forward_transform`](@ref).
"""
function inverse_transform end

"""
    spectral_backend(plan) -> SpectralBackends.AbstractSpectralBackend

The tag that would rebuild `plan`. Each plan type declares its own, so a transform can be
reconstructed faithfully on a remote worker; plans that cannot be rebuilt from a tag alone (a
device-resident FFT plan needs its device too) throw rather than report a host plan.
"""
spectral_backend(plan::AbstractScatteringPlan) = throw(ArgumentError(
    "no spectral backend tag is registered for $(typeof(plan)), so a transform using it cannot be " *
    "rebuilt from a serialisable spec — construct the transform on each worker explicitly."))

"""
    nufft_guru_make(points, type, ms, iflag, ntrans, eps, T; nthreads = 0) -> guru plan
    nufft_guru_setpts!(guru, x, y) -> guru
    nufft_guru_exec!(guru, input, output) -> output

Creation, point assignment and execution of a FINUFFT guru plan, split so that a device binding is
one method rather than a second copy of the scattered-planar plan.

Creation dispatches on the point array, since that is what decides where the transform has to run;
the other two dispatch on the returned plan, so each backend's handle carries its own execution. The
host methods live in the FINUFFT extension, the CUDA ones in the cuFINUFFT extension — only the NUFFT
is vendor-specific, because the cascade around it is broadcasts and reductions over whatever array
type the points are.

`nthreads` is the library's own thread count, baked into the plan, with `0` meaning "the library's
default" (all cores). Measured on the scattered-planar shapes: the library's threading is worth
2.2–3.3× at `M ≳ 2·10⁴` and breaks even at `M ~ 500`, so the default keeps it. A plan built per task
defaults to one instead — see [`per_task_nthreads`](@ref). A device binding has no CPU threads to set
and ignores it.
"""
function nufft_guru_make end
function nufft_guru_setpts! end
function nufft_guru_exec! end
function nufft_guru_destroy! end

"""
    close_plan!(plan) -> nothing

Release the foreign-library resources `plan` owns, now rather than at collection. No-op by default,
and safe to call more than once.

Whoever *builds* a plan per task must call this when the task is done. These destructors take a lock
— FINUFFT installs one to serialise its FFTW planner calls — and a lock cannot be taken from a GC
finalizer, so leaving them to be collected aborts the process the moment one fires inside another
task's transform. Closing eagerly leaves the finalizer nothing to do, since it checks whether the C
plan is already gone.
"""
close_plan!(::Any) = nothing

"""
    per_task_nthreads(requested) -> Int

Thread count for a library plan built inside a task: `requested` if the caller asked for one, else 1.

An explicit request is honoured here and not just at construction, or `nufft_nthreads` would be a
keyword that silently does nothing under a threaded backend. The default is one because
[`task_local_plan`](@ref) builds a plan per task, so the tasks have already claimed the cores.

Deriving the default from `Sys.CPU_THREADS` instead gets this wrong twice: those are logical cores, so
on a hyperthreaded machine running one Julia task per *physical* core — already a full machine — the
arithmetic hands every task a second library thread.
"""
per_task_nthreads(requested::Integer) = requested > 0 ? Int(requested) : 1

"""
    AnalysisNotConverged <: Exception

Thrown when an iterative analysis cannot produce a usable answer.

Not thrown merely for stopping early: LSMR decreases both `‖r‖` and `‖A†r‖` monotonically, so an
iterate truncated at `maxiter` is the best one seen and is returned. This is for the cases where the
answer means nothing — a non-finite residual, or a conditioning estimate past what the precision can
represent — because those propagate into every coefficient downstream, and a silent `NaN` is worse
than a stop.
"""
struct AnalysisNotConverged{T <: Real} <: Exception
    residual::T
    rtol::T
    iters::Int
    maxiter::Int
    detail::String
end

AnalysisNotConverged(residual::Real, rtol::Real, iters::Integer, maxiter::Integer,
                     detail::AbstractString) =
    AnalysisNotConverged{promote_type(typeof(residual), typeof(rtol))}(
        residual, rtol, Int(iters), Int(maxiter), String(detail))

function Base.showerror(io::IO, e::AnalysisNotConverged)
    print(io, "AnalysisNotConverged: reached relative residual ", e.residual, " after ", e.iters,
          " of ", e.maxiter, " iterations, against rtol = ", e.rtol, ". ")
    return print(io, e.detail)
end

# `istop` 3 and 6 are the conditioning limits — `cond(A)` past `conlim`, and past what the precision
# can represent. A non-finite residual means the recurrence itself broke down. Everything else,
# `istop = 7` (ran out of iterations) included, is a usable iterate.
function _check_solve(info, M::Integer, ms::Tuple, rtol::Real, maxiter::Integer)
    (isfinite(info.normr) && info.istop != 3 && info.istop != 6) && return nothing
    n = prod(ms)
    advice = n > M ?
        "The mode grid asks for more coefficients than there are samples ($n modes, $M points), so " *
        "$(n - M) mode directions are constrained by no sample: reduce `ms`, supply more points, or " *
        "raise `damp` to regularise them." :
        "The sampling may not determine the mode grid ($n modes, $M points) — check the point set " *
        "for gaps, reduce `ms`, or raise `damp`."
    throw(AnalysisNotConverged(info.normr, rtol, info.iters, maxiter,
                              "cond(A) ≈ $(Float32(info.condA)), istop = $(info.istop). " * advice))
end

# ---------------------------------------------------------------------------
# Least-squares solver (LSMR, Fong & Saunders 2011)
#
# Finds modes `x` minimising `‖A x − b‖² + λ²‖x‖²` for the nonuniform transform pair `A` (modes →
# points, Type-2) and `A†` (points → modes, Type-1), given only closures that apply them.
#
# LSMR rather than conjugate gradients on `A†A`: `A` is a nonuniform DFT, never Hermitian positive
# definite at any size, so CG is only available on the normal equations — which costs the same two
# transforms per iteration while squaring the condition number. On a rank-deficient point set (fewer
# samples than modes, or a gap) that CG is semiconvergent: it reaches its best answer in a couple of
# iterations and then diverges, unguarded, into `Inf − Inf`. LSMR decreases both `‖r‖` and `‖A†r‖`
# monotonically, so stopping anywhere — including at `maxiter` — returns the best iterate seen, and its
# stopping quantities come out of the recurrence scalars for free.
# ---------------------------------------------------------------------------

#     _sym_ortho(a, b) -> (c, s, r)
#
# Givens rotation zeroing `b` into `a`: `c*a + s*b == r`, `-s*a + c*b == 0`, with `r ≥ 0`.
#
# `hypot` rather than `sqrt(a^2 + b^2)`, which overflows for `a, b` above `sqrt(floatmax(T))` —
# reachable in `Float32` on the norms this is fed.
@inline function _sym_ortho(a::T, b::T) where {T<:Real}
    r = hypot(a, b)
    r == 0 && return (one(T), zero(T), r)
    return (a / r, b / r, r)
end

"""
    LSMRState{T}

The scalar state of an LSMR iteration: the bidiagonalisation and rotation quantities, the running
estimates of `‖r‖`, `‖A†r‖`, `‖A‖` and `cond(A)`, and the stopping code.

`normA` accumulates `Σ(α² + β²)`, so it estimates the **Frobenius** norm of the bidiagonalisation
built so far — bounded above by `‖A‖_F` and generally above `‖A‖₂`. That is what the optimality test
`‖A†r‖ ≤ atol·normA·‖r‖` is scaled by, which makes it slightly conservative rather than wrong.

Immutable and advanced by [`lsmr_step`](@ref), which touches no arrays — so the batched solver can hold
one of these per column and drive them from the host while the vector work stays on the device.
"""
struct LSMRState{T<:Real}
    alphabar::T
    rho::T
    rhobar::T
    cbar::T
    sbar::T
    zeta::T
    zetabar::T
    betadd::T           # ‖r‖ estimation (Fong & Saunders §3.3)
    betad::T
    rhodold::T
    tautildeold::T
    thetatilde::T
    d::T
    normA2::T           # ‖A‖ and cond(A) estimation
    maxrbar::T
    minrbar::T
    normb::T
    normr::T
    normar::T
    normA::T
    condA::T
    istop::Int
    iters::Int
end

"""
    lsmr_init(alpha, beta, ::Type{T}) -> LSMRState{T}

Initial state from the first bidiagonalisation pair, `beta = ‖b‖` and `alpha = ‖A†b/beta‖`.
"""
function lsmr_init(alpha::T, beta::T) where {T<:Real}
    return LSMRState{T}(alpha, one(T), one(T), one(T), zero(T), zero(T), alpha * beta,
                        beta, zero(T), one(T), zero(T), zero(T), zero(T),
                        alpha * alpha, zero(T), typemax(T),
                        beta, beta, alpha * beta, sqrt(alpha * alpha), one(T), 0, 0)
end

"""
    lsmr_step(state, alpha, beta, damp, atol, btol, ctol) -> (state, chbar, cx, ch)

Advance the scalar recurrence one iteration and return the three coefficients the vector updates need:

    hbar .= h .+ chbar .* hbar
    x    .= x .+ cx    .* hbar
    h    .= v .+ ch    .* h

`alpha`/`beta` are this iteration's bidiagonalisation norms. `damp` is the Tikhonov `λ`, which enters
only through one extra rotation, so `λ = 0` costs a rotation of a zero and nothing else.
"""
function lsmr_step(st::LSMRState{T}, alpha::T, beta::T, damp::T,
                   atol::T, btol::T, ctol::T) where {T<:Real}
    # Damping folds in as a rotation of (alphabar, λ) — the augmented system [A; λI] without forming it.
    chat, shat, alphahat = _sym_ortho(st.alphabar, damp)

    rhoold = st.rho
    c, s, rho = _sym_ortho(alphahat, beta)
    thetanew = s * alpha
    alphabar = c * alpha

    rhobarold = st.rhobar
    zetaold = st.zeta
    thetabar = st.sbar * rho
    rhotemp = st.cbar * rho
    cbar, sbar, rhobar = _sym_ortho(st.cbar * rho, thetanew)
    zeta = cbar * st.zetabar
    zetabar = -sbar * st.zetabar

    # Exact termination. When the bidiagonalisation runs out of Krylov space it returns
    # `alpha = beta = 0`, which sends `rho` — and with it `rhobar` — to zero, and all three update
    # coefficients below divide by them. The iterate at that point is the answer the process reached,
    # so this reports a stop rather than dividing.
    #
    # It has to be caught here, not by the caller's loop. The stopping tests are computed *after* these
    # coefficients, so a division by zero produces `NaN` before any `istop` can be set; and a batched
    # solve shares one loop across its columns, freezing a column only once its `istop` is nonzero, so
    # the `NaN` would then be carried through every later iteration. Reached whenever the system is
    # consistent enough to be solved exactly — a smooth field on a well-sampled point set is enough.
    if iszero(rho) || iszero(rhobar)
        return (LSMRState{T}(alphabar, rho, rhobar, cbar, sbar, zeta, zetabar,
                             st.betadd, st.betad, st.rhodold, st.tautildeold, st.thetatilde, st.d,
                             st.normA2, st.maxrbar, st.minrbar, st.normb, st.normr, st.normar,
                             st.normA, st.condA, st.istop == 0 ? 1 : st.istop, st.iters + 1),
                zero(T), zero(T), zero(T))
    end

    chbar = -(thetabar * rho / (rhoold * rhobarold))
    cx = zeta / (rho * rhobar)
    ch = -(thetanew / rho)

    # ‖r‖ without recomputing it: the residual of the bidiagonal subproblem, which stays tied to the
    # true residual instead of drifting from it the way a recursively updated `r` does.
    betaacute = chat * st.betadd
    betacheck = -shat * st.betadd
    betahat = c * betaacute
    betadd = -s * betaacute

    thetatildeold = st.thetatilde
    ctildeold, stildeold, rhotildeold = _sym_ortho(st.rhodold, thetabar)
    thetatilde = stildeold * rhobar
    rhodold = ctildeold * rhobar
    betad = -stildeold * st.betad + ctildeold * betahat

    tautildeold = (zetaold - thetatildeold * st.tautildeold) / rhotildeold
    taud = (zeta - thetatilde * tautildeold) / rhodold
    d = st.d + betacheck * betacheck
    normr = sqrt(d + (betad - taud)^2 + betadd^2)

    normA2 = st.normA2 + beta * beta
    normA = sqrt(normA2)
    normA2 += alpha * alpha

    maxrbar = max(st.maxrbar, rhobarold)
    # `rhobarold` is meaningless on the first pass (it is the initial 1), so cond(A) ignores it.
    minrbar = st.iters == 0 ? st.minrbar : min(st.minrbar, rhobarold)
    condA = max(maxrbar, rhotemp) / min(minrbar, rhotemp)

    normar = abs(zetabar)

    # `1 + t <= 1` are the "as small as this precision can express" forms, free to test.
    test1 = st.normb == 0 ? zero(T) : normr / st.normb
    test2 = (normA * normr) == 0 ? T(Inf) : normar / (normA * normr)
    test3 = inv(condA)
    istop = 0
    (1 + test3 <= 1) && (istop = 6)
    (1 + test2 <= 1) && (istop = 5)
    (test3 <= ctol) && (istop = 3)
    (test2 <= atol) && (istop = 2)
    (test1 <= btol) && (istop = 1)

    return (LSMRState{T}(alphabar, rho, rhobar, cbar, sbar, zeta, zetabar,
                         betadd, betad, rhodold, tautildeold, thetatilde, d,
                         normA2, maxrbar, minrbar,
                         st.normb, normr, normar, normA, condA, istop, st.iters + 1),
            chbar, cx, ch)
end

"""
    lsmr_solve!(x, applyA!, applyAt!, b, u, t, v, w, h, hbar;
                damp, atol, btol, conlim, maxiter) -> (; istop, iters, normr, normar, normA, condA)

Minimise `‖A x − b‖² + damp²‖x‖²` in place, where `applyA!(dst_pts, src_modes)` applies `A` and
`applyAt!(dst_modes, src_pts)` applies `A†`.

`b` is read only. `u`/`t` are point-space scratch and `v`/`w`/`h`/`hbar` mode-space scratch; the two
destinations `t` and `w` exist because the transforms overwrite rather than accumulate. Every array
operation is `copyto!`, `fill!`, `norm` or a fused broadcast, so the buffers may live on a device.

`istop` says why it stopped: 1 the residual met `btol`, 2 the least-squares optimality met `atol`,
3 the condition estimate hit `conlim`, 5/6 those quantities reached the precision floor, 7 `maxiter`.
Stopping at 7 is not a failure — both `‖r‖` and `‖A†r‖` decrease monotonically, so the iterate is the
best one seen.
"""
function lsmr_solve!(x, applyA!::FA, applyAt!::FT, b, u, t, v, w, h, hbar;
                     damp::Real = 0, atol::Real, btol::Real, conlim::Real,
                     maxiter::Integer) where {FA, FT}
    T = real(eltype(x))
    λ = T(damp)
    at = T(atol)
    bt = T(btol)
    ct = conlim > 0 ? T(inv(conlim)) : zero(T)

    fill!(x, zero(eltype(x)))
    copyto!(u, b)
    beta = T(LinearAlgebra.norm(u))
    beta > 0 && (u .*= inv(beta))
    applyAt!(v, u)
    alpha = T(LinearAlgebra.norm(v))
    alpha > 0 && (v .*= inv(alpha))

    # A zero right-hand side, or a right-hand side entirely in the null space of `A†`, leaves `x = 0`
    # as the exact answer — there is no direction to descend.
    (beta == 0 || alpha == 0) &&
        return (istop = 0, iters = 0, normr = beta, normar = zero(T), normA = alpha, condA = one(T))

    copyto!(h, v)
    fill!(hbar, zero(eltype(hbar)))
    state = lsmr_init(alpha, beta)
    iters = 0

    for k in 1:maxiter
        iters = k
        applyA!(t, v)
        @. u = t - alpha * u
        beta = T(LinearAlgebra.norm(u))
        if beta > 0
            u .*= inv(beta)
            applyAt!(w, u)
            @. v = w - beta * v
            alpha = T(LinearAlgebra.norm(v))
            alpha > 0 && (v .*= inv(alpha))
        end

        state, chbar, cx, ch = lsmr_step(state, alpha, beta, λ, at, bt, ct)

        @. hbar = h + chbar * hbar
        @. x = x + cx * hbar
        @. h = v + ch * h

        state.istop == 0 || break
    end
    istop = state.istop == 0 ? 7 : state.istop
    return (istop = istop, iters = iters, normr = state.normr, normar = state.normar,
            normA = state.normA, condA = state.condA)
end

"""
    BatchedLSMRWork{A2,A3,HV,SV}

Per-column bookkeeping for [`lsmr_solve_batched!`](@ref): the two reduction targets, the three
coefficient arrays the vector updates broadcast against, host mirrors of each of those five, and one
[`LSMRState`](@ref) per column.

The device arrays are `(1, B)` over points and `(1, 1, B)` over modes so they broadcast against the
stacks with no reshape in the loop, and they are preallocated because `sum(abs2, x; dims = …)`
allocates on every call. Each norm is held twice — as both ranks, over the same memory — because the
point stack is rank 2 and the mode stack rank 3, and a `(1, 1, B)` array broadcast against `(M, B)`
would expand to `(M, B, B)` rather than scaling columns. Each *quantity* nonetheless gets its own
buffer: the recurrence needs this iteration's `alpha` while the previous one is still live.
"""
struct BatchedLSMRWork{A2, A3, HV, SV}
    nrm_p::A2      # (1, B)        per-column ‖u‖, i.e. `beta`
    nrm_p3::A3     # the same memory as (1, 1, B), to broadcast `beta` against the mode stack
    nrm_m::A3      # (1, 1, B)     per-column ‖v‖, i.e. `alpha`
    nrm_m2::A2     # the same memory as (1, B), to broadcast `alpha` against the point stack
    c_hbar::A3
    c_x::A3
    c_h::A3
    beta::HV       # host mirrors of the norms …
    alpha::HV
    chbar::HV      # … and of the coefficients on their way back to the device
    cx::HV
    ch::HV
    states::SV
end

# Per-column `‖·‖`, fused and allocation-free: `mapreducedim!` over the leading axes, then a
# broadcast `sqrt` in place — the idiom `Batched.slice_modulus_mean!` already uses.
function _col_norm!(dst, a)
    fill!(dst, zero(eltype(dst)))
    Base.mapreducedim!(abs2, +, dst, a)
    @. dst = sqrt(dst)
    return dst
end

# `x ./= nrm` per column, leaving a zero-norm column alone rather than dividing by zero: such a column
# must stay finite, because the transform it shares with the others would otherwise be handed an `Inf`.
_col_scale!(x, nrm) = (@. x = x * ifelse(nrm > 0, inv(nrm), zero(eltype(nrm))); x)

"""
    lsmr_solve_batched!(x, applyA!, applyAt!, b, u, t, v, w, h, hbar, work;
                        damp, atol, btol, conlim, maxiter) -> (; istop, iters, normr, normar, …)

[`lsmr_solve!`](@ref) over a stack of `B` right-hand sides sharing one operator.

A batched NUFFT plan's width is fixed when it is built, so no column can be transformed on its own and
the stack must advance together — which turns every scalar in the recurrence into one per column. They
advance on the host through the same [`lsmr_step`](@ref) the single-column path uses, and return to the
device as three coefficient arrays.

A column that has stopped is *frozen*: its coefficients go to zero, so it contributes nothing further.
It is not compacted out of the stack, because the transform width cannot shrink — removing it would
save no work while costing the bookkeeping that makes a permuted, partially-retired stack correct.

`istop`/`normr`/`normar` describe the worst column, so a caller that checks them sees the whole stack.
"""
function lsmr_solve_batched!(x, applyA!::FA, applyAt!::FT, b, u, t, v, w, h, hbar,
                             work::BatchedLSMRWork; damp::Real = 0, atol::Real, btol::Real,
                             conlim::Real, maxiter::Integer) where {FA, FT}
    T = real(eltype(x))
    λ = T(damp); at = T(atol); bt = T(btol)
    ct = conlim > 0 ? T(inv(conlim)) : zero(T)
    B = length(work.states)

    fill!(x, zero(eltype(x)))
    copyto!(u, b)
    _col_norm!(work.nrm_p, u)
    _col_scale!(u, work.nrm_p)
    copyto!(work.beta, work.nrm_p)
    applyAt!(v, u)
    _col_norm!(work.nrm_m, v)
    _col_scale!(v, work.nrm_m)
    copyto!(work.alpha, work.nrm_m)

    @inbounds for c in 1:B
        work.states[c] = lsmr_init(work.alpha[c], work.beta[c])
    end
    copyto!(h, v)
    fill!(hbar, zero(eltype(hbar)))
    iters = 0

    for k in 1:maxiter
        iters = k
        # `nrm_m` still holds the previous iteration's `alpha` here, which is what the `u` update
        # needs; `nrm_p` then becomes this iteration's `beta` before the `v` update reads it.
        applyA!(t, v)
        @. u = t - work.nrm_m2 * u
        _col_norm!(work.nrm_p, u)
        _col_scale!(u, work.nrm_p)
        copyto!(work.beta, work.nrm_p)
        applyAt!(w, u)
        @. v = w - work.nrm_p3 * v
        _col_norm!(work.nrm_m, v)
        _col_scale!(v, work.nrm_m)
        copyto!(work.alpha, work.nrm_m)

        done = true
        @inbounds for c in 1:B
            if work.states[c].istop != 0
                work.chbar[c] = zero(T); work.cx[c] = zero(T); work.ch[c] = zero(T)
                continue
            end
            st, chbar, cx, ch = lsmr_step(work.states[c], work.alpha[c], work.beta[c], λ, at, bt, ct)
            work.states[c] = st
            work.chbar[c] = chbar
            work.cx[c] = cx
            work.ch[c] = ch
            done &= st.istop != 0
        end
        copyto!(work.c_hbar, work.chbar)
        copyto!(work.c_x, work.cx)
        copyto!(work.c_h, work.ch)

        @. hbar = h + work.c_hbar * hbar
        @. x = x + work.c_x * hbar
        @. h = v + work.c_h * h

        done && break
    end

    worst = argmax(c -> work.states[c].normar, 1:B)
    st = work.states[worst]
    return (istop = st.istop == 0 ? 7 : st.istop, iters = iters, normr = st.normr,
            normar = st.normar, normA = st.normA, condA = st.condA)
end

"""
    default_nufft_eps(::Type{T}) -> Float64

The NUFFT tolerance a fast backend uses when the caller names none: `1e-6` for `Float32`, `1e-9`
otherwise. One definition, because both fast backends must ask their library for the same accuracy.
"""
default_nufft_eps(::Type{T}) where {T} = real(float(T)) === Float32 ? 1.0e-6 : 1.0e-9

"""
    default_solver_rtol(::Type{T}, spectral, eps) -> T

Stopping tolerance for the scattered least-squares solve, chosen from the accuracy of the transform
the solve is built on rather than from a fixed constant.

Dispatched on the spectral backend, exactly as `SphericalCore.default_rtol` is: the in-core direct
summation evaluates an exact conjugate pair, so the arithmetic is the only limit and `sqrt(Base.eps(T))`
is the conventional least-squares target. A fast NUFFT's Type-1 and Type-2 are adjoints of each other
only to about `eps`, so the operator the solver sees is Hermitian only to that accuracy and a target
below it asks for something the transform cannot express — which is how a `Float32` plan came to be
asked for `1e-8`, two orders under its own `1e-6`.

`Base.eps` is spelled out because `eps` is a keyword-argument name in every plan constructor here.
"""
default_solver_rtol(::Type{T}, ::SB.AbstractDirectSumSpectralBackend, eps) where {T} =
    T(sqrt(Base.eps(real(float(T)))))
default_solver_rtol(::Type{T}, ::SB.AbstractSpectralBackend, eps) where {T} =
    T(max(10 * Float64(eps === nothing ? default_nufft_eps(T) : eps),
          sqrt(Base.eps(real(float(T))))))

"""
    warn_underdetermined(M, ms, solve, damp) -> nothing

Warn, once per session, that a solve has more modes than samples and no damping to resolve them.

With `prod(ms) > M` at least `prod(ms) - M` mode directions are constrained by no sample, so the
least-squares problem has no unique solution and the answer depends on the regularisation. LSMR
returns the minimum-norm iterate and its early termination is itself a regulariser, which is why
this warns rather than throws — but the caller is the only one who can say whether that is the
answer they wanted, or whether they should coarsen `ms`, add samples, or set `damp` for explicit
Tikhonov. Both numbers are named because the ratio is what decides how ill-posed the solve is.

No `λ` is chosen for the caller. Sweeping `λ` against a dense Tikhonov reference on this operator
shows why: on quasi-uniform points `A` is a scaled isometry and every `λ` across six decades returns
the same coefficients to `1e-14`, while on clustered points a `λ` small enough to be safe changes
neither the residual nor `‖x‖` in the first five digits. A default that cannot be observed to do
anything is worse than none, since it would still be a number the caller has to reason about.
"""
function warn_underdetermined(M::Integer, ms::Tuple, solve::Bool, damp::Real)
    n = prod(ms)
    (!solve || n <= M || damp > 0) && return nothing
    @warn "Scattered solve is underdetermined: $n modes from $M samples leaves at least $(n - M) " *
          "mode directions unconstrained. LSMR returns the minimum-norm solution; pass `damp` for " *
          "Tikhonov regularisation, or reduce `ms`." maxlog = 1
    return nothing
end

"""
    batch_width(plan) -> Int

Number of co-located fields this plan transforms per execution — `1` unless it was built for a batch.

A nonuniform plan's batch width is fixed when its guru plan is built and cannot vary per execution,
so a plan is either single-field or batched. Callers use this to decide whether a stack can go
through one execution per cascade step instead of one per field.
"""
batch_width(::Any) = 1

"""
    task_local_plan(plan) -> plan

A plan equivalent to `plan` that is safe to use concurrently with it. Stateless plans (FFTW,
`AbstractFFTs`) return themselves; plans carrying mutable scratch return a copy that shares their
read-only tables and owns fresh scratch. Called once per task by the parallel backends.

Three obligations follow from *where* this is called. A method that **builds** rather than shares must
do so under [`PLANNER_LOCK`](@ref), because it runs inside a spawned task by construction; it must
take its thread count from [`per_task_nthreads`](@ref), since the caller has already claimed the cores;
and whoever built it must [`close_plan!`](@ref) it when the task ends.
"""
task_local_plan(plan::AbstractScatteringPlan) = plan

"""
    PLANNER_LOCK

Serialises plan construction across every backend in the package.

FFTW documents `fftw_execute` as its only thread-safe entry point, so two plan builds running at once
fault inside the planner. One lock covers them all rather than one per backend, because the planner
is shared far more widely than any single backend: FFTW.jl, FINUFFT, NonuniformFFTs and
FastTransforms all plan through the same libfftw3, so a lock private to one of them excludes nothing.
Concurrent construction is routine — [`task_local_plan`](@ref) and `SphericalCore.batch_plan` build
inside spawned tasks — and a build racing a build of a *different* library was observed as a segfault
in `fftw_mkapiplan`.

Only construction takes this lock; transforms are never serialised, so a plan per task still executes
in parallel. FastTransforms additionally needs its OpenMP thread count pinned across construction
*and* execution — see `SphericalCore.with_serial_ft`.
"""
const PLANNER_LOCK = ReentrantLock()

"""
    with_fft_nthreads(f, n) -> f()

Run `f` with the FFT library's global thread count set to `n`, restoring it afterwards.

FFTW's thread count is process-global and is raised as a side effect of loading unrelated packages,
so a plan built without pinning it inherits whatever was last set — and a plan built for more threads
than it needs spawns (and allocates) a task per thread on *every* execution. Plan builders wrap
construction in this so a plan's threading is a property of the plan, not of load order.

Because the count is one global, callers hold [`PLANNER_LOCK`](@ref) across this: two builds pinning
it concurrently would each restore the other's value.

The default is a no-op: only the FFTW extension has a global count to set. It takes `args...` so the
extension's fixed-arity method is strictly more specific and *adds* a method rather than overwriting
this one — overwriting is an error during precompilation.
"""
with_fft_nthreads(f, args...) = f()

"""
    fftw_plan(T, dims; nbatch, planning, fft_nthreads) -> AbstractScatteringPlan

Build the FFTW-backed plan. The real method lives in the FFTW extension; the definition here is a
throwing stub, so an explicit `FFTSpectralBackend()` request dispatches straight to the extension
when it is loaded and to an actionable error when it is not — with no capability lookup on either
path.
"""
fftw_plan(args...; kwargs...) = throw(ArgumentError(
    "FFTSpectralBackend requires the FFTW extension. Run `using FFTW`."))

"""
    abstractffts_plan(dummy_device_array; region=1:ndims(dummy_device_array))

Vendor-neutral device FFT plan constructor. Declaration only — the sole method is provided by the
KernelAbstractions extension, which builds forward/inverse plans via `AbstractFFTs.plan_fft` /
`plan_ifft` on `dummy_device_array`. Because those dispatch on the array *type*, the same builder
yields cuFFT (`CuArray`), rocFFT (`ROCArray`), or FFTW (plain `Array`) plans — so the GPU scattering
path is device-agnostic. `region` selects the transformed dimensions (e.g. `(1, 2)` for a batched
`(Ny, Nx, B)` stack, leaving the batch axis untouched).
"""
function abstractffts_plan end

# Which fast paths exist right now. Only `Auto*` and the "whichever NUFFT library is loaded"
# resolution call these, because only they have a choice to make; an explicitly named backend
# dispatches straight to its extension's plan builder — or to that builder's throwing stub — and
# performs no lookup at all. `hasmethod` cannot substitute here: the stubs make it always true.
_have_fftw() = Base.get_extension(parentmodule(@__MODULE__), :ScatteringTransformsFFTWExt) !== nothing
_have_finufft() = Base.get_extension(parentmodule(@__MODULE__), :ScatteringTransformsFINUFFTExt) !== nothing
_have_nonuniformffts() =
    Base.get_extension(parentmodule(@__MODULE__), :ScatteringTransformsNonuniformFFTsExt) !== nothing

# ---------------------------------------------------------------------------
# Spectral backend tags
#
# Uniform-grid transforms take `SpectralBackends` tags directly. The two NUFFT libraries need
# distinguishing, which a single `NUFFTSpectralBackend` cannot do, so each gets its own tag under
# the shared abstract supertype.
# ---------------------------------------------------------------------------

"FINUFFT fast path for scattered/nonuniform planar points; requires `using FINUFFT`."
struct FINUFFTBackend <: SB.AbstractNUFFTSpectralBackend end

"NonuniformFFTs.jl fast path for scattered/nonuniform planar points; requires `using NonuniformFFTs`."
struct NonuniformFFTsBackend <: SB.AbstractNUFFTSpectralBackend end

"""
    make_plan(spectral, T, dims; nbatch=1, kwargs...) -> AbstractScatteringPlan

Build the spectral plan selected by `spectral` for arrays whose leading dimensions are `dims` and
whose element type is `Complex{T}`, with a trailing batch axis of length `nbatch`.

`SpectralBackends.DirectSumSpectralBackend` is the dependency-free in-core default;
`SpectralBackends.FFTSpectralBackend` requires `using FFTW`;
`SpectralBackends.AutoSpectralBackend` takes the FFTW fast path when its extension is loaded and
otherwise the in-core direct sum.
"""
make_plan(::SB.AbstractDirectSumSpectralBackend, ::Type{T}, dims; nbatch::Int = 1, kwargs...) where {T} =
    DirectSumPlan(T, dims; nbatch = nbatch)

make_plan(::SB.AbstractFFTSpectralBackend, ::Type{T}, dims; nbatch::Int = 1, kwargs...) where {T} =
    fftw_plan(T, dims; nbatch = nbatch, kwargs...)

make_plan(::SB.AbstractAutoSpectralBackend, ::Type{T}, dims; nbatch::Int = 1, kwargs...) where {T} =
    _have_fftw() ? fftw_plan(T, dims; nbatch = nbatch, kwargs...) :
    DirectSumPlan(T, dims; nbatch = nbatch)

"""
    plan_like(plan, prototype) -> plan′

A plan of the same kind as `plan`, sized for arrays like `prototype`.

Sizing from an *array* rather than from dimensions plus a backend tag is what lets one piece of code
plan on host and device alike: a device plan has no spectral-backend tag to look up — `spectral_backend`
deliberately throws on it — but it does have arrays, and `AbstractFFTs` dispatches on their type. Used
where a cascade needs plans at resolutions not known when the transform was built.
"""
plan_like(plan::AbstractScatteringPlan, prototype::AbstractArray) =
    make_plan(spectral_backend(plan), real(eltype(prototype)), size(prototype))

"""
    plan_analysis(plan) -> (; solve, maxiter, rtol, damp, eps, nufft_nthreads)

How a scattered-planar plan turns samples into modes, in the form its constructor takes.

Carried rather than re-derived, because none of it follows from the points: a plan rebuilt on another
process without `solve` analyses with the plain Type-1 adjoint where the original ran a least-squares
inversion, and on irregular sampling those are different transforms,
not one slightly less accurate than the other. `eps` is `nothing` for the exact direct sum, which has
no tolerance to honour.
"""
function plan_analysis end

"""
    finufft_scattered_plan(x, y, ms, T; period, solve, maxiter, rtol, eps, nufft_nthreads)
    nonuniformffts_scattered_plan(x, y, ms, T; period, solve, maxiter, rtol, eps, nufft_nthreads)

Fast-path scattered-planar plan constructors. The real methods live in the FINUFFT and
NonuniformFFTs extensions; the definitions here are throwing stubs, so naming one of those backends
explicitly costs a plain dispatch rather than a capability lookup.
"""
finufft_scattered_plan(args...; kwargs...) = throw(ArgumentError(
    "FINUFFTBackend requires the FINUFFT extension. Run `using FINUFFT`."))

"See [`finufft_scattered_plan`](@ref)."
nonuniformffts_scattered_plan(args...; kwargs...) = throw(ArgumentError(
    "NonuniformFFTsBackend requires the NonuniformFFTs extension. Run `using NonuniformFFTs`."))

"""
    make_scattered_plan(spectral, x, y, ms, T; period, solve, maxiter, rtol, eps,
                        nufft_nthreads) -> AbstractScatteringPlan

Build the scattered/nonuniform planar plan selected by `spectral` over points `(x, y)` and a uniform
mode grid of size `ms`. `SpectralBackends.DirectSumSpectralBackend` is the dependency-free exact
NUDFT; [`FINUFFTBackend`](@ref) and [`NonuniformFFTsBackend`](@ref) select a specific fast library;
`SpectralBackends.NUFFTSpectralBackend` takes whichever fast library is loaded, and
`SpectralBackends.AutoSpectralBackend` falls back to the exact direct sum when neither is.

`nufft_nthreads` sets the fast library's own thread count (`0`, the default, leaves it to the
library). The direct sum accepts it and ignores it, as it does `eps`, so a caller can pass one set of
options without first knowing which backend it will get.
"""
make_scattered_plan(spectral::SB.AbstractDirectSumSpectralBackend, x, y, ms, ::Type{T};
                    period = nothing, solve::Bool = false, maxiter::Int = 100,
                    rtol::Real = default_solver_rtol(T, spectral, eps), damp::Real = 0,
                    eps = nothing, ntrans::Int = 1, nufft_nthreads::Int = 0) where {T} =
    DirectNUFFTPlan(x, y, ms, T; period, solve, maxiter, rtol, damp, ntrans)

make_scattered_plan(::FINUFFTBackend, x, y, ms, ::Type{T}; kwargs...) where {T} =
    finufft_scattered_plan(x, y, ms, T; kwargs...)

make_scattered_plan(::NonuniformFFTsBackend, x, y, ms, ::Type{T}; kwargs...) where {T} =
    nonuniformffts_scattered_plan(x, y, ms, T; kwargs...)

function make_scattered_plan(::SB.AbstractNUFFTSpectralBackend, x, y, ms, ::Type{T}; kwargs...) where {T}
    _have_finufft() && return finufft_scattered_plan(x, y, ms, T; kwargs...)
    _have_nonuniformffts() && return nonuniformffts_scattered_plan(x, y, ms, T; kwargs...)
    throw(ArgumentError("NUFFTSpectralBackend requires a fast NUFFT library. Run `using FINUFFT` " *
                        "or `using NonuniformFFTs`, or pass DirectSumSpectralBackend() for the " *
                        "exact O(M·prod(ms)) direct summation."))
end

function make_scattered_plan(::SB.AbstractAutoSpectralBackend, x, y, ms, ::Type{T}; kwargs...) where {T}
    _have_finufft() && return finufft_scattered_plan(x, y, ms, T; kwargs...)
    _have_nonuniformffts() && return nonuniformffts_scattered_plan(x, y, ms, T; kwargs...)
    return DirectNUFFTPlan(x, y, ms, T; kwargs...)
end

# ---------------------------------------------------------------------------
# In-core direct-summation DFT plan
#
# This is the dependency-free reference: `O(N²)` per axis by construction, so it is the ground
# truth every fast plan is validated against — not a fast path. Load FFTW (or a device FFT) for
# `O(N log N)`. Within that `O(N²)` contract the kernel is written to be optimal: `O(N)` memory (a
# per-axis table of roots of unity, plus its conjugate so the inverse carries no branch), an
# incrementally-advanced twiddle index rather than a modulo in the inner loop, and a lane-major
# traversal so non-leading axes and the batch axis vectorise instead of striding.
# ---------------------------------------------------------------------------

"""
    DirectSumPlan{T,V,D,S}

Direct-summation DFT plan: evaluates `X_k = Σ_n x_n e^{-2πi kn/N}` (and its inverse) by direct
summation, separably over the leading `D` dimensions, leaving a trailing batch axis of length
`nbatch` untouched. Memory is `O(N)` — a per-axis table of roots of unity and its conjugate, never
an `N×N` matrix. `scratch` is `nothing` for `D == 1` and a full-size complex array otherwise.
Containers stay parametric.
"""
struct DirectSumPlan{T, V <: AbstractVector{Complex{T}}, D, S} <: AbstractScatteringPlan
    twiddle::NTuple{D, V}       # twiddle[d][m+1] = exp(-2πi m / N_d)
    twiddle_conj::NTuple{D, V}  # its conjugate — the inverse direction, branch-free
    dims::NTuple{D, Int}        # transformed (leading) dimensions
    nbatch::Int                 # length of the trailing batch axis
    scratch::S
    invscale::T                 # 1 / prod(dims)
    # Explicit inner constructor: binds `T,V,D,S` from the call rather than inferring them from a
    # possibly-empty tuple, and enforces that each per-axis twiddle table has exactly its axis length.
    function DirectSumPlan{T, V, D, S}(twiddle::NTuple{D, V}, twiddle_conj::NTuple{D, V},
                                       dims::NTuple{D, Int}, nbatch::Int, scratch::S,
                                       invscale::T) where {T, V <: AbstractVector{Complex{T}}, D, S}
        for d in 1:D
            length(twiddle[d]) == dims[d] || throw(ArgumentError(
                "DirectSumPlan: twiddle table for axis $d has length $(length(twiddle[d])), " *
                "expected dims[$d] = $(dims[d])"))
        end
        return new{T, V, D, S}(twiddle, twiddle_conj, dims, nbatch, scratch, invscale)
    end
end

function _twiddles(::Type{T}, N::Int) where {T}
    tw = Vector{Complex{T}}(undef, N)
    @inbounds for m in 0:(N - 1)
        tw[m + 1] = cispi(-2 * T(m) / T(N))
    end
    return tw
end

"""
    DirectSumPlan(T, dims; nbatch = 1) -> DirectSumPlan

Build a direct-summation DFT plan over leading dimensions `dims` (an `Int` or `NTuple{D,Int}`) with
a trailing batch axis of length `nbatch`.
"""
function DirectSumPlan(::Type{T}, dims::NTuple{D, Int}; nbatch::Int = 1) where {T, D}
    nbatch >= 1 || throw(ArgumentError("nbatch must be positive, got $nbatch"))
    tw = ntuple(d -> _twiddles(T, dims[d]), D)
    twc = ntuple(d -> conj.(tw[d]), D)
    scratch = D == 1 ? nothing : zeros(Complex{T}, prod(dims) * nbatch)
    return DirectSumPlan{T, Vector{Complex{T}}, D, typeof(scratch)}(
        tw, twc, dims, nbatch, scratch, inv(T(prod(dims))))
end
DirectSumPlan(::Type{T}, N::Int; kwargs...) where {T} = DirectSumPlan(T, (N,); kwargs...)

Base.show(io::IO, p::DirectSumPlan{T, V, D}) where {T, V, D} =
    print(io, "DirectSumPlan{", T, "}(dims=", p.dims, ", nbatch=", p.nbatch, ")")

spectral_backend(::DirectSumPlan) = SB.DirectSumSpectralBackend()

task_local_plan(p::DirectSumPlan{T, V, D, S}) where {T, V, D, S} =
    DirectSumPlan{T, V, D, S}(p.twiddle, p.twiddle_conj, p.dims, p.nbatch,
                              p.scratch === nothing ? nothing : similar(p.scratch), p.invscale)

# Transform the middle axis of the (nb, n, na) view of `x` into `out`, out-of-place.
#
#     out[l, k, ia] = Σ_j tw[(k·j mod n) + 1] · x[l, j, ia]
#
# The twiddle exponent advances by `k` per step and wraps at most once, so the inner loop needs a
# compare-subtract rather than an integer division. `nb == 1` (a leading-axis transform) accumulates
# in a register; `nb > 1` accumulates into the contiguous lane run, which stays in L1 and vectorises.
function _dft_axis!(out, x, tw::AbstractVector, nb::Int, n::Int, na::Int)
    if nb == 1
        @inbounds for ia in 0:(na - 1)
            off = ia * n
            for k in 0:(n - 1)
                acc = zero(eltype(out))
                idx = 0
                for j in 1:n
                    acc += tw[idx + 1] * x[off + j]
                    idx += k
                    idx >= n && (idx -= n)
                end
                out[off + k + 1] = acc
            end
        end
    else
        @inbounds for ia in 0:(na - 1)
            aoff = ia * nb * n
            for k in 0:(n - 1)
                di = aoff + k * nb
                for l in 1:nb
                    out[di + l] = zero(eltype(out))
                end
                idx = 0
                for j in 0:(n - 1)
                    w = tw[idx + 1]
                    si = aoff + j * nb
                    for l in 1:nb
                        out[di + l] += w * x[si + l]
                    end
                    idx += k
                    idx >= n && (idx -= n)
                end
            end
        end
    end
    return out
end

# Separable multi-axis execution, ping-ponging so the final axis writes into `out`.
function _execute!(out, x, p::DirectSumPlan{T, V, D}, twt::NTuple{D}) where {T, V, D}
    total = length(out)
    total == length(x) || throw(DimensionMismatch(
        "output has $(length(out)) elements, input has $(length(x))"))
    total == prod(p.dims) * p.nbatch || throw(DimensionMismatch(
        "plan is for dims $(p.dims) × nbatch $(p.nbatch), got $total elements"))
    if D == 1
        n = p.dims[1]
        return _dft_axis!(out, x, twt[1], 1, n, total ÷ n)
    end
    sc = p.scratch
    dst = isodd(D) ? out : sc
    src = x
    nb = 1
    for d in 1:D
        n = p.dims[d]
        _dft_axis!(dst, src, twt[d], nb, n, total ÷ (nb * n))
        src = dst
        dst = dst === out ? sc : out
        nb *= n
    end
    return out
end

forward_transform!(out::AbstractArray, p::DirectSumPlan, x::AbstractArray) =
    _execute!(out, x, p, p.twiddle)

function inverse_transform!(out::AbstractArray, p::DirectSumPlan, x::AbstractArray)
    _execute!(out, x, p, p.twiddle_conj)
    s = p.invscale
    @inbounds for i in eachindex(out)
        out[i] *= s
    end
    return out
end

# ---------------------------------------------------------------------------
# Non-mutating, autodiff-friendly direct-sum transforms.
#
# The sum is contracted against the plan's `O(N)` twiddle table rather than through a dense `N×N`
# DFT matrix. Both are `O(N²)` in time, but a matrix formulation cannot serve this path's purpose:
# Enzyme has no derivative rule for complex `gemm`, so `W*x` on `ComplexF64` is undifferentiable.
# Contracting the table instead is plain scalar arithmetic every AD backend can trace, allocates
# only the output (measured at N=4096: 64 KiB and 56 ms, against 256 MiB and 143 ms for building the
# matrix per call), and stays element-type generic so `Dual`/`Float32` flow through.
# ---------------------------------------------------------------------------

# Contract dimension `d` of `A` against the DFT kernel from `tw`. The 1↔d transposition is an
# involution, so the same permutation restores the layout.
function _dft_along(tw::AbstractVector, n::Int, A::AbstractArray, d::Int)
    perm = ntuple(i -> i == 1 ? d : (i == d ? 1 : i), ndims(A))
    Ap = d == 1 ? A : permutedims(A, perm)
    sz = size(Ap)
    M = reshape(Ap, n, :)
    # Written as an explicit loop over a fresh output rather than a comprehension: the generator of
    # a comprehension closes over both the constant twiddle table and the active input, which
    # Enzyme's static activity analysis cannot separate. A loop reads both as plain arguments.
    R = similar(M, promote_type(eltype(tw), eltype(M)))
    @inbounds for c in axes(M, 2)
        for k in 0:(n - 1)
            acc = zero(eltype(R))
            idx = 0
            for j in 1:n
                acc += tw[idx + 1] * M[j, c]
                idx += k
                idx >= n && (idx -= n)
            end
            R[k + 1, c] = acc
        end
    end
    Rr = reshape(R, sz...)
    return d == 1 ? Rr : permutedims(Rr, perm)
end

function forward_transform(p::DirectSumPlan{T, V, D}, x::AbstractArray) where {T, V, D}
    y = x
    for d in 1:D
        y = _dft_along(p.twiddle[d], p.dims[d], y, d)
    end
    return y
end

function inverse_transform(p::DirectSumPlan{T, V, D}, x::AbstractArray) where {T, V, D}
    y = x
    for d in 1:D
        y = _dft_along(p.twiddle_conj[d], p.dims[d], y, d)
    end
    return y .* p.invscale
end

# ---------------------------------------------------------------------------
# Dependency-free scattered / nonuniform planar transform (exact direct-summation NUDFT).
#
# The `O(M·prod(ms))` reference that lets `scattered_planar_scattering` run with no external NUFFT
# library — the nonuniform counterpart of `DirectSumPlan`. Same numeric contract as the fast
# `NUFFTScatteringPlan`s: `forward_transform!` is the Type-1 adjoint (points → uniform mode grid),
# or an LSMR least-squares inversion when `solve`; `inverse_transform!` is the Type-2
# synthesis (modes → points) scaled by `1/prod(ms)`. The mode grid uses FFT ordering, so on a
# uniform `0:m-1` grid these reduce exactly to `fft`/`ifft` and the lattice matches the wavelet
# bank's `fftfreq` layout.
#
# Exact NUDFT by direct summation, separated per axis (`s = 2π(p−min)/period` scaled coordinates,
# `f_d` the FFT-ordered integer frequencies):
#   Type-1:  X[k₁,k₂] = Σ_n c_n · e^{-i f₁[k₁]·sx_n} · e^{-i f₂[k₂]·sy_n}
#   Type-2:  c_n      = Σ_{k₁,k₂} X[k₁,k₂] · e^{+i f₁[k₁]·sx_n} · e^{+i f₂[k₂]·sy_n}
#
# `Ex`/`Exc` are stored `(ms₁, M)` and `Ey`/`Eyc` `(M, ms₂)`/`(ms₂, M)` — each in the layout its
# consumer reads contiguously, so neither the `Sbuf` fill nor the type-2 accumulation gathers.
# ---------------------------------------------------------------------------

# FFT-ordered integer frequencies for a length-`m` axis: 0,1,…,⌈m/2⌉−1, −⌊m/2⌋,…,−1.
_fftfreqs(m::Int) = Int[i <= (m - 1) ÷ 2 ? i : i - m for i in 0:(m - 1)]

struct DirectNUFFTPlan{T, EM <: AbstractMatrix{Complex{T}},
                       CV <: AbstractVector{Complex{T}},
                       RV <: AbstractVector{T}} <: AbstractScatteringPlan
    ms::NTuple{2, Int}
    M::Int
    invN::T                 # 1/prod(ms) — makes synthesis the ifft-convention inverse
    solve::Bool
    maxiter::Int
    rtol::T
    damp::T                 # Tikhonov λ; 0 unless the mode grid is over-specified
    sx::RV                  # (M) points on the 2π-periodic domain, retained so the plan can be
    sy::RV                  #     rebuilt elsewhere — see `plan_points`
    Ex::EM                  # (ms[1], M)  e^{-i f₁[k]·sx_n}
    Ey::EM                  # (M, ms[2])  e^{-i f₂[k]·sy_n}   (point index contiguous)
    Exc::EM                 # (ms[1], M)  conj(Ex)
    Eyc::EM                 # (ms[2], M)  conj(Ey)ᵀ
    cj::CV                  # (M) values buffer (shared by Type-1/Type-2)
    Sbuf::EM                # (M, ms[2]) Type-1 scratch
    T1::EM                  # (ms[1], M) Type-2 scratch
    ls_v::EM                # (ms) solver scratch: the four LSMR mode vectors. `cj`/`ls_t` are its
    ls_w::EM                #      two point vectors — the transforms overwrite rather than
    ls_h::EM                #      accumulate, so `A·v` and `A†·u` each need a destination.
    ls_hbar::EM
    ls_t::CV                # (M)
end

# `ntrans` is accepted so a caller can request a batch width without first asking which backend it
# will get, and ignored because direct summation transforms one field per call. The plan reports
# `batch_width == 1`, so nothing downstream feeds it a stack it cannot take. `eps` and
# `nufft_nthreads` are accepted and ignored for the same reason: they configure a fast library that
# this plan does not use.
function DirectNUFFTPlan(x::AbstractVector, y::AbstractVector, ms::NTuple{2, Int}, ::Type{T};
                         period = nothing, solve::Bool = false, maxiter::Int = 100,
                         rtol::Real = default_solver_rtol(T, SB.DirectSumSpectralBackend(), nothing),
                         damp::Real = 0, eps = nothing, ntrans::Int = 1,
                         nufft_nthreads::Int = 0) where {T}
    M = length(x)
    length(y) == M || throw(DimensionMismatch("x and y must have equal length"))
    warn_underdetermined(M, ms, solve, damp)
    xmin, ymin = T(minimum(x)), T(minimum(y))
    # Default period so a uniform 0:m-1 grid (span m-1) maps to the exact DFT nodes 2π·(0:m-1)/m.
    px = period === nothing ? (T(maximum(x)) - xmin) * ms[1] / (ms[1] - 1) : T(period[1])
    py = period === nothing ? (T(maximum(y)) - ymin) * ms[2] / (ms[2] - 1) : T(period[2])
    sx = T(2π) .* (T.(x) .- xmin) ./ px
    sy = T(2π) .* (T.(y) .- ymin) ./ py
    f1, f2 = _fftfreqs(ms[1]), _fftfreqs(ms[2])
    Ex = Complex{T}[cis(-f1[k] * sx[n]) for k in 1:ms[1], n in 1:M]
    Ey = Complex{T}[cis(-f2[k] * sy[n]) for n in 1:M, k in 1:ms[2]]
    return DirectNUFFTPlan{T, Matrix{Complex{T}}, Vector{Complex{T}}, Vector{T}}(
        ms, M, one(T) / prod(ms), solve, maxiter, T(rtol), T(damp), sx, sy,
        Ex, Ey, conj.(Ex), Matrix(conj.(transpose(Ey))),
        Vector{Complex{T}}(undef, M),
        Matrix{Complex{T}}(undef, M, ms[2]), Matrix{Complex{T}}(undef, ms[1], M),
        Matrix{Complex{T}}(undef, ms), Matrix{Complex{T}}(undef, ms), Matrix{Complex{T}}(undef, ms),
        Matrix{Complex{T}}(undef, ms), Vector{Complex{T}}(undef, M))
end

Base.show(io::IO, p::DirectNUFFTPlan{T}) where {T} =
    print(io, "DirectNUFFTPlan{", T, "}(ms=", p.ms, ", M=", p.M, ", solve=", p.solve, ")")

spectral_backend(::DirectNUFFTPlan) = SB.DirectSumSpectralBackend()

function task_local_plan(p::DirectNUFFTPlan{T, EM, CV, RV}) where {T, EM, CV, RV}
    return DirectNUFFTPlan{T, EM, CV, RV}(
        p.ms, p.M, p.invN, p.solve, p.maxiter, p.rtol, p.damp, p.sx, p.sy,
        p.Ex, p.Ey, p.Exc, p.Eyc,
        similar(p.cj), similar(p.Sbuf), similar(p.T1),
        similar(p.ls_v), similar(p.ls_w), similar(p.ls_h), similar(p.ls_hbar), similar(p.ls_t))
end

"""
    plan_points(plan) -> (x, y)

The scattered sample locations a nonuniform plan was built on, already mapped onto its `2π`-periodic
domain. This is what lets a transform be rebuilt on another process, where the plan itself cannot
travel. Rebuilding from these requires `period = (2π, 2π)`, since they are already scaled.
"""
plan_points(p::DirectNUFFTPlan) = (p.sx, p.sy)

# Direct summation is exact and single-threaded by construction, so it reports no tolerance and no
# thread count — both are `nothing`/`0`, the values its constructor ignores.
plan_analysis(p::DirectNUFFTPlan) =
    (solve = p.solve, maxiter = p.maxiter, rtol = p.rtol, damp = p.damp, eps = nothing,
     nufft_nthreads = 0)

# Type-1 (points → modes): X = Ex · (c ⊙ Ey), all preallocated.
function _nudft_type1!(X::AbstractMatrix, plan::DirectNUFFTPlan, c::AbstractVector)
    @inbounds for k2 in 1:plan.ms[2], n in 1:plan.M
        plan.Sbuf[n, k2] = c[n] * plan.Ey[n, k2]
    end
    LinearAlgebra.mul!(X, plan.Ex, plan.Sbuf)
    return X
end

# Type-2 (modes → points): c_n = Σ_{k₁} Exc[k₁,n]·(X·Eyc)[k₁,n].
function _nudft_type2!(c::AbstractVector, plan::DirectNUFFTPlan{T}, X::AbstractMatrix) where {T}
    LinearAlgebra.mul!(plan.T1, X, plan.Eyc)
    @inbounds for n in 1:plan.M
        acc = zero(Complex{T})
        for k1 in 1:plan.ms[1]
            acc += plan.Exc[k1, n] * plan.T1[k1, n]
        end
        c[n] = acc
    end
    return c
end

inverse_transform!(out_pts::AbstractVector, plan::DirectNUFFTPlan, Xmodes::AbstractMatrix) =
    (_nudft_type2!(plan.cj, plan, Xmodes); @. out_pts = plan.cj * plan.invN; out_pts)

function forward_transform!(Xmodes::AbstractMatrix, plan::DirectNUFFTPlan, x_pts::AbstractVector)
    if plan.solve
        _lsmr_solve_nudft!(Xmodes, plan, x_pts)
    else
        copyto!(plan.cj, x_pts)
        _nudft_type1!(Xmodes, plan, plan.cj)
    end
    return Xmodes
end

# Least-squares inversion: find modes `f` with `A f ≈ N·x`, `A` = Type-2, `A† ` = Type-1, so that
# synthesis (Type-2 scaled by `invN`) reproduces the samples.
#
# `A f̃ = x` is solved and the answer scaled by `N` afterwards, rather than feeding the solver an
# `N`-inflated right-hand side: at `ms = 200²` that factor is 4·10⁴, which in `Float32` puts the
# squared quantities within reach of `floatmax` — and it makes the reported residual a misfit in the
# field's own units.
function _lsmr_solve_nudft!(f::AbstractMatrix, plan::DirectNUFFTPlan{T},
                            x_pts::AbstractVector) where {T}
    info = lsmr_solve!(f,
                       (dst, src) -> _nudft_type2!(dst, plan, src),
                       (dst, src) -> _nudft_type1!(dst, plan, src),
                       x_pts, plan.cj, plan.ls_t,
                       plan.ls_v, plan.ls_w, plan.ls_h, plan.ls_hbar;
                       # A relative perturbation `eps` in the data becomes a relative error up to
                       # `cond(A)·eps` in the solution, so at `cond(A) = 1/eps` the smallest singular
                       # direction carries nothing this precision can represent. That is the limit.
                       damp = plan.damp, atol = plan.rtol, btol = plan.rtol,
                       conlim = inv(Base.eps(T)), maxiter = plan.maxiter)
    _check_solve(info, plan.M, plan.ms, plan.rtol, plan.maxiter)
    f .*= one(T) / plan.invN
    return f
end

end # module Plans
