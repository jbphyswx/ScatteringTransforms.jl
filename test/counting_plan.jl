# Count spectral-plan executions instead of timing them.
#
# A scattering transform's cost is dominated by how many forward/inverse spectral transforms its
# cascade issues, so a complexity claim ("the second order no longer recomputes the first order")
# is asserted exactly by counting those calls. Counting is deterministic and instant, which is why
# it belongs in the test suite where a wall-clock assertion would be GC- and load-flaky.
#
# `CountingPlan` is immutable like every other plan in the package; only its two counters are
# `Ref`s, since the tallies are the sole mutable state.

"""
    CountingPlan(inner)

Pass-through spectral plan that tallies `forward`/`inverse` executions of `inner`, and the
butterflies those executions cost.

An execution *count* is invariant to decimation — the periodized cascade issues exactly the same
number of transforms as the undecimated one, just at smaller sizes — so counting calls alone cannot
tell whether the cascade got cheaper. `work` tallies `n·⌈log₂n⌉` per execution, which is what
actually shrinks, and is as deterministic as the call count.
"""
struct CountingPlan{P <: ScatteringTransforms.Plans.AbstractScatteringPlan,
                    R <: Base.RefValue{Int}} <: ScatteringTransforms.Plans.AbstractScatteringPlan
    inner::P
    forward::R
    inverse::R
    work::R
end
CountingPlan(inner::ScatteringTransforms.Plans.AbstractScatteringPlan) =
    CountingPlan(inner, Ref(0), Ref(0), Ref(0))

Base.show(io::IO, p::CountingPlan) =
    print(io, "CountingPlan(", p.inner, "; forward=", p.forward[], ", inverse=", p.inverse[], ")")

# n·⌈log₂n⌉ for n ≥ 2 — an FFT's butterfly count, in exact integer arithmetic.
@inline _butterflies(x) = (n = length(x); n * (sizeof(Int) * 8 - leading_zeros(max(n - 1, 1))))

function ScatteringTransforms.Plans.forward_transform!(out, p::CountingPlan, x)
    p.forward[] += 1
    p.work[] += _butterflies(x)
    return ScatteringTransforms.Plans.forward_transform!(out, p.inner, x)
end
function ScatteringTransforms.Plans.inverse_transform!(out, p::CountingPlan, x)
    p.inverse[] += 1
    p.work[] += _butterflies(x)
    return ScatteringTransforms.Plans.inverse_transform!(out, p.inner, x)
end
function ScatteringTransforms.Plans.forward_transform(p::CountingPlan, x)
    p.forward[] += 1
    p.work[] += _butterflies(x)
    return ScatteringTransforms.Plans.forward_transform(p.inner, x)
end
function ScatteringTransforms.Plans.inverse_transform(p::CountingPlan, x)
    p.inverse[] += 1
    p.work[] += _butterflies(x)
    return ScatteringTransforms.Plans.inverse_transform(p.inner, x)
end
ScatteringTransforms.Plans.spectral_backend(p::CountingPlan) =
    ScatteringTransforms.Plans.spectral_backend(p.inner)
# Must pass through: the cascade workspace aliases each resolution's inverse destination onto its
# source based on this, and it decided that before being instrumented.
ScatteringTransforms.Plans.inplace_inverse(p::CountingPlan) =
    ScatteringTransforms.Plans.inplace_inverse(p.inner)
# Tasks share the counters deliberately, so a threaded run reports the whole cascade's total.
ScatteringTransforms.Plans.task_local_plan(p::CountingPlan) =
    CountingPlan(ScatteringTransforms.Plans.task_local_plan(p.inner), p.forward, p.inverse, p.work)

reset_counts!(p::CountingPlan) = (p.forward[] = 0; p.inverse[] = 0; p.work[] = 0; p)

"""
    with_counting_plan(st) -> (st′, plans)

A copy of transform `st` with every spectral plan wrapped in a [`CountingPlan`](@ref) sharing one
set of counters, plus one of those plans. Every other field — filter bank, tree, workspace buffers —
is shared with `st`, so the copy costs nothing and runs the identical cascade.

Wrapping the cascade workspace's plans and not just `st.plan` is the whole point: the cascade runs
almost entirely through the per-resolution plans, so instrumenting `st.plan` alone tallies the
input's forward transform and nothing else.
"""
function with_counting_plan(st)
    T = typeof(st)
    :plan in fieldnames(T) || throw(ArgumentError("$T has no `plan` field to instrument"))
    counting = CountingPlan(getfield(st, :plan))
    wrap(p) = CountingPlan(p, counting.forward, counting.inverse, counting.work)
    args = ntuple(fieldcount(T)) do i
        f = fieldname(T, i)
        f === :plan ? counting :
        f === :pw ? ScatteringTransforms.Cascade.with_plans(wrap, getfield(st, i)) :
        getfield(st, i)
    end
    return T.name.wrapper(args...), counting
end

"""
    count_executions(f, st, args...) -> (; forward, inverse, total, work)

Spectral executions that `f(st′, args...)` issues, where `st′` is `st` instrumented with a
[`CountingPlan`](@ref). Runs once to warm up, then once to measure.
"""
function count_executions(f, st, args...)
    stc, plan = with_counting_plan(st)
    f(stc, args...)
    reset_counts!(plan)
    f(stc, args...)
    return (forward = plan.forward[], inverse = plan.inverse[],
            total = plan.forward[] + plan.inverse[], work = plan.work[])
end
