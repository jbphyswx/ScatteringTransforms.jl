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

Pass-through spectral plan that tallies `forward`/`inverse` executions of `inner`.
"""
struct CountingPlan{P <: ScatteringTransforms.Plans.AbstractScatteringPlan,
                    R <: Base.RefValue{Int}} <: ScatteringTransforms.Plans.AbstractScatteringPlan
    inner::P
    forward::R
    inverse::R
end
CountingPlan(inner::ScatteringTransforms.Plans.AbstractScatteringPlan) =
    CountingPlan(inner, Ref(0), Ref(0))

Base.show(io::IO, p::CountingPlan) =
    print(io, "CountingPlan(", p.inner, "; forward=", p.forward[], ", inverse=", p.inverse[], ")")

function ScatteringTransforms.Plans.forward_transform!(out, p::CountingPlan, x)
    p.forward[] += 1
    return ScatteringTransforms.Plans.forward_transform!(out, p.inner, x)
end
function ScatteringTransforms.Plans.inverse_transform!(out, p::CountingPlan, x)
    p.inverse[] += 1
    return ScatteringTransforms.Plans.inverse_transform!(out, p.inner, x)
end
function ScatteringTransforms.Plans.forward_transform(p::CountingPlan, x)
    p.forward[] += 1
    return ScatteringTransforms.Plans.forward_transform(p.inner, x)
end
function ScatteringTransforms.Plans.inverse_transform(p::CountingPlan, x)
    p.inverse[] += 1
    return ScatteringTransforms.Plans.inverse_transform(p.inner, x)
end
ScatteringTransforms.Plans.spectral_backend(p::CountingPlan) =
    ScatteringTransforms.Plans.spectral_backend(p.inner)
# Tasks share the counters deliberately, so a threaded run reports the whole cascade's total.
ScatteringTransforms.Plans.task_local_plan(p::CountingPlan) =
    CountingPlan(ScatteringTransforms.Plans.task_local_plan(p.inner), p.forward, p.inverse)

reset_counts!(p::CountingPlan) = (p.forward[] = 0; p.inverse[] = 0; p)

"""
    with_counting_plan(st) -> (st′, plan)

A copy of transform `st` whose `plan` field is a [`CountingPlan`](@ref), plus that plan. Every other
field — filter bank, tree, workspace buffers — is shared with `st`, so the copy costs nothing and
runs the identical cascade.
"""
function with_counting_plan(st)
    T = typeof(st)
    :plan in fieldnames(T) || throw(ArgumentError("$T has no `plan` field to instrument"))
    counting = CountingPlan(getfield(st, :plan))
    args = ntuple(i -> fieldname(T, i) === :plan ? counting : getfield(st, i), fieldcount(T))
    return T.name.wrapper(args...), counting
end

"""
    count_executions(f, st, args...) -> (; forward, inverse, total)

Spectral executions that `f(st′, args...)` issues, where `st′` is `st` instrumented with a
[`CountingPlan`](@ref). Runs once to warm up, then once to measure.
"""
function count_executions(f, st, args...)
    stc, plan = with_counting_plan(st)
    f(stc, args...)
    reset_counts!(plan)
    f(stc, args...)
    return (forward = plan.forward[], inverse = plan.inverse[],
            total = plan.forward[] + plan.inverse[])
end
