# Collections of variables

Real problems combine variables of different kinds: an angle (``S^1``), a translation (``\mathbb{R}^2``) and
a positive scale (``\mathbb{R}^+``), say. A collection of variable blocks is itself a variable. Its
[`nvars`](@ref) is the sum of the blocks' counts, and its step is the blocks' steps concatenated, in order:
[`update`](@ref) hands each block its own slice, by passing it the right `start`.

Four containers are supported: `Vector`, `StaticVector` (e.g. `SVector`), `Tuple` and `NamedTuple`.

```@setup collections
using OptimizationVariableInterface, Static, StaticArrays
import OptimizationVariableInterface: nvars, update
struct CircleAngle{T<:AbstractFloat}
    theta::T
end
nvars(::CircleAngle) = static(1)
update(var::CircleAngle, step, start=1) = CircleAngle(mod2pi(var.theta + step[start]))
```

The examples use the `CircleAngle` type of [Defining a variable type](@ref).

## `Vector`: dynamic collections

A `Vector` can hold any mix of blocks. Its `nvars` is an `Int`, computed at run time:

```@example collections
vars = [CircleAngle(0.0), SVector(1.0, 2.0), ZeroToInfScalar(1.0)]
nvars(vars)   # 1 + 2 + 1
```

A `Vector` is updated in place by [`update!`](@ref): each element of `to` is replaced by the `update` of the
corresponding element of `from`. The scratch buffer comes from [`copyvars`](@ref):

```@example collections
step = [0.1, 0.5, -0.5, log(2.0)]   # a 4-dimensional tangent step from an optimizer
vars_next = copyvars(vars)
vars_next = update!(vars_next, vars, step)
vars_next[1]
```

```@example collections
vars_next[2]
```

```@example collections
vars_next[3].val
```

A `Vector{<:Number}` is a single variable, not a collection of scalar blocks: its `nvars` is its length,
and `update!` adds the step elementwise in place.

## `StaticVector`: static collections with no allocation

When every block has a static size, the blocks can be held in an `SVector`. Then `nvars` adds up at compile
time and returns a `StaticInt`, and `update` is unrolled across the blocks and allocates nothing:

```@example collections
svec = SVector(ZeroToInfScalar(1.0), ZeroToInfScalar(2.0))
nvars(svec)
```

```@example collections
svec_next = update(svec, SVector(log(2.0), log(3.0)))
svec_next[1].val, svec_next[2].val
```

```@example collections
allocs(v, s) = @allocated update(v, s)
s = SVector(log(2.0), log(3.0))
allocs(svec, s)       # compile
allocs(svec, s) == 0
```

A `StaticVector` of numbers is a single variable (`nvars` is `static(N)`, and `update` is addition), not a
collection of scalar blocks.

## `Tuple` and `NamedTuple`: a few differently typed blocks

A `Tuple` or `NamedTuple` holds a few blocks of different types without the type instability of a
`Vector{Any}`. `nvars` is a `StaticInt` when every block's count is, and `update` returns a new tuple of
the same shape:

```@example collections
state = (angle = CircleAngle(0.0), offset = SVector(1.0, 2.0), scale = ZeroToInfScalar(1.0))
nvars(state)
```

```@example collections
state_next = update(state, [0.1, 0.5, -0.5, log(2.0)])
state_next.angle, state_next.offset, state_next.scale.val
```

A tuple is immutable, so `update!` on it returns the new tuple (which is why its result is always assigned,
`to = update!(to, from, step)`); a tuple whose blocks are all mutable arrays is updated in place block by
block by [`syncvars!`](@ref) (see [In-place variable types](@ref)).

## Composition

The rules apply recursively: a block can itself be a collection, and its own `nvars` and `update` follow the
same rules. Every container passes each block the `start` of its own slice, so no step is ever copied.
