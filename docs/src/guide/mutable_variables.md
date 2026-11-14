# In-place variable types

An optimizer keeps more than one copy of the variables: the current iterate, a trial point, perhaps the
best point so far. For a value type, a copy is free and an update returns a new value. For a type that is
updated in place, a copy has to be made, and a trial step must be written into a separate buffer, or a
rejected step would overwrite the current iterate.

## The four methods

A type updated in place defines, besides [`nvars`](@ref) and [`update`](@ref):

| Method | What it does |
| :--- | :--- |
| [`update!(to, from, step)`](@ref update!) | Writes the retraction of `from` along `step` into `to`, and returns `to`. |
| [`copyvars(x)`](@ref copyvars) | Returns an independent copy of `x`, to use as a scratch buffer. |
| [`syncvars!(to, from)`](@ref syncvars!) | Copies the state of `from` into `to`, and returns `to`. |
| [`ismutablevar(::Type{<:MyType}) = true`](@ref ismutablevar) | Declares that the type is updated in place. |

An out-of-place type needs none of them. Their fallbacks, `update!(to, from, step) = update(from, step)`,
`copyvars(x) = x`, `syncvars!(to, from) = from` and `ismutablevar(x) = false`, treat an unknown type as an
immutable value, which is correct exactly when its update is out of place. Arrays need none of them either:
the package defines them for mutable arrays, `Vector`s of blocks, tuples and named tuples.

Because the same code must work for both kinds, an optimizer always uses the return values:
`to = update!(to, from, step)` and `to = syncvars!(to, from)`.

## `ismutablevar` and its defining property

[`ismutablevar(x)`](@ref ismutablevar) says whether `x` is updated in place. Its defining property: for `y`
of the same type as `x` with different contents, let `r = syncvars!(x, y)`. If `ismutablevar(x)`, the
caller's `x` itself now holds `y`'s values, and `r === x`. Otherwise `x` is not (fully) updated, and only
`r` holds `y`'s values.

This is not `Base.ismutable`. What matters is whether the variable's state is written in place, not whether
the outer object is mutable:

```@example mutable
using OptimizationVariableInterface, StaticArrays

ismutablevar(([1.0], [2.0]))   # a tuple is immutable, but its arrays are synced in place
```

```@example mutable
mutable struct NotOptedIn
    a::Vector{Float64}
end
ismutablevar(NotOptedIn([1.0]))   # a mutable struct is a value until it opts in
```

The result is a compile-time constant, so an optimizer can branch on it at no cost.

## A worked in-place type

A vector of positive values, stored as its logarithm so that every update stays positive, and updated in
place:

```@example mutable
import OptimizationVariableInterface: nvars, update, update!, copyvars, syncvars!, ismutablevar

struct PositiveVector
    logv::Vector{Float64}   # state
    v::Vector{Float64}      # cached exp.(logv)
end
PositiveVector(v::Vector{Float64}) = PositiveVector(log.(v), copy(v))

nvars(x::PositiveVector) = length(x.logv)

# Out of place: a new value
update(x::PositiveVector, step, start=1) =
    PositiveVector(exp.(x.logv .+ view(step, start:start+nvars(x)-1)))

# In place: the four methods
function update!(to::PositiveVector, from::PositiveVector, step::AbstractVector)
    to.logv .= from.logv .+ step
    to.v .= exp.(to.logv)
    return to
end
copyvars(x::PositiveVector) = PositiveVector(copy(x.logv), copy(x.v))
function syncvars!(to::PositiveVector, from::PositiveVector)
    copyto!(to.logv, from.logv)
    copyto!(to.v, from.v)
    return to
end
ismutablevar(::Type{<:PositiveVector}) = true
nothing # hide
```

It passes the defining property of `ismutablevar`:

```@example mutable
x = PositiveVector([1.0, 2.0])
y = PositiveVector([3.0, 4.0])
r = syncvars!(x, y)
ismutablevar(x), r === x, x.v == y.v
```

and a trial step is written into a copy, leaving the original as it was:

```@example mutable
trial = copyvars(x)
trial = update!(trial, x, [log(2.0), 0.0])
trial.v, x.v
```

## Requiring in-place variables

A mutating entry point that can only work in place checks up front, before doing any work, and says what
to use instead:

```@example mutable
function optimize_in_place!(x0)
    ismutablevar(x0) || throw(ArgumentError("optimize_in_place! needs variables it can update in place; " *
        "$(typeof(x0)) is treated as an immutable value. Use the non-mutating form instead."))
    # ... update x0 in place ...
    return x0
end

optimize_in_place!([1.0, 2.0])
```

```@example mutable
try
    optimize_in_place!(SVector(1.0, 2.0))
catch err
    sprint(showerror, err)
end
```
