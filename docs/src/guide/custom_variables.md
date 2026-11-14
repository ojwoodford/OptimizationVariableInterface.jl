# Defining a variable type

A variable type that is updated out of place (every update returns a new value) defines two methods,
[`nvars`](@ref) and [`update`](@ref). It needs nothing else: the defaults of [`update!`](@ref),
[`copyvars`](@ref), [`syncvars!`](@ref) and [`ismutablevar`](@ref) are correct for it. A type that is updated
in place needs more; see [In-place variable types](@ref).

## `nvars`: the degrees of freedom

`nvars(var)` is the dimension of the variable's tangent space: the number of elements of the step vector
that belong to it. This is the intrinsic number of degrees of freedom, which can be smaller than the number
of numbers stored (an angle stored as a unit 2-vector still has one).

Return a `StaticInt` (`static(N)`) when the count is fixed by the type. The counts of a static collection of
blocks then add up at compile time, and an optimizer can size its gradient and Hessian, or choose a
static linear solver, without running anything. Return an `Int` when the count depends on the value.

## `update`: the retraction

`update(var, step, start=1)` returns the variable moved along the tangent vector
`step[start:start+nvars(var)-1]`. It returns a new value and does not change `var`.

- `step` is a vector in the tangent space at `var`, in coordinates the type chooses. The optimizer knows
  nothing about them: it only computes a step of the right length.
- `start` is the index in `step` of the variable's first coordinate. An optimizer with several variables
  computes one concatenated step for all of them, and passes each variable its offset, so no slice is
  allocated. A method reads only its own elements, `step[start]` to `step[start+nvars(var)-1]`.

## Example: an angle on the circle

The circle ``S^1`` has one degree of freedom, and angles wrap modulo ``2\pi``:

```@example custom
using OptimizationVariableInterface, Static
import OptimizationVariableInterface: nvars, update

# Custom variable type: an angle on the circle S¹
struct CircleAngle{T<:AbstractFloat}
    theta::T
end

# 1. Tangent space dimension (intrinsic degrees of freedom).
#    Returning `static(1)` allows compile-time sizing in static solvers.
nvars(::CircleAngle) = static(1)

# 2. Retraction: the variable's step starts at `start` in the tangent step vector.
update(var::CircleAngle, step, start=1) = CircleAngle(mod2pi(var.theta + step[start]))

angle = CircleAngle(0.0)
nvars(angle)
```

```@example custom
update(angle, [pi])          # retract by a tangent step of π radians
```

```@example custom
update(angle, [0.0, 7.0], 2) # this variable's coordinate is the second element; 7 wraps to 7 - 2π
```

The type is now usable by any optimizer written against the interface, on its own or in a collection
(see [Collections of variables](@ref)).

## The fallback for types with `length` and `+`

A type whose update is plain vector addition needs no methods of this package at all. The fallbacks are

- `nvars(x) = length(x)`, and
- `update(var, step, start=1) = var + view(step, start:start+nvars(var)-1)`,

so a type that defines `Base.length` and `Base.:+(::MyType, ::AbstractVector)` works as it is:

```@example custom
struct Offsets
    data::Vector{Float64}
end
Base.length(v::Offsets) = length(v.data)
Base.:+(v::Offsets, s::AbstractVector) = Offsets(v.data + s)

o = Offsets([1.0, 2.0])
nvars(o), update(o, [0.0, 10.0, 20.0], 2).data
```

Define `nvars(::MyType) = static(N)` only to give the size at compile time, and `update` only if the
retraction is not addition.
