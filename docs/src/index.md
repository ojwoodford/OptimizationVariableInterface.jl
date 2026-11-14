# OptimizationVariableInterface.jl

A lightweight, zero-overhead variable and retraction interface for numerical and nonlinear optimization in
Julia.

In nonlinear optimization, the variables often do not live in a flat Euclidean space ``\mathbb{R}^n``. They may
be points on a manifold (rotations, poses, directions on a sphere) or parameters confined to an interval or
to the positive reals. Moving such a variable along a search direction is not vector addition
``x + \Delta x``: the optimizer computes a step ``\Delta x`` in the **tangent space**
``T_x\mathcal{M} \cong \mathbb{R}^d`` and maps it back onto the variable's domain with a **retraction**,
``x_{k+1} = \mathcal{R}_x(\Delta x)``.

OptimizationVariableInterface.jl is the contract between **variable representations** and **optimization
algorithms**. An optimizer works only with flat tangent vectors, sized by [`nvars`](@ref) and applied by
[`update`](@ref) or [`update!`](@ref), so it needs no knowledge of the variables' geometry. The interface
supports compile-time static sizes (Static.jl, StaticArrays.jl), allocation-free scratch buffers, mutable and
immutable variables alike, and heterogeneous collections of variable blocks treated as one state.

## Installation

```julia
using Pkg
Pkg.add("OptimizationVariableInterface")
```

## A first variable type

An angle on the circle ``S^1`` has one degree of freedom, and wraps modulo ``2\pi``. Two methods make it a
variable:

```@example index
using OptimizationVariableInterface, Static
import OptimizationVariableInterface: nvars, update

struct CircleAngle{T<:AbstractFloat}
    theta::T
end

nvars(::CircleAngle) = static(1)
update(var::CircleAngle, step, start=1) = CircleAngle(mod2pi(var.theta + step[start]))

angle = CircleAngle(0.0)
nvars(angle)
```

```@example index
update(angle, [pi])   # retract by a tangent step of π radians
```

Any optimizer written against the interface can now optimize a `CircleAngle`, alone or as one block of a
larger state.

## Who uses it

[SecondOrderOptimization.jl](https://github.com/ojwoodford/SecondOrderOptimization.jl) (Newton,
Levenberg–Marquardt, dogleg and gradient-descent optimizers) and
[NLLSModel.jl](https://github.com/ojwoodford/NLLSModel.jl) (nonlinear least-squares problems) are built on
this interface: any variable type that implements it can be optimized by them.

## Contents

- [Defining a variable type](@ref): the contract for an out-of-place type, `nvars` and `update`.
- [Collections of variables](@ref): vectors, static vectors, tuples and named tuples of variable blocks.
- [In-place variable types](@ref): the four extra methods a type updated in place needs.
- [Built-in variable types](@ref): positive and bounded scalars and bounded vectors.
- [ManifoldsBase integration](@ref): any `ManifoldsBase.AbstractManifold` as a variable.
- [Using the interface in an optimizer](@ref): a minimal Newton optimizer, and the trial-and-accept pattern.
- [API reference](@ref): every exported name.
