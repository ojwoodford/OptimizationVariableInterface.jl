# OptimizationVariableInterface.jl

[![Documentation (stable)](https://img.shields.io/badge/docs-stable-blue.svg)](https://ojwoodford.github.io/OptimizationVariableInterface.jl/stable/)
[![Documentation (dev)](https://img.shields.io/badge/docs-dev-blue.svg)](https://ojwoodford.github.io/OptimizationVariableInterface.jl/dev/)
[![Build Status](https://github.com/ojwoodford/OptimizationVariableInterface.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/ojwoodford/OptimizationVariableInterface.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/ojwoodford/OptimizationVariableInterface.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/ojwoodford/OptimizationVariableInterface.jl)

A lightweight, zero-overhead variable and retraction interface for numerical and nonlinear optimization in Julia.

In nonlinear optimization, the variables often do not live in a flat Euclidean space: they may be points on a manifold (rotations, poses, directions) or parameters confined to an interval. An optimizer computes a step in the **tangent space** of the variables and maps it back with a **retraction**. `OptimizationVariableInterface.jl` is the contract between variable representations and optimization algorithms: an optimizer works only with flat tangent vectors, sized by `nvars` and applied by `update`, with compile-time static sizes, allocation-free scratch buffers, and heterogeneous collections of variable blocks. [SecondOrderOptimization.jl](https://github.com/ojwoodford/SecondOrderOptimization.jl) and [NLLSModel.jl](https://github.com/ojwoodford/NLLSModel.jl) are built on it.

## Installation

```julia
using Pkg
Pkg.add("OptimizationVariableInterface")
```

## Example: an angle on the circle

```julia
using OptimizationVariableInterface, Static
import OptimizationVariableInterface: nvars, update

struct CircleAngle{T<:AbstractFloat}
    theta::T
end

nvars(::CircleAngle) = static(1)   # one degree of freedom, known at compile time
update(var::CircleAngle, step, start=1) = CircleAngle(mod2pi(var.theta + step[start]))

update(CircleAngle(0.0), [pi])     # CircleAngle{Float64}(3.141592653589793)
```

Any optimizer written against the interface can now optimize a `CircleAngle`, alone or as one block of a larger state.

## Documentation

The [documentation](https://ojwoodford.github.io/OptimizationVariableInterface.jl/dev/) covers:

- [defining a variable type](https://ojwoodford.github.io/OptimizationVariableInterface.jl/dev/guide/custom_variables/) (`nvars`, `update`, and the fallbacks);
- [collections of variables](https://ojwoodford.github.io/OptimizationVariableInterface.jl/dev/guide/collections/) (`Vector`, `SVector`, `Tuple`, `NamedTuple`);
- [in-place variable types](https://ojwoodford.github.io/OptimizationVariableInterface.jl/dev/guide/mutable_variables/) (`update!`, `copyvars`, `syncvars!`, `ismutablevar`);
- [built-in variable types](https://ojwoodford.github.io/OptimizationVariableInterface.jl/dev/guide/builtin_types/) (positive and bounded scalars, bounded vectors);
- [ManifoldsBase integration](https://ojwoodford.github.io/OptimizationVariableInterface.jl/dev/guide/manifolds/) (`ManifoldVar`);
- [using the interface in an optimizer](https://ojwoodford.github.io/OptimizationVariableInterface.jl/dev/guide/writing_an_optimizer/);
- the [API reference](https://ojwoodford.github.io/OptimizationVariableInterface.jl/dev/api/).

## Licence

MIT; see [LICENSE](LICENSE).
