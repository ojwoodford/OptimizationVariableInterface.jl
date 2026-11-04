# OptimizationVariableInterface.jl

[![Build Status](https://github.com/ojwoodford/OptimizationVariableInterface.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/ojwoodford/OptimizationVariableInterface.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/ojwoodford/OptimizationVariableInterface.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/ojwoodford/OptimizationVariableInterface.jl)

A lightweight, zero-overhead variable and manifold retraction interface for numerical and nonlinear optimization in Julia.

---

## Purpose & Motivation

In nonlinear optimization, decision variables often do not live in a standard flat Euclidean space $\mathbb{R}^n$. Instead, they may belong to non-Euclidean manifolds or constrained domains—such as 3D rotations ($SO(3)$), rigid-body poses ($SE(3)$), directions on a unit sphere ($S^2$), or strictly positive / interval-bounded physical parameters.

On a manifold $\mathcal{M}$, moving in search directions cannot be done with simple vector addition ($x + \Delta x$). Instead, optimization algorithms compute a step vector $\Delta x$ in the **tangent space** $T_x\mathcal{M} \cong \mathbb{R}^d$ and map it back onto the manifold using a **retraction** (or exponential map):

$$x_{k+1} = \mathcal{R}_x(\Delta x)$$

`OptimizationVariableInterface.jl` provides the shared contract between **variable representations** and **optimization algorithms**:
1. **Decouples solvers from representations**: Solvers (such as gradient descent, Gauss-Newton, or Levenberg-Marquardt) only operate on tangent vectors in $\mathbb{R}^n$ without needing knowledge of specific manifold geometries.
2. **Zero-overhead & allocation-friendly**: Built for high performance, supporting compile-time static dimensions (`Static.jl`, `StaticArrays.jl`), allocation-free scratch buffers, and unified handling of both mutable arrays and immutable structs.
3. **Composable variable blocks**: Automatically handles heterogeneous collections of variable blocks (e.g. camera poses, 3D landmarks, calibration parameters) as a single composite state.

Packages such as [`SecondOrderOptimization.jl`](https://github.com/ojwoodford/SecondOrderOptimization.jl) and [`NLLSModels.jl`](https://github.com/ojwoodford/NLLSModels.jl) build upon this interface.

---

## 1. Defining a Custom Variable Type

To create a custom manifold or constrained variable, define your type and extend two methods:
1. `OptimizationVariableInterface.nvars`: returns the intrinsic tangent-space dimensionality (degrees of freedom).
2. `OptimizationVariableInterface.update`: computes the retraction step $\mathcal{R}_x(\Delta x)$.

### Example: A 2D Rotation / Circle Angle ($S^1$)

The circle $S^1$ has 1 degree of freedom, but angles wrap modulo $2\pi$:

```julia
using OptimizationVariableInterface, Static
import OptimizationVariableInterface: nvars, update

# Custom variable type: an angle on the circle S¹
struct CircleAngle{T<:AbstractFloat}
    theta::T
end

# 1. Tangent space dimension (intrinsic degrees of freedom)
# Returning `static(1)` allows compile-time sizing in static solvers
nvars(::CircleAngle) = static(1)

# 2. Retraction: step starts at `start` in the tangent step vector
update(var::CircleAngle, step, start=1) = CircleAngle(mod2pi(var.theta + step[start]))
```

That's it! Your custom variable is now fully compatible with any optimizer or problem structure using `OptimizationVariableInterface.jl`.

```julia
angle = CircleAngle(0.0)
nvars(angle) # static(1)

# Retract by a tangent step of π radians
update(angle, [pi]) # CircleAngle(3.14159...)
```

> **Why the `start` argument?**
> In optimization algorithms with multiple variables, the tangent step $\Delta x$ is computed as a single concatenated vector. The solver passes the variable's slice offset via `start`, avoiding intermediate allocations.

---

## 2. Using Blocks of Heterogeneous Variables

Real-world optimization problems often involve composite states with different variable types—for example, a camera angle ($S^1$), a 2D Euclidean translation ($\mathbb{R}^2$), and a positive scale parameter ($\mathbb{R}^+$).

`OptimizationVariableInterface.jl` supports collections of variables out of the box:

```julia
using OptimizationVariableInterface, StaticArrays

# Problem state combining different variable types:
vars = [CircleAngle(0.0), SVector(1.0, 2.0), ZeroToInfScalar(1.0)]

# Total tangent space dimension: 1 + 2 + 1 = 4
n = nvars(vars) # 4

# Suppose an optimizer computed a 4-dimensional tangent step:
step = [0.1, 0.5, -0.5, log(2.0)]

# Unified retraction: updates each block according to its individual nvars
vars_next = copyvars(vars)
update!(vars_next, vars, step)

# vars_next[1] is CircleAngle(0.1)
# vars_next[2] is SVector(1.5, 1.5)
# vars_next[3] is ZeroToInfScalar(2.0)
```

---

## 3. How an Optimizer Uses the Interface

Optimization solvers use four core functions to manage state, allocate buffers, and apply steps:

- `nvars(x)`: Determines the dimension of the tangent space to size linear systems, gradients, or Hessian matrices.
- `copyvars(x)`: Allocates scratch buffers for candidate states (zero allocations for immutable types; shallow copies for `isbits` vectors).
- `update!(to, from, step)`: Retracts candidate variables along the tangent step.
- `syncvars!(to, from)`: Synchronizes state when a step is accepted.

### Example: A Minimal Manifold Newton Optimizer

Here is a complete, working example showing how a second-order optimizer interacts with variables via the interface. Notice how `nvars(x0)` is used to size and preallocate the tangent gradient vector and Hessian matrix:

```julia
using OptimizationVariableInterface, LinearAlgebra

"""
    manifold_newton(fdf!, x0; max_iters=20, tol=1e-6)

Optimizes problem state `x0` over an arbitrary manifold using Newton's method.
`fdf!(H, g, x)` evaluates the objective and fills the tangent-space Hessian `H`
and gradient `g` in-place.
"""
function manifold_newton(fdf!, x0; max_iters=20, tol=1e-6)
    # 1. Use nvars to preallocate tangent-space Hessian and gradient
    n = dynamic(nvars(x0))
    g = zeros(Float64, n)
    H = zeros(Float64, n, n)

    # 2. Allocate variable buffers without unnecessary allocations
    x = copyvars(x0)
    x_next = copyvars(x0)

    for iter in 1:max_iters
        # Fill Hessian and gradient in-place in tangent space
        cost = fdf!(H, g, x)
        if norm(g) < tol
            break
        end

        # Solve Newton step in tangent space: H * step = -g
        step = -(H \ g)

        # 3. Retraction step: x_next = x ⊕ step
        # Always assign back: handles immutable structs and in-place vectors uniformly
        x_next = update!(x_next, x, step)

        # 4. Accept step: synchronize current state with candidate state
        x = syncvars!(x, x_next)
    end

    return x
end
```

Because `update!`, `copyvars`, and `syncvars!` handle immutable and mutable types transparently, this single optimizer function works seamlessly whether `x0` is:
- A scalar `Float64`
- A static vector `SVector{N, Float64}`
- A standard dynamic `Vector{Float64}`
- A custom manifold struct (like `CircleAngle`)
- A heterogeneous `Vector` of variable blocks

#### Running the Optimizer on `CircleAngle`:

```julia
# Minimize f(θ) = 1 - cos(θ - 1.0)
# Tangent gradient: sin(θ - 1.0)
# Tangent Hessian: cos(θ - 1.0)
function cost_grad_hess!(H, g, a::CircleAngle)
    d = a.theta - 1.0
    g[1] = sin(d)
    H[1, 1] = max(cos(d), 1e-4) # ensure positive curvature
    return 1.0 - cos(d)
end

x_opt = manifold_newton(cost_grad_hess!, CircleAngle(0.2))
# Result: CircleAngle(1.0)
```

---

## Built-in Variable Types

The package includes default implementations for standard types as well as common bounded scalar manifolds:

| Type | Tangent Dim (`nvars`) | Retraction Behavior |
| :--- | :--- | :--- |
| `Number` | `static(1)` | Standard scalar addition: $x + \Delta x$ |
| `StaticVector{N}` | `static(N)` | Vector addition with `StaticArrays.jl` |
| `Vector{<:Number}` | `length(v)` | In-place vector addition |
| `ZeroToInfScalar{T}(val)` | `static(1)` | Constrained to $(0, \infty)$ via multiplicative exponential map: x \cdot e^{\Delta x}$ |
| `ZeroToOneScalar{T}(val)` | `static(1)` | Constrained to $(0, 1)$ via logistic-exponential retraction |

---

## Interface Reference

| Method | Role | Invariant / Behavior |
| :--- | :--- | :--- |
| `nvars(x)` | Tangent dimension | Returns `StaticInt` or `Int` indicating intrinsic degrees of freedom. |
| `update(var, step, start=1)` | Out-of-place retraction | Returns new variable instance retracted along `step[start : start+nvars(var)-1]`. |
| `update!(to, from, step)` | Unified / in-place retraction | Mutates `to` in-place for `Vector`s; returns `update(from, step)` for immutables. Always use as `to = update!(to, from, step)`. |
| `copyvars(x)` | Buffer allocation | Returns `x` for immutables (0 allocations); returns shallow `copy` for `isbits` element vectors; `deepcopy` otherwise. |
| `syncvars!(to, from)` | State synchronization | Copies elements in-place for mutable `Vector`s; returns `from` for immutables. Always use as `to = syncvars!(to, from)`. |
| `isbitsvartype(T)` | Type introspection | Returns `true` if `T` is an `isbitstype` `DataType` or a `Union` of `isbitstype` types. |
