# Using the interface in an optimizer

An optimizer uses five functions to manage the variables:

- [`nvars(x)`](@ref nvars): the dimension of the tangent space, to size the gradient, the Hessian or the
  linear system.
- [`copyvars(x)`](@ref copyvars): a scratch buffer for a candidate point (free for immutable variables).
- [`update!(to, from, step)`](@ref update!): the candidate, `from` retracted along the step.
- [`syncvars!(to, from)`](@ref syncvars!): copies an accepted candidate into the current point.
- [`ismutablevar(x)`](@ref ismutablevar): whether `x` is updated in place, so that an optimizer can choose
  between a mutating and a non-mutating entry point.

The return values of `update!` and `syncvars!` are always assigned back. That one rule makes the same code
correct for immutable values, which are returned, and for mutable ones, which are written in place.

## A minimal Newton optimizer

```@setup optimizer
using OptimizationVariableInterface, Static, StaticArrays
import OptimizationVariableInterface: nvars, update
struct CircleAngle{T<:AbstractFloat}
    theta::T
end
nvars(::CircleAngle) = static(1)
update(var::CircleAngle, step, start=1) = CircleAngle(mod2pi(var.theta + step[start]))
```

```@example optimizer
using OptimizationVariableInterface, LinearAlgebra
using Static: dynamic

"""
    manifold_newton(fdf!, x0; max_iters=20, tol=1e-6)

Minimizes over the variables `x0` with Newton's method. `fdf!(H, g, x)` returns the cost at `x`,
and fills the tangent-space Hessian `H` and gradient `g` in place.
"""
function manifold_newton(fdf!, x0; max_iters=20, tol=1e-6)
    # 1. Size the tangent-space Hessian and gradient with nvars
    n = dynamic(nvars(x0))
    g = zeros(Float64, n)
    H = zeros(Float64, n, n)

    # 2. Variable buffers, copied only where needed
    x = copyvars(x0)
    x_next = copyvars(x0)

    for iter in 1:max_iters
        fdf!(H, g, x)
        norm(g) < tol && break

        # Newton step in the tangent space: H * step = -g
        step = -(H \ g)

        # 3. Retraction, x_next = x ⊕ step; always assign the result back
        x_next = update!(x_next, x, step)

        # 4. Accept the step
        x = syncvars!(x, x_next)
    end
    return x
end
nothing # hide
```

Because `copyvars`, `update!` and `syncvars!` handle immutable and mutable variables alike, this one function
works whether `x0` is a `Float64`, an `SVector`, a `Vector{Float64}`, a custom type such as the `CircleAngle`
of [Defining a variable type](@ref), or a collection of variable blocks.

Run on the circle angle, minimizing ``f(\theta) = 1 - \cos(\theta - 1)``:

```@example optimizer
function cost_grad_hess!(H, g, a::CircleAngle)
    d = a.theta - 1.0
    g[1] = sin(d)                 # tangent gradient
    H[1, 1] = max(cos(d), 1e-4)   # tangent Hessian, kept positive
    return 1.0 - cos(d)
end

x_opt = manifold_newton(cost_grad_hess!, CircleAngle(0.2))
```

```@example optimizer
isapprox(x_opt.theta, 1.0; atol = 1e-6)
```

## Trial and best points

An optimizer that can reject a step keeps the current point and a trial point. The trial is computed into
its own buffer, and copied into the current point only when it is accepted, so a rejected step leaves the
current point untouched, whether the variables are values or are updated in place:

```@example optimizer
function descend(cost, grad, x0; iters = 50, stepsize = 1.0)
    x = copyvars(x0)
    trial = copyvars(x0)
    c = cost(x)
    for _ in 1:iters
        trial = update!(trial, x, -stepsize * grad(x))
        ct = cost(trial)
        if ct < c                # accept: the trial becomes the current point
            x = syncvars!(x, trial)
            c = ct
            stepsize *= 2
        else                     # reject: x is unchanged
            stepsize /= 2
        end
    end
    return x
end

cost(x) = sum(abs2, x .- [1.0, 2.0])
grad(x) = 2 .* (x .- [1.0, 2.0])
x0 = [0.0, 0.0]
descend(cost, grad, x0)          # in place on a Vector
```

```@example optimizer
x0                               # the starting point is not modified
```

```@example optimizer
descend(cost, grad, SVector(0.0, 0.0))   # the same code on an immutable SVector
```

The same pattern keeps a best-so-far point: `best = copyvars(x0)`, then `best = syncvars!(best, x)` each
time `x` improves on it.

## Traits read by optimizers

- [`isbitsvartype(T)`](@ref isbitsvartype) is read by `copyvars` and `syncvars!` on a `Vector{T}` of
  variables: when the element type is `isbits`, the elements are values and the vector is copied in one go;
  otherwise each element is copied or synced separately. An optimizer that manages its own buffers of
  variables can make the same choice.
- [`iseliminable(V)`](@ref iseliminable) marks a variable type as eligible for Schur complement
  elimination. It is read by optimizers that eliminate variables from a bipartite problem (such as
  bundle adjustment) before solving for the rest; NLLSModel uses it to decide whether to eliminate. A type
  opts in with `OptimizationVariableInterface.iseliminable(::Type{<:MyType}) = static(true)`.
