# ManifoldsBase integration

Any manifold defined with [ManifoldsBase.jl](https://github.com/JuliaManifolds/ManifoldsBase.jl) (including
those of [Manifolds.jl](https://github.com/JuliaManifolds/Manifolds.jl)) can be used as a variable through
[`ManifoldVar`](@ref). The methods live in the package extension `OVIManifoldsBaseExt`, which loads when
ManifoldsBase is loaded:

```@example manifolds
using OptimizationVariableInterface, ManifoldsBase, StaticArrays
```

Without ManifoldsBase, the types [`ManifoldVar`](@ref) and [`ManifoldContext`](@ref) exist, but the
function [`manifoldcontext`](@ref) and the interface methods for `ManifoldVar` have no methods.

## `ManifoldVar` and `ManifoldContext`

A [`ManifoldVar`](@ref) wraps a point on an `AbstractManifold`. Its step is the vector of coordinates of a
tangent vector in a basis of the tangent space, and [`update`](@ref) maps it to a tangent vector with
`get_vector` and retracts the point along it with `retract`. The examples use a flat manifold, but any
`AbstractManifold` works:

```@example manifolds
M = ManifoldsBase.DefaultManifold(3; parameter = :type)   # ℝ³, with the size in the type
p = @SVector [1.0, 0.0, 0.0]

var = ManifoldVar(p, M)
nvars(var)   # the manifold dimension, always static
```

```@example manifolds
var_new = update(var, [0.1, 0.2, -0.1])   # a 3-dimensional tangent step
var_new.point
```

The manifold, the basis (default `DefaultOrthonormalBasis()`), the retraction (default
`default_retraction_method(M, typeof(p))`) and the manifold dimension are held in a
[`ManifoldContext`](@ref). The dimension is stored as a `StaticInt`, which is why `nvars` of a `ManifoldVar`
is always static. `ManifoldVar(point, manifold; basis, retraction, byref)` builds the context;
`ManifoldContext(manifold; basis, retraction, pointtype)` builds one on its own, always stored by value.

## By-value and by-reference storage

When thousands of variables share one manifold, the context should not be duplicated in each.
[`manifoldcontext`](@ref) builds a context and chooses how it is stored:

- **By value** (`byref = false`): the `ManifoldContext` is stored inline in each variable, with no pointer
  to follow. The default for a context that is `isbits` and at most 16 bytes, such as one for a manifold with
  its size in the type (`parameter = :type`), which takes no bytes at all.
- **By reference** (`byref = true`): the context is stored once, in a `Base.RefValue`, and each variable
  holds a pointer to it. The default for a larger or non-`isbits` context.

`ManifoldVar(point, manifold)` and `manifoldcontext(manifold)` choose automatically, and `byref` forces the
choice:

```@example manifolds
manifoldcontext(M) isa ManifoldContext   # 0 bytes: by value
```

To share one context between many variables, create it once and pass it to each:

```@example manifolds
ctx = manifoldcontext(M; byref = true)
points = [SVector(1.0, 0.0, Float64(i)) for i in 1:10_000]
vars = [ManifoldVar(pt, ctx) for pt in points]

vars[1].context === vars[2].context === ctx   # one shared context
```

[`copyvars`](@ref) makes new scratch buffers without copying the shared context:

```@example manifolds
vars_copy = copyvars(vars)
vars_copy[1].context === ctx
```

A collection of manifold variables is a collection like any other (see [Collections of variables](@ref)):

```@example manifolds
nvars(vars)
```

## In-place manifold points

When the point is a mutable array (a `Vector`, say), the variable is updated in place:
[`update!`](@ref) retracts into the existing point with `retract!`, and [`syncvars!`](@ref) copies into it.
[`ismutablevar`](@ref) of a `ManifoldVar` is that of its point:

```@example manifolds
mvar = ManifoldVar([1.0, 0.0, 0.0], ManifoldsBase.DefaultManifold(3))
ismutablevar(mvar), ismutablevar(var)
```

```@example manifolds
trial = copyvars(mvar)
trial = update!(trial, mvar, [0.1, 0.2, -0.1])
trial.point, mvar.point
```
