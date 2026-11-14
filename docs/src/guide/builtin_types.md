# Built-in variable types

## Summary

| Type | Tangent dimension (`nvars`) | Retraction |
| :--- | :--- | :--- |
| `Number` | `static(1)` | Addition: ``x + \Delta x`` |
| `StaticVector{N, <:Number}` | `static(N)` | Vector addition |
| `StaticVector` of blocks | sum over blocks (`StaticInt` if every block's is) | Unrolled per block, no allocation |
| `Tuple` / `NamedTuple` of blocks | sum over blocks (`StaticInt` if every block's is) | Per block; returns a new tuple of the same shape |
| `Vector{<:Number}` | `length(v)` | Addition; in place with `update!` |
| `Vector` of blocks | sum over blocks (`Int`) | Per block; in place with `update!` |
| [`ZeroToInfScalar`](@ref) | `static(1)` | On ``(0, \infty)``: ``x \cdot e^{\Delta x}`` |
| [`ZeroToOneScalar`](@ref) | `static(1)` | On ``(0, 1)``: adds the step to the logit |
| [`BoundedReal`](@ref) | `static(1)` | On ``(lo, hi)``: adds the step to the logit |
| [`BoundedVector`](@ref) | `static(N)` for static storage, else `length` | Per-component bounds; adds the step to the logits |
| [`ManifoldVar`](@ref) | `static(manifold_dimension(M))` | `retract` along `get_vector` (needs ManifoldsBase; see [ManifoldsBase integration](@ref)) |
| Any other type | `length(x)` | `x + view(step, ...)` |

The containers are described in [Collections of variables](@ref). This page is about the bounded types.

## Bounded types

The bounded types store an unconstrained **state**, the logarithm or the logit of the value, and cache the
value derived from it. A retraction is one exact addition to the state, so the result can never leave the
interval, however large the step, and a step followed by its negative returns exactly to the start.

```@setup builtin
using OptimizationVariableInterface, StaticArrays, LinearAlgebra
```

### `ZeroToInfScalar`: positive scalars

[`ZeroToInfScalar`](@ref) is a positive scalar. Its state is `logval = log(val)`, and a step ``s``
multiplies the value by ``e^s``. Read the value from `val`:

```@example builtin
x = ZeroToInfScalar(2.0)
x.val, update(x, [log(3.0)]).val
```

The cached value is clamped to `[floatmin, floatmax]`, so it stays positive and finite after any step:

```@example builtin
update(x, [-1000.0]).val, update(x, [1000.0]).val
```

### `ZeroToOneScalar`: the unit interval

[`ZeroToOneScalar`](@ref) is on the open interval ``(0, 1)``. Its state is the logit, ``\log(p / (1 - p))``;
the value `val = p` and its complement `comp = 1 - p` are both cached, each computed without cancellation,
so `comp` stays accurate where `val` has rounded to 1:

```@example builtin
w = ZeroToOneScalar(0.5)
w2 = update(w, [40.0])
w2.val, w2.comp
```

```@example builtin
update(w2, [-40.0]).val   # back exactly
```

### `BoundedReal`: an interval

[`BoundedReal`](@ref) is on ``(lo, hi)``, for finite ``lo < hi``, with the same logit state as
`ZeroToOneScalar`: ``val = lo + (hi - lo)\,p``. The bounds are fields, not type parameters, so they can be
set at run time; they are never optimized, and `update` copies them unchanged. [`distlo`](@ref) and
[`disthi`](@ref) give the distance to each bound, computed from the logit so that they are accurate near it:

```@example builtin
r = BoundedReal(2.0, 1.0, 5.0)
r.val, distlo(r), disthi(r)
```

```@example builtin
r2 = update(r, [0.3])
1.0 < r2.val < 5.0
```

### `BoundedVector`: per-component bounds

[`BoundedVector`](@ref) is the vector analogue of `BoundedReal`, with bounds `lo[i] < hi[i]` for each
component; a scalar bound applies to every component. The value can be any `AbstractVector` (an `SVector`,
`Vector`, `MVector`, ...), and its type is the storage type for the state, the cached values and the bounds.
With static storage, `nvars` is static:

```@example builtin
bv = BoundedVector(SVector(0.2, 3.0), [0.0, 1.0], [1.0, 5.0])   # static storage
nvars(bv)
```

```@example builtin
bd = BoundedVector([0.2, 3.0], 0.0, 5.0)   # Vector storage, scalar bounds
nvars(bd)
```

```@example builtin
update(bv, SVector(0.1, -0.1)).val
```

`update` is out of place and shares `lo` and `hi` with the original.

### Construction, bounds and `NaN`

- Construction always takes a *value*, including the parameterized forms (`ZeroToInfScalar{Float32}(2.0)`,
  `BoundedReal{Float64, Float64}(v, lo, hi)`). The raw state is reachable only through the tagged forms,
  `ZeroToInfScalar(z, Val(:log))`, `ZeroToOneScalar(z, Val(:logit))`, `BoundedReal(z, lo, hi, Val(:logit))`
  and `BoundedVector(z, lo, hi, Val(:logit))`, and is read back from the `logval` or `logit` field.
- The constructors clamp the value to the open interval, so a value on a bound gives a finite state:

  ```@example builtin
  BoundedReal(1.0, 1.0, 5.0).logit
  ```

- `NaN` propagates rather than being masked:

  ```@example builtin
  ZeroToOneScalar(NaN).val, update(BoundedReal(2.0, 1.0, 5.0), [NaN]).val
  ```

- Invalid bounds are an `ArgumentError`:

  ```@example builtin
  try
      BoundedReal(2.0, 5.0, 1.0)
  catch err
      sprint(showerror, err)
  end
  ```

- In exact arithmetic the bounds are never reached. In floating point, `val` rounds to a bound once the
  distance to it is below the spacing of floating-point numbers there; the state, `comp`, `distlo` and
  `disthi` stay accurate:

  ```@example builtin
  r3 = update(r, [100.0])
  r3.val, disthi(r3)
  ```

- The derivative of `val` with respect to the step is ``(hi - lo)\,p(1-p)``, which vanishes at the bounds:
  an optimum on a bound is approached slowly, rather than becoming an active constraint.
- `update` works with automatic differentiation: with a `ForwardDiff.Dual` step, the state and cached
  values become duals while the bounds stay plain numbers.

## Operator forwarding

`ZeroToInfScalar`, `ZeroToOneScalar`, `BoundedReal` and `BoundedVector` forward the standard arithmetic,
comparison and mathematical functions (`+`, `-`, `*`, `/`, `^`, `sin`, `exp`, `<`, `==`, `≈`, ...) to their
cached `val`, so they can be used directly in residual code. The result is a plain number or vector, not a
wrapper:

```@example builtin
w = ZeroToOneScalar(0.25)
2w, sin(w), w < 0.5
```

`==` and `<` compare `val` only. `BoundedVector` also forwards indexing, iteration, reductions and
broadcasting:

```@example builtin
bv[1], sum(bv), sqrt.(bv)
```

When `LinearAlgebra` is loaded, the package extension `OVILinearAlgebraExt` forwards `dot`, `cross`,
`norm` and `normalize` too:

```@example builtin
norm(bv), dot(bv, [1.0, 1.0])
```
