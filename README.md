# OptimizationVariableInterface.jl

[![Build Status](https://github.com/ojwoodford/OptimizationVariableInterface.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/ojwoodford/OptimizationVariableInterface.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/ojwoodford/OptimizationVariableInterface.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/ojwoodford/OptimizationVariableInterface.jl)

Common variable and manifold retraction interface for nonlinear optimization in Julia.

## Overview

`OptimizationVariableInterface.jl` defines the shared variable interface (`nvars`, `update`, `update!`, `copyvars`, `syncvars!`, `isbitsvartype`) along with default implementations for Euclidean scalars (`Number`), static vectors (`StaticVector`), dynamic vectors (`Vector`), and bounded scalar manifolds (`ZeroToInfScalar`, `ZeroToOneScalar`).

Custom manifold variable types only need to extend `OptimizationVariableInterface.nvars` and `OptimizationVariableInterface.update` to work seamlessly across optimization and modeling packages such as `SecondOrderOptimization.jl` and `NLLSModels.jl`.

## Externally Available Types and Methods

### `nvars`

```julia
nvars(::Number) -> StaticInt{1}
nvars(::StaticVector{N}) -> StaticInt{N}
nvars(v::AbstractVector{<:Number}) -> Int
nvars(vars::AbstractVector) -> Int or StaticInt
nvars(::ZeroToInfScalar) -> StaticInt{1}
nvars(::ZeroToOneScalar) -> StaticInt{1}
```

Returns the intrinsic degrees of freedom (tangent-space dimensionality) of a variable or vector of variable blocks.

### `update`

```julia
update(var::Number, step, start=1) -> Number
update(var::StaticVector{N}, step::StaticVector{N}) -> StaticVector{N}
update(var::StaticVector{N}, step, start=1) -> StaticVector{N}
update(var::Vector{T<:Number}, step, start=1) -> Vector{T}
update(var::ZeroToInfScalar, step, start=1) -> ZeroToInfScalar
update(var::ZeroToOneScalar, step, start=1) -> ZeroToOneScalar
```

Computes out-of-place retraction of `var` along the tangent step vector `step` starting at index `start`.

### `update!`

```julia
update!(to, from, step::AbstractVector) -> to
update!(to::Vector{T<:Number}, from::Vector{T<:Number}, step::AbstractVector) -> Vector{T}
update!(to::Vector, from::Vector, step::AbstractVector) -> Vector
```

Unified variable retraction used inside optimization loops:
- For immutable variables (`Number`, `SVector`, `ZeroToInfScalar`, `ZeroToOneScalar`, or custom structs), returns `update(from, step)`.
- For flat mutable `Vector{<:Number}`, updates `to[i] = from[i] + step[i]` in-place.
- For a heterogeneous or homogeneous `Vector` of variable blocks, updates each block `to[i] = update(from[i], step, offset)` advancing `offset` by `nvars(from[i])`.

### `ZeroToInfScalar{T}`

```julia
ZeroToInfScalar(val::T)
```

Scalar variable constrained to $(0, \infty)$ using multiplicative exponential retraction:
$\text{val}_{\text{new}} = \max(\text{val}, \text{floatmin}) \cdot \exp(\Delta x)$.

### `ZeroToOneScalar{T}`

```julia
ZeroToOneScalar(val::T)
```

Scalar variable constrained to $(0, 1)$ using logistic-exponential retraction.

### `isbitsvartype`

```julia
isbitsvartype(T) -> Bool
```

Returns `true` if `T` is an `isbitstype` `DataType` or a `Union` of `isbitstype` types, and `false` otherwise.

### `copyvars`

```julia
copyvars(x)
copyvars(vars::Vector{T}) -> Vector{T}
```

Allocates a scratch variable buffer matching `x`:
- Returns `x` directly (zero allocations) when `x` is not a `Vector`.
- Uses shallow `copy(vars)` when `isbitsvartype(T)` is `true`, and `deepcopy(vars)` otherwise.

### `syncvars!`

```julia
syncvars!(to, from) -> to
syncvars!(to::Vector, from::Vector) -> Vector
```

Synchronizes scratch variable buffer `to` with `from` (`copyto!(to, from)` for `Vector`, or returns `from` for immutable variables).
