# API reference

Every exported name. Names that are not exported are internal and may change without notice.

```@docs
OptimizationVariableInterface
```

## The variable contract

Every variable type implements these; see [Defining a variable type](@ref).

```@docs
nvars
update
update!
```

## In-place support

Needed only by a type that is updated in place; see [In-place variable types](@ref).

```@docs
copyvars
syncvars!
ismutablevar
```

## Traits

```@docs
isbitsvartype
iseliminable
```

## Built-in types

See [Built-in variable types](@ref).

```@docs
ZeroToInfScalar
ZeroToOneScalar
BoundedReal
BoundedVector
distlo
disthi
```

## Manifolds

Methods for these are defined when ManifoldsBase is loaded; see [ManifoldsBase integration](@ref).

```@docs
ManifoldVar
ManifoldContext
manifoldcontext
```
