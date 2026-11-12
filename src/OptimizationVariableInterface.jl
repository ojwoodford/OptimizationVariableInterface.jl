module OptimizationVariableInterface

using Static, StaticArrays

export nvars, update, update!, copyvars, syncvars!, isbitsvartype, iseliminable
export ZeroToInfScalar, ZeroToOneScalar, BoundedReal, BoundedVector
export distlo, disthi
export ManifoldVar, ManifoldContext, manifoldcontext

SR(first, last) = StaticArrays.SUnitRange(dynamic(first), dynamic(last))

# 1. Intrinsic dimensionality (degrees of freedom)
"""
    nvars(v) -> StaticInt or Int

Returns the intrinsic number of degrees of freedom (tangent-space dimension) of a variable or collection of variables.
Falls back to `Base.length(v)` for custom types and generic arrays.
For `StaticVector` and `Vector` collections of variable blocks, returns the sum of their individual degrees of freedom.

# Example
```julia
using OptimizationVariableInterface, StaticArrays

nvars(1.0) # static(1)
nvars(@SVector [1.0, 2.0, 3.0]) # static(3)
nvars(zeros(5)) # 5
nvars(SVector(ZeroToInfScalar(1.0), ZeroToInfScalar(2.0))) # static(2)
```
"""
nvars(x) = length(x)
nvars(::Number) = static(1)
nvars(::StaticVector{N, <:Number}) where N = static(N)
@inline nvars(vars::StaticVector) = sum(nvars, Tuple(vars); init=static(0))
nvars(vars::Vector) = sum(x -> Int(dynamic(nvars(x))), vars; init=0)
nvars(vars::Vector{<:Number}) = length(vars)
@inline nvars(t::Union{Tuple, NamedTuple}) = sum(nvars, Tuple(t); init=static(0))

# 2. Out-of-place variable update: update(var, step, start=1) -> newvar
"""
    update(var, step, start=1) -> newvar

Computes out-of-place retraction of `var` along tangent vector `step[start:start+nvars(var)-1]`.
Falls back to `var + @inbounds(view(step, ...))` using `Base.:+` if not explicitly defined.
For `StaticVector` collections of variable blocks, unrolls retraction across each block with zero allocations.

# Example
```julia
using OptimizationVariableInterface, StaticArrays

x = @SVector [1.0, 2.0]
step = @SVector [0.1, -0.1]
x_new = update(x, step)
```
"""
update(var::Number, step, start=1) = var + @inbounds(step[start])
update(var::StaticVector{N, <:Number}, step::StaticVector{N, <:Number}) where N = var + step

@generated function update(vars::StaticVector{N}, step, start=1) where N
    if N == 0
        return :(vars)
    end
    exprs = Symbol[Symbol("u_", i) for i in 1:N]
    stmts = Expr[]
    offset_expr = :(start)
    for i in 1:N
        u_i = exprs[i]
        push!(stmts, :($u_i = update(vars[$i], step, $offset_expr)))
        offset_expr = :($offset_expr + dynamic(nvars(vars[$i])))
    end
    types_expr = Expr(:call, :promote_type, [:(typeof($u)) for u in exprs]...)
    tuple_expr = Expr(:tuple, exprs...)
    return quote
        $(Expr(:meta, :inline))
        $(stmts...)
        T = $types_expr
        return similar_type(vars, T)($tuple_expr)
    end
end

@inline update(t::Tuple{}, step, start=1) = t
@inline function update(t::Tuple, step, start=1)
    h = first(t)
    return (update(h, step, start), update(Base.tail(t), step, start + dynamic(nvars(h)))...)
end
@inline update(nt::NamedTuple{K}, step, start=1) where K = NamedTuple{K}(update(Tuple(nt), step, start))

@inline function update(var, step, start=1)
    N = nvars(var)
    rng = N isa StaticInt ? (SR(0, dynamic(N) - 1) .+ start) : (start:(start + dynamic(N) - 1))
    return var + @inbounds(view(step, rng))
end

# 3. In-place / unified retraction: update!(to, from, step) -> to
"""
    update!(to, from, step::AbstractVector) -> to

Updates variable state `to` from `from` along tangent step `step`.
For immutable variables, returns the updated immutable instance via [`update(from, step)`](@ref update).
For mutable vectors (either flat `Vector{<:Number}` or `Vector` of variable blocks), mutates `to` in-place and returns `to`.
"""
@inline update!(to, from, step::AbstractVector) = update(from, step)

@inline function update!(to::Vector{T}, from::Vector{T}, step::AbstractVector) where {T<:Number}
    @inbounds for i in eachindex(to, from, step)
        to[i] = from[i] + step[i]
    end
    return to
end

@inline function update!(to::Vector, from::Vector, step::AbstractVector)
    offset = 1
    @inbounds for i in eachindex(to, from)
        v = from[i]
        to[i] = update(v, step, offset)
        offset += dynamic(nvars(v))
    end
    return to
end

# 4. Built-in variable types
# Smallest/largest positive values used to keep `exp`-derived values finite and non-zero.
# Non-`AbstractFloat` reals (e.g. autodiff duals) fall back to the `Float64` limits.
_tiny(::Type{T}) where {T<:AbstractFloat} = floatmin(T)
_tiny(::Type{<:Real}) = floatmin(Float64)
_huge(::Type{T}) where {T<:AbstractFloat} = floatmax(T)
_huge(::Type{<:Real}) = floatmax(Float64)
# Clamp to [lo, hi], letting NaN propagate
@inline _clampnan(x, lo, hi) = ifelse(x < lo, lo, ifelse(x > hi, hi, x))

# Stable logistic function: returns (p, 1 - p) computed from one `exp`, without cancellation.
@inline function _logistic_pair(z)
    e = exp(-abs(z))             # in [0, 1]
    a, b = inv(1 + e), e / (1 + e)   # larger and smaller of p, 1 - p
    return z >= 0 ? (a, b) : (b, a)
end

"""
    ZeroToInfScalar(val)

Positive scalar variable parameterized on `(0, Inf)` via multiplicative exponential retraction.
The state is stored as `logval = log(val)`, so retraction is one exact addition; `val = exp(logval)`
is cached (clamped to `[floatmin, floatmax]`). `NaN` propagates.
"""
struct ZeroToInfScalar{T}
    logval::T   # source of truth
    val::T      # cached exp(logval)
    function ZeroToInfScalar{T}(z::T) where T
        return new{T}(z, _clampnan(exp(z), _tiny(T), _huge(T)))
    end
end
ZeroToInfScalar(z::T, ::Val{:log}) where T = ZeroToInfScalar{T}(z)
function ZeroToInfScalar(val::Real)
    v = float(val)
    return ZeroToInfScalar(log(_clampnan(v, _tiny(typeof(v)), _huge(typeof(v)))), Val(:log))
end
nvars(::ZeroToInfScalar) = static(1)
update(var::ZeroToInfScalar, updatevec, start=1) = ZeroToInfScalar(var.logval + @inbounds(updatevec[start]), Val(:log))

"""
    ZeroToOneScalar(val)

Bounded scalar variable parameterized on the open interval `(0, 1)` via logistic retraction
(`logit(p') = logit(p) + step`). The state is stored as `logit`; `val = p` and `comp = 1 - p` are
cached and computed without cancellation, so both are accurate near either end. `NaN` propagates.
"""
struct ZeroToOneScalar{T}
    logit::T   # source of truth
    val::T     # cached p = 1 / (1 + exp(-logit))
    comp::T    # cached 1 - p, computed directly
    function ZeroToOneScalar{T}(z::T) where T
        p, q = _logistic_pair(z)
        return new{T}(z, p, q)
    end
end
ZeroToOneScalar(z::T, ::Val{:logit}) where T = ZeroToOneScalar{T}(z)
function ZeroToOneScalar(p::Real)   # clamped to the open interval
    f = float(p)
    q = _clampnan(f, nextfloat(zero(f)), prevfloat(one(f)))
    return ZeroToOneScalar(log(q) - log1p(-q), Val(:logit))
end
nvars(::ZeroToOneScalar) = static(1)
update(var::ZeroToOneScalar, updatevec, start=1) = ZeroToOneScalar(var.logit + @inbounds(updatevec[start]), Val(:logit))

# lo + (hi - lo)·logistic(z), measured from the nearer bound so it is accurate at both ends
@inline function _bounded_val(z, lo, hi)
    p, q = _logistic_pair(z)
    w = hi - lo
    return p <= q ? lo + w * p : hi - w * q
end

"""
    BoundedReal(val, lo, hi)

Scalar variable bounded to the open interval `(lo, hi)` (finite `lo < hi`), using the same logit
state as [`ZeroToOneScalar`](@ref): `val = lo + (hi - lo)·p` with `p = logistic(logit)`.
`val` is cached and measured from the nearer bound, so it stays accurate and strictly inside the
interval. The bounds `lo` and `hi` are fields (never optimized, copied unchanged by [`update`](@ref)).

Note that the bounds are never reached, and the derivative of `val` with respect to the step is
`(hi - lo)·p(1 - p)`, so an optimum on a bound is approached slowly rather than becoming active.
"""
struct BoundedReal{T, B <: Real}
    logit::T   # state: z, p = logistic(z)
    val::T     # cached lo + (hi - lo)·p
    lo::B
    hi::B
    function BoundedReal{T, B}(z::T, lo::B, hi::B) where {T, B <: Real}
        return new{T, B}(z, _bounded_val(z, lo, hi), lo, hi)
    end
end
BoundedReal(z::T, lo::B, hi::B, ::Val{:logit}) where {T, B <: Real} = BoundedReal{T, B}(z, lo, hi)
function BoundedReal(val::Real, lo::Real, hi::Real)
    lo, hi = promote(float(lo), float(hi))
    (isfinite(lo) && isfinite(hi) && lo < hi) || throw(ArgumentError("need finite lo < hi, got ($lo, $hi)"))
    v = _clampnan(float(val), nextfloat(lo), prevfloat(hi))   # open interval
    return BoundedReal(log(v - lo) - log(hi - v), lo, hi, Val(:logit))   # separate logs: no cancellation
end
nvars(::BoundedReal) = static(1)
update(v::BoundedReal, δ, start=1) = BoundedReal(v.logit + @inbounds(δ[start]), v.lo, v.hi, Val(:logit))

"""
    distlo(v::BoundedReal), disthi(v::BoundedReal)

Accurate distance from `v.val` to the lower / upper bound, computed from the logit (one `exp` per call).
"""
function distlo(v::BoundedReal)
    p, _ = _logistic_pair(v.logit)
    return (v.hi - v.lo) * p
end
function disthi(v::BoundedReal)
    _, q = _logistic_pair(v.logit)
    return (v.hi - v.lo) * q
end

"""
    BoundedVector(val, lo, hi)

Vector variable with per-component open bounds `(lo[i], hi[i])`: the vector analogue of [`BoundedReal`](@ref).
`val` is any `AbstractVector` (e.g. `SVector`, `Vector`, `MVector`) and is used as the storage type for
the logit state and the cached values; `lo` and `hi` are vectors of the same length or scalars applied to
every component. Bounds are never optimized and are shared (not copied) by [`update`](@ref), which is
out-of-place. For a `StaticVector`, `nvars` is static.
"""
struct BoundedVector{V <: AbstractVector, B <: AbstractVector}
    logit::V   # state
    val::V     # cached lo + (hi - lo)·p
    lo::B
    hi::B
    function BoundedVector{V, B}(z::V, lo::B, hi::B) where {V <: AbstractVector, B <: AbstractVector}
        val = convert(V, map(_bounded_val, z, lo, hi))
        return new{V, B}(z, val, lo, hi)
    end
end
BoundedVector(z::V, lo::B, hi::B, ::Val{:logit}) where {V <: AbstractVector, B <: AbstractVector} = BoundedVector{V, B}(z, lo, hi)

_conform(val, x::Real) = map(_ -> x, val)
_conform(val::StaticVector, x::AbstractVector) = similar_type(val)(x)
_conform(val, x::AbstractVector) = collect(x)

function BoundedVector(val::AbstractVector, lo, hi)
    val = map(float, val)
    lo, hi = _conform(val, lo), _conform(val, hi)
    length(lo) == length(val) == length(hi) || throw(DimensionMismatch("val, lo and hi must have equal length"))
    T = promote_type(eltype(lo), eltype(hi))
    lo, hi = map(float ∘ T, lo), map(float ∘ T, hi)
    all(isfinite, lo) && all(isfinite, hi) && all(map(<, lo, hi)) ||
        throw(ArgumentError("need finite lo < hi componentwise"))
    z = map(val, lo, hi) do v, l, h
        v = _clampnan(v, nextfloat(l), prevfloat(h))   # open interval
        log(v - l) - log(h - v)
    end
    return BoundedVector(z, lo, hi, Val(:logit))
end
nvars(::BoundedVector{<:StaticVector{N}}) where N = static(N)
nvars(v::BoundedVector) = length(v.logit)
@inline function update(v::BoundedVector, δ, start=1)
    n = nvars(v)
    rng = n isa StaticInt ? (SR(0, dynamic(n) - 1) .+ start) : (start:(start + dynamic(n) - 1))
    return BoundedVector(v.logit + @inbounds(view(δ, rng)), v.lo, v.hi, Val(:logit))
end

# 4b. Forwarding of standard operators to the cached `val` of the bounded types
const BoundedScalar = Union{ZeroToInfScalar, ZeroToOneScalar, BoundedReal}
const BoundedAny = Union{BoundedScalar, BoundedVector}
@inline _v(x::BoundedAny) = x.val
@inline _v(x) = x

# Scalars: unary functions
for f in (:+, :-, :abs, :abs2, :sqrt, :cbrt, :exp, :exp2, :exp10, :expm1, :log, :log2, :log10, :log1p,
          :sin, :cos, :tan, :sec, :csc, :cot, :asin, :acos, :atan, :sinh, :cosh, :tanh, :asinh, :acosh, :atanh,
          :sinpi, :cospi, :sind, :cosd, :tand, :deg2rad, :rad2deg, :floor, :ceil, :round, :trunc, :sign, :signbit,
          :inv, :conj, :real, :imag, :float, :isnan, :isinf, :isfinite, :iszero, :isone, :adjoint, :transpose)
    @eval Base.$f(x::BoundedScalar) = $f(x.val)
end
# Scalars: binary functions (scalar-scalar, scalar-number, number-scalar)
for f in (:+, :-, :*, :/, :\, :^, :min, :max, :atan, :hypot, :mod, :rem, :div, :fld, :cld, :copysign,
          :(==), :<, :<=, :isless)
    @eval begin
        Base.$f(a::BoundedScalar, b::BoundedScalar) = $f(a.val, b.val)
        Base.$f(a::BoundedScalar, b::Number) = $f(a.val, b)
        Base.$f(a::Number, b::BoundedScalar) = $f(a, b.val)
    end
end
# Scalars with arrays
Base.:*(a::BoundedScalar, B::AbstractArray) = a.val * B
Base.:*(A::AbstractArray, b::BoundedScalar) = A * b.val
Base.:/(A::AbstractArray, b::BoundedScalar) = A / b.val
Base.:\(a::BoundedScalar, B::AbstractArray) = a.val \ B
Base.isapprox(a::BoundedScalar, b::Union{Number, BoundedScalar}; kw...) = isapprox(a.val, _v(b); kw...)
Base.isapprox(a::Number, b::BoundedScalar; kw...) = isapprox(a, b.val; kw...)
Base.broadcastable(x::BoundedScalar) = x.val

# Vectors: container interface and unary operations
Base.length(v::BoundedVector) = length(v.val)
Base.size(v::BoundedVector) = size(v.val)
Base.axes(v::BoundedVector) = axes(v.val)
Base.eltype(::Type{<:BoundedVector{V}}) where V = eltype(V)
Base.firstindex(v::BoundedVector) = firstindex(v.val)
Base.lastindex(v::BoundedVector) = lastindex(v.val)
Base.getindex(v::BoundedVector, i...) = getindex(v.val, i...)
Base.iterate(v::BoundedVector, st...) = iterate(v.val, st...)
Base.broadcastable(v::BoundedVector) = v.val
for f in (:+, :-, :conj, :real, :imag, :adjoint, :transpose, :sum, :prod, :maximum, :minimum, :extrema,
          :first, :last, :copy, :reverse, :cumsum, :isempty)
    @eval Base.$f(x::BoundedVector) = $f(x.val)
end
# Vectors: binary operations
for f in (:+, :-, :*, :/, :\, :(==))
    @eval begin
        Base.$f(a::BoundedVector, b::BoundedVector) = $f(a.val, b.val)
        Base.$f(a::BoundedVector, b::AbstractArray) = $f(a.val, b)
        Base.$f(a::AbstractArray, b::BoundedVector) = $f(a, b.val)
    end
end
for f in (:*, :/)
    @eval Base.$f(a::BoundedVector, b::Number) = $f(a.val, b)
end
Base.:*(a::Number, b::BoundedVector) = a * b.val
Base.:\(a::Number, b::BoundedVector) = a \ b.val
Base.isapprox(a::BoundedVector, b::Union{AbstractArray, BoundedVector}; kw...) = isapprox(a.val, _v(b); kw...)
Base.isapprox(a::AbstractArray, b::BoundedVector; kw...) = isapprox(a, b.val; kw...)
# `dot`, `cross`, `norm` and `normalize` are forwarded in the LinearAlgebra extension (OVILinearAlgebraExt).

# 5. Scratch buffer allocation & syncing
"""
    isbitsvartype(T) -> Bool

Returns `true` if `T` is an `isbitstype` or a `Union` of `isbitstype` types.
"""
isbitsvartype(T::DataType) = isbitstype(T)
isbitsvartype(T::Union) = isbitsvartype(T.a) && isbitsvartype(T.b)
isbitsvartype(T) = false

"""
    copyvars(vars)

Returns a copy of `vars` suitable as a scratch buffer:
- For immutable variables, returns `vars` itself (0 allocations).
- For a mutable `AbstractArray` (e.g. `Matrix`, `MVector`), returns `copy(vars)`.
- For a `Vector{T}`, uses shallow `copy(vars)` when `isbitsvartype(T)` is `true`; otherwise copies each
  element with `copyvars` (so the policy is per-element and consistent with [`syncvars!`](@ref)).
- For a `Tuple` or `NamedTuple` of variables, copies each element with `copyvars`.

# Contract
A variable type with an in-place `update!` (i.e. one that is mutable and not a `Vector`) must also define
`copyvars` and `syncvars!`: the fallbacks treat unknown types as immutable values, which is only correct
when `update!` is out-of-place.
"""
copyvars(x) = x
copyvars(x::AbstractArray) = ismutable(x) ? copy(x) : x
copyvars(vars::Vector{T}) where T = isbitsvartype(T) ? copy(vars) : T[copyvars(v) for v in vars]
copyvars(t::Union{Tuple, NamedTuple}) = map(copyvars, t)

"""
    syncvars!(to, from) -> to

Synchronizes variable buffer `to` with `from`. Returns `from` for immutable variables or copies elements in-place for mutable arrays.
For a `Vector` of non-isbits variables each element is synced with `syncvars!`. For a `Tuple`/`NamedTuple`, returns the tuple of per-element results.
"""
syncvars!(to, from) = from
function syncvars!(to::AbstractArray, from::AbstractArray)
    ismutable(to) || return from
    copyto!(to, from)
    return to
end
function syncvars!(to::Vector{T}, from::Vector) where T
    if isbitsvartype(T)
        copyto!(to, from)
    else
        @inbounds for i in eachindex(to, from)   # per-element, consistent with `copyvars`
            to[i] = syncvars!(to[i], from[i])
        end
    end
    return to
end
syncvars!(to::Union{Tuple, NamedTuple}, from::Union{Tuple, NamedTuple}) = map(syncvars!, to, from)

# 6. Manifold variable wrappers (methods defined in extension OVIManifoldsBaseExt)
"""
    ManifoldContext(manifold, basis, retraction, dim)

Bundles a manifold, tangent basis, retraction method and the manifold dimension `dim` (a `StaticInt`,
so that `nvars` of a manifold variable is always static) for a manifold variable.
When `ManifoldsBase` is loaded, `ManifoldContext(manifold, basis, retraction)` computes `dim` itself, and
`ManifoldContext(manifold; basis=..., retraction=..., pointtype=...)` fills in defaults. Both always return a
`ManifoldContext`; use [`manifoldcontext`](@ref) to choose between by-value and by-reference storage.
"""
struct ManifoldContext{M, B, R, D}
    manifold::M
    basis::B
    retraction::R
    dim::D
end

"""
    manifoldcontext(manifold; basis=..., retraction=..., pointtype=nothing, byref=nothing)

Factory for a context to share between manifold variables. Returns a `ManifoldContext` stored by value
(for small isbits contexts), or a `Base.RefValue{<:ManifoldContext}` for by-reference storage (larger or
non-isbits contexts); `byref` forces the choice. If `pointtype` is given, the default retraction is
`default_retraction_method(manifold, pointtype)`, otherwise `default_retraction_method(manifold)`.
Defined in the `ManifoldsBase` extension.
"""
function manifoldcontext end

"""
    ManifoldVar(point, context)

Variable wrapper for a point on a manifold, with associated manifold, basis, and retraction context.
When `ManifoldsBase` is loaded, constructors accepting an `AbstractManifold` are provided:
`ManifoldVar(point, manifold; basis=..., retraction=..., byref=...)`.
"""
struct ManifoldVar{P, C}
    point::P
    context::C
end

Base.convert(::Type{ManifoldVar{P, C}}, v::ManifoldVar) where {P, C} =
    ManifoldVar{P, C}(convert(P, v.point), convert(C, v.context))

@inline get_context(v::ManifoldVar{P, <:ManifoldContext}) where P = v.context
@inline get_context(v::ManifoldVar{P, <:Base.RefValue}) where P = v.context[]

# 8. Eliminable trait for Schur complement marginalization
"""
    iseliminable(::Type{V}) -> StaticBool

Returns `static(true)` if variable type `V` is structurally eligible for Schur complement elimination
(marginalization in bipartite factor graphs like Bundle Adjustment / SLAM).
Defaults to `static(false)`.

# Example
```julia
struct Landmark3D; x::SVector{3, Float64}; end
OptimizationVariableInterface.iseliminable(::Type{<:Landmark3D}) = static(true)
```
"""
iseliminable(::Type) = static(false)
iseliminable(x) = iseliminable(typeof(x))

end # module OptimizationVariableInterface
