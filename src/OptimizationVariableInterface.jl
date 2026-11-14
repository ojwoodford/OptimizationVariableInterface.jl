"""
    OptimizationVariableInterface

The contract between the variables of an optimization problem and the optimizer that updates them.
An optimizer computes a step in the tangent space of the variables, a flat vector of length
[`nvars`](@ref), and asks the variables to apply it (a retraction), without knowing what they are:
scalars, static or dynamic vectors, bounded parameters, points on a manifold, or collections of
differently typed blocks of these.

A variable type implements:
- [`nvars`](@ref): the number of degrees of freedom (a `StaticInt` when it is fixed by the type);
- [`update`](@ref): the out-of-place retraction along a slice of the step.

A type that is updated in place implements, in addition, [`update!`](@ref), [`copyvars`](@ref),
[`syncvars!`](@ref) and [`ismutablevar`](@ref). The trait [`iseliminable`](@ref) is optional.
"""
module OptimizationVariableInterface

using Static, StaticArrays

export nvars, update, update!, copyvars, syncvars!, ismutablevar, isbitsvartype, iseliminable
export ZeroToInfScalar, ZeroToOneScalar, BoundedReal, BoundedVector
export distlo, disthi
export ManifoldVar, ManifoldContext, manifoldcontext

SR(first, last) = StaticArrays.SUnitRange(dynamic(first), dynamic(last))

# 1. Intrinsic dimensionality (degrees of freedom)
"""
    nvars(v) -> StaticInt or Int

Returns the intrinsic number of degrees of freedom (tangent-space dimension) of a variable or collection of variables.

A number has `static(1)`, and a `StaticVector` of numbers `static(N)`. For a `StaticVector`, `Vector`, `Tuple`
or `NamedTuple` of variable blocks, returns the sum over the blocks: a `StaticInt` when every block's count is
static and the container is static (`StaticVector`, `Tuple`, `NamedTuple`), otherwise an `Int`.

**Fallback:** `nvars(x) = length(x)`, used for any type without its own method (generic arrays, and custom
types that define `Base.length`).

**A custom type** defines `nvars(::MyType)`, unless the fallback is right for it. Return a `StaticInt`
(`static(N)`) when the count is fixed by the type, so that an optimizer can size its linear system at
compile time.

# Example
```jldoctest
julia> nvars(1.0)
static(1)

julia> nvars(@SVector [1.0, 2.0, 3.0])
static(3)

julia> nvars(zeros(5))
5

julia> nvars(SVector(ZeroToInfScalar(1.0), ZeroToInfScalar(2.0)))
static(2)

julia> nvars((1.0, SVector(1.0, 2.0)))
static(3)
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

Computes the out-of-place retraction of `var` along the tangent vector `step[start:start+nvars(var)-1]`, and
returns the new variable; `var` is not changed. `step` is the optimizer's whole step vector, and `start` is the
index of `var`'s first element in it, so that no slice is allocated.

For a `StaticVector`, `Tuple` or `NamedTuple` of variable blocks, updates each block from its own slice of
`step`, in order, and returns a container of the same kind; for a `StaticVector` this is unrolled and does not
allocate.

**Fallback:** `update(var, step, start=1) = var + @inbounds(view(step, start:start+nvars(var)-1))`, used for
any type without its own method. It is correct for a type whose retraction is addition and that defines
`Base.:+(::MyType, ::AbstractVector)` (and `Base.length`, or [`nvars`](@ref)).

**A custom type** whose retraction is not addition defines `update(var::MyType, step, start=1)`, reading only
`step[start:start+nvars(var)-1]`.

# Example
```jldoctest
julia> update(@SVector([1.0, 2.0]), @SVector([0.1, -0.1]))
2-element SVector{2, Float64} with indices SOneTo(2):
 1.1
 1.9

julia> update(1.0, [10.0, 0.5], 2)   # the slice starting at index 2
1.5
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

Updates variable state `to` from `from` along the tangent step `step` (starting at `step[1]`), writing into
`to` where it can. For a `Vector{<:Number}`, or a `Vector` of variable blocks (each element replaced by
[`update`](@ref) of the corresponding element of `from`), mutates `to` in place and returns `to`.

**Always use the return value**, as `to = update!(to, from, step)`: for an immutable variable `to` cannot
be changed, and the result is returned instead.

**Fallback:** `update!(to, from, step) = update(from, step)`, used for any type without its own method: `to`
is ignored and the new value is returned. This is correct for every out-of-place type.

**A custom type that is updated in place** defines `update!(to::MyType, from::MyType, step::AbstractVector)`,
writing the result into `to` and returning `to`, and must then also define [`copyvars`](@ref),
[`syncvars!`](@ref) and [`ismutablevar`](@ref).

# Example
```jldoctest
julia> to = [0.0, 0.0];

julia> update!(to, [1.0, 2.0], [0.5, 0.5]) === to   # written in place
true

julia> to
2-element Vector{Float64}:
 1.5
 2.5

julia> x = update!(1.0, 2.0, [0.5])   # immutable: only the returned value is updated
2.5
```
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
is cached (clamped to `[floatmin, floatmax]`). A step `s` multiplies the value by `exp(s)`. The constructor
takes the value; `ZeroToInfScalar(z, Val(:log))` takes the state. `NaN` propagates.

# Example
```jldoctest
julia> x = ZeroToInfScalar(2.0);

julia> x.val
2.0

julia> update(x, [log(3.0)]).val
6.0
```
"""
struct ZeroToInfScalar{T}
    logval::T   # source of truth
    val::T      # cached exp(logval)
    # The raw state is only reachable through the tagged form, so `ZeroToInfScalar{T}(x)` always takes a value
    function ZeroToInfScalar{T}(z::T, ::Val{:log}) where T
        return new{T}(z, _clampnan(exp(z), _tiny(T), _huge(T)))
    end
end
ZeroToInfScalar(z::T, ::Val{:log}) where T = ZeroToInfScalar{T}(z, Val(:log))
function _log_of(val)
    v = float(val)
    return log(_clampnan(v, _tiny(typeof(v)), _huge(typeof(v))))
end
ZeroToInfScalar(val::Real) = ZeroToInfScalar(_log_of(val), Val(:log))
ZeroToInfScalar{T}(val::Real) where T = ZeroToInfScalar{T}(convert(T, _log_of(val)), Val(:log))
nvars(::ZeroToInfScalar) = static(1)
update(var::ZeroToInfScalar, updatevec, start=1) = ZeroToInfScalar(var.logval + @inbounds(updatevec[start]), Val(:log))

"""
    ZeroToOneScalar(val)

Bounded scalar variable parameterized on the open interval `(0, 1)` via logistic retraction
(`logit(p') = logit(p) + step`). The state is stored as `logit`; `val = p` and `comp = 1 - p` are
cached and computed without cancellation, so both are accurate near either end. The constructor takes the
value (clamped to the open interval); `ZeroToOneScalar(z, Val(:logit))` takes the state. `NaN` propagates.

# Example
```jldoctest
julia> w = update(ZeroToOneScalar(0.5), [40.0]);

julia> w.val, w.comp   # comp = 1 - val, accurate where val rounds to 1
(1.0, 4.248354255291589e-18)

julia> update(w, [-40.0]).val   # the state is exact, so the step is undone
0.5
```
"""
struct ZeroToOneScalar{T}
    logit::T   # source of truth
    val::T     # cached p = 1 / (1 + exp(-logit))
    comp::T    # cached 1 - p, computed directly
    # The raw state is only reachable through the tagged form, so `ZeroToOneScalar{T}(p)` always takes a value
    function ZeroToOneScalar{T}(z::T, ::Val{:logit}) where T
        p, q = _logistic_pair(z)
        return new{T}(z, p, q)
    end
end
ZeroToOneScalar(z::T, ::Val{:logit}) where T = ZeroToOneScalar{T}(z, Val(:logit))
function _logit_of(p)   # clamped to the open interval
    f = float(p)
    q = _clampnan(f, nextfloat(zero(f)), prevfloat(one(f)))
    return log(q) - log1p(-q)
end
ZeroToOneScalar(p::Real) = ZeroToOneScalar(_logit_of(p), Val(:logit))
ZeroToOneScalar{T}(p::Real) where T = ZeroToOneScalar{T}(convert(T, _logit_of(p)), Val(:logit))
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
`val` is cached and measured from the nearer bound, so it is accurate near both ends. The bounds `lo` and
`hi` are fields (never optimized, copied unchanged by [`update`](@ref)).

However large the step, the state stays finite and `val` never leaves `[lo, hi]`. In exact arithmetic the
bounds are never reached; in floating point `val` rounds to a bound once the distance to it is below its
spacing, while [`distlo`](@ref) and [`disthi`](@ref) stay positive and accurate. The derivative of `val` with
respect to the step is `(hi - lo)·p(1 - p)`, so an optimum on a bound is approached slowly rather than
becoming active.

The constructor takes the value, clamped to the open interval, and throws an `ArgumentError` unless
`lo` and `hi` are finite with `lo < hi`; `BoundedReal(z, lo, hi, Val(:logit))` takes the state. `NaN` propagates.

# Example
```jldoctest
julia> r = BoundedReal(2.0, 1.0, 5.0);

julia> r.val, distlo(r), disthi(r)
(2.0, 1.0, 3.0)

julia> r2 = update(r, [100.0]);

julia> r2.val, disthi(r2)   # val has rounded to hi; the distance has not
(5.0, 4.464091171224992e-43)
```
"""
struct BoundedReal{T, B <: Real}
    logit::T   # state: z, p = logistic(z)
    val::T     # cached lo + (hi - lo)·p
    lo::B
    hi::B
    # The raw state is only reachable through the tagged form, so `BoundedReal{T, B}(val, lo, hi)` always takes a value
    function BoundedReal{T, B}(z::T, lo::B, hi::B, ::Val{:logit}) where {T, B <: Real}
        return new{T, B}(z, _bounded_val(z, lo, hi), lo, hi)
    end
end
BoundedReal(z::T, lo::B, hi::B, ::Val{:logit}) where {T, B <: Real} = BoundedReal{T, B}(z, lo, hi, Val(:logit))
function _bounded_state(val, lo, hi)   # -> (logit, lo, hi), validated; value clamped to the open interval
    lo, hi = promote(float(lo), float(hi))
    (isfinite(lo) && isfinite(hi) && lo < hi) || throw(ArgumentError("need finite lo < hi, got ($lo, $hi)"))
    v = _clampnan(float(val), nextfloat(lo), prevfloat(hi))
    return log(v - lo) - log(hi - v), lo, hi   # separate logs: no cancellation
end
function BoundedReal(val::Real, lo::Real, hi::Real)
    z, l, h = _bounded_state(val, lo, hi)
    return BoundedReal(z, l, h, Val(:logit))
end
function BoundedReal{T, B}(val::Real, lo::Real, hi::Real) where {T, B <: Real}
    z, l, h = _bounded_state(val, lo, hi)
    return BoundedReal{T, B}(convert(T, z), convert(B, l), convert(B, h), Val(:logit))
end
nvars(::BoundedReal) = static(1)
update(v::BoundedReal, δ, start=1) = BoundedReal(v.logit + @inbounds(δ[start]), v.lo, v.hi, Val(:logit))

"""
    distlo(v::BoundedReal)

Distance from the value of `v` to its lower bound, `v.val - v.lo`, computed from the logit (one `exp`) so
that it is accurate, and positive, even where `v.val` has rounded to the bound. See also [`disthi`](@ref).

# Example
```jldoctest
julia> distlo(BoundedReal(2.0, 1.0, 5.0))
1.0
```
"""
function distlo(v::BoundedReal)
    p, _ = _logistic_pair(v.logit)
    return (v.hi - v.lo) * p
end
"""
    disthi(v::BoundedReal)

Distance from the value of `v` to its upper bound, `v.hi - v.val`, computed from the logit (one `exp`) so
that it is accurate, and positive, even where `v.val` has rounded to the bound. See also [`distlo`](@ref).

# Example
```jldoctest
julia> disthi(BoundedReal(2.0, 1.0, 5.0))
3.0
```
"""
function disthi(v::BoundedReal)
    _, q = _logistic_pair(v.logit)
    return (v.hi - v.lo) * q
end

"""
    BoundedVector(val, lo, hi)

Vector variable with per-component open bounds `(lo[i], hi[i])`: the vector analogue of [`BoundedReal`](@ref).
`val` is any `AbstractVector` (e.g. `SVector`, `Vector`, `MVector`) and is used as the storage type for
the logit state and the cached values; `lo` and `hi` are vectors of the same length or scalars applied to
every component, and are stored in the same kind of container as `val`. Bounds are never optimized and are shared
(not copied) by [`update`](@ref), which is out-of-place. For a `StaticVector`, `nvars` is static.

Values are clamped to the open intervals, `NaN` propagates, and an `ArgumentError` is thrown unless every
bound is finite with `lo[i] < hi[i]`. `BoundedVector(z, lo, hi, Val(:logit))` takes the state. A
`BoundedVector` forwards indexing, iteration, reductions, broadcasting and arithmetic to its cached `val`,
and `dot`, `cross`, `norm` and `normalize` when `LinearAlgebra` is loaded.

# Example
```jldoctest
julia> bv = BoundedVector(SVector(0.2, 3.0), [0.0, 1.0], [1.0, 5.0]);

julia> nvars(bv)
static(2)

julia> bv[2]
3.0
```
"""
struct BoundedVector{V <: AbstractVector, B <: AbstractVector}
    logit::V   # state
    val::V     # cached lo + (hi - lo)·p
    lo::B
    hi::B
    # The raw state is only reachable through the tagged form, so `BoundedVector{V, B}(val, lo, hi)` always takes values
    function BoundedVector{V, B}(z::V, lo::B, hi::B, ::Val{:logit}) where {V <: AbstractVector, B <: AbstractVector}
        val = convert(V, map(_bounded_val, z, lo, hi))
        return new{V, B}(z, val, lo, hi)
    end
end
BoundedVector(z::V, lo::B, hi::B, ::Val{:logit}) where {V <: AbstractVector, B <: AbstractVector} = BoundedVector{V, B}(z, lo, hi, Val(:logit))

_conform(val, x::Real) = map(_ -> x, val)
_conform(val::StaticVector, x::AbstractVector) = similar_type(val)(x)
_conform(val, x::AbstractVector) = collect(x)

function _bounded_vector_state(val, lo, hi)   # -> (logit, lo, hi), validated; values clamped to the open intervals
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
    return z, lo, hi
end
function BoundedVector(val::AbstractVector, lo, hi)
    z, l, h = _bounded_vector_state(val, lo, hi)
    return BoundedVector(z, l, h, Val(:logit))
end
function BoundedVector{V, B}(val::AbstractVector, lo, hi) where {V <: AbstractVector, B <: AbstractVector}
    z, l, h = _bounded_vector_state(val, lo, hi)
    return BoundedVector{V, B}(convert(V, z), convert(B, l), convert(B, h), Val(:logit))
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

Returns `true` if `T` is an `isbitstype` `DataType` or a `Union` of `isbitstype` types. [`copyvars`](@ref) and
[`syncvars!`](@ref) read it on a `Vector{T}` of variables: when it is `true` the elements are values, and the
vector is copied in one go; otherwise each element is copied or synced with `copyvars` or `syncvars!`.

# Example
```jldoctest
julia> isbitsvartype(SVector{3, Float64}), isbitsvartype(Union{Float64, Int}), isbitsvartype(Vector{Float64})
(true, true, false)
```
"""
isbitsvartype(T) = T isa DataType ? isbitstype(T) : (T isa Union && isbitsvartype(T.a) && isbitsvartype(T.b))

"""
    ismutablevar(x) -> Bool
    ismutablevar(::Type{T}) -> Bool

Whether a variable (or container of variables) of this type is updated in place: that is, whether
`syncvars!(x, y)` writes `y`'s values into `x` and returns `x` itself, so that the caller's `x` holds
the result. When it is `false`, `x` is treated as an immutable value: `copyvars(x)` returns `x`, and
`syncvars!(x, y)` returns `y`.

This is not `Base.ismutable`. A tuple of mutable arrays is updated in place although a tuple is
immutable; a `mutable struct` that has not opted in is not.

The defining property: for `y` of the same type as `x` but with different contents, let `r = syncvars!(x, y)`.
If `ismutablevar(x)`, then afterwards the caller's `x` itself holds `y`'s values, and `r === x`. Otherwise `x`
is not fully updated (a value is left unchanged) and only `r` holds `y`'s values.

**Fallback:** `false` for any type without its own method. Mutable arrays are `true`, other arrays `false`;
a `Tuple` or `NamedTuple` is `true` when every element is; a [`ManifoldVar`](@ref) is `true` when its point is.
The result is a compile-time constant.

**A custom type** with an in-place [`update!`](@ref) must define `copyvars`, `syncvars!` and
`ismutablevar(::Type{<:MyType}) = true`.

# Example
```jldoctest
julia> ismutablevar([1.0, 2.0]), ismutablevar(SVector(1.0, 2.0)), ismutablevar(MVector(1.0, 2.0))
(true, false, true)

julia> ismutablevar(([1.0], [2.0])), ismutablevar(([1.0], SVector(2.0)))
(true, false)

julia> ismutablevar(Ref(1.0))   # mutable, but not a variable type that has opted in
false
```
"""
ismutablevar(x) = ismutablevar(typeof(x))
ismutablevar(@nospecialize(::Type)) = false                                             # values, and types that have not opted in
ismutablevar(::Type{T}) where {T <: AbstractArray} = Base.ismutabletype(T)
ismutablevar(::Type{T}) where {T <: Tuple} = _all_mutablevar(T)
ismutablevar(::Type{NamedTuple{K, T}}) where {K, T} = ismutablevar(T)

# True when every element type of the tuple type is `ismutablevar` (true for the empty tuple).
Base.@assume_effects :foldable function _all_mutablevar(@nospecialize(T::Type))
    Base.isvatuple(T) && return false   # abstract: length unknown
    return all(ismutablevar, Base.unwrap_unionall(T).parameters)
end

"""
    copyvars(vars)

Returns a copy of `vars` suitable as a scratch buffer:
- For immutable variables, returns `vars` itself (0 allocations).
- For a mutable `AbstractArray` (e.g. `Matrix`, `MVector`), returns `copy(vars)`.
- For a `Vector{T}`, uses shallow `copy(vars)` when `isbitsvartype(T)` is `true`; otherwise copies each
  element with `copyvars` (so the policy is per-element and consistent with [`syncvars!`](@ref)).
- For a `Tuple` or `NamedTuple` of variables, copies each element with `copyvars`.

**Fallback:** `copyvars(x) = x`, used for any type without its own method: the type is treated as an
immutable value.

**A custom type** with an in-place [`update!`](@ref) (one that is mutable and not an `AbstractArray`) must
define `copyvars`, returning an independent copy, and also [`syncvars!`](@ref): the fallbacks treat unknown
types as immutable values, which is only correct when `update!` is out-of-place. Such a type must also define
`ismutablevar(::Type{<:MyType}) = true` (see [`ismutablevar`](@ref)), so that
`ismutablevar(x) == (syncvars!(x, y) === x)` holds.

# Example
```jldoctest
julia> x = [1.0, 2.0]; copyvars(x) !== x   # a new buffer
true

julia> s = SVector(1.0, 2.0); copyvars(s) === s   # a value: nothing to copy
true
```
"""
copyvars(x) = x
copyvars(x::AbstractArray) = ismutablevar(x) ? copy(x) : x
copyvars(vars::Vector{T}) where T = isbitsvartype(T) ? copy(vars) : T[copyvars(v) for v in vars]
copyvars(t::Union{Tuple, NamedTuple}) = map(copyvars, t)

"""
    syncvars!(to, from) -> to

Synchronizes the variable buffer `to` with `from`, so that afterwards the result holds `from`'s values. For a
mutable array, copies `from` into `to` and returns `to`; for a `Vector` of non-isbits variables, each element
is synced with `syncvars!` (per element, as [`copyvars`](@ref) copies them). For a `Tuple` or `NamedTuple`,
returns the tuple of per-element results. For an immutable variable, returns `from`.

**Always use the return value**, as `to = syncvars!(to, from)`: only when [`ismutablevar(to)`](@ref ismutablevar)
is `to` itself updated.

**Fallback:** `syncvars!(to, from) = from`, used for any type without its own method: the type is treated as
an immutable value.

**A custom type** with an in-place [`update!`](@ref) must define `syncvars!(to::MyType, from::MyType)`,
copying `from`'s state into `to` and returning `to`, as well as `copyvars` and `ismutablevar`.

# Example
```jldoctest
julia> to = [0.0, 0.0]; syncvars!(to, [1.0, 2.0]) === to
true

julia> to
2-element Vector{Float64}:
 1.0
 2.0

julia> x = syncvars!(SVector(0.0), SVector(1.0))   # immutable: only the returned value is synced
1-element SVector{1, Float64} with indices SOneTo(1):
 1.0
```
"""
syncvars!(to, from) = from
function syncvars!(to::AbstractArray, from::AbstractArray)
    ismutablevar(to) || return from
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
The default `basis` is `DefaultOrthonormalBasis()`, and the default `retraction` is
`default_retraction_method(manifold, pointtype)` if `pointtype` is given, otherwise
`default_retraction_method(manifold)`.

# Example
```jldoctest; setup = :(using ManifoldsBase)
julia> ctx = ManifoldContext(ManifoldsBase.DefaultManifold(3));

julia> ctx.dim
static(3)
```
"""
struct ManifoldContext{M, B, R, D}
    manifold::M
    basis::B
    retraction::R
    dim::D
end

"""
    manifoldcontext(manifold; basis=..., retraction=..., pointtype=nothing, byref=nothing)

Factory for a context to share between manifold variables. Builds
`ctx = ManifoldContext(manifold; basis, retraction, pointtype)` (see [`ManifoldContext`](@ref) for the
defaults), and returns `ctx` itself, stored by value, or `Ref(ctx)`, a `Base.RefValue{<:ManifoldContext}`
for by-reference storage. With `byref = nothing` (the default), by-reference is chosen when the context is
not `isbits` or is larger than 16 bytes; `byref = true` or `false` forces the choice.

Defined in the `ManifoldsBase` extension: it has methods only once `ManifoldsBase` is loaded.

# Example
```jldoctest; setup = :(using ManifoldsBase)
julia> M = ManifoldsBase.DefaultManifold(3; parameter = :type);   # a 0-byte manifold

julia> manifoldcontext(M) isa ManifoldContext
true

julia> manifoldcontext(M; byref = true) isa Base.RefValue{<:ManifoldContext}
true
```
"""
function manifoldcontext end

"""
    ManifoldVar(point, context)

Variable wrapper for a point on a manifold, with its context: a [`ManifoldContext`](@ref) (by value) or a
`Base.RefValue` of one (by reference), as returned by [`manifoldcontext`](@ref). Variables can share a context.

When `ManifoldsBase` is loaded, `ManifoldVar(point, manifold; basis=..., retraction=..., byref=...)` builds the
context too, with the default retraction for the type of `point`, and the methods of the interface are defined:
[`nvars`](@ref) is the manifold dimension, always a `StaticInt`; [`update`](@ref) retracts the point along the
tangent vector with coordinates `step[start:start+nvars-1]` in the context's basis; [`update!`](@ref),
[`copyvars`](@ref) and [`syncvars!`](@ref) work in place when the point is a mutable array, and
`ismutablevar(v) == ismutablevar(v.point)`.

# Example
```jldoctest; setup = :(using ManifoldsBase)
julia> v = ManifoldVar(SVector(1.0, 0.0, 0.0), ManifoldsBase.DefaultManifold(3; parameter = :type));

julia> nvars(v)
static(3)

julia> update(v, [0.1, 0.2, -0.1]).point
3-element SVector{3, Float64} with indices SOneTo(3):
  1.1
  0.2
 -0.1
```
"""
struct ManifoldVar{P, C}
    point::P
    context::C
end
ismutablevar(::Type{ManifoldVar{P, C}}) where {P, C} = ismutablevar(P)

Base.convert(::Type{ManifoldVar{P, C}}, v::ManifoldVar) where {P, C} =
    ManifoldVar{P, C}(convert(P, v.point), convert(C, v.context))

@inline get_context(v::ManifoldVar{P, <:ManifoldContext}) where P = v.context
@inline get_context(v::ManifoldVar{P, <:Base.RefValue}) where P = v.context[]

# 8. Eliminable trait for Schur complement marginalization
"""
    iseliminable(::Type{V}) -> StaticBool

Returns `static(true)` if variable type `V` is structurally eligible for Schur complement elimination
(marginalization in bipartite problems such as bundle adjustment). Read by optimizers that eliminate
variables, such as NLLSModel. `iseliminable(x)` is `iseliminable(typeof(x))`.

**Fallback:** `static(false)`. **A custom type** opts in by defining
`OptimizationVariableInterface.iseliminable(::Type{<:MyType}) = static(true)`.

# Example
```jldoctest
julia> struct Point3D; x::SVector{3, Float64}; end

julia> OptimizationVariableInterface.iseliminable(::Type{<:Point3D}) = static(true)

julia> iseliminable(Point3D(SVector(0.0, 0.0, 0.0))), iseliminable(1.0)
(static(true), static(false))
```
"""
iseliminable(::Type) = static(false)
iseliminable(x) = iseliminable(typeof(x))

end # module OptimizationVariableInterface
