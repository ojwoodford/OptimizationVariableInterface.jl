module OptimizationVariableInterface

using Static, StaticArrays

export nvars, update, update!, copyvars, syncvars!, isbitsvartype, iseliminable
export ZeroToInfScalar, ZeroToOneScalar
export ManifoldVar, ManifoldContext

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
"""
    ZeroToInfScalar(val)

Positive scalar variable parameterized on `(0, Inf)` via multiplicative exponential retraction.
"""
struct ZeroToInfScalar{T}
    val::T
end
nvars(::ZeroToInfScalar) = static(1)
update(var::ZeroToInfScalar, updatevec, start=1) = ZeroToInfScalar(ifelse(var.val > 0, var.val, floatmin(var.val)) * exp(@inbounds(updatevec[start])))

"""
    ZeroToOneScalar(val)

Bounded scalar variable parameterized on `(0, 1)` via logistic-exponential retraction.
"""
struct ZeroToOneScalar{T}
    val::T
end
nvars(::ZeroToOneScalar) = static(1)
function update(var::ZeroToOneScalar, updatevec, start=1)
    val = ifelse(var.val > 0, var.val, floatmin(var.val)) * exp(@inbounds(updatevec[start]))
    return ZeroToOneScalar(ifelse(val < Inf, val / (1 + (val - var.val)), one(val)))
end

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
- For a `Vector{T}`, uses shallow `copy(vars)` when `isbitsvartype(T)` is `true` and `deepcopy(vars)` otherwise.
"""
copyvars(x) = x
copyvars(x::AbstractArray) = ismutable(x) ? copy(x) : x
copyvars(vars::Vector{T}) where T = isbitsvartype(T) ? copy(vars) : deepcopy(vars)

"""
    syncvars!(to, from) -> to

Synchronizes variable buffer `to` with `from`. Returns `from` for immutable variables or copies elements in-place for mutable arrays.
"""
syncvars!(to, from) = from
function syncvars!(to::AbstractArray, from::AbstractArray)
    ismutable(to) || return from
    copyto!(to, from)
    return to
end
function syncvars!(to::Vector, from::Vector)
    copyto!(to, from)
    return to
end

# 6. Manifold variable wrappers (methods defined in extension OVIManifoldsBaseExt)
"""
    ManifoldContext(manifold, basis, retraction)

Bundles a manifold, tangent basis, and retraction method for a manifold variable.
When `ManifoldsBase` is loaded, constructors accepting an `AbstractManifold` are provided,
supporting both by-value storage (for small isbits manifolds) and by-reference storage
(via `Base.RefValue`) for larger or non-isbits manifolds.
"""
struct ManifoldContext{M, B, R}
    manifold::M
    basis::B
    retraction::R
end

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
