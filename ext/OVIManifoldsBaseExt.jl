module OVIManifoldsBaseExt

using OptimizationVariableInterface
using ManifoldsBase
using Static

import OptimizationVariableInterface: ManifoldVar, ManifoldContext, get_context, SR

# 1. Constructors
function OptimizationVariableInterface.ManifoldContext(
    manifold::ManifoldsBase.AbstractManifold;
    basis = ManifoldsBase.DefaultOrthonormalBasis(),
    retraction = ManifoldsBase.default_retraction_method(manifold),
    byref::Union{Bool, Nothing} = nothing
)
    ctx = ManifoldContext(manifold, basis, retraction)
    use_ref = isnothing(byref) ? (!isbitstype(typeof(ctx)) || sizeof(ctx) > 16) : byref
    return use_ref ? Ref(ctx) : ctx
end

function OptimizationVariableInterface.ManifoldVar(
    point::P,
    manifold::ManifoldsBase.AbstractManifold;
    basis = ManifoldsBase.DefaultOrthonormalBasis(),
    retraction = ManifoldsBase.default_retraction_method(manifold, typeof(point)),
    byref::Union{Bool, Nothing} = nothing
) where {P}
    ctx = ManifoldContext(manifold; basis=basis, retraction=retraction, byref=byref)
    return ManifoldVar{P, typeof(ctx)}(point, ctx)
end

# 2. nvars
OptimizationVariableInterface.nvars(M::ManifoldsBase.AbstractManifold) = ManifoldsBase.manifold_dimension(M)
OptimizationVariableInterface.nvars(::ManifoldsBase.DefaultManifold{𝔽, ManifoldsBase.TypeParameter{Tuple{N}}}) where {𝔽, N} = static(N)
OptimizationVariableInterface.nvars(v::ManifoldVar) = OptimizationVariableInterface.nvars(get_context(v).manifold)

# 3. update (out-of-place retraction)
@inline function OptimizationVariableInterface.update(v::ManifoldVar{P}, step, start=1) where {P}
    ctx = get_context(v)
    M = ctx.manifold
    d = OptimizationVariableInterface.nvars(v)
    dim = dynamic(d)
    coords = if d isa Static.StaticInt
        @inbounds view(step, SR(0, dim - 1) .+ start)
    else
        @inbounds view(step, start:start+dim-1)
    end
    X = ManifoldsBase.get_vector(M, v.point, coords, ctx.basis)
    p_ret = ManifoldsBase.retract(M, v.point, X, ctx.retraction)
    p_new = if eltype(p_ret) === eltype(P)
        p_ret isa P ? p_ret : convert(P, p_ret)
    else
        p_ret
    end
    return ManifoldVar(p_new, v.context)
end

# 4. update! (unified retraction)
@inline function OptimizationVariableInterface.update!(to::ManifoldVar{P}, from::ManifoldVar{P}, step::AbstractVector) where {P}
    if isbitstype(P)
        return OptimizationVariableInterface.update(from, step)
    else
        ctx = get_context(from)
        M = ctx.manifold
        d = dynamic(OptimizationVariableInterface.nvars(from))
        coords = @inbounds view(step, 1:d)
        X = ManifoldsBase.get_vector(M, from.point, coords, ctx.basis)
        ManifoldsBase.retract!(M, to.point, from.point, X, ctx.retraction)
        return to
    end
end

@inline function OptimizationVariableInterface.update!(to::Vector{<:ManifoldVar{P}}, from::Vector{<:ManifoldVar{P}}, step::AbstractVector) where {P}
    offset = 1
    if isbitstype(P)
        @inbounds for i in eachindex(to, from)
            v = from[i]
            to[i] = OptimizationVariableInterface.update(v, step, offset)
            offset += dynamic(OptimizationVariableInterface.nvars(v))
        end
    else
        @inbounds for i in eachindex(to, from)
            v_from = from[i]
            ctx = get_context(v_from)
            M = ctx.manifold
            d = dynamic(OptimizationVariableInterface.nvars(v_from))
            coords = @inbounds view(step, offset:offset+d-1)
            X = ManifoldsBase.get_vector(M, v_from.point, coords, ctx.basis)
            ManifoldsBase.retract!(M, to[i].point, v_from.point, X, ctx.retraction)
            offset += d
        end
    end
    return to
end

# 5. copyvars
@inline OptimizationVariableInterface.copyvars(v::ManifoldVar) =
    ManifoldVar(OptimizationVariableInterface.copyvars(v.point), v.context)

function OptimizationVariableInterface.copyvars(vars::Vector{<:ManifoldVar{P, C}}) where {P, C}
    if isbitstype(P) && C <: Base.RefValue
        return copy(vars)
    elseif OptimizationVariableInterface.isbitsvartype(ManifoldVar{P, C})
        return copy(vars)
    else
        return [OptimizationVariableInterface.copyvars(v) for v in vars]
    end
end

# 6. syncvars!
@inline function OptimizationVariableInterface.syncvars!(to::ManifoldVar, from::ManifoldVar)
    ismutable(to.point) || return from
    OptimizationVariableInterface.syncvars!(to.point, from.point)
    return to
end

function OptimizationVariableInterface.syncvars!(to::Vector{<:ManifoldVar}, from::Vector{<:ManifoldVar})
    if OptimizationVariableInterface.isbitsvartype(eltype(to))
        copyto!(to, from)
    else
        @inbounds for i in eachindex(to, from)
            to[i] = OptimizationVariableInterface.syncvars!(to[i], from[i])
        end
    end
    return to
end

end # module
