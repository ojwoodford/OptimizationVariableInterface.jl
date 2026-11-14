using OptimizationVariableInterface, Test

# The public API is exactly what is exported, and every exported name is documented. Adding or
# dropping an export is a change to this list.
const PUBLIC_API = sort([
    :BoundedReal,
    :BoundedVector,
    :ManifoldContext,
    :ManifoldVar,
    :OptimizationVariableInterface,
    :ZeroToInfScalar,
    :ZeroToOneScalar,
    :copyvars,
    :disthi,
    :distlo,
    :isbitsvartype,
    :iseliminable,
    :ismutablevar,
    :manifoldcontext,
    :nvars,
    :syncvars!,
    :update,
    :update!,
])

@testset "Public API" begin
    @test issorted(PUBLIC_API)
    @test sort(names(OptimizationVariableInterface)) == PUBLIC_API
    # (Base.Docs.hasdoc is not in Julia 1.10, so look the binding up in the module's docstrings)
    docs = Base.Docs.meta(OptimizationVariableInterface)
    @testset "$name has a docstring" for name in PUBLIC_API
        @test haskey(docs, Base.Docs.Binding(OptimizationVariableInterface, name))
    end
end
