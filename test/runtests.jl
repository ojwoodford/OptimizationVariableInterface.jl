using OptimizationVariableInterface, Static, StaticArrays, Test

@testset "OptimizationVariableInterface.jl" begin
    # 1. nvars
    @test nvars(2.5) === static(1)
    @test nvars(42) === static(1)
    @test nvars(SVector(1.0, 2.0, 3.0)) === static(3)
    @test nvars(SVector{2, Float64}(1.0, 2.0)) === static(2)
    @test nvars([1.0, 2.0, 3.0, 4.0]) == 4
    @test nvars(ZeroToInfScalar(1.5)) === static(1)
    @test nvars(ZeroToOneScalar(0.5)) === static(1)
    @test nvars([SVector(1.0, 2.0), SVector(3.0, 4.0)]) == 4
    @test nvars(Any[1.0, SVector(1.0, 2.0, 3.0), [5.0, 6.0]]) == 6
    @test nvars(Float64[]) == 0
    @test nvars(Any[]) == 0

    # 2. update (out-of-place)
    @test update(3.0, [0.5, 1.5]) == 3.5
    @test update(3.0, [0.5, 1.5], 2) == 4.5
    v_static = SVector(1.0, 2.0)
    @test update(v_static, SVector(0.5, -0.5)) === SVector(1.5, 1.5)
    @test update(v_static, [10.0, 0.5, -0.5], 2) === SVector(1.5, 1.5)
    v_dyn = [1.0, 2.0, 3.0]
    @test update(v_dyn, [0.0, 10.0, 20.0, 30.0], 2) == [11.0, 22.0, 33.0]

    # ZeroToInfScalar
    s_pos = ZeroToInfScalar(2.0)
    @test update(s_pos, [0.0]).val ≈ 2.0
    @test update(s_pos, [log(3.0)]).val ≈ 6.0
    @test update(ZeroToInfScalar(0.0), [0.0]).val == floatmin(Float64)

    # ZeroToOneScalar
    s_unit = ZeroToOneScalar(0.5)
    @test update(s_unit, [0.0]).val ≈ 0.5
    s_inc = update(s_unit, [1.0])
    @test 0.5 < s_inc.val < 1.0
    s_dec = update(s_unit, [-1.0])
    @test 0.0 < s_dec.val < 0.5
    @test update(s_unit, [1000.0]).val == 1.0
    @test update(ZeroToOneScalar(0.0), [0.0]).val > 0.0

    # 3. update! (in-place / unified)
    # Immutable fallback
    @test update!(0.0, 2.0, [0.5]) == 2.5
    @test update!(SVector(0.0, 0.0), SVector(1.0, 2.0), [0.5, -0.5]) === SVector(1.5, 1.5)

    # Flat mutable Vector{<:Number}
    to_flat = zeros(3)
    from_flat = [1.0, 2.0, 3.0]
    step_flat = [0.1, 0.2, 0.3]
    @test update!(to_flat, from_flat, step_flat) === to_flat
    @test to_flat ≈ [1.1, 2.2, 3.3]

    # Vector of heterogeneous variable blocks
    from_blocks = Union{SVector{2, Float64}, ZeroToInfScalar{Float64}}[SVector(1.0, 2.0), ZeroToInfScalar(2.0)]
    to_blocks = copyvars(from_blocks)
    step_blocks = [0.5, -0.5, log(2.0)]
    @test update!(to_blocks, from_blocks, step_blocks) === to_blocks
    @test to_blocks[1] === SVector(1.5, 1.5)
    @test to_blocks[2].val ≈ 4.0

    # 4. isbitsvartype, copyvars, syncvars!
    @test isbitsvartype(SVector{3, Float64})
    @test isbitsvartype(Union{SVector{3, Float64}, SVector{2, Float64}, Float64})
    @test !isbitsvartype(Vector{Float64})
    @test !isbitsvartype(Union{SVector{3, Float64}, Vector{Float64}})
    @test !isbitsvartype(Any)
    @test !isbitsvartype(SVector{3})
    @test !isbitsvartype(Complex)

    # Immutable copyvars / syncvars!
    @test copyvars(SVector(1.0, 2.0)) === SVector(1.0, 2.0)
    @test syncvars!(SVector(0.0, 0.0), SVector(1.0, 2.0)) === SVector(1.0, 2.0)

    # UnionAll Vector copyvars
    unionallvars = SVector{3}[SVector(1.0, 2.0, 3.0), SVector(1, 2, 3)]
    unionallcopy = copyvars(unionallvars)
    @test unionallcopy == unionallvars && unionallcopy !== unionallvars

    # Isbits Union Vector copyvars allocation check
    vars = Union{SVector{3, Float64}, SVector{2, Float64}}[i % 2 == 0 ? SVector(1.0, 2.0, 3.0) : SVector(1.0, 2.0) for i in 1:200]
    varscopy = copyvars(vars)
    @test varscopy == vars && varscopy !== vars
    copyvars(vars)
    @test (@allocations copyvars(vars)) <= 4

    # Dynamic Vector{Vector{Float64}} deep copy
    dynvars = [[1.0, 2.0], [3.0]]
    dyncopy = copyvars(dynvars)
    @test dyncopy == dynvars && dyncopy !== dynvars && dyncopy[1] !== dynvars[1]

    # syncvars! on Vector
    dest_vec = [0.0, 0.0]
    src_vec = [4.0, 5.0]
    @test syncvars!(dest_vec, src_vec) === dest_vec
    @test dest_vec == [4.0, 5.0]
end

if Base.find_package("ManifoldsBase") !== nothing
    include("test_manifolds_base.jl")
end
