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

    # Custom AbstractVector where elements are not sub-variables
    struct CustomVectorParam <: AbstractVector{String}
        data::Vector{String}
    end
    Base.size(cv::CustomVectorParam) = size(cv.data)
    Base.getindex(cv::CustomVectorParam, i::Int) = cv.data[i]
    @test nvars(CustomVectorParam(["a", "b", "c"])) == 3
    @test nvars(view([1.0, 2.0, 3.0, 4.0], 1:3)) == 3

    # 2. update (out-of-place)
    @test update(3.0, [0.5, 1.5]) == 3.5
    @test update(3.0, [0.5, 1.5], 2) == 4.5
    v_static = SVector(1.0, 2.0)
    @test update(v_static, SVector(0.5, -0.5)) === SVector(1.5, 1.5)
    @test update(v_static, [10.0, 0.5, -0.5], 2) === SVector(1.5, 1.5)
    v_dyn = [1.0, 2.0, 3.0]
    @test update(v_dyn, [0.0, 10.0, 20.0, 30.0], 2) == [11.0, 22.0, 33.0]

    # Custom types implementing only Base.length and Base.:+
    struct CustomAddVar
        data::Vector{Float64}
    end
    Base.length(v::CustomAddVar) = length(v.data)
    Base.:+(v::CustomAddVar, s::AbstractVector) = CustomAddVar(v.data + s)

    struct CustomStaticAddVar
        data::SVector{2, Float64}
    end
    OptimizationVariableInterface.nvars(::CustomStaticAddVar) = static(2)
    Base.:+(v::CustomStaticAddVar, s::AbstractVector) = CustomStaticAddVar(v.data + s)

    c_dyn = CustomAddVar([1.0, 2.0])
    @test nvars(c_dyn) == 2
    @test update(c_dyn, [0.0, 10.0, 20.0], 2).data == [11.0, 22.0]

    c_stat = CustomStaticAddVar(SVector(1.0, 2.0))
    @test nvars(c_stat) === static(2)
    @test update(c_stat, [0.0, 0.5, -0.5], 2).data === SVector(1.5, 1.5)

    # StaticVector of OVI variables
    allocs_update(v, s) = @allocated update(v, s)
    svec_ovi = SVector(ZeroToInfScalar(1.0), ZeroToInfScalar(2.0))
    @test nvars(svec_ovi) === static(2)
    svec_ovi_upd = update(svec_ovi, SVector(log(2.0), log(3.0)))
    @test svec_ovi_upd[1].val ≈ 2.0 && svec_ovi_upd[2].val ≈ 6.0
    svec_ovi_upd2 = update(svec_ovi, [99.0, log(2.0), log(3.0)], 2)
    @test svec_ovi_upd2[1].val ≈ 2.0 && svec_ovi_upd2[2].val ≈ 6.0
    @test allocs_update(svec_ovi, SVector(0.1, 0.2)) == 0

    svec_stat = SVector(c_stat, c_stat)
    @test nvars(svec_stat) === static(4)
    svec_stat_upd = update(svec_stat, SVector(0.5, -0.5, 1.0, 2.0))
    @test svec_stat_upd[1].data === SVector(1.5, 1.5)
    @test svec_stat_upd[2].data === SVector(2.0, 4.0)
    @test allocs_update(svec_stat, SVector(0.5, -0.5, 1.0, 2.0)) == 0

    # Heterogeneous StaticVector
    svec_hetero = SVector(ZeroToInfScalar(1.0), ZeroToOneScalar(0.5))
    @test nvars(svec_hetero) === static(2)
    svec_hetero_upd = update(svec_hetero, SVector(log(3.0), 0.0))
    @test svec_hetero_upd[1].val ≈ 3.0 && svec_hetero_upd[2].val ≈ 0.5

    # 0-length StaticVector
    svec_0 = SVector{0, ZeroToInfScalar{Float64}}()
    @test update(svec_0, Float64[]) === svec_0

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

    # 7. iseliminable trait
    @test iseliminable(Float64) === static(false)
    @test iseliminable(1.0) === static(false)
    @test iseliminable(SVector(1.0, 2.0, 3.0)) === static(false)
    struct CustomPoint3D; x::SVector{3, Float64}; end
    OptimizationVariableInterface.iseliminable(::Type{<:CustomPoint3D}) = static(true)
    @test iseliminable(CustomPoint3D) === static(true)
    @test iseliminable(CustomPoint3D(SVector(1.0, 2.0, 3.0))) === static(true)
end

if Base.find_package("ManifoldsBase") !== nothing
    include("test_manifolds_base.jl")
end
