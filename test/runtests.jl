using OptimizationVariableInterface, Static, StaticArrays, Test, ForwardDiff

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
    @test update(ZeroToInfScalar(0.0), [0.0]).val ≈ floatmin(Float64) rtol=1e-10
    @test ZeroToInfScalar(3).val ≈ 3.0                      # integer input
    @test update(s_pos, [1e4]).val == floatmax(Float64)      # B5: clamped, not Inf
    @test update(update(s_pos, [1e4]), [-1e4]).val ≈ 2.0     # ...and not stuck
    @test update(s_pos, [-1e4]).val == floatmin(Float64)
    @test isnan(update(s_pos, [NaN]).val)                    # B4: NaN propagates
    @test isnan(ZeroToInfScalar(NaN).val)

    # ZeroToOneScalar
    s_unit = ZeroToOneScalar(0.5)
    @test update(s_unit, [0.0]).val ≈ 0.5
    @test s_unit.val ≈ 0.5 && s_unit.comp ≈ 0.5
    s_inc = update(s_unit, [1.0])
    @test 0.5 < s_inc.val < 1.0
    s_dec = update(s_unit, [-1.0])
    @test 0.0 < s_dec.val < 0.5
    @test update(s_unit, [1000.0]).val == 1.0 && update(s_unit, [1000.0]).comp == 0.0
    @test update(update(s_unit, [1000.0]), [-1000.0]).val ≈ 0.5   # not stuck at 1
    @test update(s_unit, [30.0]).val < 1.0
    @test update(s_unit, [30.0]).comp ≈ exp(-30) rtol=1e-12
    @test update(ZeroToOneScalar(0.0), [0.0]).val > 0.0
    @test 0.0 < ZeroToOneScalar(1.0).val <= 1.0 && isfinite(ZeroToOneScalar(1.0).logit)
    @test ZeroToOneScalar(0.3).val ≈ 0.3 && ZeroToOneScalar(0.3).comp ≈ 0.7
    @test ZeroToOneScalar(1).val > 0.5                       # integer input
    # B3: from the top end, steps give the exact logistic result, never leaving (0, 1)
    top = ZeroToOneScalar(prevfloat(1.0))
    for (step, expected) in ((-1.0, 0.9999999999999997), (-36.0, 0.676), (-37.0, 0.435), (-50.0, 1.737e-6))
        r = update(top, [step])
        @test 0 < r.val < 1
        @test r.val ≈ expected rtol=5e-3
    end
    # Consistency: val + comp == 1 and val = logistic(logit) over a range of logits
    for z in (-800.0, -50.0, -1.0, 0.0, 1.0, 50.0, 800.0)
        v = ZeroToOneScalar(z, Val(:logit))
        @test v.val + v.comp ≈ 1.0
        @test !isnan(v.val) && !isnan(v.comp)
    end
    @test isnan(update(s_unit, [NaN]).val)                   # B4
    @test isbitstype(typeof(s_unit)) && isbitstype(typeof(s_pos))

    # BoundedReal
    br = BoundedReal(2.0, 1.0, 5.0)
    @test nvars(br) === static(1) && isbitstype(typeof(br))
    @test br.val ≈ 2.0 && br.lo == 1.0 && br.hi == 5.0
    for v in (-3.0, 1.5, 1.0 + 1e-9, 4.999, 0.0)
        @test BoundedReal(v, -3.0, 5.0).val ≈ v rtol=1e-8 atol=1e-12
    end
    for step in (-1e3, -50.0, -1.0, 0.0, 1.0, 50.0, 1e3)
        r = update(br, [step])
        @test 1.0 <= r.val <= 5.0 && r.lo == 1.0 && r.hi == 5.0
    end
    @test update(br, [1e3]).val > 4.999 && update(br, [-1e3]).val < 1.001
    @test update(update(br, [1e3]), [-1e3]).val ≈ 2.0        # not stuck
    @test update(br, [0.7], 1).val ≈ update(br, [9.0, 0.7], 2).val
    # accuracy near hi when |hi| >> hi - lo
    bt = BoundedReal(1e6 + 0.5, 1e6, 1e6 + 1)
    @test bt.val ≈ 1e6 + 0.5
    bt2 = update(bt, [30.0])
    @test disthi(bt2) ≈ exp(-30) rtol=1e-9
    @test distlo(bt2) + disthi(bt2) ≈ 1.0
    @test bt2.val <= 1e6 + 1
    # clamping of out-of-range / endpoint values
    @test 1.0 < BoundedReal(0.0, 1.0, 5.0).val < 5.0 && isfinite(BoundedReal(0.0, 1.0, 5.0).logit)
    @test 1.0 < BoundedReal(5.0, 1.0, 5.0).val < 5.0 && isfinite(BoundedReal(5.0, 1.0, 5.0).logit)
    @test_throws ArgumentError BoundedReal(0.5, 1.0, 1.0)
    @test_throws ArgumentError BoundedReal(0.5, 2.0, 1.0)
    @test_throws ArgumentError BoundedReal(0.5, 0.0, Inf)
    @test BoundedReal(1, 0, 2).val ≈ 1.0                      # integer input
    @test isnan(update(br, [NaN]).val)
    # ForwardDiff: dual T with Float64 B constructs, infers concretely, and has the right derivative
    f(d) = update(br, [d]).val
    p_ = (br.val - br.lo) / (br.hi - br.lo)
    @test ForwardDiff.derivative(f, 0.0) ≈ (br.hi - br.lo) * p_ * (1 - p_)
    D = ForwardDiff.Dual{Nothing, Float64, 1}
    @test (@inferred update(br, [D(0.0, ForwardDiff.Partials((1.0,)))])) isa BoundedReal{D, Float64}
    # ZeroToOneScalar/ZeroToInfScalar autodiff
    @test ForwardDiff.derivative(d -> update(s_unit, [d]).val, 0.0) ≈ 0.25
    @test ForwardDiff.derivative(d -> update(s_unit, [d]).comp, 0.0) ≈ -0.25
    @test ForwardDiff.derivative(d -> update(s_pos, [d]).val, 0.0) ≈ 2.0
    # Heterogeneous static vector with BoundedReal
    sv_b = SVector(br, ZeroToInfScalar(1.0))
    @test nvars(sv_b) === static(2)
    sv_b2 = update(sv_b, SVector(0.0, log(2.0)))
    @test sv_b2[1].val ≈ 2.0 && sv_b2[2].val ≈ 2.0

    # M5: Tuple / NamedTuple of variables
    tup = (1.0, SVector(1.0, 2.0), ZeroToInfScalar(2.0), br)
    @test nvars(tup) === static(1 + 2 + 1 + 1)
    @test nvars(()) === static(0)
    tup_new = update(tup, [0.5, 0.1, 0.2, log(2.0), 0.0])
    @test tup_new isa Tuple && length(tup_new) == 4
    @test tup_new[1] == 1.5 && tup_new[2] === SVector(1.1, 2.2)
    @test tup_new[3].val ≈ 4.0 && tup_new[4].val ≈ 2.0
    @test update(tup, [9.0, 0.5, 0.1, 0.2, log(2.0), 0.0], 2)[1] == 1.5
    @test update((), Float64[]) === ()
    nt = (a = 1.0, b = SVector(1.0, 2.0))
    @test nvars(nt) === static(3)
    nt_new = update(nt, [0.5, 0.1, 0.2])
    @test nt_new isa NamedTuple{(:a, :b)} && nt_new.a == 1.5 && nt_new.b === SVector(1.1, 2.2)
    @test (@allocated update(tup, [0.5, 0.1, 0.2, 0.0, 0.0])) < 200   # no Vector{Union}; small
    @test update!(copyvars(tup), tup, [0.5, 0.1, 0.2, log(2.0), 0.0])[1] == 1.5
    tup_mut = (zeros(2), 1.0)
    tup_mut_copy = copyvars(tup_mut)
    @test tup_mut_copy[1] == tup_mut[1] && tup_mut_copy[1] !== tup_mut[1]
    tup_sync = syncvars!(tup_mut_copy, ([5.0, 6.0], 2.0))
    @test tup_sync == ([5.0, 6.0], 2.0) && tup_sync[1] === tup_mut_copy[1]

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

    # Mutable non-Vector arrays are copied; immutable ones are not
    mat = [1.0 2.0; 3.0 4.0]
    matcopy = copyvars(mat)
    @test matcopy == mat && matcopy !== mat
    mv = MVector(1.0, 2.0)
    @test copyvars(mv) == mv && copyvars(mv) !== mv
    @test syncvars!(zeros(2, 2), mat) isa Matrix
    mdest = zeros(2, 2)
    @test syncvars!(mdest, mat) === mdest && mdest == mat
    @test syncvars!(MVector(0.0, 0.0), mv) isa MVector

    # M1: per-element policy for Vectors of non-isbits variables
    inner = [[1.0, 2.0], [3.0]]
    inner_copy = copyvars(inner)
    @test inner_copy == inner && inner_copy[1] !== inner[1]
    inner_dest = copyvars(inner)
    inner_src = [[7.0, 8.0], [9.0]]
    syncvars!(inner_dest, inner_src)
    @test inner_dest == inner_src
    @test inner_dest[1] !== inner_src[1]     # elements are synced, not aliased
    mixed_dest = Any[[0.0, 0.0], SVector(0.0, 0.0)]
    syncvars!(mixed_dest, Any[[1.0, 2.0], SVector(3.0, 4.0)])
    @test mixed_dest == Any[[1.0, 2.0], SVector(3.0, 4.0)]

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
