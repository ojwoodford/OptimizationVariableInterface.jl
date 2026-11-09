using OptimizationVariableInterface, Static, StaticArrays, Test
using ManifoldsBase

@testset "OVIManifoldsBaseExt" begin
    # 1. Zero-byte singleton manifold (parameter = :type)
    M_type = ManifoldsBase.DefaultManifold(3; parameter = :type)
    @test isbitstype(typeof(M_type))
    @test sizeof(M_type) == 0

    p_static = SVector(1.0, 2.0, 3.0)
    v_type = ManifoldVar(p_static, M_type)
    @test v_type.context isa ManifoldContext
    @test isbitsvartype(typeof(v_type))
    # Context should be zero-overhead for singleton manifold and default basis/retraction
    @test sizeof(v_type) == sizeof(p_static)
    @test nvars(v_type) === static(3)

    # Retraction with StaticInt dimension
    step = [0.1, 0.2, -0.3]
    v_type_new = update(v_type, step)
    @test v_type_new.point ≈ SVector(1.1, 2.2, 2.7)

    # Retraction with different element type (e.g. BigFloat hitting line 53)
    v_type_big = update(v_type, BigFloat[0.1, 0.2, -0.3])
    @test eltype(v_type_big.point) === BigFloat
    @test v_type_big.point ≈ SVector(1.1, 2.2, 2.7)

    # Convert ManifoldVar
    v_type_conv = convert(ManifoldVar{Vector{Float64}, typeof(v_type.context)}, v_type)
    @test v_type_conv.point isa Vector{Float64}
    @test v_type_conv.point ≈ [1.0, 2.0, 3.0]

    # Offset update with StaticInt dimension
    v_type_offset = update(v_type, [10.0, 0.1, 0.2, -0.3], 2)
    @test v_type_offset.point ≈ SVector(1.1, 2.2, 2.7)

    # 2. Field-parameterized manifold (parameter = :field)
    M_field = ManifoldsBase.DefaultManifold(3; parameter = :field)
    v_field = ManifoldVar(p_static, M_field)
    @test nvars(v_field) == 3 && !(nvars(v_field) isa Static.StaticInt)
    v_field_new = update(v_field, step)
    @test v_field_new.point ≈ SVector(1.1, 2.2, 2.7)

    # 3. Automatic byref selection
    # Small isbits context <= 16 bytes defaults to byref = false
    v_auto_small = ManifoldVar(p_static, M_type)
    @test v_auto_small.context isa ManifoldContext

    # Large context > 16 bytes defaults to byref = true
    M_large = ManifoldsBase.DefaultManifold(2, 3, 4; parameter = :field)
    v_large = ManifoldVar(zeros(2, 3, 4), M_large)
    @test v_large.context isa Base.RefValue

    # Explicit byref = true
    v_ref = ManifoldVar(p_static, M_type; byref = true)
    @test v_ref.context isa Base.RefValue
    @test sizeof(v_ref) == sizeof(p_static) + sizeof(Ptr{Cvoid})

    # Explicit byref = false
    v_val = ManifoldVar(p_static, M_type; byref = false)
    @test v_val.context isa ManifoldContext

    # 4. Context sharing across multiple variables
    ctx_ref = ManifoldContext(M_type; byref = true)
    vars_shared = [ManifoldVar(SVector(Float64(i), 0.0, 0.0), ctx_ref) for i in 1:10]
    for i in 1:10
        @test vars_shared[i].context === ctx_ref
    end

    # 5. update preserves context reference
    step_shared = fill(0.5, 30)
    to_shared = copyvars(vars_shared)
    update!(to_shared, vars_shared, step_shared)
    for i in 1:10
        @test to_shared[i].point ≈ SVector(Float64(i) + 0.5, 0.5, 0.5)
        @test to_shared[i].context === ctx_ref
    end

    # 6. copyvars behavior
    # For byref with isbits point: fast array copy, same RefValue pointer
    copied_shared = copyvars(vars_shared)
    @test copied_shared == vars_shared && copied_shared !== vars_shared
    @test copied_shared[1].context === ctx_ref

    # For by-value with isbits point: copyvars hits isbits branch (line 98)
    vars_byval = [ManifoldVar(SVector(Float64(i), 0.0, 0.0), M_type; byref = false) for i in 1:5]
    copied_byval = copyvars(vars_byval)
    @test copied_byval == vars_byval && copied_byval !== vars_byval

    # For mutable points (e.g. Vector{Float64}): independent point copy, shared context
    v_mut1 = ManifoldVar([1.0, 2.0, 3.0], ctx_ref)
    v_mut2 = copyvars(v_mut1)
    @test v_mut2.point == v_mut1.point && v_mut2.point !== v_mut1.point
    @test v_mut2.context === v_mut1.context

    # Vector of mutable points copyvars
    mut_vars = [ManifoldVar([Float64(i), 0.0, 0.0], ctx_ref) for i in 1:5]
    mut_copied = copyvars(mut_vars)
    @test mut_copied !== mut_vars
    for i in 1:5
        @test mut_copied[i].point == mut_vars[i].point && mut_copied[i].point !== mut_vars[i].point
        @test mut_copied[i].context === ctx_ref
    end

    # 7. update! tests
    # Single update! with immutable isbits point
    single_dest_isbits = copyvars(v_val)
    @test update!(single_dest_isbits, v_val, [0.1, 0.2, 0.3]).point ≈ SVector(1.1, 2.2, 3.3)

    # Single update! with mutable point (in-place)
    single_mut_from = ManifoldVar([1.0, 2.0, 3.0], ctx_ref)
    single_mut_to = copyvars(single_mut_from)
    @test update!(single_mut_to, single_mut_from, [0.1, 0.2, 0.3]) === single_mut_to
    @test single_mut_to.point ≈ [1.1, 2.2, 3.3]

    # Vector in-place update! on mutable points
    mut_dest = copyvars(mut_vars)
    step_mut = fill(1.0, 15)
    update!(mut_dest, mut_vars, step_mut)
    for i in 1:5
        @test mut_dest[i].point ≈ [Float64(i) + 1.0, 1.0, 1.0]
    end

    # 8. syncvars! tests
    # Vector syncvars! on isbits variables (line 112)
    sync_isbits_dest = copyvars(vars_byval)
    sync_isbits_src = [ManifoldVar(SVector(10.0, 20.0, 30.0), M_type; byref = false) for _ in 1:5]
    @test syncvars!(sync_isbits_dest, sync_isbits_src) === sync_isbits_dest
    @test sync_isbits_dest == sync_isbits_src

    # Single syncvars! on mutable variable
    single_sync_dest = copyvars(single_mut_from)
    @test syncvars!(single_sync_dest, single_mut_to) === single_sync_dest
    @test single_sync_dest.point ≈ [1.1, 2.2, 3.3]

    # Vector syncvars! on mutable points
    sync_dest = copyvars(mut_vars)
    syncvars!(sync_dest, mut_dest)
    for i in 1:5
        @test sync_dest[i].point ≈ mut_dest[i].point
    end

    # 9. Heterogeneous problem state combining Euclidean, bounded scalar, and ManifoldVar
    mixed_vars = Any[
        1.5,
        ZeroToInfScalar(2.0),
        SVector(1.0, 2.0),
        ManifoldVar(SVector(0.0, 1.0, 0.0), M_type)
    ]
    @test nvars(mixed_vars) == 1 + 1 + 2 + 3 # 7
    mixed_step = [0.5, log(2.0), 0.1, -0.1, 0.2, 0.2, 0.2]
    mixed_to = copyvars(mixed_vars)
    update!(mixed_to, mixed_vars, mixed_step)
    @test mixed_to[1] == 2.0
    @test mixed_to[2].val ≈ 4.0
    @test mixed_to[3] === SVector(1.1, 1.9)
    @test mixed_to[4].point ≈ SVector(0.2, 1.2, 0.2)
end
