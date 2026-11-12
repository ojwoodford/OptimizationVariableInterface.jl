module OVILinearAlgebraExt

using OptimizationVariableInterface: BoundedVector
using LinearAlgebra

# Forward LinearAlgebra functions on a `BoundedVector` to its cached `val`
LinearAlgebra.norm(x::BoundedVector, p::Real=2) = norm(x.val, p)
LinearAlgebra.normalize(x::BoundedVector, p::Real=2) = normalize(x.val, p)
for f in (:dot, :cross)
    @eval begin
        LinearAlgebra.$f(a::BoundedVector, b::BoundedVector) = $f(a.val, b.val)
        LinearAlgebra.$f(a::BoundedVector, b::AbstractArray) = $f(a.val, b)
        LinearAlgebra.$f(a::AbstractArray, b::BoundedVector) = $f(a, b.val)
    end
end

end # module
