using Documenter
using OptimizationVariableInterface
# Loading these triggers both package extensions (OVILinearAlgebraExt, OVIManifoldsBaseExt) for the examples
using LinearAlgebra, ManifoldsBase

DocMeta.setdocmeta!(OptimizationVariableInterface, :DocTestSetup, :(using OptimizationVariableInterface, Static, StaticArrays); recursive=true)

makedocs(
    sitename = "OptimizationVariableInterface.jl",
    modules = [OptimizationVariableInterface],
    checkdocs = :exports,   # every exported name must be in the API reference
    warnonly = false,
    doctest = true,
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
    ),
    pages = [
        "Home" => "index.md",
        "Guide" => [
            "Defining a variable type" => "guide/custom_variables.md",
            "Collections of variables" => "guide/collections.md",
            "In-place variable types" => "guide/mutable_variables.md",
            "Built-in variable types" => "guide/builtin_types.md",
            "ManifoldsBase integration" => "guide/manifolds.md",
            "Using the interface in an optimizer" => "guide/writing_an_optimizer.md",
        ],
        "API reference" => "api.md",
    ],
)

deploydocs(
    repo = "github.com/ojwoodford/OptimizationVariableInterface.jl.git",
    devbranch = "main",
    push_preview = true,
)
