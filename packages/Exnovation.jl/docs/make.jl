# SPDX-License-Identifier: MPL-2.0
using Documenter
using Exnovation

makedocs(
    sitename = "Exnovation.jl",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        canonical = "https://hyperpolymath.github.io/Exnovation.jl",
    ),
    modules = [Exnovation],
    pages = ["Home" => "index.md", "API" => "api.md"],
)

deploydocs(
    repo = "github.com/hyperpolymath/Exnovation.jl.git",
    devbranch = "main",
)
