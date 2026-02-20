# SPDX-License-Identifier: PMPL-1.0-or-later
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
