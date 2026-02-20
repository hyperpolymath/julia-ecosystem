# SPDX-License-Identifier: PMPL-1.0-or-later
using Documenter
using SMTLib

makedocs(
    sitename = "SMTLib.jl",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        canonical = "https://hyperpolymath.github.io/SMTLib.jl",
        assets = String[],
    ),
    modules = [SMTLib],
    pages = [
        "Home" => "index.md",
        "API Reference" => "api.md",
        "Examples" => "examples.md",
        "Solvers" => "solvers.md",
    ],
    checkdocs = :exports,
    strict = false,
)

deploydocs(
    repo = "github.com/hyperpolymath/SMTLib.jl.git",
    devbranch = "main",
    push_preview = true,
)
