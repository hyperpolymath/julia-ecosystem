# SPDX-License-Identifier: PMPL-1.0-or-later
using Documenter
using Causals

makedocs(
    sitename = "Causals.jl",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        canonical = "https://hyperpolymath.github.io/Causals.jl",
        assets = String[],
    ),
    modules = [Causals],
    pages = [
        "Home" => "index.md",
        "Modules" => [
            "Dempster-Shafer" => "dempster_shafer.md",
            "Bradford Hill" => "bradford_hill.md",
            "Causal DAGs" => "causal_dag.md",
            "Granger Causality" => "granger.md",
            "Propensity Scores" => "propensity.md",
            "Do-Calculus" => "do_calculus.md",
            "Counterfactuals" => "counterfactuals.md",
        ],
        "Examples" => "examples.md",
        "API Reference" => "api.md",
    ],
    checkdocs = :exports,
    strict = false,
)

deploydocs(
    repo = "github.com/hyperpolymath/Causals.jl.git",
    devbranch = "main",
    push_preview = true,
)
