# SPDX-License-Identifier: PMPL-1.0-or-later

using Documenter
using Cliodynamics

makedocs(
    sitename = "Cliodynamics.jl",
    modules = [Cliodynamics],
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        canonical = "https://hyperpolymath.github.io/Cliodynamics.jl",
        assets = String[],
    ),
    pages = [
        "Home" => "index.md",
        "Tutorial" => "tutorial.md",
        "Models" => [
            "Population Dynamics" => "models/population.md",
            "Elite Dynamics" => "models/elites.md",
            "Political Instability" => "models/instability.md",
            "Secular Cycles" => "models/cycles.md",
            "State Formation" => "models/state.md",
            "Spatial Models" => "models/spatial.md",
        ],
        "Data Integration" => "data.md",
        "Model Fitting" => "fitting.md",
        "Plotting" => "plotting.md",
        "API Reference" => "api.md",
    ],
    checkdocs = :exports,
    warnonly = true,
)

deploydocs(
    repo = "github.com/hyperpolymath/Cliodynamics.jl.git",
    devbranch = "main",
)
