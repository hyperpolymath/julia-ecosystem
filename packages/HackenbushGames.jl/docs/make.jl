# SPDX-License-Identifier: PMPL-1.0-or-later
using Documenter
using HackenbushGames

makedocs(
    sitename = "HackenbushGames.jl",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        canonical = "https://hyperpolymath.github.io/HackenbushGames.jl",
    ),
    modules = [HackenbushGames],
    pages = ["Home" => "index.md", "API" => "api.md"],
)

deploydocs(
    repo = "github.com/hyperpolymath/HackenbushGames.jl.git",
    devbranch = "main",
)
