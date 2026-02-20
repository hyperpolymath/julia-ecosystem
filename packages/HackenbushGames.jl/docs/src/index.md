# HackenbushGames.jl

Documentation for HackenbushGames.jl

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/hyperpolymath/HackenbushGames.jl")
```

## Quick Start

```julia
using HackenbushGames

# Red-Blue stalk value (ground -> top)
colors = [Blue, Red, Blue]
value = stalk_value(colors)
println(value) # dyadic rational

# Green impartial position via graph + Grundy number
edges = [
    Edge(0, 1, Green),
    Edge(1, 2, Green),
    Edge(1, 3, Green),
]
position = HackenbushGraph(edges, [0])
println(green_grundy(position))
```

## API Reference

See [API](api.md) for complete reference.
