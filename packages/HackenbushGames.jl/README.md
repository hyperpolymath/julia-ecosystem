# HackenbushGames.jl

[![Project Topology](https://img.shields.io/badge/Project-Topology-9558B2)](TOPOLOGY.md)
[![Completion Status](https://img.shields.io/badge/Completion-100%25-green)](TOPOLOGY.md)
[![License](https://img.shields.io/badge/License-PMPL--1.0-blue.svg)](LICENSE)




HackenbushGames.jl is a Julia toolkit for experimenting with Hackenbush
positions and the combinatorial game theory ideas from Padraic Bartlett’s
“A Short Guide to Hackenbush” (VIGRE REU 2006). It provides a small API for
building graphs, cutting edges, evaluating basic positions, and exporting
visualizations.

This project emphasizes transparent rules over heavy automation. It includes
simple evaluators for stalks and small green graphs plus utilities for nimbers
and dyadic rationals.

## Installation

### From Julia REPL
```julia
using Pkg
Pkg.add("HackenbushGames")
```

### From Git (Development)
```julia
using Pkg
Pkg.add(url="https://github.com/hyperpolymath/HackenbushGames.jl")
```

## Features

- Build Red/Blue/Green Hackenbush graphs and generate legal moves.
- Compute dyadic values for **Red-Blue stalks** (linear chains).
- Compute nimbers for **Green stalks** and **small Green graphs**.
- Helpers for nim-sum and minimal excluded value (mex).
- Graph sum composition and GraphViz export.
- Canonical {L|R} notation and numeric evaluation for small games.
- ASCII visualization helper.

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

```julia
g = simple_stalk([Blue, Red, Blue])
println(game_value(g))       # dyadic value if numeric
println(to_ascii(g))
```

## Model Notes (Based on the Bartlett Guide)

- **Red-Blue Hackenbush** uses {L | R} game values and the Simplicity Rule
  to identify dyadic rationals for stalks.
- **Green Hackenbush** is impartial and can be evaluated via nimbers.
  This package uses a Grundy-number search for small graphs.
- **Graph sums** are supported for building disjoint unions.
- **Colon and Fusion** rules and the “flower/jungle” ideas are described in
  the guide; in this package they are documented as concepts and can be
  layered into higher-level analysis later.

## Limitations

The `green_grundy` evaluator is exponential in the number of edges and is
intended for small positions and teaching. Larger graphs should use specialized
algorithms or structural simplifications.

## Development

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. -e 'using Pkg; Pkg.test()'
```

## API Snapshot

```julia
EdgeColor, Edge, HackenbushGraph
prune_disconnected, cut_edge, moves, game_sum
simple_stalk, to_ascii, to_graphviz
GameForm, canonical_game, simplify_game, game_value
stalk_value, simplest_dyadic_between
mex, nim_sum, green_stalk_nimber, green_grundy
```
