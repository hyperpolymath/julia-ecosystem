# SPDX-License-Identifier: PMPL-1.0-or-later
# Basic usage examples for HackenbushGames.jl

using HackenbushGames

println("=== HackenbushGames.jl Examples ===\n")

# Example 1: Red-Blue stalk value
println("Example 1: Red-Blue Stalk Value")
println("-" ^ 40)
colors = [Blue, Red, Blue]
value = stalk_value(colors)
println("Colors: ", colors)
println("Stalk value: ", value)
println()

# Example 2: Green impartial position
println("Example 2: Green Grundy Number")
println("-" ^ 40)
edges = [
    Edge(0, 1, Green),
    Edge(1, 2, Green),
    Edge(1, 3, Green),
]
position = HackenbushGraph(edges, [0])
grundy = green_grundy(position)
println("Edges: ", edges)
println("Grundy number: ", grundy)
println()

# Example 3: Simple graph operations
println("Example 3: Graph Operations")
println("-" ^ 40)
# Create a simple position
g = HackenbushGraph([
    Edge(0, 1, Blue),
    Edge(1, 2, Red),
    Edge(0, 3, Green)
], [0])
println("Original graph has ", length(g.edges), " edges")

# Cut an edge
g2 = cut_edge(g, 1)
println("After cutting edge 1: ", length(g2.edges), " edges")

# Prune disconnected
g3 = prune_disconnected(g2)
println("After pruning: ", length(g3.edges), " edges")
println()

# Example 4: Nimber operations
println("Example 4: Nimber Operations")
println("-" ^ 40)
nimbers = [3, 5, 2]
sum_result = nim_sum(nimbers)
println("nim_sum($nimbers) = ", sum_result)

values = [0, 2, 1, 3]
mex_result = mex(values)
println("mex($values) = ", mex_result)
println()

println("All examples completed successfully!")
