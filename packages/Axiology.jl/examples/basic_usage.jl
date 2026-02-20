# SPDX-License-Identifier: PMPL-1.0-or-later
# Basic usage example for Axiology.jl

using Axiology
using Statistics

println("Axiology.jl Basic Usage Examples")
println("=" ^ 50)

# Example 1: Fairness values
println("\n1. Creating and checking Fairness values:")
fairness = Fairness(metric=:demographic_parity, threshold=0.1)
state_fair = Dict(
    :predictions => [1, 1, 0, 0],
    :protected => [:a, :a, :b, :b]
)
println("  Fairness satisfied: ", satisfy(fairness, state_fair))
println("  Fairness score: ", value_score(fairness, state_fair))

# Example 2: Welfare values
println("\n2. Creating and checking Welfare values:")
welfare = Welfare(metric=:utilitarian)
state_welfare = Dict(:utilities => [10.0, 8.0, 12.0], :min_welfare => 25.0)
println("  Welfare satisfied: ", satisfy(welfare, state_welfare))
println("  Welfare score: ", maximize(welfare, Dict(:utilities => [10.0, 8.0, 12.0])))

# Example 3: Multi-objective optimization
println("\n3. Multi-objective optimization:")
values = [
    Fairness(metric=:demographic_parity, threshold=0.1, weight=0.5),
    Welfare(metric=:utilitarian, weight=0.5)
]
state_multi = Dict(
    :predictions => [1, 1, 0, 0],
    :protected => [:a, :a, :b, :b],
    :utilities => [10.0, 8.0, 12.0]
)
println("  Weighted score: ", weighted_score(values, state_multi))

# Example 4: Pareto frontier
println("\n4. Pareto frontier analysis:")
solutions = [
    Dict(:predictions => [1, 1, 0, 0], :protected => [:a, :a, :b, :b], :utilities => [10.0, 8.0, 12.0]),
    Dict(:predictions => [1, 0, 1, 0], :protected => [:a, :a, :b, :b], :utilities => [12.0, 12.0, 8.0])
]
frontier = pareto_frontier(solutions, values)
println("  Pareto optimal solutions: ", length(frontier))

println("\n" * "=" ^ 50)
println("All examples completed successfully!")
