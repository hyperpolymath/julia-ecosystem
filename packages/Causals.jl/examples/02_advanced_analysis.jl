# SPDX-License-Identifier: PMPL-1.0-or-later
"""
Advanced Causal Analysis Example for Causals.jl

This example demonstrates advanced causal inference techniques including:
- Granger causality for time series
- Propensity score matching for observational studies
- Do-calculus for interventional queries
- Counterfactual reasoning
"""

using Causals
using Causals.Granger
using Causals.PropensityScore
using Causals.DoCalculus
using Causals.Counterfactuals
using Causals.CausalDAG
using Statistics
using Random

Random.seed!(42)

println("=" ^ 60)
println("Causals.jl - Advanced Analysis Example")
println("=" ^ 60)
println()

# Example 1: Granger Causality for Time Series
println("1. Granger Causality Analysis")
println("-" ^ 40)

# Generate synthetic time series: X causes Y with lag
n_observations = 100
time_series_x = randn(n_observations)
time_series_y = zeros(n_observations)

# Y depends on past values of X (with noise)
for time_index in 3:n_observations
    time_series_y[time_index] = 0.7 * time_series_x[time_index-1] +
                                 0.3 * time_series_x[time_index-2] +
                                 0.3 * randn()
end

# Test for Granger causality
println("Testing if X Granger-causes Y...")
causes, f_stat, p_value, best_lag = granger_test(time_series_x, time_series_y, 5)
println("  F-statistic: $(round(f_stat, digits=3))")
println("  p-value: $(round(p_value, digits=4))")
println("  Granger causes? $(causes)")
println()

# Find optimal lag
optimal = optimal_lag(time_series_x, time_series_y, 8)
println("  Optimal lag: $(optimal) periods")
println()

# Example 2: Propensity Score Matching
println("2. Propensity Score Matching")
println("-" ^ 40)

# Simulated observational study: effect of training program on salary
num_subjects = 200

# Confounders: education level and prior experience
education = rand(num_subjects) .* 10  # 0-10 years
experience = rand(num_subjects) .* 15  # 0-15 years

# Treatment assignment (training) depends on confounders
propensity_true = 1.0 ./ (1.0 .+ exp.(-(0.3 .* education .+ 0.2 .* experience .- 4.0)))
treatment = rand(num_subjects) .< propensity_true

# Outcome (salary) depends on treatment and confounders
control_outcomes = 30.0 .+ 2.0 .* education .+ 1.5 .* experience .+ randn(num_subjects) .* 3.0
treated_outcomes = control_outcomes .+ 5.0 .+ randn(num_subjects) .* 2.0
observed_outcomes = ifelse.(treatment, treated_outcomes, control_outcomes)

# Estimate propensity scores
covariates = hcat(education, experience)
propensity_scores = propensity_score(treatment, covariates)

println("Propensity score statistics:")
println("  Mean (treated): $(round(mean(propensity_scores[treatment]), digits=3))")
println("  Mean (control): $(round(mean(propensity_scores[.!treatment]), digits=3))")
println()

# Perform matching
matches, ate_matched, se_matched = matching(treatment, observed_outcomes, propensity_scores)
println("Matching results:")
println("  Number of matched pairs: $(length(matches))")
println("  Average Treatment Effect (ATE): $(round(ate_matched, digits=2)) thousand")
println("  Standard Error: $(round(se_matched, digits=2)) thousand")
println("  (True effect: 5.0 thousand)")
println()

# Example 3: Do-Calculus and Interventions
println("3. Do-Calculus and Interventional Queries")
println("-" ^ 40)

# Build a causal model with confounding
#   Z (confounder) → X (treatment) → Y (outcome)
#   Z → Y (confounding path)
confounded_graph = CausalGraph([:Z, :X, :Y])
CausalDAG.add_edge!(confounded_graph, :Z, :X)  # Z → X
CausalDAG.add_edge!(confounded_graph, :Z, :Y)  # Z → Y
CausalDAG.add_edge!(confounded_graph, :X, :Y)  # X → Y

println("Causal model: Z → X → Y, Z → Y")
println()

# Check if effect is identifiable
identifiable, method, adjustment_set = identify_effect(confounded_graph, :X, :Y)
println("Effect identification:")
println("  Can identify causal effect of X on Y? $(identifiable)")
println("  Method: $(method)")
println("  Adjustment set: $(adjustment_set)")
println()

# Example of using confounding adjustment with data
# Generate synthetic data matching the causal graph
z_vals = randn(num_subjects)
x_binary = Float64.(z_vals .+ randn(num_subjects) .> 0.0)  # Binary treatment based on Z
y_vals = 0.8 .* x_binary .+ 0.6 .* z_vals .+ randn(num_subjects) .* 0.3

confounding_data = Dict{Symbol, Vector{Float64}}(
    :Z => z_vals,
    :X => x_binary,
    :Y => y_vals
)

adjusted_ate = confounding_adjustment(:X, :Y, Set([:Z]), confounding_data)
println("Confounding adjustment:")
println("  Adjusted causal effect: $(round(adjusted_ate, digits=3))")
println("  (True effect: 0.8, adjusted for confounder Z)")
println()

# Example 4: Counterfactual Reasoning
println("4. Counterfactual Reasoning")
println("-" ^ 40)

# Build causal graph: X → Y
counterfactual_graph = CausalGraph([:X, :Y])
CausalDAG.add_edge!(counterfactual_graph, :X, :Y)

# Simple structural equation model: Y = 2X + U_Y
structural_equations = Dict{Symbol, Function}(
    :X => (parents, noise) -> get(noise, :U_X, 0.0),
    :Y => (parents, noise) -> 2.0 * get(parents, :X, 0.0) + get(noise, :U_Y, 0.0)
)

# Observed values: X=3, Y=6.5 (implies U_Y = 0.5)
observations = Dict{Symbol, Any}(:X => 3.0, :Y => 6.5)
factual_Y = observations[:Y]

println("Factual scenario:")
println("  X = $(observations[:X])")
println("  Y = $(factual_Y)")
println()

# Counterfactual: What if X had been 5.0 instead?
counterfactual_Y = counterfactual(
    counterfactual_graph,
    :Y,
    :X => 5.0,
    observations;
    equations=structural_equations
)

println("Counterfactual scenario (if X = 5.0):")
println("  Y would have been: $(round(counterfactual_Y, digits=2))")
println("  (Computation: 2*5 + 0.5 = 10.5)")
println()

# Probability of necessity: Was X=3 necessary for Y>5?
println("Causal responsibility analysis:")
println("  Factual: X=3 → Y=$(factual_Y)")
println("  Question: Was X=3 necessary for Y>5?")
if factual_Y > 5.0
    counterfactual_Y_if_X_0 = counterfactual(
        counterfactual_graph,
        :Y,
        :X => 0.0,
        observations;
        equations=structural_equations
    )
    was_necessary = counterfactual_Y_if_X_0 <= 5.0
    println("  Answer: $(was_necessary)")
    println("  (If X had been 0, Y would be $(round(counterfactual_Y_if_X_0, digits=2)))")
else
    println("  Not applicable (Y ≤ 5)")
end
println()

println("=" ^ 60)
println("Advanced analysis example completed successfully!")
println("=" ^ 60)
