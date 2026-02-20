# SPDX-License-Identifier: PMPL-1.0-or-later
# Copyright (c) 2026 Jonathan D.A. Jewell <jonathan.jewell@open.ac.uk>

"""
    Axiology

The Axiology.jl module provides a robust framework for integrating value theory
into machine learning models. In an era where AI systems increasingly influence
critical decisions, understanding and explicitly managing their ethical and
economic implications is paramount. This module allows practitioners and researchers
to define, optimize, and formally verify various values within their ML pipelines.

The term "Axiology" (from Greek ἀξία, axía, "value"; and -λογία, -logia, "theory")
refers to the philosophical study of value. In this context, Axiology.jl brings
this study into the computational domain, providing tools to operationalize and
reason about values such as fairness, welfare, profit, efficiency, and safety.

Key Features & Applications:
----------------------------
1.  **Value Type System**: Define diverse ethical and economic values (e.g., `Fairness`,
    `Welfare`, `Profit`, `Efficiency`, `Safety`) as first-class citizens. This allows
    for explicit representation of desired model behaviors beyond traditional performance metrics.
2.  **Value Satisfaction Checking**: Assess whether a given ML model's behavior or
    outcome `satisfy` predefined value criteria (e.g., is a model "fair" according
    to a specific metric and threshold?).
3.  **Multi-Objective Optimization**: Address the inherent trade-offs between
    competing values (e.g., fairness vs. efficiency, profit vs. safety). The module
    supports finding optimal model configurations that balance multiple objectives.
4.  **Pareto Frontier Analysis**: Explore the set of non-dominated solutions in
    multi-objective optimization, providing insights into the best possible trade-offs
    between different values.
5.  **Value Verification**: Lay the groundwork for formal verification of value properties,
    ensuring that models adhere to their intended ethical and economic principles
    under various conditions.

Axiology.jl aims to foster more transparent, accountable, and ethically aligned AI systems
by moving beyond black-box optimization towards explicit value-driven design. It provides
a bridge between philosophical considerations of value and the practical implementation
of machine learning.

# Example

```julia
using Axiology

# Define a fairness criterion
fairness = Fairness(
    metric = :demographic_parity,
    protected_attributes = [:gender, :race],
    threshold = 0.05
)

# Simulate a model's state or evaluation results
model_evaluation_state = Dict(
    :predictions => [0.8, 0.7, 0.6, 0.9, 0.5, 0.75],
    :protected => [:male, :female, :male, :female, :male, :female],
    :actual_labels => [1, 1, 0, 1, 0, 1],
    :disparity => 0.03 # Assuming a pre-calculated disparity for demographic_parity
)

# Check if the model satisfies the defined fairness criterion
@assert satisfy(fairness, model_evaluation_state)  # True if disparity < threshold

# Define multiple competing values for optimization
values_to_optimize = [
    Welfare(metric = :utilitarian, weight = 0.4),
    Fairness(metric = :equalized_odds, weight = 0.3),
    Efficiency(metric = :computation_time, weight = 0.3)
]

# (Conceptual) Find solutions on the Pareto frontier for a given system
# In practice, `system` would represent an ML model or decision-making process
# and `pareto_frontier` would search its configuration space.
# solutions = pareto_frontier(my_ml_system_configs, values_to_optimize)
```
"""
module Axiology

using Statistics
using LinearAlgebra

# Core value types
export Value, Fairness, Welfare, Profit, Efficiency, Safety
export FairnessMetric, WelfareMetric, EfficiencyMetric
export satisfy, maximize, verify_value
export pareto_frontier, dominated, value_score
export weighted_score, normalize_scores

# Fairness metrics
export demographic_parity, equalized_odds, equal_opportunity
export disparate_impact, individual_fairness

# Welfare functions
export utilitarian_welfare, rawlsian_welfare, egalitarian_welfare

# Value types and implementations
include("types.jl")
include("fairness.jl")
include("welfare.jl")
include("optimization.jl")

end # module Axiology
