# SPDX-License-Identifier: PMPL-1.0-or-later
"""
Basic Usage Example for Exnovation.jl

This example demonstrates how to use the Exnovation package to evaluate
whether to phase out a legacy system, practice, or product. It covers:
- Building an exnovation assessment
- Computing exnovation scores
- Getting recommendations
- Identifying debiasing actions
"""

using Exnovation

println("=" ^ 60)
println("Exnovation.jl - Basic Usage Example")
println("=" ^ 60)
println()

# Scenario: A company is considering phasing out a legacy monolithic application
# in favor of a modern microservices architecture

println("Scenario: Evaluating Legacy Monolith vs. Microservices Migration")
println("-" ^ 60)
println()

# Define the item under consideration
legacy_system = ExnovationItem(
    :legacy_monolith,
    "Monolithic application built 10 years ago",
    "Core business application handling orders, inventory, and customer management"
)

# Define drivers pushing toward exnovation (phase-out)
drivers = [
    Driver(:technical_debt, 0.8, "High maintenance costs and slow feature velocity"),
    Driver(:scalability, 0.6, "Cannot scale individual components independently"),
    Driver(:cloud_native, 0.5, "Not optimized for cloud infrastructure"),
    Driver(:talent_retention, 0.4, "Difficulty recruiting developers for legacy stack")
]

# Define barriers resisting exnovation
barriers = [
    Barrier(Cognitive, 0.5, "Team familiar with existing system"),
    Barrier(Emotional, 0.3, "Attachment to system we built"),
    Barrier(Structural, 0.6, "Existing contracts and SLAs depend on it"),
    Barrier(Behavioral, 0.4, "Operational procedures built around monolith")
]

# Define decision criteria weights
criteria = DecisionCriteria(
    sunk_cost_weight = 0.2,
    strategic_fit_weight = 0.3,
    performance_weight = 0.3,
    risk_weight = 0.2
)

# Define assessment values
assessment = ExnovationAssessment(
    legacy_system,
    drivers,
    barriers,
    criteria,
    sunk_cost = 2_000_000.0,        # $2M already invested
    forward_value = 500_000.0,      # $500K estimated remaining value
    replacement_value = 3_000_000.0, # $3M to build microservices
    strategic_fit = 0.3,             # Low fit with cloud-first strategy
    performance = 0.5,               # Moderate performance issues
    risk = 0.6                       # Moderate risk of system failure
)

# Compute exnovation score
summary = exnovation_score(assessment)

println("Exnovation Analysis Results:")
println("-" ^ 40)
println("Driver Score:    $(round(summary.driver_score, digits=2))")
println("Barrier Score:   $(round(summary.barrier_score, digits=2))")
println("Criteria Score:  $(round(summary.criteria_score, digits=2))")
println("Total Score:     $(round(summary.total_score, digits=2))")
println()

println("Sunk Cost Bias:  $(round(summary.sunk_cost_bias * 100, digits=1))%")
if summary.sunk_cost_bias > 0.6
    println("  ⚠ Warning: High sunk cost bias detected!")
end
println()

# Get recommendation
rec = recommendation(assessment)
println("Recommendation:  $(uppercase(String(rec)))")
println()

interpretation = Dict(
    :exnovate => "Proceed with phase-out and migration",
    :pilot => "Run a pilot migration of one subsystem first",
    :defer => "Defer decision; continue with current system"
)
println("Action: $(interpretation[rec])")
println()

# Identify debiasing actions to overcome resistance
println("Suggested Debiasing Actions:")
println("-" ^ 40)
actions = debiasing_actions(barriers)
for (index, action) in enumerate(actions)
    println("$(index). $(action)")
end
println()

# Show breakdown by barrier type
println("Barrier Analysis by Type:")
println("-" ^ 40)
barrier_types = Dict(
    Cognitive => filter(b -> b.kind == Cognitive, barriers),
    Emotional => filter(b -> b.kind == Emotional, barriers),
    Behavioral => filter(b -> b.kind == Behavioral, barriers),
    Structural => filter(b -> b.kind == Structural, barriers)
)

for (barrier_type, type_barriers) in sort(collect(barrier_types), by=x->sum(b.weight for b in x[2]), rev=true)
    if !isempty(type_barriers)
        total_weight = sum(b.weight for b in type_barriers)
        println("$(barrier_type): $(round(total_weight, digits=2))")
        for barrier in type_barriers
            println("  - $(barrier.description) ($(barrier.weight))")
        end
    end
end
println()

println("=" ^ 60)
println("Basic usage example completed successfully!")
println("=" ^ 60)
