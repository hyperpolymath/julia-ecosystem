# SPDX-License-Identifier: PMPL-1.0-or-later
"""
Portfolio Management Example for Exnovation.jl

This example demonstrates how to use Exnovation for portfolio-level
decision-making across multiple candidates. It covers:
- Evaluating multiple exnovation candidates
- Running stage-gate processes
- Portfolio prioritization
- Budget allocation
- Decision reporting with JSON export
"""

using Exnovation
using JSON3

println("=" ^ 60)
println("Exnovation.jl - Portfolio Management Example")
println("=" ^ 60)
println()

# Scenario: An IT department is evaluating multiple legacy systems
# for potential phase-out, with limited budget for modernization

println("Scenario: IT Portfolio Rationalization")
println("-" ^ 60)
println()

# Helper function to create standardized assessments
function create_assessment(name::Symbol, description::String,
                          driver_weights::Vector{Float64},
                          barrier_weights::Vector{Float64},
                          sunk::Float64, forward::Float64,
                          fit::Float64, perf::Float64, risk::Float64)

    item = ExnovationItem(name, description, "Legacy system portfolio")

    drivers = [
        Driver(:tech_debt, driver_weights[1], "Technical debt and maintenance burden"),
        Driver(:performance, driver_weights[2], "Performance and scalability issues"),
        Driver(:strategic, driver_weights[3], "Strategic misalignment")
    ]

    barriers = [
        Barrier(Cognitive, barrier_weights[1], "Team knowledge and familiarity"),
        Barrier(Structural, barrier_weights[2], "Organizational dependencies"),
        Barrier(Behavioral, barrier_weights[3], "Operational routines")
    ]

    criteria = DecisionCriteria(0.2, 0.3, 0.3, 0.2)

    ExnovationAssessment(item, drivers, barriers, criteria,
                        sunk, forward, 0.0, fit, perf, risk)
end

# Define portfolio candidates
println("Evaluating 5 legacy systems:")
println()

candidates = [
    # System A: Old reporting tool (good candidate)
    create_assessment(:reporting_tool, "Legacy Crystal Reports system",
                     [0.8, 0.6, 0.7], [0.3, 0.2, 0.2],
                     500_000.0, 100_000.0, 0.2, 0.4, 0.5),

    # System B: Core ERP (risky to change)
    create_assessment(:core_erp, "15-year-old ERP system",
                     [0.7, 0.5, 0.6], [0.8, 0.9, 0.7],
                     5_000_000.0, 2_000_000.0, 0.4, 0.6, 0.8),

    # System C: Email archiver (easy win)
    create_assessment(:email_archive, "On-premise email archiving",
                     [0.9, 0.7, 0.8], [0.2, 0.1, 0.1],
                     200_000.0, 50_000.0, 0.1, 0.3, 0.3),

    # System D: Custom CRM (moderate)
    create_assessment(:custom_crm, "Homegrown CRM system",
                     [0.6, 0.5, 0.5], [0.5, 0.4, 0.5],
                     1_500_000.0, 800_000.0, 0.3, 0.5, 0.6),

    # System E: File server (defer)
    create_assessment(:file_server, "Legacy file server",
                     [0.4, 0.3, 0.3], [0.4, 0.3, 0.4],
                     300_000.0, 400_000.0, 0.6, 0.7, 0.4)
]

# Evaluate each candidate
for (index, assessment) in enumerate(candidates)
    summary = exnovation_score(assessment)
    rec = recommendation(assessment)

    println("$(index). $(assessment.item.name)")
    println("   Score: $(round(summary.total_score, digits=2)) → $(uppercase(String(rec)))")
end
println()

# Stage-gate process
println("Stage-Gate Evaluation:")
println("-" ^ 40)

gates = [
    StageGate(:initial_screen, 0.3),
    StageGate(:detailed_analysis, 0.6),
    StageGate(:final_approval, 0.9)
]

for assessment in candidates
    println("$(assessment.item.name):")
    results = run_stage_gates(assessment, gates)
    for gate in gates
        status = results[gate.name] ? "✓ PASS" : "✗ FAIL"
        println("  $(gate.name) (≥$(gate.threshold)): $(status)")
    end
    println()
end

# Portfolio prioritization with impact models
println("Portfolio Prioritization:")
println("-" ^ 40)

# Add impact models (capex, opex, public value)
portfolio = [
    PortfolioItem(
        ExnovationCase(candidates[1], nothing, RiskGovernance(0.6, 0.4, :balanced)),
        ImpactModel(300_000.0, 50_000.0, 0.7)  # Reporting tool
    ),
    PortfolioItem(
        ExnovationCase(candidates[2], nothing, RiskGovernance(0.3, 0.8, :minimize)),
        ImpactModel(8_000_000.0, 500_000.0, 0.9)  # Core ERP
    ),
    PortfolioItem(
        ExnovationCase(candidates[3], nothing, RiskGovernance(0.8, 0.2, :maximize)),
        ImpactModel(150_000.0, 20_000.0, 0.5)  # Email archive
    ),
    PortfolioItem(
        ExnovationCase(candidates[4], nothing, RiskGovernance(0.5, 0.5, :balanced)),
        ImpactModel(2_000_000.0, 200_000.0, 0.8)  # Custom CRM
    ),
    PortfolioItem(
        ExnovationCase(candidates[5], nothing, RiskGovernance(0.4, 0.3, :minimize)),
        ImpactModel(400_000.0, 80_000.0, 0.6)  # File server
    )
]

# Get prioritized list
priorities = portfolio_scores(portfolio)

println("Ranked by combined exnovation + impact score:")
for (rank, (name, exnov_score, impact_score)) in enumerate(priorities)
    combined = exnov_score + impact_score
    println("$(rank). $(name)")
    println("   Exnovation: $(round(exnov_score, digits=2)), Impact: $(round(impact_score, digits=2)), Combined: $(round(combined, digits=2))")
end
println()

# Budget allocation
println("Budget Allocation:")
println("-" ^ 40)

total_budget = 5_000_000.0  # $5M available for modernization
selected = allocate_budget(portfolio; capex_budget=total_budget)

println("Budget: \$$(Int(total_budget / 1_000_000))M available")
println("Selected projects:")
total_allocated = 0.0
for name in selected
    item = first(filter(p -> p.case.assessment.item.name == name, portfolio))
    total_allocated += item.impact.capex
    println("  - $(name): \$$(Int(item.impact.capex / 1_000))K")
end
println("Total allocated: \$$(round(total_allocated / 1_000_000, digits=2))M")
println("Remaining: \$$(round((total_budget - total_allocated) / 1_000_000, digits=2))M")
println()

# Generate detailed decision report for top candidate
println("Detailed Decision Report (Top Candidate):")
println("-" ^ 40)

top_case = portfolio[1].case
report = decision_pipeline(top_case)

println("Item: $(top_case.assessment.item.name)")
println("Recommendation: $(uppercase(String(report.recommendation)))")
println()
println("Scores:")
println("  Exnovation total: $(round(report.exnovation.total_score, digits=2))")
println("  Driver score: $(round(report.exnovation.driver_score, digits=2))")
println("  Barrier score: $(round(report.exnovation.barrier_score, digits=2))")
println("  Criteria score: $(round(report.exnovation.criteria_score, digits=2))")
println()

if !isempty(report.notes)
    println("Decision notes:")
    for note in report.notes
        println("  - $(note)")
    end
    println()
end

# Export report to JSON
report_file = "/tmp/exnovation_report.json"
write_report_json(report_file, report)
println("Full report exported to: $(report_file)")
println()

# Show JSON structure
println("JSON Report Structure:")
report_json = JSON3.read(read(report_file, String))
println(JSON3.pretty(report_json))
println()

println("=" ^ 60)
println("Portfolio management example completed successfully!")
println("=" ^ 60)
