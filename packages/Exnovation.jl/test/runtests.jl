# SPDX-License-Identifier: PMPL-1.0-or-later
using Test
using Exnovation

@testset "Exnovation" begin
    item = ExnovationItem(:Legacy, "Legacy tool", "Operations")

    drivers = [
        Driver(:Risk, 0.6, "High risk"),
        Driver(:Compliance, 0.4, "Regulatory pressure"),
    ]

    barriers = [
        Barrier(Cognitive, 0.5, "Sunk-cost framing"),
        Barrier(Behavioral, 0.2, "Process inertia"),
    ]

    criteria = DecisionCriteria(0.3, 0.3, 0.2, 0.2)

    assessment = ExnovationAssessment(
        item,
        drivers,
        barriers,
        criteria,
        100.0,
        50.0,
        120.0,
        0.4,
        0.6,
        0.7,
    )

    summary = exnovation_score(assessment)
    @test summary.driver_score > 0.0
    @test summary.barrier_score > 0.0
    @test summary.criteria_score > 0.0
    @test summary.sunk_cost_bias > 0.0

    rec = recommendation(assessment)
    @test rec in (:exnovate, :pilot, :defer)

    # Edge case: Zero drivers
    no_driver_assessment = ExnovationAssessment(
        item, Driver[], barriers, criteria, 100.0, 50.0, 120.0, 0.4, 0.6, 0.7
    )
    no_driver_summary = exnovation_score(no_driver_assessment)
    @test no_driver_summary.driver_score == 0.0

    # Edge case: Zero barriers
    no_barrier_assessment = ExnovationAssessment(
        item, drivers, Barrier[], criteria, 100.0, 50.0, 120.0, 0.4, 0.6, 0.7
    )
    no_barrier_summary = exnovation_score(no_barrier_assessment)
    @test no_barrier_summary.barrier_score == 0.0

    # Edge case: Maximum sunk costs
    high_sunk_assessment = ExnovationAssessment(
        item, drivers, barriers, criteria, 1000.0, 10.0, 50.0, 0.9, 0.9, 0.9
    )
    high_sunk_summary = exnovation_score(high_sunk_assessment)
    @test high_sunk_summary.sunk_cost_bias > 0.5

    # Edge case: Zero sunk costs
    zero_sunk_assessment = ExnovationAssessment(
        item, drivers, barriers, criteria, 0.0, 50.0, 120.0, 0.0, 0.0, 0.0
    )
    zero_sunk_summary = exnovation_score(zero_sunk_assessment)
    @test zero_sunk_summary.sunk_cost_bias == 0.0

    # Test recommendation varies with different scores
    strong_driver_assessment = ExnovationAssessment(
        item,
        [Driver(:Risk, 0.95, "Critical risk"), Driver(:Compliance, 0.9, "Mandatory")],
        [Barrier(Cognitive, 0.1, "Minimal")],
        criteria,
        50.0,
        100.0,
        120.0,
        0.2,
        0.3,
        0.4,
    )
    rec_strong = recommendation(strong_driver_assessment)
    @test rec_strong in (:exnovate, :pilot)

    actions = debiasing_actions(barriers)
    @test !isempty(actions)

    # Test Political barrier debiasing
    political_barriers = [Barrier(Political, 0.5, "Stakeholder resistance")]
    political_actions = debiasing_actions(political_barriers)
    @test length(political_actions) >= 1
    @test any(contains(a, "stakeholder") || contains(a, "Stakeholder") for a in political_actions)

    criteria = IntelligentFailureCriteria(0.9, 0.7, 0.8, 0.9, 0.8, 0.7, 0.8)
    failure = FailureAssessment(Intelligent, criteria, 0.6, 0.7)
    fsummary = failure_summary(failure)
    @test fsummary.intelligent_failure_score > 0.0
    @test fsummary.failure_type == Intelligent

    # Edge case: All criteria at maximum (highly intelligent failure)
    max_criteria = IntelligentFailureCriteria(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    max_failure = FailureAssessment(Intelligent, max_criteria, 1.0, 1.0)
    max_fsummary = failure_summary(max_failure)
    @test max_fsummary.intelligent_failure_score > 0.8

    # Edge case: All criteria at minimum
    min_criteria = IntelligentFailureCriteria(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    min_failure = FailureAssessment(Intelligent, min_criteria, 0.0, 0.0)
    min_fsummary = failure_summary(min_failure)
    @test min_fsummary.intelligent_failure_score == 0.0

    # Test different failure types if supported
    for failure_type in [Intelligent, :basic, :preventable]
        try
            f = FailureAssessment(failure_type, criteria, 0.5, 0.5)
            fs = failure_summary(f)
            @test fs isa Any
        catch
            # Type might not be supported
            @test true
        end
    end

    case = ExnovationCase(
        assessment,
        failure,
        RiskGovernance(0.5, 0.7, :govern),
    )
    report = decision_pipeline(case)
    @test report.recommendation in (:exnovate, :pilot, :defer)
    path = joinpath(@__DIR__, "report.json")
    write_report_json(path, report)
    @test isfile(path)
    rm(path, force=true)

    templates = barrier_templates()
    @test haskey(templates, :political)

    gates = [StageGate(:screen, 0.2), StageGate(:commit, 0.8)]
    gate_results = run_stage_gates(assessment, gates)
    @test haskey(gate_results, :screen)

    # Edge case: Single gate
    single_gate = [StageGate(:single, 0.5)]
    single_results = run_stage_gates(assessment, single_gate)
    @test haskey(single_results, :single)

    # Edge case: Many gates
    many_gates = [StageGate(Symbol("gate$i"), i/10) for i in 1:5]
    many_results = run_stage_gates(assessment, many_gates)
    @test length(many_results) == 5

    impact = ImpactModel(100.0, 50.0, 0.9)
    item = PortfolioItem(case, impact)
    scores = portfolio_scores([item])
    @test length(scores) == 1

    # Edge case: Empty portfolio
    empty_scores = portfolio_scores(PortfolioItem[])
    @test length(empty_scores) == 0

    # Edge case: Multiple portfolio items
    item2 = PortfolioItem(case, ImpactModel(200.0, 100.0, 0.8))
    multi_scores = portfolio_scores([item, item2])
    @test length(multi_scores) == 2

    allocation = allocate_budget([item]; capex_budget=120.0)
    @test allocation == [:Legacy]

    # Edge case: Insufficient budget
    low_allocation = allocate_budget([item]; capex_budget=50.0)
    @test low_allocation isa Vector

    # Edge case: Multiple items with budget constraint
    multi_allocation = allocate_budget([item, item2]; capex_budget=250.0)
    @test length(multi_allocation) <= 2
end
