# SPDX-License-Identifier: PMPL-1.0-or-later
using Test
using Causals
using Causals.CausalDAG: add_edge!
using Graphs: inneighbors
using Statistics: mean

@testset "Causals.jl" begin

    @testset "Dempster-Shafer" begin
        frame = [:A, :B, :C]
        masses = Dict(
            Set([:A]) => 0.4,
            Set([:B]) => 0.3,
            Set([:A, :B, :C]) => 0.3
        )
        m = MassAssignment(frame, masses)

        @test mass(m, Set([:A])) == 0.4
        @test belief(m, Set([:A])) == 0.4
        @test plausibility(m, Set([:A])) >= belief(m, Set([:A]))

        # Combination
        m2 = MassAssignment(frame, Dict(Set([:A]) => 0.6, Set([:A, :B, :C]) => 0.4))
        m_combined = combine_dempster(m, m2)
        @test sum(values(m_combined.masses)) ≈ 1.0

        # Discount
        m_discounted = discount(m, 0.8)
        @test sum(values(m_discounted.masses)) ≈ 1.0

        # Edge case: Empty frame
        empty_frame = Symbol[]
        empty_masses = Dict{Set{Symbol}, Float64}()
        m_empty = MassAssignment(empty_frame, empty_masses)
        @test length(m_empty.frame) == 0

        # Edge case: Single element frame
        single_frame = [:X]
        single_masses = Dict(Set([:X]) => 1.0)
        m_single = MassAssignment(single_frame, single_masses)
        @test belief(m_single, Set([:X])) == 1.0
        @test plausibility(m_single, Set([:X])) == 1.0

        # Edge case: Zero discount (no change)
        m_no_discount = discount(m, 1.0)
        @test mass(m_no_discount, Set([:A])) ≈ 0.4

        # Edge case: Full discount (all mass to uncertainty)
        m_full_discount = discount(m, 0.0)
        @test mass(m_full_discount, Set(frame)) ≈ 1.0

        # Test belief <= plausibility always holds
        @test belief(m, Set([:A, :B])) <= plausibility(m, Set([:A, :B]))
        @test belief(m, Set([:C])) <= plausibility(m, Set([:C]))

        # Test combination with conflicting evidence
        m_conflict = MassAssignment(frame, Dict(Set([:B]) => 0.9, Set([:A, :B, :C]) => 0.1))
        m_combined_conflict = combine_dempster(m, m_conflict)
        @test sum(values(m_combined_conflict.masses)) ≈ 1.0
    end

    @testset "Bradford Hill" begin
        criteria = BradfordHillCriteria(
            strength = 0.8,
            consistency = 0.9,
            temporality = 1.0,
            plausibility = 0.7
        )

        verdict, confidence = assess_causality(criteria)
        @test verdict in [:strong, :moderate, :weak, :insufficient, :none]
        @test 0.0 <= confidence <= 1.0

        # Temporality required
        no_temporal = BradfordHillCriteria(strength=0.9, temporality=0.0)
        verdict_no_temp, _ = assess_causality(no_temporal)
        @test verdict_no_temp == :none

        # Edge case: All criteria at maximum
        max_criteria = BradfordHillCriteria(
            strength=1.0, consistency=1.0, temporality=1.0,
            specificity=1.0, biological_gradient=1.0, plausibility=1.0,
            coherence=1.0, experiment=1.0, analogy=1.0
        )
        verdict_max, conf_max = assess_causality(max_criteria)
        @test verdict_max == :strong
        @test conf_max == 1.0

        # Edge case: All criteria at minimum
        min_criteria = BradfordHillCriteria(
            strength=0.0, consistency=0.0, temporality=0.0
        )
        verdict_min, _ = assess_causality(min_criteria)
        @test verdict_min == :none

        # Weak evidence (low values but temporality present)
        weak_criteria = BradfordHillCriteria(
            strength=0.2, consistency=0.3, temporality=0.5
        )
        verdict_weak, _ = assess_causality(weak_criteria)
        @test verdict_weak in [:weak, :insufficient, :none]

        # Strong temporality but weak other criteria
        temporal_only = BradfordHillCriteria(
            strength=0.1, consistency=0.1, temporality=1.0
        )
        verdict_temporal, _ = assess_causality(temporal_only)
        @test verdict_temporal in [:weak, :insufficient]
    end

    @testset "Granger Causality" begin
        # Generate test data: x causes y with lag
        n = 100
        x = randn(n)
        y = zeros(n)
        for t in 2:n
            y[t] = 0.5 * y[t-1] + 0.3 * x[t-1] + 0.1 * randn()
        end

        causes, F_stat, p_val, lag = granger_test(x, y, 5)
        @test F_stat >= 0.0
        @test lag >= 1

        strength = granger_causality(x, y, 5)
        @test 0.0 <= strength <= 1.0

        # Edge case: Minimal series length
        x_short = randn(10)
        y_short = randn(10)
        strength_short = granger_causality(x_short, y_short, 2)
        @test 0.0 <= strength_short <= 1.0

        # Edge case: No causality (independent series)
        x_indep = randn(100)
        y_indep = randn(100)
        causes_indep, F_indep, _, _ = granger_test(x_indep, y_indep, 3)
        @test F_indep >= 0.0

        # Edge case: Perfect correlation with no lag
        x_corr = randn(100)
        y_corr = x_corr .+ 0.01 .* randn(100)
        strength_corr = granger_causality(x_corr, y_corr, 5)
        @test 0.0 <= strength_corr <= 1.0

        # Edge case: Strong feedback loop
        n_fb = 100
        x_fb = zeros(n_fb)
        y_fb = zeros(n_fb)
        for t in 2:n_fb
            x_fb[t] = 0.4 * y_fb[t-1] + randn() * 0.1
            y_fb[t] = 0.4 * x_fb[t-1] + randn() * 0.1
        end
        strength_fb = granger_causality(x_fb, y_fb, 3)
        @test 0.0 <= strength_fb <= 1.0

        # Test with different lag values
        for max_lag in [1, 3, 10]
            strength_lag = granger_causality(x, y, max_lag)
            @test 0.0 <= strength_lag <= 1.0
        end
    end

    @testset "Causal DAG" begin
        # Create simple DAG: X → M → Y, C → X, C → Y
        g = CausalGraph([:X, :M, :Y, :C])
        add_edge!(g, :X, :M)
        add_edge!(g, :M, :Y)
        add_edge!(g, :C, :X)
        add_edge!(g, :C, :Y)

        # Test ancestors/descendants
        anc_y = ancestors(g, :Y)
        @test :X in anc_y
        @test :M in anc_y
        @test :C in anc_y

        desc_x = descendants(g, :X)
        @test :M in desc_x
        @test :Y in desc_x

        # Backdoor criterion
        @test backdoor_criterion(g, :X, :Y, Set([:C]))

        # Edge case: Single node graph
        g_single = CausalGraph([:A])
        @test isempty(ancestors(g_single, :A))
        @test isempty(descendants(g_single, :A))

        # Edge case: Two node chain
        g_chain = CausalGraph([:A, :B])
        add_edge!(g_chain, :A, :B)
        @test :A in ancestors(g_chain, :B)
        @test :B in descendants(g_chain, :A)
        @test isempty(ancestors(g_chain, :A))
        @test isempty(descendants(g_chain, :B))

        # Edge case: Triangle graph (A → B, B → C, A → C)
        g_triangle = CausalGraph([:A, :B, :C])
        add_edge!(g_triangle, :A, :B)
        add_edge!(g_triangle, :B, :C)
        add_edge!(g_triangle, :A, :C)
        anc_c = ancestors(g_triangle, :C)
        @test :A in anc_c
        @test :B in anc_c

        # Test backdoor with no confounders
        g_simple = CausalGraph([:X, :Y])
        add_edge!(g_simple, :X, :Y)
        @test backdoor_criterion(g_simple, :X, :Y, Set{Symbol}())

        # Test backdoor with insufficient adjustment set (empty set should fail)
        @test !backdoor_criterion(g, :X, :Y, Set{Symbol}())

        # Complex graph with multiple paths
        g_complex = CausalGraph([:A, :B, :C, :D, :E])
        add_edge!(g_complex, :A, :B)
        add_edge!(g_complex, :B, :C)
        add_edge!(g_complex, :A, :D)
        add_edge!(g_complex, :D, :E)
        add_edge!(g_complex, :E, :C)
        desc_a = descendants(g_complex, :A)
        @test :C in desc_a
        @test :E in desc_a
    end

    @testset "Propensity Score" begin
        n = 100
        treatment = rand(Bool, n)
        outcome = treatment .* 2.0 .+ randn(n)
        propensity = propensity_score(treatment, randn(n, 3))

        @test length(propensity) == n
        @test all(0.0 .<= propensity .<= 1.0)

        # IPW
        ate, se = inverse_probability_weighting(treatment, outcome, propensity)
        @test !isnan(ate)
        @test se >= 0.0

        # Edge case: All treated
        all_treated = trues(n)
        outcome_treated = randn(n)
        propensity_treated = propensity_score(all_treated, randn(n, 3))
        @test all(propensity_treated .>= 0.5)

        # Edge case: All control
        all_control = falses(n)
        propensity_control = propensity_score(all_control, randn(n, 3))
        @test all(propensity_control .<= 0.5)

        # Edge case: Balanced treatment
        balanced = vcat(trues(50), falses(50))
        outcome_balanced = randn(100)
        propensity_balanced = propensity_score(balanced, randn(100, 2))
        @test length(propensity_balanced) == 100
        @test all(0.0 .<= propensity_balanced .<= 1.0)

        # Test IPW with different propensity distributions
        ate_balanced, se_balanced = inverse_probability_weighting(balanced, outcome_balanced, propensity_balanced)
        @test !isnan(ate_balanced)
        @test se_balanced >= 0.0

        # Edge case: Single covariate
        propensity_single = propensity_score(treatment, randn(n, 1))
        @test length(propensity_single) == n
        @test all(0.0 .<= propensity_single .<= 1.0)

        # Edge case: Many covariates
        propensity_many = propensity_score(treatment, randn(n, 10))
        @test length(propensity_many) == n
        @test all(0.0 .<= propensity_many .<= 1.0)

        # Test with extreme propensity scores (near 0 or 1)
        extreme_propensity = vcat(fill(0.01, 50), fill(0.99, 50))
        ate_extreme, se_extreme = inverse_probability_weighting(balanced, outcome_balanced, extreme_propensity)
        @test !isnan(ate_extreme)
        @test se_extreme >= 0.0
    end

    @testset "Propensity Score Matching" begin
        using Causals.PropensityScore: matching

        n = 100
        treatment = rand(Bool, n)
        outcome = Float64.(treatment) .* 2.0 .+ randn(n)
        propensity = fill(0.5, n)

        matches, ate, se = matching(treatment, outcome, propensity)

        @test length(matches) > 0
        @test length(matches) <= sum(treatment)  # Can't exceed treated count
        @test !isnan(ate)
        @test se >= 0.0

        # Edge case: No matched pairs due to strict caliper
        matches_strict, _, _ = matching(treatment, outcome, propensity; caliper=0.001)
        @test length(matches_strict) >= 0  # May have no matches
    end

    @testset "Stratification" begin
        using Causals.PropensityScore: stratification

        n = 100
        treatment = rand(Bool, n)
        outcome = Float64.(treatment) .* 2.0 .+ randn(n)
        propensity = rand(n)

        ate, stratum_effects, stratum_weights = stratification(treatment, outcome, propensity)

        @test !isnan(ate)
        @test length(stratum_effects) == length(stratum_weights)
        @test sum(stratum_weights) ≈ 1.0

        # Test with different number of strata
        ate5, _, _ = stratification(treatment, outcome, propensity; n_strata=5)
        ate10, _, _ = stratification(treatment, outcome, propensity; n_strata=10)
        @test !isnan(ate5)
        @test !isnan(ate10)
    end

    @testset "Doubly Robust" begin
        using Causals.PropensityScore: doubly_robust

        n = 100
        treatment = rand(Bool, n)
        outcome = Float64.(treatment) .* 2.0 .+ randn(n)
        propensity = fill(0.5, n)

        # Simple outcome models
        outcome_model_1 = fill(mean(outcome[treatment]), n)
        outcome_model_0 = fill(mean(outcome[.!treatment]), n)

        ate = doubly_robust(treatment, outcome, propensity, outcome_model_1, outcome_model_0)

        @test !isnan(ate)
        @test abs(ate - 2.0) < 2.0  # Should be near true effect

        # Edge case: Perfect outcome models
        perfect_1 = outcome .* Float64.(treatment)
        perfect_0 = outcome .* Float64.(.!treatment)
        ate_perfect = doubly_robust(treatment, outcome, propensity, perfect_1, perfect_0)
        @test !isnan(ate_perfect)
    end

    @testset "D-Separation" begin
        using Causals.CausalDAG: CausalGraph, add_edge!, d_separation

        # Test chain: X → M → Y
        g = CausalGraph([:X, :M, :Y])
        add_edge!(g, :X, :M)
        add_edge!(g, :M, :Y)

        # X and Y are NOT d-separated (connected via chain)
        @test !d_separation(g, Set([:X]), Set([:Y]), Set{Symbol}())

        # X and Y ARE d-separated given M
        @test d_separation(g, Set([:X]), Set([:Y]), Set([:M]))

        # Test fork: X ← M → Y
        g2 = CausalGraph([:X, :M, :Y])
        add_edge!(g2, :M, :X)
        add_edge!(g2, :M, :Y)

        # X and Y are NOT d-separated
        @test !d_separation(g2, Set([:X]), Set([:Y]), Set{Symbol}())

        # X and Y ARE d-separated given M
        @test d_separation(g2, Set([:X]), Set([:Y]), Set([:M]))

        # Test collider: X → M ← Y
        g3 = CausalGraph([:X, :M, :Y])
        add_edge!(g3, :X, :M)
        add_edge!(g3, :Y, :M)

        # X and Y ARE d-separated (collider blocks path)
        @test d_separation(g3, Set([:X]), Set([:Y]), Set{Symbol}())

        # X and Y are NOT d-separated given M (collider unblocked)
        @test !d_separation(g3, Set([:X]), Set([:Y]), Set([:M]))
    end

    @testset "Frontdoor Criterion" begin
        using Causals.CausalDAG: CausalGraph, add_edge!, frontdoor_criterion

        # Classic frontdoor: X → M → Y, U → X, U → Y
        g = CausalGraph([:X, :M, :Y, :U])
        add_edge!(g, :X, :M)
        add_edge!(g, :M, :Y)
        add_edge!(g, :U, :X)
        add_edge!(g, :U, :Y)

        @test frontdoor_criterion(g, :X, :Y, Set([:M]))

        # Negative case: M does not intercept all paths
        g2 = CausalGraph([:X, :M, :Y, :U])
        add_edge!(g2, :X, :M)
        add_edge!(g2, :M, :Y)
        add_edge!(g2, :X, :Y)  # Direct path X → Y
        add_edge!(g2, :U, :X)
        add_edge!(g2, :U, :Y)

        @test !frontdoor_criterion(g2, :X, :Y, Set([:M]))
    end

    @testset "Markov Blanket" begin
        using Causals.CausalDAG: CausalGraph, add_edge!, markov_blanket

        # Graph: A → B ← C, B → D
        g = CausalGraph([:A, :B, :C, :D])
        add_edge!(g, :A, :B)
        add_edge!(g, :C, :B)
        add_edge!(g, :B, :D)

        blanket = markov_blanket(g, :B)

        # Should contain parents (A, C), children (D), and co-parents (none here)
        @test :A in blanket
        @test :C in blanket
        @test :D in blanket
        @test :B ∉ blanket

        # Co-parent test: A → D ← B
        g2 = CausalGraph([:A, :B, :D])
        add_edge!(g2, :A, :D)
        add_edge!(g2, :B, :D)

        blanket2 = markov_blanket(g2, :B)
        @test :D in blanket2  # Child
        @test :A in blanket2  # Co-parent (other parent of D)
    end

    @testset "DoCalculus" begin
        using Causals.DoCalculus: do_intervention, identify_effect, adjustment_formula, confounding_adjustment, Query, do_calculus_rules
        using Causals.CausalDAG: CausalGraph, add_edge!

        # Test do_intervention
        g = CausalGraph([:X, :Y, :Z])
        add_edge!(g, :Z, :X)
        add_edge!(g, :X, :Y)

        g_mutilated, _, _ = do_intervention(g, :X, 1.0)
        # Mutilated graph should have edge Z → X removed
        @test length(inneighbors(g_mutilated.graph, g_mutilated.name_to_index[:X])) == 0

        # Test identify_effect
        g2 = CausalGraph([:X, :Y, :Z])
        add_edge!(g2, :X, :Y)
        add_edge!(g2, :Z, :X)
        add_edge!(g2, :Z, :Y)

        identifiable, method, set = identify_effect(g2, :X, :Y, Set([:Z]))
        @test identifiable == true
        @test method == :backdoor

        # Test adjustment_formula
        adj_set = adjustment_formula(g2, :X, :Y, Set([:Z]))
        @test :Z in adj_set

        # Test confounding_adjustment
        n = 100
        z_vals = randn(n)
        x_vals = Float64.(z_vals .+ randn(n) .> 0.0)
        y_vals = 0.8 .* x_vals .+ 0.6 .* z_vals .+ randn(n) .* 0.3

        data = Dict{Symbol, Vector{Float64}}(
            :Z => z_vals,
            :X => x_vals,
            :Y => y_vals
        )

        ate = confounding_adjustment(:X, :Y, Set([:Z]), data)
        @test !isnan(ate)
        @test abs(ate - 0.8) < 0.5  # Should be near true effect

        # Test do_calculus_rules
        q = Query([:Y], [:X], [:Z])
        result = do_calculus_rules(g2, q)
        @test result !== :cannot_simplify || result isa Query
    end

    @testset "Counterfactuals" begin
        using Causals.Counterfactuals: counterfactual, twin_network, probability_of_necessity, probability_of_sufficiency, probability_of_necessity_and_sufficiency
        using Causals.CausalDAG: CausalGraph, add_edge!

        # Test counterfactual
        g = CausalGraph([:X, :Y])
        add_edge!(g, :X, :Y)

        equations = Dict{Symbol, Function}(
            :X => (parents, noise) -> get(noise, :U_X, 0.0),
            :Y => (parents, noise) -> 2.0 * get(parents, :X, 0.0) + get(noise, :U_Y, 0.0)
        )

        observations = Dict{Symbol, Any}(:X => 3.0, :Y => 6.5)
        result = counterfactual(g, :Y, :X => 5.0, observations; equations=equations)

        @test result !== nothing
        @test abs(result - 10.5) < 0.01  # 2*5 + 0.5 = 10.5

        # Test twin_network
        twin_g = twin_network(g)
        @test length(twin_g.names) == 4  # X, Y, X', Y'

        # Test probability of necessity
        n = 100
        treatment = rand(Bool, n)
        outcome = rand(Bool, n)
        data = Dict{Symbol, Vector{Bool}}(:Treatment => treatment, :Outcome => outcome)

        pn = probability_of_necessity(:Treatment, :Outcome, data)
        @test 0.0 <= pn <= 1.0

        # Test probability of sufficiency
        ps = probability_of_sufficiency(:Treatment, :Outcome, data)
        @test 0.0 <= ps <= 1.0

        # Test probability of necessity and sufficiency
        pns = probability_of_necessity_and_sufficiency(:Treatment, :Outcome, data)
        @test 0.0 <= pns <= 1.0
    end

end
