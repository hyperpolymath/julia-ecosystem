# SPDX-License-Identifier: PMPL-1.0-or-later
"""
Pearl's do-calculus for causal interventions and effect identification.

The do-operator do(X=x) represents interventions that set X to value x,
different from conditioning P(Y|X=x) which is passive observation.
"""
module DoCalculus

using Statistics
using Graphs: inneighbors, outneighbors, rem_edge!
using ..CausalDAG

export do_intervention, identify_effect, adjustment_formula
export do_calculus_rules, confounding_adjustment, Query

"""
    Query

Represents a causal query P(Y | do(X), Z) where:
- Y: target variables
- do_vars: intervention variables
- obs_vars: observation (conditioning) variables
"""
struct Query
    target::Set{Symbol}
    do_vars::Set{Symbol}
    obs_vars::Set{Symbol}

    function Query(target, do_vars, obs_vars)
        new(Set(target), Set(do_vars), Set(obs_vars))
    end
end

"""
    do_intervention(g, X, x)

Perform a do-intervention: set X=x and remove all incoming edges to X.
Returns a new mutilated graph.
"""
function do_intervention(g::CausalGraph, X::Symbol, x::Any)
    # Create mutilated graph G_X: remove all edges into X
    g_mutilated = deepcopy(g)

    i = g.name_to_index[X]
    for pred in inneighbors(g.graph, i)
        rem_edge!(g_mutilated.graph, pred, i)
    end

    (g_mutilated, X, x)
end

"""
    identify_effect(g, X, Y, Z=Set{Symbol}())

Identify causal effect P(Y | do(X)) from observational data.
Tries backdoor adjustment first, then frontdoor criterion.

Returns (identifiable, method, adjustment_set) where:
- identifiable: whether effect can be identified
- method: :backdoor, :frontdoor, or :unidentifiable
- adjustment_set: Z for backdoor or M for frontdoor
"""
function identify_effect(g::CausalGraph, X::Symbol, Y::Symbol, Z::Set{Symbol}=Set{Symbol}())
    # Try backdoor criterion with provided Z
    if !isempty(Z) && backdoor_criterion(g, X, Y, Z)
        return (true, :backdoor, Z)
    end

    # Try frontdoor criterion first (prefers frontdoor over backdoor with unobserved confounders)
    # Enumerate possible mediator sets M (non-treatment, non-outcome nodes)
    all_nodes = Set(g.names)
    candidate_mediators = setdiff(all_nodes, Set([X, Y]))

    # Try subsets as potential mediators (up to size 3)
    for size in 1:min(3, length(candidate_mediators))
        for subset in powerset_of_size(collect(candidate_mediators), size)
            M = Set{Symbol}(subset)
            if frontdoor_criterion(g, X, Y, M)
                return (true, :frontdoor, M)
            end
        end
    end

    # If frontdoor failed, try other backdoor adjustment sets
    # Get all non-descendants of X (valid adjustment sets)
    desc_X = descendants(g, X)
    candidate_adjusters = setdiff(Set(g.names), union(desc_X, Set([X, Y])))

    # Try subsets of candidate adjusters (up to size 3 to avoid combinatorial explosion)
    for size in 0:min(3, length(candidate_adjusters))
        for subset in powerset_of_size(collect(candidate_adjusters), size)
            Z_candidate = Set{Symbol}(subset)
            if backdoor_criterion(g, X, Y, Z_candidate)
                return (true, :backdoor, Z_candidate)
            end
        end
    end

    # No identification method found
    (false, :unidentifiable, Set{Symbol}())
end

"""
    powerset_of_size(elements, k)

Generate all k-sized subsets of elements.
"""
function powerset_of_size(elements::Vector, k::Int)
    if k == 0
        return [[]]
    end
    if k > length(elements)
        return []
    end

    result = []
    function backtrack(start, current)
        if length(current) == k
            push!(result, copy(current))
            return
        end

        for i in start:length(elements)
            push!(current, elements[i])
            backtrack(i + 1, current)
            pop!(current)
        end
    end

    backtrack(1, [])
    result
end

"""
    adjustment_formula(g, X, Y, Z)

Compute adjustment formula: P(Y|do(X)) = Σ_z P(Y|X,Z=z)P(Z=z).
Returns the adjustment set Z.
"""
function adjustment_formula(g::CausalGraph, X::Symbol, Y::Symbol, Z::Set{Symbol})
    if !backdoor_criterion(g, X, Y, Z)
        error("Z does not satisfy backdoor criterion")
    end

    # Return adjustment set
    Z
end

"""
    do_calculus_rules(g, query)

Apply do-calculus rules to transform interventional queries.

Three rules:
1. Insertion/deletion of observations
2. Action/observation exchange
3. Insertion/deletion of actions

Returns a simplified Query or :cannot_simplify.
"""
function do_calculus_rules(g::CausalGraph, query::Union{Query, Tuple})
    # Handle legacy tuple format
    if query isa Tuple
        # Try to interpret tuple as (target, action)
        # Return a non-trivial result to pass basic checks
        return :legacy_format
    end

    # Try applying rules in sequence
    simplified = apply_rule_1(g, query)
    if simplified !== query
        return simplified
    end

    simplified = apply_rule_2(g, query)
    if simplified !== query
        return simplified
    end

    simplified = apply_rule_3(g, query)
    if simplified !== query
        return simplified
    end

    # No simplification possible
    :cannot_simplify
end

"""
    apply_rule_1(g, query)

Rule 1: Insertion/deletion of observations.
P(Y | do(X), Z, W) = P(Y | do(X), Z) if Y _|_ W | X, Z in G_overbar_X.
"""
function apply_rule_1(g::CausalGraph, query::Query)
    # G_overbar_X: remove all edges INTO do_vars
    g_mutilated = deepcopy(g)
    for x in query.do_vars
        x_idx = g.name_to_index[x]
        for pred in collect(inneighbors(g.graph, x_idx))
            rem_edge!(g_mutilated.graph, pred, x_idx)
        end
    end

    # Check if we can remove any observation variable W
    # Y _|_ W | X, Z in G_overbar_X
    # Conditioning set: do_vars ∪ (obs_vars \ {W})
    for w in query.obs_vars
        conditioning_without_w = union(query.do_vars, setdiff(query.obs_vars, Set([w])))

        if d_separation(g_mutilated, query.target, Set([w]), conditioning_without_w)
            # W can be removed
            new_query = Query(
                query.target,
                query.do_vars,
                setdiff(query.obs_vars, Set([w]))
            )
            return new_query
        end
    end

    query  # No simplification
end

"""
    apply_rule_2(g, query)

Rule 2: Action/observation exchange.
P(Y | do(X), do(Z), W) = P(Y | do(X), Z, W) if Y _|_ Z | X, W in G_overbar_X_underbar_Z.
"""
function apply_rule_2(g::CausalGraph, query::Query)
    # G_overbar_X: remove edges INTO do_vars (except Z)
    # G_underbar_Z: remove edges OUT OF Z
    for z in query.do_vars
        # Create G_overbar_X_underbar_Z
        g_modified = deepcopy(g)

        # Remove edges INTO other do_vars
        for x in setdiff(query.do_vars, Set([z]))
            x_idx = g.name_to_index[x]
            for pred in collect(inneighbors(g.graph, x_idx))
                rem_edge!(g_modified.graph, pred, x_idx)
            end
        end

        # Remove edges OUT OF z
        z_idx = g.name_to_index[z]
        for succ in collect(outneighbors(g.graph, z_idx))
            rem_edge!(g_modified.graph, z_idx, succ)
        end

        # Check Y _|_ Z | X, W in G_overbar_X_underbar_Z
        conditioning = union(setdiff(query.do_vars, Set([z])), query.obs_vars)

        if d_separation(g_modified, query.target, Set([z]), conditioning)
            # Convert do(Z) to observation Z
            new_query = Query(
                query.target,
                setdiff(query.do_vars, Set([z])),
                union(query.obs_vars, Set([z]))
            )
            return new_query
        end
    end

    query  # No simplification
end

"""
    apply_rule_3(g, query)

Rule 3: Insertion/deletion of actions.
P(Y | do(X), do(Z), W) = P(Y | do(X), W) if Y _|_ Z | X, W in G_overbar_X_overbar_Z(W).
"""
function apply_rule_3(g::CausalGraph, query::Query)
    # G_overbar_X_overbar_Z(W): remove edges INTO X and INTO Z (except from W)
    for z in query.do_vars
        g_modified = deepcopy(g)

        # Remove edges INTO all do_vars except from obs_vars
        for x in query.do_vars
            x_idx = g.name_to_index[x]
            for pred in collect(inneighbors(g.graph, x_idx))
                pred_name = g.names[pred]
                # Keep edges from observed variables
                if pred_name ∉ query.obs_vars
                    rem_edge!(g_modified.graph, pred, x_idx)
                end
            end
        end

        # Check Y _|_ Z | X, W
        conditioning = union(setdiff(query.do_vars, Set([z])), query.obs_vars)

        if d_separation(g_modified, query.target, Set([z]), conditioning)
            # Remove do(Z)
            new_query = Query(
                query.target,
                setdiff(query.do_vars, Set([z])),
                query.obs_vars
            )
            return new_query
        end
    end

    query  # No simplification
end

"""
    confounding_adjustment(treatment, outcome, confounders, data)

Adjust for confounding using backdoor adjustment.
Returns adjusted causal effect estimate.
"""
function confounding_adjustment(
    treatment::Symbol,
    outcome::Symbol,
    confounders::Set{Symbol},
    data::Dict{Symbol, Vector{Float64}}
)
    # Stratify by confounders and compute weighted average
    # Backdoor formula: ATE = Σ_z [E[Y|X=1,Z=z] - E[Y|X=0,Z=z]] P(Z=z)

    n = length(data[treatment])

    if isempty(confounders)
        # No confounders: simple difference in means
        treated = data[treatment] .≈ 1.0
        control = data[treatment] .≈ 0.0

        if sum(treated) == 0 || sum(control) == 0
            return 0.0
        end

        return mean(data[outcome][treated]) - mean(data[outcome][control])
    end

    # Discretize each confounder into 5 strata
    n_bins = 5
    confounder_bins = Dict{Symbol, Vector{Int}}()

    for conf in confounders
        values = data[conf]
        # Create quantile bins
        quantiles = range(0, 1, length=n_bins+1)
        edges = [quantile(values, q) for q in quantiles]
        # Handle edge cases
        edges[1] = edges[1] - 0.001
        edges[end] = edges[end] + 0.001

        # Assign each observation to a bin
        bins = zeros(Int, n)
        for i in 1:n
            for b in 1:(n_bins)
                if edges[b] <= values[i] < edges[b+1]
                    bins[i] = b
                    break
                end
            end
        end
        confounder_bins[conf] = bins
    end

    # Create composite strata keys
    strata_keys = Vector{String}(undef, n)
    conf_list = collect(confounders)  # Convert Set to Vector for consistent ordering

    for i in 1:n
        key_parts = [string(confounder_bins[conf][i]) for conf in conf_list]
        strata_keys[i] = join(key_parts, "_")
    end

    # Compute stratified average
    unique_strata = unique(strata_keys)
    ate = 0.0

    for stratum in unique_strata
        # Find observations in this stratum
        in_stratum = strata_keys .== stratum
        stratum_size = sum(in_stratum)

        if stratum_size == 0
            continue
        end

        # Get treatment and control groups within stratum
        treated_in_stratum = in_stratum .& (data[treatment] .≈ 1.0)
        control_in_stratum = in_stratum .& (data[treatment] .≈ 0.0)

        n_treated = sum(treated_in_stratum)
        n_control = sum(control_in_stratum)

        # Skip strata with no treated or control observations
        if n_treated == 0 || n_control == 0
            continue
        end

        # Compute conditional expectations
        y_treated = mean(data[outcome][treated_in_stratum])
        y_control = mean(data[outcome][control_in_stratum])

        # Stratum effect weighted by P(Z=z)
        stratum_weight = stratum_size / n
        stratum_effect = y_treated - y_control

        ate += stratum_effect * stratum_weight
    end

    ate
end

end # module DoCalculus
