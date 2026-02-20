# SPDX-License-Identifier: PMPL-1.0-or-later
# Copyright (c) 2026 Jonathan D.A. Jewell <jonathan.jewell@open.ac.uk>

function value_score(value::Value, state::Dict)::Float64
    if value isa Fairness
        predictions = get(state, :predictions, nothing)
        protected = get(state, :protected, nothing)
        labels = get(state, :labels, nothing)
        similarity_matrix = get(state, :similarity_matrix, nothing)

        isnothing(predictions) && error("State must contain :predictions for Fairness value_score.")
        isnothing(protected) && value.metric != :individual_fairness && error("State must contain :protected or :protected_attributes for group Fairness value_score.")
        isnothing(similarity_matrix) && value.metric == :individual_fairness && error("State must contain :similarity_matrix for individual Fairness value_score.")


        disparity = if value.metric == :demographic_parity
            demographic_parity(predictions, protected)
        elseif value.metric == :equalized_odds
            isnothing(labels) && error("State must contain :labels for equalized_odds value_score.")
            equalized_odds(predictions, labels, protected)
        elseif value.metric == :equal_opportunity
            isnothing(labels) && error("State must contain :labels for equal_opportunity value_score.")
            equal_opportunity(predictions, labels, protected)
        elseif value.metric == :disparate_impact
            # For disparate impact, a ratio of 1.0 is optimal. Score is ratio / 1.0 (clamped).
            di_ratio = disparate_impact(predictions, protected)
            return min(1.0, max(0.0, di_ratio)) # Clamped to [0,1], 1.0 is best. Threshold (0.8) should be handled by satisfy
        elseif value.metric == :individual_fairness
            # For individual fairness, 0.0 is optimal (no difference for similar individuals).
            # Convert to a score where 1.0 is optimal. Assume max possible diff is 1.0.
            ind_fairness = individual_fairness(predictions, similarity_matrix)
            return max(0.0, 1.0 - ind_fairness)
        else
            error("Unknown fairness metric: $(value.metric) for value_score.")
        end

        # For disparity metrics, a lower disparity is better. Convert to score where 1.0 is optimal.
        # Normalize disparity relative to threshold. If disparity > threshold, score becomes < 0.
        return max(0.0, 1.0 - disparity / value.threshold)

    elseif value isa Welfare
        utilities = get(state, :utilities, nothing)
        isnothing(utilities) && error("State must contain :utilities for Welfare value_score.")

        # Handle empty utilities gracefully as in welfare functions
        if isempty(utilities)
             welfare_val = 0.0
        else
            welfare_val = if value.metric == :utilitarian
                utilitarian_welfare(utilities)
            elseif value.metric == :rawlsian
                rawlsian_welfare(utilities)
            elseif value.metric == :egalitarian
                egalitarian_welfare(utilities)
            else
                error("Unknown welfare metric: $(value.metric) for value_score.")
            end
        end

        # Normalize welfare value - simple approach: return welfare_val directly
        # More sophisticated normalization could be added based on expected ranges
        return welfare_val

    elseif value isa Profit
        profit = get(state, :profit, nothing)
        isnothing(profit) && error("State must contain :profit for Profit value_score.")
        # Normalize profit relative to target
        return profit / value.target

    elseif value isa Efficiency
        if value.metric == :computation_time
            time = get(state, :computation_time, nothing)
            isnothing(time) && error("State must contain :computation_time for Efficiency value_score.")
            # Lower time is better - invert and normalize
            return max(0.0, 1.0 - time / value.target)
        elseif value.metric == :pareto
            is_pareto = get(state, :is_pareto_efficient, nothing)
            isnothing(is_pareto) && error("State must contain :is_pareto_efficient for Efficiency value_score.")
            return is_pareto ? 1.0 : 0.0
        elseif value.metric == :kaldor_hicks
            net_gain = get(state, :net_gain, nothing)
            isnothing(net_gain) && error("State must contain :net_gain for Efficiency value_score.")
            return net_gain / value.target
        else
            error("Unknown efficiency metric: $(value.metric) for value_score.")
        end

    elseif value isa Safety
        is_safe = get(state, :is_safe, true)
        invariant_holds = get(state, :invariant_holds, true)
        return (is_safe && invariant_holds) ? 1.0 : 0.0

    else
        error("Unknown value type: $(typeof(value)) for value_score.")
    end
end

function weighted_score(values::Vector{<:Value}, state::Dict)::Float64
    total_weight = sum(v.weight for v in values)

    if total_weight == 0.0
        # If all weights are zero, the aggregated score is 0.0 as no value contributes.
        return 0.0
    end

    weighted_sum = sum(value_score(v, state) * v.weight for v in values)
    return weighted_sum / total_weight
end

function normalize_scores(scores::AbstractVector{<:Real})::Vector{Float64}
    if isempty(scores)
        throw(ArgumentError("Cannot normalize an empty vector of scores."))
    end

    min_score = minimum(scores)
    max_score = maximum(scores)

    if max_score == min_score
        return ones(length(scores))
    end

    return [(s - min_score) / (max_score - min_score) for s in scores]
end

function dominated(solution_a::Dict, solution_b::Dict, values::AbstractVector{<:Value})::Bool
    better_on_all = true
    strictly_better_on_one = false

    for value in values
        score_a = value_score(value, solution_a)
        score_b = value_score(value, solution_b)

        if score_b < score_a # solution_b is worse on this value
            better_on_all = false
            break # No need to check further, A is not dominated by B
        elseif score_b > score_a # solution_b is strictly better on this value
            strictly_better_on_one = true
        end
    end

    return better_on_all && strictly_better_on_one
end

function pareto_frontier(solutions::Vector{<:Dict}, values::AbstractVector{<:Value})::Vector{Dict}
    if isempty(solutions)
        return eltype(solutions)[]
    end

    pareto_optimal = eltype(solutions)[]

    for solution in solutions
        is_dominated = false

        for other in solutions
            # Ensure solution !== other to avoid self-comparison
            if solution !== other && dominated(solution, other, values)
                is_dominated = true
                break
            end
        end

        if !is_dominated
            push!(pareto_optimal, solution)
        end
    end

    return pareto_optimal
end

function pareto_frontier(system::Dict, values::AbstractVector{<:Value})::Vector{Dict}
    # Generate candidate solutions by exploring the parameter space
    solutions = Dict[]

    # If system provides candidate solutions, use them
    if haskey(system, :solutions)
        solutions = system[:solutions]
    else
        # Otherwise, just evaluate the current system state
        push!(solutions, system)
    end

    # Call the primary pareto_frontier method
    return pareto_frontier(solutions, values)
end

