# SPDX-License-Identifier: PMPL-1.0-or-later
# Copyright (c) 2026 Jonathan D.A. Jewell <jonathan.jewell@open.ac.uk>

using Statistics

function utilitarian_welfare(utilities::AbstractVector{<:Real})::Float64
    isempty(utilities) && return 0.0 # Return 0.0 for empty utility vector
    return sum(utilities)
end

function rawlsian_welfare(utilities::AbstractVector{<:Real})::Float64
    isempty(utilities) && error("Cannot compute Rawlsian welfare for an empty utility vector.")
    return minimum(utilities)
end

function egalitarian_welfare(utilities::AbstractVector{<:Real})::Float64
    if length(utilities) < 2
        # Variance of a single element or empty set is undefined or 0 (which is equality).
        # For meaningful comparison, we might want to error or return 0.0
        # for zero variance for length 1.
        return 0.0 # No inequality if only one or zero individuals
    end
    return -var(utilities)
end

function satisfy(value::Welfare, state::Dict)::Bool
    utilities = get(state, :utilities, nothing)
    isnothing(utilities) && error("State must contain :utilities for Welfare satisfaction check.")

    min_welfare = get(state, :min_welfare, 0.0) # Default minimum welfare to 0.0 if not specified

    computed_welfare = if value.metric == :utilitarian
        utilitarian_welfare(utilities)
    elseif value.metric == :rawlsian
        rawlsian_welfare(utilities)
    elseif value.metric == :egalitarian
        egalitarian_welfare(utilities)
    else
        error("Unknown welfare metric: $(value.metric). Must be one of $(instances(WelfareMetric)).")
    end

    return computed_welfare >= min_welfare
end

function satisfy(value::Profit, state::Dict)::Bool
    current_profit = get(state, :profit, nothing) # Use nothing to distinguish from actual 0.0 profit
    isnothing(current_profit) && error("State must contain :profit for Profit satisfaction check.")

    # Check profit target
    profit_ok = current_profit >= value.target

    # Check all constraints
    constraints_ok = all(satisfy(c, state) for c in value.constraints)

    return profit_ok && constraints_ok
end

function satisfy(value::Efficiency, state::Dict)::Bool
    if value.metric == :computation_time
        time = get(state, :computation_time, nothing)
        isnothing(time) && error("State must contain :computation_time for :computation_time efficiency check.")
        return time <= value.target
    elseif value.metric == :pareto
        is_pareto = get(state, :is_pareto_efficient, nothing)
        isnothing(is_pareto) && error("State must contain :is_pareto_efficient for :pareto efficiency check.")
        return is_pareto
    elseif value.metric == :kaldor_hicks
        net_gain = get(state, :net_gain, nothing)
        isnothing(net_gain) && error("State must contain :net_gain for :kaldor_hicks efficiency check.")
        return net_gain >= value.target
    else
        error("Unknown efficiency metric: $(value.metric). Must be one of $(instances(EfficiencyMetric)).")
    end
end

function satisfy(value::Safety, state::Dict)::Bool
    # Check if safety invariant is satisfied in state
    is_safe = get(state, :is_safe, true) # Optimistic assumption if not provided
    invariant_holds = get(state, :invariant_holds, true) # Optimistic assumption if not provided

    return is_safe && invariant_holds
end

function maximize(value::Welfare, initial_state::Dict)::Float64
    utilities = get(initial_state, :utilities, nothing)
    isnothing(utilities) && error("initial_state must contain :utilities for Welfare maximization.")

    if value.metric == :utilitarian
        return utilitarian_welfare(utilities)
    elseif value.metric == :rawlsian
        return rawlsian_welfare(utilities)
    elseif value.metric == :egalitarian
        return egalitarian_welfare(utilities)
    else
        error("Unknown welfare metric: $(value.metric). Must be one of $(instances(WelfareMetric)).")
    end
end

function maximize(value::Profit, initial_state::Dict)::Float64
    profit = get(initial_state, :profit, nothing)
    isnothing(profit) && error("initial_state must contain :profit for Profit maximization.")
    return profit
end

function maximize(value::Efficiency, initial_state::Dict)::Float64
    if value.metric == :computation_time
        time = get(initial_state, :computation_time, nothing)
        isnothing(time) && error("initial_state must contain :computation_time for :computation_time efficiency maximization.")
        return -time  # Negative because we want to minimize time
    elseif value.metric == :pareto
        is_pareto = get(initial_state, :is_pareto_efficient, nothing)
        isnothing(is_pareto) && error("initial_state must contain :is_pareto_efficient for :pareto efficiency maximization.")
        return is_pareto ? 1.0 : 0.0
    elseif value.metric == :kaldor_hicks
        net_gain = get(initial_state, :net_gain, nothing)
        isnothing(net_gain) && error("initial_state must contain :net_gain for :kaldor_hicks efficiency maximization.")
        return net_gain
    else
        error("Unknown efficiency metric: $(value.metric). Must be one of $(instances(EfficiencyMetric)).")
    end
end

function maximize(value::Fairness, initial_state::Dict)::Float64
    # Extract data from state
    predictions = get(initial_state, :predictions, nothing)
    protected = get(initial_state, :protected, get(initial_state, :protected_attributes, nothing))
    labels = get(initial_state, :labels, nothing)
    similarity_matrix = get(initial_state, :similarity_matrix, nothing)

    if isnothing(predictions) || (isnothing(protected) && value.metric != :individual_fairness) || (isnothing(similarity_matrix) && value.metric == :individual_fairness)
        # Cannot compute fairness score without required data
        return 0.0 # Return a low score to indicate poor fairness or inability to compute
    end

    # Compute disparity based on metric
    if value.metric == :demographic_parity
        disparity = demographic_parity(predictions, protected)
        return 1.0 - disparity  # Convert disparity to maximization score
    elseif value.metric == :equalized_odds
        isnothing(labels) && return 0.0 # Cannot compute without labels
        disparity = equalized_odds(predictions, labels, protected)
        return 1.0 - disparity
    elseif value.metric == :equal_opportunity
        isnothing(labels) && return 0.0 # Cannot compute without labels
        disparity = equal_opportunity(predictions, labels, protected)
        return 1.0 - disparity
    elseif value.metric == :disparate_impact
        return disparate_impact(predictions, protected) # DI is a ratio, 1.0 is best
    elseif value.metric == :individual_fairness
        return 1.0 - individual_fairness(predictions, similarity_matrix) # Convert individual fairness to a maximization score
    else
        error("Unknown fairness metric: $(value.metric). Please ensure it is a valid metric from FairnessMetric enum.")
    end
end

function maximize(value::Safety, initial_state::Dict)::Float64
    is_safe = get(initial_state, :is_safe, true)
    invariant_holds = get(initial_state, :invariant_holds, true)
    return (is_safe && invariant_holds) ? 1.0 : 0.0
end

function verify_value(value::Value, proof::Dict)::Bool
    # Require verified field to be a Bool
    verified = get(proof, :verified, nothing)
    isnothing(verified) && error("proof must contain :verified field")
    isa(verified, Bool) || error("proof[:verified] must be a Bool, got $(typeof(verified))")

    return verified
end

function verify_value(value::Safety, proof::Dict)::Bool
    # Require verified field to be a Bool
    verified = get(proof, :verified, nothing)
    isnothing(verified) && error("proof must contain :verified field")
    isa(verified, Bool) || error("proof[:verified] must be a Bool, got $(typeof(verified))")

    # For critical safety values, require a prover
    if value.critical
        prover = get(proof, :prover, nothing)
        isnothing(prover) && error("Critical safety proofs must contain :prover field")
        # Log prover information if details are available
        details = get(proof, :details, nothing)
        if !isnothing(details)
            @info "Safety proof from prover: $prover" details
        end
    end

    return verified
end

