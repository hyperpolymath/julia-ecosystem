# SPDX-License-Identifier: PMPL-1.0-or-later
# Copyright (c) 2026 Jonathan D.A. Jewell <jonathan.jewell@open.ac.uk>

function demographic_parity(predictions::AbstractVector, protected_attributes::AbstractVector)::Float64
    @assert length(predictions) == length(protected_attributes) "Lengths must match."
    unique_groups = unique(protected_attributes)
    if length(unique_groups) < 2
        return 0.0
    end
    group_rates = Dict{Any,Float64}()
    for group in unique_groups
        group_mask = protected_attributes .== group
        group_preds = predictions[group_mask]
        group_rates[group] = isempty(group_preds) ? 0.0 : mean(group_preds)
    end
    rates = collect(values(group_rates))
    return maximum(rates) - minimum(rates)
end

function equalized_odds(predictions::AbstractVector{<:Real}, labels::AbstractVector{<:Real},
                        protected_attributes::AbstractVector)::Float64
    @assert length(predictions) == length(labels) == length(protected_attributes) "Lengths must match."
    unique_groups = unique(protected_attributes)
    if length(unique_groups) < 2
        return 0.0
    end
    tpr_disparities = Float64[]
    fpr_disparities = Float64[]
    for group in unique_groups
        group_mask = protected_attributes .== group
        group_preds = predictions[group_mask]
        group_labels = labels[group_mask]
        tp = sum((group_preds .== 1) .& (group_labels .== 1))
        fp = sum((group_preds .== 1) .& (group_labels .== 0))
        tn = sum((group_preds .== 0) .& (group_labels .== 0))
        fn = sum((group_preds .== 0) .& (group_labels .== 1))
        tpr = (tp + fn) > 0 ? tp / (tp + fn) : 0.0
        fpr = (fp + tn) > 0 ? fp / (fp + tn) : 0.0
        push!(tpr_disparities, tpr)
        push!(fpr_disparities, fpr)
    end
    max_tpr_disparity = maximum(tpr_disparities) - minimum(tpr_disparities)
    max_fpr_disparity = maximum(fpr_disparities) - minimum(fpr_disparities)
    return max(max_tpr_disparity, max_fpr_disparity)
end

function equal_opportunity(predictions::AbstractVector{<:Real}, labels::AbstractVector{<:Real},
                          protected_attributes::AbstractVector)::Float64
    @assert length(predictions) == length(labels) == length(protected_attributes) "Lengths must match."
    unique_groups = unique(protected_attributes)
    if length(unique_groups) < 2
        return 0.0
    end
    tprs = Float64[]
    for group in unique_groups
        group_mask = protected_attributes .== group
        group_preds = predictions[group_mask]
        group_labels = labels[group_mask]
        tp = sum((group_preds .== 1) .& (group_labels .== 1))
        fn = sum((group_preds .== 0) .& (group_labels .== 1))
        tpr = (tp + fn) > 0 ? tp / (tp + fn) : 0.0
        push!(tprs, tpr)
    end
    return maximum(tprs) - minimum(tprs)
end

function disparate_impact(predictions::AbstractVector, protected_attributes::AbstractVector)::Float64
    @assert length(predictions) == length(protected_attributes) "Lengths must match."
    unique_groups = unique(protected_attributes)
    if length(unique_groups) < 2
        return 1.0
    end
    group_rates = Dict{Any,Float64}()
    for group in unique_groups
        group_mask = protected_attributes .== group
        group_preds = predictions[group_mask]
        group_rates[group] = isempty(group_preds) ? 0.0 : mean(group_preds)
    end
    rates = collect(values(group_rates))
    min_rate = minimum(rates)
    max_rate = maximum(rates)
    return max_rate > 0.0 ? min_rate / max_rate : 1.0
end

function individual_fairness(predictions::AbstractVector, similarity_matrix::AbstractMatrix;
                            similarity_threshold::Float64 = 0.8)::Float64
    n = length(predictions)
    @assert size(similarity_matrix) == (n, n) "Similarity matrix must be n×n."
    @assert similarity_threshold >= 0.0 && similarity_threshold <= 1.0 "Threshold must be in [0, 1]."
    total_diff = 0.0
    count = 0
    for i in 1:n
        for j in (i+1):n
            if similarity_matrix[i, j] > similarity_threshold
                total_diff += abs(predictions[i] - predictions[j])
                count += 1
            end
        end
    end
    return count > 0 ? total_diff / count : 0.0
end

function satisfy(value::Fairness, state::Dict)::Bool
    predictions = get(state, :predictions, nothing)
    protected = get(state, :protected, get(state, :protected_attributes, nothing))
    labels = get(state, :labels, nothing)
    similarity_matrix = get(state, :similarity_matrix, nothing)

    isnothing(predictions) && error("State must contain :predictions for fairness evaluation.")
    isnothing(protected) && value.metric != :individual_fairness && error("State must contain :protected for group fairness.")
    isnothing(similarity_matrix) && value.metric == :individual_fairness && error("State must contain :similarity_matrix for individual fairness.")

    disparity = if value.metric == :demographic_parity
        demographic_parity(predictions, protected)
    elseif value.metric == :equalized_odds
        isnothing(labels) && error("equalized_odds metric requires :labels in state.")
        equalized_odds(predictions, labels, protected)
    elseif value.metric == :equal_opportunity
        isnothing(labels) && error("equal_opportunity metric requires :labels in state.")
        equal_opportunity(predictions, labels, protected)
    elseif value.metric == :disparate_impact
        di_ratio = disparate_impact(predictions, protected)
        return di_ratio >= value.threshold
    elseif value.metric == :individual_fairness
        individual_fairness(predictions, similarity_matrix)
    else
        error("Unknown fairness metric: $(value.metric).")
    end

    return disparity <= value.threshold
end
