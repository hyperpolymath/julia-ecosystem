# SPDX-License-Identifier: PMPL-1.0-or-later
# Copyright (c) 2026 Jonathan D.A. Jewell <jonathan.jewell@open.ac.uk>

"""
    demographic_parity(predictions::AbstractVector, protected_attributes::AbstractVector)::Float64

Calculates the demographic parity disparity.
"""
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
