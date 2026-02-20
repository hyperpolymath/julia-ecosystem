# SPDX-License-Identifier: PMPL-1.0-or-later
"""
Propensity score methods for causal inference from observational data.

Propensity scores balance treatment and control groups on observed covariates,
mimicking randomization when experiments are infeasible.
"""
module PropensityScore

using Statistics
using StatsBase
using LinearAlgebra

export propensity_score, matching, inverse_probability_weighting
export stratification, doubly_robust

"""
    propensity_score(treatment, covariates)

Estimate propensity score: P(treatment=1 | covariates).
Uses logistic regression (simplified implementation).
"""
function propensity_score(treatment::AbstractVector{Bool}, covariates::Matrix{Float64})
    # Logistic regression using iteratively reweighted least squares (IRLS)
    # Model: P(treatment=1 | X) = logistic(X*β) = 1 / (1 + exp(-X*β))

    n = length(treatment)
    k = size(covariates, 2)

    # Add intercept column
    X = hcat(ones(n), covariates)

    # Initialize coefficients
    β = zeros(k + 1)

    # Convert treatment to Float64 for numerical stability
    y = Float64.(treatment)

    # IRLS iterations
    max_iter = 25
    tol = 1e-8

    for iter in 1:max_iter
        # Compute predictions: p = 1 / (1 + exp(-X*β))
        linear_pred = X * β
        # Numerical stability: clip to avoid overflow
        linear_pred = clamp.(linear_pred, -20.0, 20.0)
        p = 1.0 ./ (1.0 .+ exp.(-linear_pred))

        # Compute weights: W = diag(p .* (1 - p))
        w = p .* (1.0 .- p)
        # Avoid division by zero
        w = clamp.(w, 1e-10, Inf)

        # Working response: z = X*β + (y - p) / w
        # But we solve directly: β_new = β + (X'WX)^-1 * X' * (y - p)

        # Weighted design matrix
        W = Diagonal(w)
        XtWX = X' * W * X

        # Add small regularization for numerical stability
        XtWX += 1e-6 * I

        # Gradient
        grad = X' * (y .- p)

        # Newton update
        try
            β_new = β + XtWX \ grad

            # Check convergence
            if maximum(abs.(β_new - β)) < tol
                β = β_new
                break
            end

            β = β_new
        catch e
            # If matrix is singular, stop iteration
            break
        end
    end

    # Compute final propensity scores
    linear_pred = X * β
    linear_pred = clamp.(linear_pred, -20.0, 20.0)
    p = 1.0 ./ (1.0 .+ exp.(-linear_pred))

    # Clip propensity scores to avoid extreme weights in IPW
    p = clamp.(p, 0.01, 0.99)

    return p
end

"""
    matching(treatment, outcome, propensity; method=:nearest, caliper=0.1)

Propensity score matching: pair treated and control units.
Returns (matched_indices, treatment_effect, std_error).
"""
function matching(
    treatment::AbstractVector{Bool},
    outcome::Vector{Float64},
    propensity::Vector{Float64};
    method::Symbol = :nearest,
    caliper::Float64 = 0.1
)
    treated_idx = findall(treatment)
    control_idx = findall(.!treatment)

    matches = Tuple{Int, Int}[]

    for t_idx in treated_idx
        # Find nearest control
        distances = abs.(propensity[control_idx] .- propensity[t_idx])
        best_match_pos = argmin(distances)
        best_distance = distances[best_match_pos]

        if best_distance <= caliper
            c_idx = control_idx[best_match_pos]
            push!(matches, (t_idx, c_idx))
        end
    end

    # Estimate treatment effect
    effects = [outcome[t] - outcome[c] for (t, c) in matches]
    ate = mean(effects)
    se = std(effects) / sqrt(length(effects))

    (matches, ate, se)
end

"""
    inverse_probability_weighting(treatment, outcome, propensity)

IPW estimator: weight by inverse propensity score.
Returns (ATE, std_error).
"""
function inverse_probability_weighting(
    treatment::AbstractVector{Bool},
    outcome::Vector{Float64},
    propensity::Vector{Float64}
)
    n = length(treatment)

    # IPW weights
    weights = treatment ./ propensity .+ (1 .- treatment) ./ (1 .- propensity)

    # Weighted mean difference
    treated_weighted = sum(outcome .* treatment ./ propensity) / sum(treatment ./ propensity)
    control_weighted = sum(outcome .* (1 .- treatment) ./ (1 .- propensity)) / sum((1 .- treatment) ./ (1 .- propensity))

    ate = treated_weighted - control_weighted

    # Simplified standard error
    se = sqrt(var(outcome .* weights) / n)

    (ate, se)
end

"""
    stratification(treatment, outcome, propensity; n_strata=5)

Stratify by propensity score quintiles and estimate ATE.
"""
function stratification(
    treatment::AbstractVector{Bool},
    outcome::Vector{Float64},
    propensity::Vector{Float64};
    n_strata::Int = 5
)
    # Create strata by propensity quantiles
    quantiles = range(0, 1, length=n_strata+1)
    strata_bounds = [quantile(propensity, q) for q in quantiles]

    stratum_effects = Float64[]
    stratum_weights = Float64[]

    for s in 1:n_strata
        lower = strata_bounds[s]
        upper = strata_bounds[s+1]

        in_stratum = (propensity .>= lower) .& (propensity .<= upper)

        treated_in_stratum = in_stratum .& treatment
        control_in_stratum = in_stratum .& (.!treatment)

        if sum(treated_in_stratum) > 0 && sum(control_in_stratum) > 0
            effect = mean(outcome[treated_in_stratum]) - mean(outcome[control_in_stratum])
            weight = sum(in_stratum) / length(propensity)

            push!(stratum_effects, effect)
            push!(stratum_weights, weight)
        end
    end

    # Weighted average across strata
    ate = sum(stratum_effects .* stratum_weights)
    (ate, stratum_effects, stratum_weights)
end

"""
    doubly_robust(treatment, outcome, propensity, outcome_model_1, outcome_model_0)

Doubly robust estimator: consistent if either propensity or outcome model correct.

Uses Augmented IPW (AIPW) formula:
ATE = (1/n) * sum [
    (T_i * Y_i / e_i) - ((T_i - e_i) / e_i) * μ_1(X_i)
    - ((1-T_i) * Y_i / (1-e_i)) + ((T_i - e_i) / (1-e_i)) * μ_0(X_i)
]

where:
- T_i: treatment indicator
- Y_i: observed outcome
- e_i: propensity score
- μ_1(X_i): predicted outcome under treatment
- μ_0(X_i): predicted outcome under control
"""
function doubly_robust(
    treatment::AbstractVector{Bool},
    outcome::Vector{Float64},
    propensity::Vector{Float64},
    outcome_model_1::Vector{Float64},  # Predicted outcomes under treatment
    outcome_model_0::Vector{Float64}   # Predicted outcomes under control
)
    n = length(treatment)
    T = Float64.(treatment)

    # AIPW formula
    # Term 1: IPW for treated
    term1 = (T .* outcome) ./ propensity

    # Term 2: Bias correction for treated
    term2 = ((T .- propensity) ./ propensity) .* outcome_model_1

    # Term 3: IPW for control
    term3 = ((1.0 .- T) .* outcome) ./ (1.0 .- propensity)

    # Term 4: Bias correction for control
    term4 = ((T .- propensity) ./ (1.0 .- propensity)) .* outcome_model_0

    # Combine terms
    ate = mean(term1 .- term2 .- term3 .+ term4)

    ate
end

end # module PropensityScore
