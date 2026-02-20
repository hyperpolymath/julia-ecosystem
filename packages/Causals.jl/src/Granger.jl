# SPDX-License-Identifier: PMPL-1.0-or-later
"""
Granger causality for time series analysis.

Tests whether one time series helps predict another. "X Granger-causes Y"
means past values of X contain information about Y beyond what's in Y's own past.
"""
module Granger

using LinearAlgebra
using Statistics
using Distributions

export granger_test, granger_causality, optimal_lag, bidirectional_granger

"""
    granger_test(x, y, max_lag=10; α=0.05)

Test if x Granger-causes y using F-test.
Returns (causes, F_stat, p_value, optimal_lag).
"""
function granger_test(x::Vector{Float64}, y::Vector{Float64}, max_lag::Int=10; α::Float64=0.05)
    length(x) == length(y) || error("Series must have same length")
    n = length(x)

    # Find optimal lag by AIC
    best_lag = optimal_lag(x, y, max_lag)

    # Restricted model: y ~ lag(y)
    X_restricted = lag_matrix(y, best_lag)
    y_restricted = y[best_lag+1:end]
    β_restricted = X_restricted \ y_restricted
    rss_restricted = sum((y_restricted - X_restricted * β_restricted).^2)

    # Unrestricted model: y ~ lag(y) + lag(x)
    X_y = lag_matrix(y, best_lag)
    X_x = lag_matrix(x, best_lag)
    X_unrestricted = hcat(X_y, X_x)
    y_unrestricted = y[best_lag+1:end]
    β_unrestricted = X_unrestricted \ y_unrestricted
    rss_unrestricted = sum((y_unrestricted - X_unrestricted * β_unrestricted).^2)

    # F-test
    k = best_lag  # restrictions (number of x lags)
    n_obs = length(y_unrestricted)
    F_stat = ((rss_restricted - rss_unrestricted) / k) / (rss_unrestricted / (n_obs - 2*best_lag - 1))

    # Compute p-value using proper F-distribution
    df1 = k  # numerator degrees of freedom
    df2 = n_obs - 2 * best_lag - 1  # denominator degrees of freedom
    f_dist = FDist(df1, df2)
    p_value = 1.0 - cdf(f_dist, F_stat)

    causes = p_value < 0.05  # Standard significance level

    (causes, F_stat, p_value, best_lag)
end

"""
    lag_matrix(x, p)

Create lag matrix with p lags: [x[t-1] x[t-2] ... x[t-p]].
"""
function lag_matrix(x::Vector{Float64}, p::Int)
    n = length(x)
    X = zeros(n - p, p)
    for i in 1:p
        X[:, i] = x[p+1-i:n-i]
    end
    X
end

"""
    optimal_lag(x, y, max_lag)

Find optimal lag length using Akaike Information Criterion (AIC).
"""
function optimal_lag(x::Vector{Float64}, y::Vector{Float64}, max_lag::Int)
    best_aic = Inf
    best_lag = 1

    for lag in 1:max_lag
        X_y = lag_matrix(y, lag)
        X_x = lag_matrix(x, lag)
        X = hcat(X_y, X_x)
        y_vec = y[lag+1:end]

        β = X \ y_vec
        residuals = y_vec - X * β
        rss = sum(residuals.^2)

        n = length(y_vec)
        k = 2 * lag
        aic = n * log(rss / n) + 2 * k

        if aic < best_aic
            best_aic = aic
            best_lag = lag
        end
    end

    best_lag
end

"""
    granger_causality(x, y, max_lag=10)

Return strength of Granger causality (0-1) from x to y.
"""
function granger_causality(x::Vector{Float64}, y::Vector{Float64}, max_lag::Int=10)
    causes, F_stat, _, _ = granger_test(x, y, max_lag)
    causes ? min(F_stat / 10.0, 1.0) : 0.0  # Normalize F-stat
end

"""
    bidirectional_granger(x, y, max_lag=10)

Test Granger causality in both directions.
Returns (x→y, y→x) as strength scores.
"""
function bidirectional_granger(x::Vector{Float64}, y::Vector{Float64}, max_lag::Int=10)
    xy = granger_causality(x, y, max_lag)
    yx = granger_causality(y, x, max_lag)
    (xy, yx)
end

end # module Granger
