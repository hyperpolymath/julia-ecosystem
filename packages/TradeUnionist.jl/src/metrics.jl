# SPDX-License-Identifier: PMPL-1.0-or-later
module Metrics

using ..Types
using DataFrames
using Statistics
using StatsBase

export UnionMetrics, calc_density, calc_coverage, calc_leadership_ratio, wage_gini_coefficient

struct UnionMetrics
    site_id::Symbol
    density::Float64
    coverage::Float64
    leadership_ratio::Float64
    avg_wage_premium::Float64
end

"""
    calc_density(members::Int, total_eligible::Int)
Union density = (Members / Total Eligible Workers) * 100
"""
calc_density(m, t) = t == 0 ? 0.0 : (m / t) * 100

"""
    calc_coverage(covered_workers::Int, total_workforce::Int)
Collective bargaining coverage rate.
"""
calc_coverage(c, t) = t == 0 ? 0.0 : (c / t) * 100

"""
    calc_leadership_ratio(leaders::Int, members::Int)
Standard IR metric: Number of shop stewards/leaders per member.
"""
calc_leadership_ratio(l, m) = m == 0 ? 0.0 : (l / m)

"""
    wage_gini_coefficient(wages::Vector{Float64})
Calculates the Gini coefficient for wages in a unit. 
A lower Gini indicates higher wage equality (a key union goal).
"""
function wage_gini_coefficient(wages::Vector{Float64})
    if isempty(wages) return 0.0 end
    sorted_wages = sort(wages)
    n = length(wages)
    sum_diff = sum([ (2i - n - 1) * w for (i, w) in enumerate(sorted_wages) ])
    return sum_diff / (n * sum(wages))
end

"""
    grievance_rate(case_count::Int, headcount::Int)
Grievances per 100 workers. High rates may indicate management hostility or contract ambiguity.
"""
grievance_rate(cases, head) = head == 0 ? 0.0 : (cases / head) * 100

end # module
