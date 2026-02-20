# SPDX-License-Identifier: PMPL-1.0-or-later
module Analytics

using DataFrames
using Statistics

export SurveyResult, calc_nps, share_of_voice
export first_order_ratio, second_order_ratio, third_order_ratio

struct SurveyResult
    id::Symbol
    topic::String
    respondents::Int
    scores::Vector{Int} # 1-10 scale
    verbatim::Vector{String}
end

# --- Ratio Analysis Engine ---

"""
    first_order_ratio(successes, attempts)
Direct performance metrics (e.g., Click-through rate, Conversion rate).
"""
first_order_ratio(s, a) = a == 0 ? 0.0 : (s / a)

"""
    second_order_ratio(impact_ratio, cost)
Efficiency metrics (e.g., Sentiment lift per dollar spent).
"""
second_order_ratio(r, c) = c == 0 ? 0.0 : (r / c)

"""
    third_order_ratio(efficiency_a, efficiency_b, delta_time)
Strategic velocity/elasticity (e.g., Acceleration of efficiency over time).
"""
third_order_ratio(ea, eb, dt) = dt == 0 ? 0.0 : (ea - eb) / dt

"""
    calc_nps(survey::SurveyResult)

Calculates Net Promoter Score (NPS) from survey data.
"""
function calc_nps(survey::SurveyResult)
    promoters = count(x -> x >= 9, survey.scores)
    detractors = count(x -> x <= 6, survey.scores)
    total = length(survey.scores)
    
    if total == 0 return 0.0 end
    
    return ((promoters - detractors) / total) * 100
end

"""
    share_of_voice(brand_mentions, total_market_mentions)

Calculates the percentage of market conversation owned by the brand.
"""
function share_of_voice(brand::Int, market::Int)
    if market == 0 return 0.0 end
    return (brand / market) * 100
end

end # module
