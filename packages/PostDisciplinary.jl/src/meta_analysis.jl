# SPDX-License-Identifier: PMPL-1.0-or-later
module MetaAnalysis

using Statistics

export EffectSize, aggregate_effects, heterogeneity_q

struct EffectSize
    value::Float64
    variance::Float64
    weight::Float64
end

"""
    aggregate_effects(studies::Vector{EffectSize})
Calculates the weighted mean effect size (Fixed Effects Model).
"""
function aggregate_effects(studies::Vector{EffectSize})
    total_weight = sum(s.weight for s in studies)
    weighted_sum = sum(s.value * s.weight for s in studies)
    return weighted_sum / total_weight
end

"""
    heterogeneity_q(studies::Vector{EffectSize}, pooled_effect::Float64)
Calculates Cochran's Q statistic to test for heterogeneity across studies.
"""
function heterogeneity_q(studies::Vector{EffectSize}, pooled_effect::Float64)
    return sum(s.weight * (s.value - pooled_effect)^2 for s in studies)
end

end # module
