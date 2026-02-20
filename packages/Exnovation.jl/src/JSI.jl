# SPDX-License-Identifier: PMPL-1.0-or-later
"""
Just Sustainability Index (JSI): Evaluates the intersection of social equity 
and environmental sustainability to inform exnovation decisions.
Based on Agyeman's framework.
"""
module JSI

export JustSustainabilityIndex, evaluate_jsi

struct JustSustainabilityIndex
    equity_score::Float64      # 0.0 to 1.0 (Fairness/Justice)
    environment_score::Float64 # 0.0 to 1.0 (Resource preservation)
    economy_score::Float64     # 0.0 to 1.0 (Viability)
end

"""
    evaluate_jsi(jsi)
Returns a weighted aggregate score. High scores justify the exnovation 
of current unsustainable alternatives.
"""
function evaluate_jsi(j::JustSustainabilityIndex)
    # Balanced weighting: Equity and Environment prioritized over pure Economy
    return (0.4 * j.equity_score) + (0.4 * j.environment_score) + (0.2 * j.economy_score)
end

end # module
