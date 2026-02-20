# SPDX-License-Identifier: PMPL-1.0-or-later
"""
Bradford Hill criteria for assessing causality in observational studies.

Sir Austin Bradford Hill's 1965 criteria for evaluating causal relationships
when randomized experiments are impractical or unethical.
"""
module BradfordHill

export BradfordHillCriteria, assess_causality, strength_of_evidence

"""
    BradfordHillCriteria

Nine criteria for causal assessment (scored 0-1 each):

1. **Strength**: Magnitude of association (strong correlations more likely causal)
2. **Consistency**: Reproducibility across studies, populations, contexts
3. **Specificity**: One cause → one effect (more specific = stronger)
4. **Temporality**: Cause precedes effect (REQUIRED for causality)
5. **BiologicalGradient**: Dose-response relationship
6. **Plausibility**: Biological/mechanistic plausibility
7. **Coherence**: Fits with known facts and theory
8. **Experiment**: Experimental evidence (strongest)
9. **Analogy**: Similar cause-effect relationships exist

# Example
```julia
criteria = BradfordHillCriteria(
    strength = 0.8,         # Strong correlation
    consistency = 0.9,      # Replicated many times
    specificity = 0.5,      # Moderate specificity
    temporality = 1.0,      # MUST have: cause before effect
    biological_gradient = 0.7,
    plausibility = 0.8,
    coherence = 0.7,
    experiment = 0.0,       # No RCT available
    analogy = 0.6
)

strength = strength_of_evidence(criteria)  # 0.0-1.0
```
"""
struct BradfordHillCriteria
    strength::Float64                # Magnitude of association
    consistency::Float64             # Reproducibility
    specificity::Float64             # Specificity of effect
    temporality::Float64             # Temporal precedence (REQUIRED)
    biological_gradient::Float64     # Dose-response
    plausibility::Float64            # Biological plausibility
    coherence::Float64               # Fits known facts
    experiment::Float64              # Experimental evidence
    analogy::Float64                 # Analogous relationships

    function BradfordHillCriteria(;
        strength = 0.0,
        consistency = 0.0,
        specificity = 0.0,
        temporality = 0.0,
        biological_gradient = 0.0,
        plausibility = 0.0,
        coherence = 0.0,
        experiment = 0.0,
        analogy = 0.0
    )
        for (name, val) in [
            (:strength, strength), (:consistency, consistency),
            (:specificity, specificity), (:temporality, temporality),
            (:biological_gradient, biological_gradient), (:plausibility, plausibility),
            (:coherence, coherence), (:experiment, experiment), (:analogy, analogy)
        ]
            if !(0.0 <= val <= 1.0)
                error("$name must be in [0,1], got $val")
            end
        end

        new(strength, consistency, specificity, temporality,
            biological_gradient, plausibility, coherence, experiment, analogy)
    end
end

"""
    assess_causality(criteria::BradfordHillCriteria)

Assess whether evidence supports causality.
Returns (verdict, confidence) where verdict ∈ {:strong, :moderate, :weak, :none}
"""
function assess_causality(c::BradfordHillCriteria)
    # Temporality is REQUIRED
    if c.temporality < 0.5
        return (:none, 0.0)  # Cannot establish causality without temporal precedence
    end

    # Weighted scoring (experiment and temporality most important)
    weights = (
        strength = 1.0,
        consistency = 1.5,
        specificity = 0.8,
        temporality = 2.0,  # CRITICAL
        biological_gradient = 1.2,
        plausibility = 1.0,
        coherence = 1.0,
        experiment = 2.0,   # GOLD STANDARD
        analogy = 0.5
    )

    score = (
        c.strength * weights.strength +
        c.consistency * weights.consistency +
        c.specificity * weights.specificity +
        c.temporality * weights.temporality +
        c.biological_gradient * weights.biological_gradient +
        c.plausibility * weights.plausibility +
        c.coherence * weights.coherence +
        c.experiment * weights.experiment +
        c.analogy * weights.analogy
    )

    max_score = sum(values(weights))
    confidence = score / max_score

    verdict = if confidence >= 0.75
        :strong
    elseif confidence >= 0.55
        :moderate
    elseif confidence >= 0.35
        :weak
    else
        :insufficient
    end

    (verdict, confidence)
end

"""
    strength_of_evidence(criteria)

Return overall strength of causal evidence as a score ∈ [0,1].
"""
function strength_of_evidence(c::BradfordHillCriteria)
    _, confidence = assess_causality(c)
    confidence
end

end # module BradfordHill
