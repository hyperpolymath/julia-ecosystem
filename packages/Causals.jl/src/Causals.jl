# SPDX-License-Identifier: PMPL-1.0-or-later
"""
    Causals

Causal inference framework providing do-calculus, counterfactual reasoning, mediation
analysis, and multiple causal assessment methods. Integrates Dempster-Shafer belief
functions, Bradford-Hill criteria, Granger causality, and propensity score matching.

# Key Features
- DAG-based causal graphs with d-separation and backdoor criterion
- Do-calculus interventions and effect identification
- Counterfactual queries (probability of necessity/sufficiency)
- Natural direct/indirect effects (mediation analysis)
- Applied Information Economics (EVOI, uncertainty reduction)

# Example
```julia
using Causals
g = CausalGraph([:X, :Y, :Z])
add_edge!(g, :X, :Y)
d_separation(g, :X, :Z, [:Y])
```
"""
module Causals

include("backends/abstract.jl")
include("CausalDAG.jl")
include("DoCalculus.jl")
include("Counterfactuals.jl")
include("Mediation.jl")
include("AIE.jl")
include("ConsensusEngine.jl")
include("CognitiveCausality.jl") # Sloman-inspired
include("DempsterShafer.jl")
include("BradfordHill.jl")
include("Granger.jl")
include("PropensityScore.jl")

using .CausalDAG
using .DoCalculus
using .Counterfactuals
using .Mediation
using .AIE
using .ConsensusEngine
using .CognitiveCausality
using .DempsterShafer
using .BradfordHill
using .Granger
using .PropensityScore

# Re-export key operations
export CausalGraph, add_edge!, d_separation, backdoor_criterion
export do_intervention, identify_effect, Query
export counterfactual_query, probability_of_necessity, probability_of_sufficiency
export natural_direct_effect, natural_indirect_effect
export evoi, reduce_uncertainty # AIE: Applied Information Economics
export causal_consensus, ConsensusReport # Automated Inference Engine
export score_explanatory_depth, predict_intervention_effect # Sloman-inspired
export MassAssignment, combine_dempster, belief, plausibility
export BradfordHillCriteria, assess_causality
export granger_test
export propensity_score, matching

end # module Causals
