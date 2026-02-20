# SPDX-License-Identifier: PMPL-1.0-or-later
"""
Basic Usage Example for Causals.jl

This example demonstrates the core functionality of the Causals package,
including Dempster-Shafer evidence combination, Bradford Hill criteria assessment,
and simple causal DAG operations.
"""

using Causals
using Causals.DempsterShafer
using Causals.BradfordHill
using Causals.CausalDAG

println("=" ^ 60)
println("Causals.jl - Basic Usage Example")
println("=" ^ 60)
println()

# Example 1: Dempster-Shafer Evidence Combination
println("1. Dempster-Shafer Evidence Combination")
println("-" ^ 40)

# Create evidence from two independent sources about a medical diagnosis
# Frame of discernment: {disease_a, disease_b, disease_c}
frame = [:disease_a, :disease_b, :disease_c]

# Source 1: Lab test suggests disease_a or disease_b with 0.7 confidence
evidence1 = MassAssignment(
    frame,
    Dict(
        Set([:disease_a, :disease_b]) => 0.7,
        Set(frame) => 0.3  # Remaining uncertainty
    )
)

# Source 2: Symptoms suggest disease_b or disease_c with 0.6 confidence
evidence2 = MassAssignment(
    frame,
    Dict(
        Set([:disease_b, :disease_c]) => 0.6,
        Set(frame) => 0.4  # Remaining uncertainty
    )
)

# Combine evidence using Dempster's rule
combined = combine_dempster(evidence1, evidence2)

println("Individual disease beliefs:")
println("  disease_a: belief=$(round(belief(combined, Set([:disease_a])), digits=3))")
println("  disease_b: belief=$(round(belief(combined, Set([:disease_b])), digits=3))")
println("  disease_c: belief=$(round(belief(combined, Set([:disease_c])), digits=3))")
println()

# Pignistic transformation for decision-making
probs = pignistic_transform(combined)
println("Decision probabilities (pignistic):")
for (disease, prob) in sort(collect(probs), by=x->x[2], rev=true)
    println("  $(disease): $(round(prob, digits=3))")
end
println()

# Example 2: Bradford Hill Causal Assessment
println("2. Bradford Hill Causal Assessment")
println("-" ^ 40)

# Assess potential causal relationship: smoking → lung cancer
assessment = BradfordHillCriteria(
    strength = 0.9,          # Strong association (RR >> 1)
    consistency = 0.95,      # Consistent across many studies
    specificity = 0.7,       # Somewhat specific (other factors exist)
    temporality = 1.0,       # Clear temporal relationship
    biological_gradient = 0.85,  # Clear dose-response
    plausibility = 0.9,      # Biologically plausible mechanisms
    coherence = 0.9,         # Coherent with existing knowledge
    experiment = 0.6,        # Some experimental evidence (animal studies)
    analogy = 0.7           # Analogous to other carcinogen relationships
)

verdict, confidence = assess_causality(assessment)
evidence_level = strength_of_evidence(assessment)

println("Smoking → Lung Cancer Assessment:")
println("  Causality verdict: $(verdict)")
println("  Confidence: $(round(confidence, digits=3))")
println("  Evidence strength: $(evidence_level)")
println()

# Example 3: Simple Causal DAG Operations
println("3. Causal DAG Operations")
println("-" ^ 40)

# Build a simple causal graph:
#   Education → Income
#   Education → Health
#   Income → Health
#   Exercise → Health
cg = CausalGraph([:Education, :Income, :Health, :Exercise])

CausalDAG.add_edge!(cg, :Education, :Income)   # Education → Income
CausalDAG.add_edge!(cg, :Education, :Health)   # Education → Health
CausalDAG.add_edge!(cg, :Income, :Health)      # Income → Health
CausalDAG.add_edge!(cg, :Exercise, :Health)    # Exercise → Health

println("Causal graph structure:")
println("  Nodes: Education, Income, Health, Exercise")
println("  Edges: 4 (Education→Income, Education→Health, Income→Health, Exercise→Health)")
println()

# Check d-separation: Is Education ⊥ Exercise given Income?
is_dsep = d_separation(cg, Set([:Education]), Set([:Exercise]), Set([:Income]))
println("D-separation test:")
println("  Education ⊥ Exercise | Income? $(is_dsep)")
println()

# Find ancestors and descendants
println("Causal relationships:")
println("  Ancestors of Health: $(sort(collect(ancestors(cg, :Health))))")
println("  Descendants of Education: $(sort(collect(descendants(cg, :Education))))")
println()

# Check backdoor criterion for Income → Health effect
backdoor_ok = backdoor_criterion(cg, :Income, :Health, Set([:Education]))
println("Backdoor criterion:")
println("  Can estimate Income → Health effect controlling for Education? $(backdoor_ok)")
println()

println("=" ^ 60)
println("Basic usage example completed successfully!")
println("=" ^ 60)
