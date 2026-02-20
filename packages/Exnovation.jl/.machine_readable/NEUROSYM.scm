;; SPDX-License-Identifier: PMPL-1.0-or-later
;; NEUROSYM.scm for Exnovation.jl
;; Neurosymbolic integration directives

(define neurosymbolic-profile
  '((symbolic-layer
     (formal-specifications
       (type-system . "Julia strong typing")
       (invariants
         "Barrier severity must be in [0,1]"
         "Driver strength must be in [0,1]"
         "ExnovationItem must have at least one barrier or driver"
         "Scores must be non-negative")
       (contracts
         "assess_barriers returns Dict{BarrierKind,Float64}"
         "debiasing_actions returns Vector{String}"
         "assess_failure returns (Bool, String)"))

     (logical-constraints
       "Political barriers require stakeholder analysis"
       "Sunk cost barriers amplify with investment depth"
       "Intelligent failure must have learning potential"
       "Risk-at-value requires probability distribution"))

    (neural-layer
     (pattern-recognition
       "Identify common barrier combinations"
       "Learn typical debiasing strategies"
       "Recognize failure patterns"
       "Predict exnovation success factors")

     (learning-objectives
       "Which barriers co-occur frequently"
       "Which debiasing actions are most effective"
       "What constitutes intelligent vs unintelligent failure"
       "How context affects exnovation outcomes"))

    (integration-strategy
     (symbolic-guides-neural
       "Type system constrains valid inputs"
       "Invariants validate learned patterns"
       "Logic rules filter neural suggestions")

     (neural-enhances-symbolic
       "Suggest barrier weights from data"
       "Recommend context-specific mitigations"
       "Identify edge cases not in rules"
       "Learn from exnovation case studies"))

    (verification-approach
     (symbolic-checks
       "Type checking at compile time"
       "Property testing with invariants"
       "Contract verification for public API")

     (neural-validation
       "Cross-validation on known cases"
       "Ablation studies on recommendations"
       "Counterfactual analysis on predictions")

     (hybrid-assurance
       "Neural outputs must satisfy symbolic constraints"
       "Symbolic rules can be weighted by learned confidence"
       "Human in loop for high-stakes decisions"))

    (hypatia-integration
     (scan-integration
       (tool . "panic-attack")
       (frequency . "on-push")
       (action-on-findings . "create-issue"))

     (echidna-integration
       (proof-obligations . "Validate barrier scoring ranges")
       (verification-targets . "Critical API contracts"))

     (verisimdb-integration
       (weakness-tracking . #t)
       (pattern-detection . #t)
       (remediation-dispatch . #t))))
