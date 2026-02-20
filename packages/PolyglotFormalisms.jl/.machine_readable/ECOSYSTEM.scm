;; SPDX-License-Identifier: PMPL-1.0-or-later
;; ECOSYSTEM.scm - PolyglotFormalisms.jl ecosystem relationships

(ecosystem
  (version "1.0")
  (name "polyglot-formalisms.jl")
  (type "library")
  (purpose "Julia reference implementation of aggregate-library with formal verification")

  (position-in-ecosystem
    (role "reference-implementation")
    (layer "library")
    (description "Formally verified Julia implementation serving as semantic baseline for cross-language equivalence checking"))

  (related-projects
    ((name . "aggregate-library")
     (relationship . "specification")
     (url . "https://github.com/hyperpolymath/aggregate-library"))
    ((name . "Axiom.jl")
     (relationship . "verification-tool")
     (url . "https://github.com/hyperpolymath/Axiom.jl"))
    ((name . "SMTLib.jl")
     (relationship . "planned-dependency")
     (url . "https://github.com/hyperpolymath/SMTLib.jl"))
    ((name . "alib-for-rescript")
     (relationship . "sibling-implementation")
     (url . "https://github.com/hyperpolymath/alib-for-rescript"))
    ((name . "tangle")
     (relationship . "potential-consumer")
     (nature . "TANGLE-JTV bridges topological + imperative paradigms - natural target for cross-language verification"))
    ((name . "KnotTheory.jl")
     (relationship . "sibling-standard")
     (nature . "Knot theory invariants as formally verified operations"))
    ((name . "ZeroProb.jl")
     (relationship . "sibling-standard")
     (nature . "Zero-probability event handling with formal measure-theoretic properties")))

  (what-this-is
    "A Julia package implementing the minimal overlap functions from aggregate-library (aLib), with formal verification of mathematical properties and semantic equivalence guarantees across programming languages.")

  (what-this-is-not
    "Not a comprehensive standard library replacement. Not language-specific optimizations. Not a general-purpose verification framework (use Axiom.jl directly for that). Not production-ready for safety-critical systems until formal proofs are integrated."))
