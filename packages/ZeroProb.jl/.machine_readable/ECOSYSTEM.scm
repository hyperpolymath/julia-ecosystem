;; SPDX-License-Identifier: PMPL-1.0-or-later
;; ECOSYSTEM.scm - Ecosystem relationships for ZeroProb.jl
;; Media-Type: application/vnd.ecosystem+scm

(ecosystem
  (version "1.0.0")
  (name "ZeroProb.jl")
  (type "library")
  (purpose "Handle zero-probability events in continuous probability spaces")

  (position-in-ecosystem
    "Part of the hyperpolymath ecosystem's probability theory foundations. "
    "Provides tools for reasoning about zero-probability events in continuous "
    "distributions using alternative relevance measures (density ratio, Hausdorff, "
    "epsilon-neighborhood). Complements Axiom.jl and supports ECHIDNA theorem proving.")

  (related-projects
    (sibling-standard "Axiom.jl" "Broader probability reasoning framework")
    (sibling-standard "Axiology.jl" "Value theory for ML - complementary focus")
    (potential-consumer "ECHIDNA" "May integrate zero-prob reasoning in proof strategies")
    (inspiration "StatLect" "Based on StatLect probability theory articles")
    (related "tangle" "Quantum topology connections - anyonic braiding yields zero-probability measurement events")
    (sibling-standard "KnotTheory.jl" "Knot theory invariants - topological aspects of zero-probability events")
    (sibling-standard "SMTLib.jl" "SMT solver interface for verifying measure-theoretic properties"))

  (what-this-is
    "A Julia library for handling zero-probability events in continuous probability "
    "spaces. Provides alternative relevance measures since classical probability "
    "(P=0) is insufficient for continuous spaces. Includes pedagogical tools for "
    "teaching continuum paradox, Borel-Kolmogorov paradox, and black swan events.")

  (what-this-is-not
    "This is not a complete probability framework - it focuses specifically on the "
    "counterintuitive case where P(E)=0 but the event can occur. For general "
    "probability theory, see Axiom.jl."))
