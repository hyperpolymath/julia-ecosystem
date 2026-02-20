;; SPDX-License-Identifier: PMPL-1.0-or-later
;; ECOSYSTEM.scm - Ecosystem relationships for Axiology.jl
;; Media-Type: application/vnd.ecosystem+scm

(ecosystem
  (version "1.0.0")
  (name "Axiology.jl")
  (type "library")
  (purpose "Value theory for machine learning - define, optimize, and verify ethical/economic values in ML models")

  (position-in-ecosystem
    "Part of the hyperpolymath ecosystem's ML ethics and value theory foundations. "
    "Provides tools for encoding human values (fairness, welfare, profit, efficiency) "
    "into machine learning optimization objectives and verification criteria. "
    "Complements ZeroProb.jl (probability theory) and integrates with ECHIDNA (theorem proving).")

  (related-projects
    (sibling-standard "ZeroProb.jl" "Probability theory - complementary focus")
    (sibling-standard "Axiom.jl" "May integrate value-aware probability reasoning")
    (potential-consumer "ECHIDNA" "Formal verification of value properties in proofs")
    (potential-consumer "Flux.jl" "ML optimization with value constraints"))

  (what-this-is
    "A Julia library for value theory in machine learning. Provides types and functions "
    "for encoding ethical values (fairness), economic values (profit, efficiency), and "
    "social values (welfare) as optimization objectives and verification criteria. "
    "Enables developers to build ML systems that respect explicit value commitments.")

  (what-this-is-not
    "This is not a complete ML framework or ethics certification system. It provides "
    "building blocks for value-aware ML development, not end-to-end solutions."))
