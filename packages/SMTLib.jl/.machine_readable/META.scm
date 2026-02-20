;; SPDX-License-Identifier: PMPL-1.0-or-later
;; META.scm for SMTLib.jl

(define meta
  '((project-meta
     (name . "SMTLib.jl")
     (tagline . "Pure Julia SMT solver interface with SMT-LIB2 generation")
     (category . "formal-methods")
     (license . "PMPL-1.0-or-later")
     (inception-date . "2024")
     (repository . "https://github.com/hyperpolymath/SMTLib.jl"))

    (architecture-decisions
     (adr-001
       (title . "Pure Julia implementation with external solver process")
       (status . accepted)
       (decision . "Generate SMT-LIB2 text, invoke solver as subprocess, parse stdout"))
     (adr-002
       (title . "Support multiple solvers via runtime detection")
       (status . accepted)
       (decision . "Detect z3, cvc5, yices via PATH - no hard dependency"))
     (adr-003
       (title . "Macro for convenience syntax")
       (status . accepted)
       (decision . "@smt macro converts Julia expressions to SMT-LIB2 via metaprogramming")))

    (design-rationale
      (core-principles
        "Solver-agnostic (z3, cvc5, yices, etc.)"
        "Pure Julia - no C bindings required"
        "SMT-LIB2 as lingua franca"
        "Type-safe expression generation"))))
