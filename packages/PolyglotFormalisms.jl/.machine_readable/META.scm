;; SPDX-License-Identifier: PMPL-1.0-or-later
;; META.scm - PolyglotFormalisms.jl metadata and architectural decisions

(define project-meta
  `((version . "0.1.0")

    (architecture-decisions
      ((adr-001
         (status . "accepted")
         (date . "2025-01-23")
         (context . "Need Julia reference implementation for aggregate-library cross-language verification")
         (decision . "Use Julia as reference language due to: 1) Native formal verification support via Axiom.jl, 2) Clear mathematical semantics, 3) Existing hyperpolymath ecosystem packages")
         (consequences . "Julia becomes canonical reference for semantic equivalence. Other languages verify against Julia implementation via Axiom.jl + SMT solvers."))

       (adr-002
         (status . "accepted")
         (date . "2025-01-23")
         (context . "aLib specifies minimal overlap functions. Need to decide module organization.")
         (decision . "One module per aLib category (Arithmetic, Comparison, Logical, String, Collection, Conditional). Export all submodules from main PolyglotFormalisms module.")
         (consequences . "Users can import specific modules or all at once. Matches aLib specification structure. Future expansion follows same pattern."))

       (adr-003
         (status . "accepted")
         (date . "2025-01-23")
         (context . "Mathematical properties should be documented. Formal proof integration planned but not immediate.")
         (decision . "Document all mathematical properties in docstrings now. Add @prove macros later when Axiom.jl becomes a dependency.")
         (consequences . "Properties are clear to implementers today. Proofs added incrementally without breaking changes. Test suite validates properties empirically until formal proofs available."))))

    (development-practices
      ((code-style . "julia-standard")
       (security . "openssf-scorecard")
       (testing . "conformance-driven")
       (versioning . "semver")
       (documentation . "markdown+docstrings")
       (branching . "trunk-based")))

    (design-rationale
      ((why-julia
         "Julia combines mathematical rigor, native formal verification support (Axiom.jl), and clear semantics. Natural fit for reference implementation of formally verified cross-language library.")

       (why-minimal-overlap
         "aggregate-library philosophy: Define only functions that work across ALL target languages. Language-specific features stay in standard libraries. Ensures universal semantic equivalence.")

       (why-formal-verification
         "Cross-language semantic equivalence requires formal proof. Testing validates behavior, but only formal methods prove two implementations are truly equivalent across all inputs.")

       (why-not-complete-stdlib
         "Complete standard library would diverge across languages, defeating cross-language verification goal. Minimal overlap maximizes portability and verifiability.")))))
