;; SPDX-License-Identifier: PMPL-1.0-or-later
;; ROADMAP.scm - PolyglotFormalisms.jl development roadmap

(roadmap
  (version "1.0")
  (project "PolyglotFormalisms.jl")
  (updated "2025-01-23")

  ;; Release timeline
  (releases
    ((version . "0.1.0")
     (status . "released")
     (date . "2025-01-23")
     (scope . "Initial release - Arithmetic module")
     (highlights
       "5 arithmetic operations (add, subtract, multiply, divide, modulo)"
       "59 conformance tests matching aLib specifications"
       "Mathematical properties documented"
       "CI/CD workflows (tests, CodeQL, Scorecard)"
       "RSR-compliant structure"))

    ((version . "0.2.0")
     (status . "planned")
     (target-date . "2025-Q1")
     (scope . "Comparison and Logical modules")
     (deliverables
       "6 comparison operations (less_than, greater_than, equal, not_equal, less_equal, greater_equal)"
       "3 logical operations (and, or, not)"
       "Property documentation and conformance tests"
       "Cross-reference with aLib spec updates"))

    ((version . "0.3.0")
     (status . "planned")
     (target-date . "2025-Q1")
     (scope . "String and Collection modules")
     (deliverables
       "String operations (concat, length, substring)"
       "Collection operations (map, filter, fold, contains)"
       "Handle language-specific edge cases"
       "Performance benchmarks"))

    ((version . "0.4.0")
     (status . "planned")
     (target-date . "2025-Q2")
     (scope . "Conditional module + Axiom.jl integration")
     (deliverables
       "if_then_else operation"
       "Axiom.jl dependency integration"
       "@prove macros for all mathematical properties"
       "Compile-time property verification"
       "Formal proof certificates"))

    ((version . "1.0.0")
     (status . "planned")
     (target-date . "2025-Q2")
     (scope . "Production release")
     (deliverables
       "All 6 modules complete with formal proofs"
       "Cross-language verification examples (ReScript, Gleam, Elixir)"
       "Comprehensive documentation"
       "Performance benchmarks vs other implementations"
       "Julia General registry registration"
       "Academic paper submission (optional)")))

  ;; Feature roadmap by theme
  (themes
    ((name . "Core Implementation")
     (priority . "critical")
     (milestones
       ((v0.1 . "Arithmetic ✓")
        (v0.2 . "Comparison & Logical")
        (v0.3 . "String & Collection")
        (v0.4 . "Conditional"))))

    ((name . "Formal Verification")
     (priority . "high")
     (milestones
       ((v0.1 . "Properties documented ✓")
        (v0.4 . "Axiom.jl integration")
        (v1.0 . "All properties proven"))))

    ((name . "Cross-Language Integration")
     (priority . "high")
     (milestones
       ((v0.4 . "Equivalence checking design")
        (v1.0 . "ReScript/Gleam/Elixir verification examples"))))

    ((name . "Performance")
     (priority . "medium")
     (milestones
       ((v0.3 . "Initial benchmarks")
        (v1.0 . "Performance optimization pass"))))

    ((name . "Documentation")
     (priority . "medium")
     (milestones
       ((v0.1 . "README and basic docs ✓")
        (v0.4 . "Formal verification guide")
        (v1.0 . "Complete API docs + academic paper")))))

  ;; Long-term vision
  (future-work
    "Integration with more aggregate-library implementations (Rust, Ada, Haskell)"
    "Property-based testing using Axiom.jl"
    "Performance optimizations for production use"
    "Safety-critical systems certification"
    "Academic research on cross-language semantic equivalence"
    "Tool support for automatic verification of new languages")

  ;; Dependencies on external projects
  (blockers
    ((project . "Axiom.jl")
     (status . "development")
     (needed-for . "v0.4.0")
     (impact . "Formal proof integration delayed until Axiom.jl stable"))

    ((project . "aggregate-library")
     (status . "active")
     (needed-for . "all-versions")
     (impact . "Spec changes require corresponding implementation updates"))

    ((project . "Julia General registry")
     (status . "ready")
     (needed-for . "v1.0.0")
     (impact . "Registration requires all modules complete"))))
