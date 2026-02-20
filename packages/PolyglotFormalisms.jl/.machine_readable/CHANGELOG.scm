;; SPDX-License-Identifier: PMPL-1.0-or-later
;; CHANGELOG.scm - PolyglotFormalisms.jl version history

(changelog
  (version "1.0")
  (project "PolyglotFormalisms.jl")
  (format "keep-a-changelog")
  (updated "2026-01-23")

  ;; Current version
  (release
    (version . "0.2.0")
    (date . "2026-01-23")
    (status . "released")
    (tag . "v0.2.0")

    (added
      "Comparison module with 6 operations (less_than, greater_than, equal, not_equal, less_equal, greater_equal)"
      "Logical module with 3 operations (and, or, not)"
      "98 conformance tests for Comparison module"
      "41 conformance tests for Logical module"
      "Mathematical properties documentation for all comparison operations"
      "Boolean algebra properties documentation for all logical operations"
      "Truth tables and edge case tests for logical operations"
      "IEEE 754 edge case handling for NaN, Inf, and signed zeros")

    (changed
      "Total test count: 59 → 198 (139 new tests)")

    (deprecated . ())

    (removed . ())

    (fixed . ())

    (security . ()))

  ;; Previous version
  (release
    (version . "0.1.0")
    (date . "2025-01-23")
    (status . "released")
    (tag . "v0.1.0")
    (url . "https://github.com/hyperpolymath/PolyglotFormalisms.jl/releases/tag/v0.1.0")

    (added
      "Arithmetic module with 5 operations (add, subtract, multiply, divide, modulo)"
      "59 conformance tests matching aggregate-library specifications"
      "Mathematical property documentation in docstrings"
      "CI workflows: tests (Julia 1.9/1.10/nightly), CodeQL, OpenSSF Scorecard"
      "Community files: LICENSE (PMPL-1.0), CONTRIBUTING, SECURITY"
      "RSR-compliant structure with .machine_readable/ SCM files"
      "ECOSYSTEM.scm - project relationships"
      "META.scm - architectural decisions (3 ADRs)"
      "STATE.scm - project state tracking"
      "DEPENDENCIES.scm - dependency specification"
      "ROADMAP.scm - development roadmap"
      "CHANGELOG.scm - version history"
      "GitHub repository with 8 topics/tags"
      "SHA-pinned GitHub Actions for supply chain security")

    (changed . ())

    (deprecated . ())

    (removed . ())

    (fixed . ())

    (security
      "All GitHub Actions SHA-pinned to prevent supply chain attacks"
      "OpenSSF Scorecard integration for continuous security monitoring"
      "CodeQL analysis for vulnerability detection"))

  ;; Pre-release history
  (unreleased
    (version . "0.0.1-dev")
    (date . "2025-01-23")
    (status . "development")

    (notes
      "Initial development as 'aLib.jl'"
      "Renamed to PolyglotFormalisms.jl for clarity"
      "Package structure established"
      "Test infrastructure created"
      "First module (Arithmetic) implemented"))

  ;; Future versions (planned)
  (planned
    ((version . "0.3.0")
     (target-date . "2025-Q1")
     (scope
       "String module (3 operations)"
       "Collection module (4 operations)"
       "Performance benchmarks"))

    ((version . "0.4.0")
     (target-date . "2025-Q2")
     (scope
       "Conditional module (1 operation)"
       "Axiom.jl integration"
       "@prove macros for all properties"
       "Formal verification at compile time"))

    ((version . "1.0.0")
     (target-date . "2025-Q2")
     (scope
       "All 6 modules complete"
       "Cross-language verification examples"
       "Julia General registry registration"
       "Production-ready release")))

  ;; Changelog conventions
  (conventions
    (commit-format . "conventional-commits")
    (breaking-changes . "Clearly marked in BREAKING CHANGE section")
    (deprecations . "Announced one minor version before removal")
    (security-fixes . "Highlighted in Security section")
    (co-authorship . "Claude Sonnet 4.5 contributions acknowledged")))
