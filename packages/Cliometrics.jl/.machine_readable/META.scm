;; SPDX-License-Identifier: PMPL-1.0-or-later
;; META.scm - Architectural decisions and project meta-information
;; Media-Type: application/meta+scheme

(define-meta Cliometrics.jl
  (version "1.0.0")

  (architecture-decisions
    ;; ADR format: (adr-NNN status date context decision consequences)
    ((adr-001 accepted "2026-02-07"
      "Need to establish repository structure and standards"
      "Adopt RSR (Rhodium Standard Repository) conventions from rsr-template-repo"
      "Ensures consistency with 500+ repos in hyperpolymath ecosystem. "
      "Enables automated quality enforcement via gitbot-fleet and Hypatia.")
    (adr-002 accepted "2026-02-07"
      "Choice of language for cliometric analysis library"
      "Use Julia for numerical analysis performance and ecosystem integration"
      "Julia provides excellent performance for numerical work, native support for "
      "DataFrames, and a rich ecosystem (CSV, Statistics, StatsBase). Academic researchers "
      "familiar with both R and Python can adopt Julia easily.")
    (adr-003 accepted "2026-02-07"
      "Module structure: single-file vs multi-file"
      "Use single-file module (src/Cliometrics.jl) for initial implementation"
      "Simplifies maintenance for <1000 LOC codebase. Clear dependency graph. "
      "Easy to audit. Standard Julia practice for focused libraries. "
      "Can refactor to multi-file if complexity grows beyond 2000 LOC.")
    (adr-004 accepted "2026-02-12"
      "Treatment effect estimation methodology"
      "Implement difference-in-differences (DiD) as primary causal inference method"
      "DiD is the standard cliometric approach for estimating treatment effects in "
      "historical data. Simple to understand, widely used in economic history. "
      "Can be extended to synthetic controls or event study designs in future.")))

  (development-practices
    (code-style
      "Follow Julia style guide conventions. "
      "Use lowercase_with_underscores for functions. "
      "CamelCase for types. "
      "Comprehensive docstrings for all exported functions. "
      "@testset for all test groups.")
    (security
      "All commits signed. "
      "Hypatia neurosymbolic scanning enabled. "
      "OpenSSF Scorecard tracking.")
    (testing
      "75 tests covering all 11 exported functions. "
      "Edge cases tested (empty data, missing values, single points). "
      "CI/CD runs full test suite on all pushes.")
    (versioning
      "Semantic versioning (semver). "
      "v0.1.0 = core functionality. "
      "v0.2.0 = extended features (sigma-convergence, outliers). "
      "v1.0.0 = production-ready with Documenter.jl docs.")
    (documentation
      "README.md for overview and quick examples. "
      "Docstrings for function-level documentation. "
      "STATE.scm for current state. "
      "ECOSYSTEM.scm for relationships. "
      "CITATIONS.adoc for academic citation.")
    (branching
      "Main branch protected. "
      "Feature branches for new work. "
      "PRs required for merges."))

  (design-rationale
    (why-julia
      "Julia combines Python-like syntax with C-like performance. "
      "Essential for economic historians working with large historical datasets. "
      "Native DataFrames support and rich statistics ecosystem (StatsBase, HypothesisTests).")
    (why-growth-accounting
      "Growth accounting (Solow residual) is foundational to cliometrics. "
      "Decomposes output growth into capital, labor, and TFP contributions. "
      "Essential for understanding sources of historical economic growth.")
    (why-convergence-analysis
      "Convergence testing (beta-convergence) tests whether poor countries grow faster. "
      "Central hypothesis in development economics and economic history. "
      "Helps identify whether income gaps close over time.")
    (why-institutional-analysis
      "Institutions (rule of law, property rights) are key determinants of long-run growth. "
      "Quantifying institutional quality enables cross-country comparisons. "
      "Measuring institutional change over time reveals reform impacts.")))
