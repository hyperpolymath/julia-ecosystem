;; SPDX-License-Identifier: PMPL-1.0-or-later
;; META.scm - Architectural decisions and project meta-information
;; Media-Type: application/meta+scheme

(define-meta Cladistics.jl
  (version "1.0.0")

  (architecture-decisions
    ;; ADR format: (adr-NNN status date context decision consequences)
    ((adr-001 accepted "2026-01-30"
      "Need to establish repository structure and standards"
      "Adopt RSR (Rhodium Standard Repository) conventions"
      "Ensures consistency with 500+ repos in hyperpolymath ecosystem. "
      "Enables automated quality enforcement via gitbot-fleet and Hypatia.")
    (adr-002 accepted "2026-02-12"
      "Need parsimony scoring algorithm for character-based phylogenetics"
      "Use Fitch algorithm for maximum parsimony scoring"
      "Fitch algorithm is the standard dynamic programming approach for parsimony. "
      "Efficient O(n*m) time complexity for n taxa and m characters. "
      "Well-documented and widely used in phylogenetic software.")
    (adr-003 accepted "2026-02-12"
      "Need multiple evolutionary distance models for molecular data"
      "Support four distance metrics: Hamming, p-distance, JC69, K2P"
      "Covers basic (Hamming, p-distance) to evolutionary models (JC69, K2P). "
      "JC69 corrects for multiple substitutions. K2P distinguishes transitions/transversions. "
      "Sufficient for most molecular phylogenetics workflows.")
    (adr-004 accepted "2026-02-12"
      "Need to handle saturated sequences (too many substitutions)"
      "Return Inf for K2P distance when (1-2P-Q) or (1-2Q) non-positive"
      "Prevents log of non-positive numbers which would give NaN or DomainError. "
      "Inf signals saturation where evolutionary model assumptions break down.")
    (adr-005 accepted "2026-02-12"
      "Tree rerooting requires midpoint placement on outgroup branch"
      "Implement midpoint rerooting: place new root at half outgroup branch length"
      "Standard phylogenetic convention. Preserves branch length information. "
      "Creates biologically meaningful rooted tree for downstream analysis.")))

  (development-practices
    (code-style
      "Follow Julia conventions: "
      "snake_case for functions and variables, "
      "PascalCase for types, "
      "Descriptive names for clarity. "
      "Type annotations where it improves performance or clarity.")
    (security
      "All commits signed. "
      "Hypatia neurosymbolic scanning enabled. "
      "OpenSSF Scorecard tracking. "
      "SPDX headers on all source files.")
    (testing
      "Comprehensive test coverage required. "
      "Every exported function must have tests. "
      "CI/CD runs full test suite on all pushes. "
      "Property-based tests for tree algorithms.")
    (versioning
      "Semantic versioning (semver). "
      "v0.x.0 for active development. "
      "v1.0.0 after Julia General registry submission.")
    (documentation
      "Docstrings for all exported functions. "
      "README.adoc for overview and quick start. "
      "Examples in examples/ directory. "
      "STATE.scm for current state tracking.")
    (branching
      "Main branch protected. "
      "Feature branches for new work. "
      "PRs required for merges.")
    (performance
      "Profile before optimizing. "
      "Use type stability for critical paths. "
      "Benchmark against established tools (MEGA, PAUP)."))

  (design-rationale
    (why-fitch
      "Fitch algorithm is the standard for parsimony scoring. "
      "Elegant dynamic programming solution with minimal memory footprint.")
    (why-stepwise-addition
      "Exact parsimony search is NP-hard. Stepwise addition provides good "
      "heuristic approximation in polynomial time. Standard in phylogenetics.")
    (why-julia
      "Julia combines Python-like ease of use with C-like performance. "
      "Native support for scientific computing and linear algebra. "
      "Growing BioJulia ecosystem for bioinformatics.")))
