;; SPDX-License-Identifier: PMPL-1.0-or-later
;; META.scm - Meta-level information for ZeroProb.jl
;; Media-Type: application/meta+scheme

(meta
  (architecture-decisions
    (adr-001
      (title "Use alternative relevance measures for zero-probability events")
      (status "accepted")
      (date "2026-02-07")
      (context
        "Classical probability theory assigns P=0 to continuous point events, "
        "but this doesn't capture their relative relevance or likelihood.")
      (decision
        "Implement density ratio, Hausdorff measure, and epsilon-neighborhood "
        "as alternative measures of relevance for zero-probability events.")
      (consequences
        "Users can distinguish between different zero-probability events and "
        "make informed decisions in continuous probability spaces."))

    (adr-002
      (title "Include pedagogical paradox examples")
      (status "accepted")
      (date "2026-02-07")
      (context
        "Zero-probability events are counterintuitive and often misunderstood.")
      (decision
        "Include continuum paradox, Borel-Kolmogorov paradox, and other examples "
        "as first-class library features, not just documentation.")
      (consequences
        "Library serves both practical and educational purposes.")))

  (development-practices
    (testing "Comprehensive test suite with property-based testing")
    (documentation "Extensive examples and pedagogical content")
    (versioning "Semantic versioning following Julia ecosystem standards"))

  (design-rationale
    (philosophy
      "Embrace the counterintuitive nature of measure theory. Make the implicit "
      "explicit - expose alternative measures that practitioners use intuitively.")
    (audience
      "Researchers, practitioners, and students working with continuous probability "
      "distributions, risk analysis, black swan events, and edge cases in betting.")
    (inspiration
      "StatLect articles on zero-probability events, Nassim Taleb's work on "
      "black swans, measure-theoretic probability theory.")))
