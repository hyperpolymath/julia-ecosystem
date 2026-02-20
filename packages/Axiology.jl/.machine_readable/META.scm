;; SPDX-License-Identifier: PMPL-1.0-or-later
;; META.scm - Meta-level information for Axiology.jl
;; Media-Type: application/meta+scheme

(meta
  (architecture-decisions
    (adr-001
      (title "Use abstract Value type hierarchy for extensibility")
      (status "accepted")
      (date "2026-02-07")
      (context
        "Different application domains need different value types. "
        "Need extensible system that doesn't prescribe specific values.")
      (decision
        "Define abstract Value type with concrete implementations for common "
        "values (Fairness, Welfare, Profit, Efficiency). Users can add custom types.")
      (consequences
        "Library is flexible and extensible. Domain-specific values can be added "
        "without modifying core library."))

    (adr-002
      (title "Separate satisfaction, maximization, and verification")
      (status "accepted")
      (date "2026-02-07")
      (context
        "Values can be used in different ways: as constraints (satisfy), "
        "as objectives (maximize), or as post-hoc checks (verify).")
      (decision
        "Provide three distinct operations: satisfy() for constraint satisfaction, "
        "maximize() for optimization objectives, verify_value() for verification.")
      (consequences
        "Clear separation of concerns. Users can apply values in different contexts.")))

  (development-practices
    (testing "Property-based testing for value axioms (transitivity, etc.)")
    (documentation "Extensive examples showing ethical trade-offs")
    (versioning "Semantic versioning following Julia ecosystem standards"))

  (design-rationale
    (philosophy
      "Make values explicit and computable. Don't hide ethical choices in "
      "implicit design decisions - expose them as first-class types that can "
      "be reasoned about formally.")
    (audience
      "ML practitioners, researchers in AI ethics, economists working on "
      "mechanism design, anyone building systems that make value-laden decisions.")
    (inspiration
      "Value alignment research, AI safety, welfare economics, fairness in ML literature.")))
