;; SPDX-License-Identifier: PMPL-1.0-or-later
;; META.scm for Exnovation.jl
;; Format: https://github.com/hyperpolymath/rsr-template-repo/spec/META-FORMAT-SPEC.adoc

(define meta
  '((project-meta
     (name . "Exnovation.jl")
     (tagline . "Measuring and accelerating legacy system retirement")
     (category . "sociotechnical-systems")
     (license . "PMPL-1.0-or-later")
     (inception-date . "2024")
     (repository . "https://github.com/hyperpolymath/Exnovation.jl"))

    (architecture-decisions
     (adr-001
       (title . "Use Julia for statistical modeling")
       (date . "2024")
       (status . accepted)
       (context . "Need flexible language for quantitative sociotechnical analysis")
       (decision . "Julia provides performance + expressiveness for modeling exnovation dynamics")
       (consequences . "Access to scientific computing ecosystem, strong type system"))

     (adr-002
       (title . "Model barriers and drivers as separate concerns")
       (date . "2024")
       (status . accepted)
       (context . "Organizations face both accelerating and resisting forces")
       (decision . "Separate Barrier and Driver types with distinct scoring mechanisms")
       (consequences . "Clearer API, easier to reason about opposing forces"))

     (adr-003
       (title . "Include intelligent failure assessment")
       (date . "2024")
       (status . accepted)
       (context . "Not all failures in exnovation are equal")
       (decision . "Formalize criteria for productive vs. unproductive failure")
       (consequences . "Helps organizations learn from transition attempts"))

     (adr-004
       (title . "Add Political barrier type")
       (date . "2026-02-12")
       (status . accepted)
       (context . "Real-world exnovation always involves stakeholder dynamics")
       (decision . "Extend barrier types to include Political with specific mitigation strategies")
       (consequences . "More complete model of organizational change resistance")))

    (development-practices
     (testing-strategy . "Comprehensive unit tests (32 tests covering all features)")
     (documentation-approach . "Documenter.jl for API docs + rich examples in README")
     (versioning-scheme . "SemVer 2.0.0")
     (contribution-model . "RSR-compliant with Perimeter model")
     (code-quality-tools . ("Julia formatter" "EditorConfig" "panic-attack")))

    (design-rationale
     (core-principles
       "Evidence-based approach to organizational change"
       "Quantify what is typically qualitative"
       "Balance technical and social factors"
       "Embrace productive failure, avoid unproductive failure")

     (influences
       "Organizational economics"
       "Sociotechnical systems theory"
       "Innovation diffusion research"
       "Behavioral economics (sunk cost fallacy, etc.)")

     (constraints
       "Must remain accessible to non-technical stakeholders"
       "Should integrate with existing change management frameworks"
       "Must handle incomplete/uncertain data gracefully"))

    (philosophical-stance
     (purpose . "Enable organizations to consciously retire obsolete systems")
     (values . ("Transparency" "Empiricism" "Pragmatism" "Learning from failure"))
     (non-goals . ("Automate decision-making" "Replace human judgment" "Promote change for change's sake")))))
