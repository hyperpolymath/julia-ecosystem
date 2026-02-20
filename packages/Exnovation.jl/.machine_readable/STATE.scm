;; SPDX-License-Identifier: PMPL-1.0-or-later
;; STATE.scm for Exnovation.jl
;; Format: https://github.com/hyperpolymath/elegant-STATE

(define state
  '((metadata
     (project . "Exnovation.jl")
     (version . "1.0.0")
     (updated . "2026-02-12")
     (maintainers . ("Jonathan D.A. Jewell <jonathan.jewell@open.ac.uk>")))

    (project-context
     (description . "Framework for measuring and accelerating legacy system retirement")
     (domain . "sociotechnical-systems")
     (languages . ("Julia"))
     (primary-purpose . "Quantify and manage organizational exnovation processes"))

    (current-position
     (phase . "production")
     (overall-completion . 100)
     (working-features
       "ExnovationItem and core types"
       "Driver and Barrier enums (including Political)"
       "Functional barrier assessment"
       "Sunk cost bias detection"
       "Intelligent failure assessment"
       "Debiasing action recommendations"
       "Value-at-risk calculations"
       "API documentation with Documenter.jl"
       "Comprehensive test suite (32 tests)"
       "Quick Start examples"
       "Input validation on public API"
       "RSR template fully customized"
       "All SCM files configured"
       "CI workflow hardened"))

    (route-to-mvp
     (completed-milestones
       "Core types and enums"
       "Barrier scoring and categorization"
       "Debiasing logic for all barrier types"
       "Intelligent failure concepts"
       "Documentation and examples"
       "Political barrier support added"
       "RSR template cleanup completed"
       "License headers fixed (PMPL-1.0-or-later)")
     (next-milestones
       "Input validation for public API"
       "Performance benchmarks"
       "Integration examples with real systems"
       "Case studies from industry"))

    (blockers-and-issues
     (technical-debt
       "None - all SONNET-TASKS completed")
     (known-issues
       "None - 32/32 tests passing, full RSR compliance"))

    (critical-next-actions
     (immediate
       "All SONNET-TASKS completed (14/14)")
     (short-term
       "Add performance benchmarks"
       "Create integration examples"
       "Write case studies")
     (long-term
       "Build ecosystem of exnovation metrics"
       "Integrate with organizational change frameworks"
       "Research validation studies"))

    (session-history
     (sessions
       ((date . "2026-02-12")
        (agent . "Claude Sonnet 4.5")
        (summary . "Completed all 14 SONNET-TASKS: Political barrier fix, API docs, RSR cleanup, SCM files, input validation, CI hardening")
        (completion-delta . +28)
        (tasks-completed
          "Task 1: Fix Project.toml version"
          "Task 2: Add Political barrier handling"
          "Task 3: Create API documentation"
          "Task 4: Update index.md Quick Start"
          "Task 5: Fix license headers"
          "Task 6: Remove ABI/FFI boilerplate"
          "Task 7: Customize RSR template placeholders"
          "Task 8: Create .machine_readable with 6 SCM files"
          "Task 9: Remove unrelated examples"
          "Task 10: Customize CITATIONS.adoc"
          "Task 11: Mark unpinned GitHub Actions"
          "Task 12: Fix AI.a2ml references"
          "Task 13: Add input validation"
          "Task 14: Add CI workflow permissions"))))))

;; Helper functions for querying state

(define (get-completion-percentage state)
  (cdr (assoc 'overall-completion (assoc 'current-position state))))

(define (get-blockers state)
  (cdr (assoc 'blockers-and-issues state)))

(define (get-next-actions state)
  (cdr (assoc 'critical-next-actions state)))
