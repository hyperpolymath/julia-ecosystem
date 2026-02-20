;; SPDX-License-Identifier: PMPL-1.0-or-later
;; STATE.scm for HackenbushGames.jl
;; Format: https://github.com/hyperpolymath/elegant-STATE

(define state
  '((metadata
     (project . "HackenbushGames.jl")
     (version . "1.0.0")
     (updated . "2026-02-12")
     (maintainers . ("Jonathan D.A. Jewell <jonathan.jewell@open.ac.uk>")))

    (project-context
     (description . "Combinatorial game theory toolkit for Hackenbush positions")
     (domain . "mathematics-game-theory")
     (languages . ("Julia"))
     (primary-purpose . "Analyze and evaluate Hackenbush graphs with dyadic rational values"))

    (current-position
     (phase . "production")
     (overall-completion . 100)
     (working-features
       "Edge and HackenbushGraph types"
       "EdgeColor enum (Blue, Red, Green)"
       "Pruning disconnected components"
       "Cutting edges"
       "Valid move enumeration"
       "Game sum operations"
       "Dyadic rational arithmetic"
       "Stalk value calculation"
       "Mex and nim_sum for nimbers"
       "Green stalk nimber computation"
       "Green Grundy number evaluation"))

    (route-to-mvp
     (completed-milestones
       "Core data structures"
       "Graph manipulation primitives"
       "Blue/Red evaluation via dyadic rationals"
       "Green edge handling via Grundy nimbers"
       "Basic test suite")
     (next-milestones
       "Complete RSR template cleanup"
       "Add comprehensive API documentation"
       "Create example notebooks"
       "Add performance benchmarks"))

    (blockers-and-issues
     (technical-debt
       "None - all SONNET-TASKS completed")
     (known-issues
       "None - 35/35 tests passing"))

    (critical-next-actions
     (immediate
       "All 17 SONNET-TASKS completed")
     (short-term
       "Add Jupyter notebook examples"
       "Performance benchmarks for large graphs"
       "Expand test coverage further")
     (long-term
       "Integrate with BowtieRisk.jl for decision analysis"
       "Add visualization export (DOT format)"
       "Research GPU acceleration for large graphs"
       "Create interactive web playground"))

    (session-history
     (sessions
       ((date . "2026-02-12")
        (agent . "Claude Sonnet 4.5")
        (summary . "Completed all 17 SONNET-TASKS: version fix, SCM files, template cleanup, docs, examples, tests, CI fixes")
        (completion-delta . +38)
        (tasks-completed
          "Task 1: Fix Manifest.toml version"
          "Task 2: Create .machine_readable with 3 SCM files"
          "Task 3: Replace placeholders in CONTRIBUTING.md"
          "Task 4: Replace placeholders in CODE_OF_CONDUCT.md"
          "Task 5: Replace placeholders in SECURITY.md"
          "Task 6: Fix CITATIONS.adoc"
          "Task 7: Create api.md and update index.md"
          "Task 8: Remove irrelevant examples, create basic_usage.jl"
          "Task 9: Delete ROADMAP.adoc template"
          "Task 10: Remove ABI/FFI boilerplate"
          "Task 11: Fix AI.a2ml paths"
          "Task 12: Fix CodeQL language matrix"
          "Task 13: Fix quality.yml TODO scanner"
          "Task 14: Add 8 comprehensive test suites (13→35 tests)"
          "Task 15: Delete README.adoc template"
          "Task 16: Delete RSR_OUTLINE.adoc template"
          "Task 17: Fix ROADMAP.md false claims"))))))

;; Helper functions for querying state

(define (get-completion-percentage state)
  (cdr (assoc 'overall-completion (assoc 'current-position state))))

(define (get-blockers state)
  (cdr (assoc 'blockers-and-issues state)))

(define (get-next-actions state)
  (cdr (assoc 'critical-next-actions state)))
