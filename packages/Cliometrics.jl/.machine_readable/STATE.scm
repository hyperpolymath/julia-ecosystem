;; SPDX-License-Identifier: PMPL-1.0-or-later
;; STATE.scm - Project state tracking for Cliometrics.jl
;; Media-Type: application/vnd.state+scm

(define-state Cliometrics.jl
  (metadata
    (version "0.1.0")
    (schema-version "1.0.0")
    (created "2026-02-07")
    (updated "2026-02-12")
    (project "Cliometrics.jl")
    (repo "hyperpolymath/Cliometrics.jl"))

  (project-context
    (name "Cliometrics.jl")
    (tagline "Quantitative economic history analysis in Julia")
    (tech-stack ("Julia" "DataFrames" "CSV" "Statistics" "StatsBase")))

  (current-position
    (phase "beta")
    (overall-completion 85)
    (components
      ("Core Analysis" . 100)
      ("Data Loading" . 100)
      ("Testing" . 100)
      ("Documentation" . 80)
      ("ABI/FFI" . 90))
    (working-features
      "load_historical_data"
      "calculate_growth_rates"
      "solow_residual"
      "decompose_growth"
      "convergence_analysis"
      "institutional_quality_index"
      "quantify_institutions"
      "clean_historical_series"
      "interpolate_missing_years"
      "compare_historical_trajectories"
      "counterfactual_scenario"
      "estimate_treatment_effect"))

  (route-to-mvp
    (milestones
      ((name "Core Functionality")
       (status "complete")
       (completion 100)
       (items
         ("Growth accounting" . done)
         ("Convergence analysis" . done)
         ("Institutional analysis" . done)
         ("Data cleaning utilities" . done)
         ("Interpolation methods" . done)
         ("Counterfactual modeling" . done)
         ("Treatment effect estimation (DiD)" . done)))
      ((name "Testing & Quality")
       (status "complete")
       (completion 100)
       (items
         ("Unit tests for all functions" . done)
         ("Edge case coverage" . done)
         ("75 passing tests" . done)))
      ((name "Documentation")
       (status "in-progress")
       (completion 80)
       (items
         ("Function docstrings" . done)
         ("README with examples" . done)
         ("ROADMAP" . todo)
         ("CITATIONS" . todo)))
      ((name "v1.0 Release")
       (status "planned")
       (completion 0)
       (items
         ("Sigma-convergence testing" . todo)
         ("Long-run trend analysis" . todo)
         ("Outlier detection" . todo)
         ("Cross-country alignment" . todo)
         ("Documenter.jl docs" . todo)))))

  (blockers-and-issues
    (critical ())
    (high ())
    (medium
      ("README claims features not yet implemented"))
    (low
      ("Missing spline interpolation method")))

  (critical-next-actions
    (immediate
      "Update ROADMAP.adoc with realistic milestones"
      "Update CITATIONS.adoc with correct metadata")
    (this-week
      "Reconcile README claims with implemented features"
      "Add Julia usage examples to docs")
    (this-month
      "Plan v0.2.0 feature additions"
      "Consider Documenter.jl integration"))

  (session-history
    ((date "2026-02-12")
     (agent "claude-opus-4.6")
     (work "Implemented 4 missing functions, added 40 new tests, fixed all SPDX headers, replaced template placeholders"))))

;; Helper functions
(define (get-completion-percentage state)
  (current-position 'overall-completion state))

(define (get-blockers state severity)
  (blockers-and-issues severity state))

(define (get-milestone state name)
  (find (lambda (m) (equal? (car m) name))
        (route-to-mvp 'milestones state)))
