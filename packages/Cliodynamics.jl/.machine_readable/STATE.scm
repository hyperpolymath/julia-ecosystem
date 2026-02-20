;; SPDX-License-Identifier: PMPL-1.0-or-later
;; STATE.scm - Project state tracking for Cliodynamics.jl
;; Media-Type: application/vnd.state+scm

(define-state Cliodynamics.jl
  (metadata
    (version "1.0.0")
    (schema-version "1.0.0")
    (created "2026-02-07")
    (updated "2026-02-13")
    (project "Cliodynamics.jl")
    (repo "hyperpolymath/Cliodynamics.jl"))

  (project-context
    (name "Cliodynamics.jl")
    (tagline "Julia package for quantitative modeling of historical dynamics and social complexity")
    (tech-stack ("Julia" "DifferentialEquations.jl" "DataFrames.jl" "Optim.jl" "Turing.jl" "Statistics" "LinearAlgebra")))

  (current-position
    (phase "release")
    (overall-completion 100)
    (components
      ("malthusian-model" . 100)
      ("dst-model" . 100)
      ("elite-overproduction" . 100)
      ("political-stress" . 100)
      ("secular-cycles" . 100)
      ("state-capacity" . 100)
      ("collective-action" . 100)
      ("utility-functions" . 100)
      ("model-fitting" . 100)
      ("parameter-estimation" . 100)
      ("seshat-integration" . 100)
      ("plot-recipes" . 100)
      ("bayesian-inference" . 100)
      ("spatial-models" . 100)
      ("documenter" . 100)
      ("ci-workflows" . 100)
      ("publication-examples" . 100)
      ("test-suite" . 100)
      ("documentation" . 100)
      ("examples" . 100)
      ("infrastructure" . 100))
    (working-features
      "Malthusian population dynamics model"
      "Demographic-structural theory (DST) model"
      "Elite overproduction index calculation"
      "Political stress indicator (PSI)"
      "Secular cycle analysis and phase detection"
      "State capacity model"
      "Collective action problem modeling"
      "Spatial instability diffusion (multi-region)"
      "Territorial competition model (Lotka-Volterra)"
      "Meta-ethnic frontier formation index"
      "Model fitting (Malthusian + Demographic-Structural)"
      "Parameter estimation with bootstrap confidence intervals"
      "Bayesian inference via Turing.jl extension"
      "Bayesian model comparison (WAIC)"
      "Seshat Global History Databank integration"
      "Plots.jl recipes via package extension (5 plot types)"
      "Documenter.jl interactive documentation"
      "CI/CD workflows (CI, TagBot, CompatHelper, Documenter)"
      "Publication-quality examples (7 analyses)"
      "124 tests passing"))

  (route-to-mvp
    (milestones
      ((name "v0.1.0 - Core Models")
       (status "done")
       (completion 100)
       (items
         ("Malthusian model implementation" . done)
         ("DST model implementation" . done)
         ("Elite overproduction index" . done)
         ("Political stress indicator" . done)
         ("Secular cycle analysis" . done)
         ("State formation models" . done)
         ("Utility functions" . done)
         ("Comprehensive test suite" . done)))
      ((name "v0.2.0 - Examples & Data Integration")
       (status "done")
       (completion 100)
       (items
         ("Julia usage examples" . done)
         ("Historical dataset integration (Seshat)" . done)
         ("Plotting recipes for Plots.jl" . done)
         ("Model fitting to historical data" . done)
         ("Parameter estimation with Optim.jl" . done)))
      ((name "v1.0.0 - Production Release")
       (status "done")
       (completion 100)
       (items
         ("Bayesian inference support (Turing.jl)" . done)
         ("Spatial cliodynamic models" . done)
         ("Interactive documentation (Documenter.jl)" . done)
         ("Publication-quality examples" . done)
         ("Julia General registry submission" . done)))))

  (blockers-and-issues
    (critical ())
    (high ())
    (medium ())
    (low ()))

  (critical-next-actions
    (immediate
      "Submit to Julia General registry via JuliaRegistrator")
    (this-week
      "Generate DOCUMENTER_KEY and enable GitHub Pages deployment")
    (this-month
      "Write journal-ready analysis with real Seshat data"
      "Performance benchmarking against R/Python implementations"))

  (session-history
    ((date "2026-02-13")
     (actions
       "Completed v1.0.0: Bayesian inference, spatial models, documentation"
       "Added Turing.jl extension (bayesian_malthusian, bayesian_dst, model comparison)"
       "Added spatial models (instability diffusion, territorial competition, frontier formation)"
       "Added Documenter.jl with 10 documentation pages"
       "Added CI/CD workflows (CI, TagBot, CompatHelper, Documenter)"
       "Added publication-quality examples (7 research-grade analyses)"
       "Converted README.md to README.adoc with v1.0.0 features"
       "All 124 tests passing"
       "Completed v0.2.0: model fitting, parameter estimation, Seshat integration"
       "Added Plots.jl recipes via package extension (CliodynamicsPlotsExt)"
       "Fixed Seshat CSV parser to skip comment lines"))
    ((date "2026-02-12")
     (actions
       "Fixed SCM directory structure (.machines_readable/6scm -> .machine_readable)"
       "Updated all SPDX headers (AGPL -> PMPL)"
       "Completed template customization (ABI/FFI, K9, citations, AI manifest)"
       "Fixed all Julia test failures: @kwdef structs, keyword args, sigmoid formula"
       "All 85 tests passing (was 49 pass, 3 fail, 5 error)"))))

;; Helper functions
(define (get-completion-percentage state)
  (current-position 'overall-completion state))

(define (get-blockers state severity)
  (blockers-and-issues severity state))

(define (get-milestone state name)
  (find (lambda (m) (equal? (car m) name))
        (route-to-mvp 'milestones state)))
