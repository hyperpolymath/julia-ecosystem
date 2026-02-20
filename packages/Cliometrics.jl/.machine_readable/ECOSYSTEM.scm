;; SPDX-License-Identifier: PMPL-1.0-or-later
;; ECOSYSTEM.scm - Ecosystem relationships for Cliometrics.jl
;; Media-Type: application/vnd.ecosystem+scm

(ecosystem
  (version "1.0.0")
  (name "Cliometrics.jl")
  (type "library")
  (purpose "Quantitative economic history analysis")

  (position-in-ecosystem
    "Julia library for cliometric analysis within the hyperpolymath "
    "ecosystem. Provides growth accounting, convergence testing, and "
    "institutional analysis tools for economic historians and researchers.")

  (related-projects
    (dependency "DataFrames.jl" "Data manipulation")
    (dependency "CSV.jl" "Data loading")
    (dependency "Statistics" "Statistical analysis")
    (dependency "StatsBase" "Extended statistics")
    (sibling-related "Cliodynamics.jl" "Historical dynamics modeling")
    (potential-consumer "economic-research" "Academic research projects")
    (potential-consumer "historical-data-analysis" "Data science workflows"))

  (what-this-is
    "A Julia library for quantitative economic history analysis (cliometrics). "
    "Provides growth accounting (Solow residual, TFP), convergence analysis "
    "(beta-convergence), institutional quality indices, counterfactual modeling, "
    "and difference-in-differences estimation. Designed for economic historians, "
    "development economists, and researchers analyzing long-run economic growth.")

  (what-this-is-not
    "Not a general-purpose statistics library (use Statistics.jl/StatsBase.jl). "
    "Not a full econometric toolkit (use Econometrics.jl). "
    "Not focused on modern macroeconomic forecasting (historical data focus). "
    "Not an agent-based modeling framework (see Cliodynamics.jl for that)."))
