;; SPDX-License-Identifier: PMPL-1.0-or-later
;; ECOSYSTEM.scm - Ecosystem relationships for Cliodynamics.jl
;; Media-Type: application/vnd.ecosystem+scm

(ecosystem
  (version "1.0.0")
  (name "Cliodynamics.jl")
  (type "library")
  (purpose "Julia library for quantitative modeling of historical dynamics, social complexity, and long-term societal trends")

  (position-in-ecosystem
    "Cliodynamics.jl is a Julia package for computational cliodynamics - the scientific study of "
    "historical dynamics and social complexity using mathematical and statistical methods. "
    "It provides implementations of key cliodynamic models including Malthusian population dynamics, "
    "demographic-structural theory, elite overproduction indices, and secular cycle analysis. "
    "Integrates with the Julia scientific computing ecosystem, particularly DifferentialEquations.jl "
    "for ODE solving and DataFrames.jl for time series analysis. Part of the hyperpolymath "
    "ecosystem of 500+ repositories following RSR conventions.")

  (related-projects
    (sibling "Cliometrics.jl" "Quantitative economic history")
    (dependency "DifferentialEquations.jl" "ODE solvers for population models")
    (dependency "DataFrames.jl" "Time series and historical data handling")
    (dependency "Optim.jl" "Parameter estimation and model fitting")
    (dependency "Statistics" "Statistical analysis utilities")
    (dependency "LinearAlgebra" "Matrix operations")
    (inspiration "Seshat Global History Databank" "Empirical historical datasets")
    (inspiration "Peter Turchin's work" "Demographic-structural theory")
    (inspiration "Jack Goldstone's work" "State breakdown and revolution theory")
    (weak-dependency "Turing.jl" "Bayesian parameter inference via package extension")
    (weak-dependency "RecipesBase.jl" "Plot recipes via package extension")
    (weak-dependency "Distributions.jl" "Probability distributions for Bayesian models")
    (weak-dependency "MCMCChains.jl" "MCMC chain analysis for Bayesian extension")
    (integration "Documenter.jl" "Interactive documentation with GitHub Pages deployment"))

  (what-this-is
    "Cliodynamics.jl is a Julia library for applying mathematical and computational methods to "
    "the study of historical dynamics. It implements foundational models from cliodynamics: "
    "(1) Malthusian population dynamics with resource constraints, "
    "(2) Demographic-structural theory modeling elite-commoner dynamics, "
    "(3) Elite overproduction index measuring intra-elite competition, "
    "(4) Political stress indicator (PSI) predicting instability, "
    "(5) Secular cycle analysis for long-term societal oscillations, "
    "(6) State capacity and collective action models. "
    "The package provides both analytical solutions (where available) and numerical ODE solvers "
    "for complex coupled systems. It is designed for researchers, historians, and social scientists "
    "studying long-term patterns in human societies, state formation, collapse, and recurrent cycles "
    "of expansion and crisis.")

  (what-this-is-not
    "Cliodynamics.jl is NOT a general-purpose statistics library (use StatsBase.jl or Statistics.jl). "
    "It is NOT a generic agent-based modeling framework (use Agents.jl). "
    "It is NOT focused on short-term economic forecasting (use econometric packages). "
    "It does NOT provide pre-loaded historical datasets (users must integrate their own data or "
    "use external sources like Seshat Global History Databank). "
    "It is NOT a replacement for domain expertise in history or archaeology - it is a computational "
    "tool that requires careful interpretation by domain specialists."))
