;; SPDX-License-Identifier: PMPL-1.0-or-later
;; META.scm - Architectural decisions and project meta-information
;; Media-Type: application/meta+scheme

(define-meta Cliodynamics.jl
  (version "1.0.0")

  (architecture-decisions
    ;; ADR format: (adr-NNN status date context decision consequences)
    ((adr-001 accepted "2026-02-07"
      "Need computational framework for cliodynamic modeling"
      "Use Julia + DifferentialEquations.jl for population dynamics and social complexity models"
      "Julia provides high performance (comparable to C/Fortran), "
      "excellent ODE solvers via DifferentialEquations.jl, "
      "interactive REPL for exploration, and strong scientific computing ecosystem. "
      "DifferentialEquations.jl handles stiff equations common in population models.")
    (adr-002 accepted "2026-02-07"
      "Need to organize cliodynamic functions"
      "Single-file module design: all code in src/Cliodynamics.jl"
      "Simplifies package structure for initial release. "
      "All 16 exported functions in one file (~700 lines) is maintainable. "
      "Can refactor into modules later if package grows significantly.")
    (adr-003 accepted "2026-02-07"
      "Need to handle both analytical and numerical solutions"
      "Implement both closed-form Malthusian model and numerical DST model"
      "Malthusian model has analytical solution (exponential growth). "
      "DST model requires ODE solver due to elite-commoner coupling. "
      "Providing both demonstrates different modeling approaches.")
    (adr-004 accepted "2026-01-30"
      "Need to establish repository structure and standards"
      "Adopt RSR (Rhodium Standard Repository) conventions"
      "Ensures consistency with 500+ repos in hyperpolymath ecosystem. "
      "Enables automated quality enforcement via gitbot-fleet and Hypatia."))

  (development-practices
    (code-style
      "Follow Julia conventions: "
      "snake_case for functions, "
      "PascalCase for types, "
      "docstrings for all exports using Julia's triple-quote format, "
      "type annotations where they improve performance or clarity.")
    (security
      "All commits signed. "
      "Hypatia neurosymbolic scanning enabled. "
      "OpenSSF Scorecard tracking. "
      "SPDX headers on all source files.")
    (testing
      "Comprehensive test coverage using @testset. "
      "Every exported function has tests. "
      "CI/CD runs Pkg.test() on all pushes. "
      "Tests verify both correctness and numerical stability.")
    (versioning
      "Semantic versioning (semver). "
      "v0.x for pre-release. "
      "v1.0 after Julia General registry submission.")
    (documentation
      "Docstrings for all exported functions. "
      "README.md for overview and quick start. "
      "Examples in examples/ directory. "
      "STATE.scm for current project state.")
    (branching
      "Main branch protected. "
      "Feature branches for new work. "
      "PRs required for merges."))

  (design-rationale
    (why-julia
      "Julia combines Python-like ease of use with C-like performance. "
      "Critical for cliodynamic models that simulate centuries of historical data. "
      "DifferentialEquations.jl is state-of-the-art for ODE solving.")
    (why-single-file
      "Single-file module (src/Cliodynamics.jl) keeps initial implementation simple. "
      "All 16 functions are thematically related (cliodynamic analysis). "
      "Easy to navigate for contributors. "
      "Can refactor into submodules if scope expands significantly.")
    (why-demographic-structural-theory
      "DST (Turchin, Goldstone) is central framework in cliodynamics. "
      "Models elite overproduction and political stress leading to instability. "
      "Empirically validated across multiple civilizations.")))
