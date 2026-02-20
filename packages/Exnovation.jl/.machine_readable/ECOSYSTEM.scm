;; SPDX-License-Identifier: PMPL-1.0-or-later
;; ECOSYSTEM.scm for Exnovation.jl
;; Media Type: application/vnd.ecosystem+scm

(ecosystem
  (version "1.0")
  (name "Exnovation.jl")
  (type "julia-package")
  (purpose "Quantitative framework for legacy system retirement analysis")

  (position-in-ecosystem
    (domain "sociotechnical-systems")
    (role "analytical-tool")
    (maturity "beta")
    (adoption "research-phase"))

  (related-projects
    ((name . "Cliodynamics.jl")
     (relationship . sibling-project)
     (nature . "Historical modeling of societal change")
     (url . "https://github.com/hyperpolymath/Cliodynamics.jl"))

    ((name . "Cliometrics.jl")
     (relationship . sibling-project)
     (nature . "Economic history with quantitative methods")
     (url . "https://github.com/hyperpolymath/Cliometrics.jl"))

    ((name . "RSR-template-repo")
     (relationship . infrastructure)
     (nature . "Repository standards and templates")
     (url . "https://github.com/hyperpolymath/rsr-template-repo"))

    ((name . "hyperpolymath ecosystem")
     (relationship . part-of)
     (nature . "14 Julia packages for sociotechnical analysis")
     (url . "https://github.com/hyperpolymath")))

  (dependencies
    (runtime
      ("Julia" "1.6+")
      ("Test" "stdlib")
      ("DataFrames.jl" "potential"))
    (development
      ("Documenter.jl" "for API docs")
      ("panic-attack" "for security scanning")
      ("EditorConfig" "for formatting")))

  (potential-integrations
    "Organizational change management systems"
    "Enterprise architecture tools"
    "Technology portfolio management platforms"
    "Risk assessment frameworks"
    "Decision support systems")

  (communication-channels
    (issues . "https://github.com/hyperpolymath/Exnovation.jl/issues")
    (discussions . "https://github.com/hyperpolymath/Exnovation.jl/discussions")
    (email . "jonathan.jewell@open.ac.uk")))
