;; SPDX-License-Identifier: PMPL-1.0-or-later
;; ECOSYSTEM.scm for HackenbushGames.jl
;; Media Type: application/vnd.ecosystem+scm

(ecosystem
  (version "1.0")
  (name "HackenbushGames.jl")
  (type "julia-package")
  (purpose "Combinatorial game theory analysis for Hackenbush positions")

  (position-in-ecosystem
    (domain "mathematics-game-theory")
    (role "analytical-library")
    (maturity "beta")
    (adoption "research-phase"))

  (related-projects
    ((name . "BowtieRisk.jl")
     (relationship . sibling-project)
     (nature . "Decision analysis with risk-reward balance")
     (url . "https://github.com/hyperpolymath/BowtieRisk.jl"))

    ((name . "KnotTheory.jl")
     (relationship . sibling-project)
     (nature . "Mathematical analysis of knots and links")
     (url . "https://github.com/hyperpolymath/KnotTheory.jl"))

    ((name . "ZeroProb.jl")
     (relationship . sibling-project)
     (nature . "Zero-probability events and measure theory")
     (url . "https://github.com/hyperpolymath/ZeroProb.jl"))

    ((name . "RSR-template-repo")
     (relationship . infrastructure)
     (nature . "Repository standards and templates")
     (url . "https://github.com/hyperpolymath/rsr-template-repo"))

    ((name . "hyperpolymath ecosystem")
     (relationship . part-of)
     (nature . "14 Julia packages for mathematical/scientific analysis")
     (url . "https://github.com/hyperpolymath")))

  (dependencies
    (runtime
      ("Julia" "1.6+")
      ("Test" "stdlib"))
    (development
      ("Documenter.jl" "for API docs")
      ("panic-attack" "for security scanning")
      ("EditorConfig" "for formatting")))

  (potential-integrations
    "Game theory research platforms"
    "Mathematical proof assistants"
    "Combinatorial optimization tools"
    "Educational game theory software"
    "Decision analysis frameworks")

  (communication-channels
    (issues . "https://github.com/hyperpolymath/HackenbushGames.jl/issues")
    (discussions . "https://github.com/hyperpolymath/HackenbushGames.jl/discussions")
    (email . "jonathan.jewell@open.ac.uk")))
