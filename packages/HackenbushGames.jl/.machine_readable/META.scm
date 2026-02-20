;; SPDX-License-Identifier: PMPL-1.0-or-later
;; META.scm for HackenbushGames.jl
;; Format: https://github.com/hyperpolymath/rsr-template-repo/spec/META-FORMAT-SPEC.adoc

(define meta
  '((project-meta
     (name . "HackenbushGames.jl")
     (tagline . "Combinatorial game theory for Hackenbush positions")
     (category . "mathematics-game-theory")
     (license . "PMPL-1.0-or-later")
     (inception-date . "2024")
     (repository . "https://github.com/hyperpolymath/HackenbushGames.jl"))

    (architecture-decisions
     (adr-001
       (title . "Pure Julia implementation with no dependencies")
       (date . "2024")
       (status . accepted)
       (context . "Need lightweight library for game theory research")
       (decision . "Implement all algorithms in pure Julia using only stdlib")
       (consequences . "Minimal dependencies, easy installation, good performance"))

     (adr-002
       (title . "Dyadic rationals for Blue-Red evaluation")
       (date . "2024")
       (status . accepted)
       (context . "Hackenbush values are dyadic rationals (p/2^q)")
       (decision . "Use (numerator, power_of_two) representation")
       (consequences . "Exact arithmetic, no floating-point errors, efficient"))

     (adr-003
       (title . "Grundy numbers for Green edges")
       (date . "2024")
       (status . accepted)
       (context . "Green edges behave as impartial games")
       (decision . "Use nim arithmetic and Mex operation for Green stalks")
       (consequences . "Standard CGT techniques apply, well-understood theory"))

     (adr-004
       (title . "Graph-based representation")
       (date . "2024")
       (status . accepted)
       (context . "Hackenbush positions are rooted directed graphs")
       (decision . "Store edges with colors + connectivity to ground")
       (consequences . "Natural representation, efficient pruning, clear semantics")))

    (development-practices
     (testing-strategy . "Unit tests for all core functions")
     (documentation-approach . "Documenter.jl for API docs + mathematical background")
     (versioning-scheme . "SemVer 2.0.0")
     (contribution-model . "RSR-compliant with Perimeter model")
     (code-quality-tools . ("Julia formatter" "EditorConfig" "panic-attack")))

    (design-rationale
     (core-principles
       "Exact arithmetic (no approximations)"
       "Minimal dependencies (stdlib only)"
       "Clear separation of colored edge types"
       "Efficient graph operations (pruning, cutting)")

     (influences
       "Winning Ways for Your Mathematical Plays (Conway, Berlekamp, Guy)"
       "Combinatorial Game Theory (Siegel)"
       "On Numbers and Games (Conway)")

     (constraints
       "Must handle arbitrarily large dyadic rational values"
       "Must correctly implement nim arithmetic"
       "Must maintain connectivity invariants after cuts"
       "Must support game sums without mutation"))

    (philosophical-stance
     (purpose . "Enable rigorous analysis of Hackenbush positions")
     (values . ("Mathematical rigor" "Computational efficiency" "Pedagogical clarity"))
     (non-goals . ("Real-time gameplay" "Visualization" "AI opponents")))))
