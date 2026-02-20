;; SPDX-License-Identifier: PMPL-1.0-or-later
;; ECOSYSTEM.scm for KnotTheory.jl
;; Media-Type: application/vnd.ecosystem+scm

(ecosystem
  (version "1.0")
  (name "KnotTheory.jl")
  (type "julia-package")
  (purpose "Knot theory invariants and planar diagram manipulation")

  (position-in-ecosystem
    (domain "mathematics-topology")
    (role "analytical-library")
    (maturity "beta")
    (adoption "research-phase")
    (description
      "KnotTheory.jl provides computational knot theory tools for Julia. "
      "It computes Alexander, Jones, Conway, and HOMFLY-PT polynomials, "
      "Seifert matrices, signatures, and determinants from planar diagram "
      "representations. Includes braid word support for TANGLE interop."))

  (related-projects
    ((name . "tangle")
     (relationship . primary-interop)
     (nature . "Topological programming language where programs are isotopy classes of tangles")
     (integration . "from_braid_word/to_braid_word functions provide direct conversion between TANGLE braid notation and KnotTheory.jl Knot objects"))
    ((name . "HackenbushGames.jl")
     (relationship . sibling-project)
     (nature . "Combinatorial game theory"))
    ((name . "SMTLib.jl")
     (relationship . potential-consumer)
     (nature . "SMT solver interface - could verify knot invariant properties"))
    ((name . "ZeroProb.jl")
     (relationship . sibling-standard)
     (nature . "Zero-probability events - quantum topology connections via anyonic braiding"))
    ((name . "PolyglotFormalisms.jl")
     (relationship . sibling-standard)
     (nature . "Cross-language formal verification - knot invariants as verified operations")))

  (dependencies
    (runtime
      ("Julia" "1.9+")
      ("LinearAlgebra" "stdlib"))))
