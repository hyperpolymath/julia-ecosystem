;; SPDX-License-Identifier: PMPL-1.0-or-later
;; META.scm for KnotTheory.jl

(define meta
  '((project-meta
     (name . "KnotTheory.jl")
     (tagline . "Knot theory invariants and planar diagram analysis")
     (category . "mathematics-topology")
     (license . "PMPL-1.0-or-later")
     (inception-date . "2024")
     (repository . "https://github.com/hyperpolymath/KnotTheory.jl"))

    (architecture-decisions
     (adr-001
       (title . "Use Kauffman bracket for Jones polynomial")
       (status . accepted)
       (decision . "Implement Jones via Kauffman bracket recursion"))
     (adr-002
       (title . "Return Dict{Int,Int} for polynomials")
       (status . accepted)
       (decision . "Sparse representation with to_polynomial converter")))

    (design-rationale
     (core-principles
       "Exact computation (no floating point for invariants)"
       "Recursion limits for safety"
       "PD code as primary representation"))))
