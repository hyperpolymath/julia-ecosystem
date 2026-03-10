;; SPDX-License-Identifier: PMPL-1.0-or-later
;; ECOSYSTEM.scm for Lithoglyph.jl
;; Media Type: application/vnd.ecosystem+scm

(ecosystem
  (version "1.0")
  (name "Lithoglyph.jl")
  (type "julia-package")
  (purpose "Native Julia client for the Lithoglyph symbolic database")

  (position-in-ecosystem
    (domain "databases")
    (role "Julia bindings and query interface for the Lithoglyph symbolic database")
    (maturity "alpha"))

  (related-projects
    ((name . "hyperpolymath ecosystem")
     (relationship . part-of)
     (nature . "Julia packages for interdisciplinary computing"))))
