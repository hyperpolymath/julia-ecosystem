;; SPDX-License-Identifier: PMPL-1.0-or-later
;; ECOSYSTEM.scm for PRComms.jl
;; Media Type: application/vnd.ecosystem+scm

(ecosystem
  (version "1.0")
  (name "PRComms.jl")
  (type "julia-package")
  (purpose "Strategic public relations and communications operations framework")
  (position-in-ecosystem
    (domain "communications")
    (role "PR campaign management and media communications operations")
    (maturity "alpha"))
  (related-projects
    ((name . "hyperpolymath ecosystem")
     (relationship . part-of)
     (nature . "Julia packages for interdisciplinary computing"))))
