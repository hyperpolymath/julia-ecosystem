;; SPDX-License-Identifier: PMPL-1.0-or-later
;; ECOSYSTEM.scm for TradeUnionist.jl
;; Media Type: application/vnd.ecosystem+scm

(ecosystem
  (version "1.0")
  (name "TradeUnionist.jl")
  (type "julia-package")
  (purpose "Labor organizing toolkit with membership, grievances, and geographic analysis")
  (position-in-ecosystem
    (domain "labor-relations")
    (role "Union membership management, grievance tracking, and geographic organizing analysis")
    (maturity "alpha"))
  (related-projects
    ((name . "hyperpolymath ecosystem")
     (relationship . part-of)
     (nature . "Julia packages for interdisciplinary computing"))))
