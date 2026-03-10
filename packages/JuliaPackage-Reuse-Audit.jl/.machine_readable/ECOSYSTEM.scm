;; SPDX-License-Identifier: PMPL-1.0-or-later
;; ECOSYSTEM.scm for JuliaPackage-Reuse-Audit.jl
;; Media Type: application/vnd.ecosystem+scm

(ecosystem
  (version "1.0")
  (name "JuliaPackage-Reuse-Audit.jl")
  (type "julia-package")
  (purpose "Automated Julia package scaffolding generator")
  (position-in-ecosystem
    (domain "developer-tools")
    (role "Package scaffolding and reuse auditing for Julia ecosystem")
    (maturity "beta"))
  (related-projects
    ((name . "hyperpolymath ecosystem")
     (relationship . part-of)
     (nature . "Julia packages for interdisciplinary computing"))))
