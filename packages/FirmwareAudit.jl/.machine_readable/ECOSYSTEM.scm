;; SPDX-License-Identifier: PMPL-1.0-or-later
;; ECOSYSTEM.scm for FirmwareAudit.jl
;; Media Type: application/vnd.ecosystem+scm

(ecosystem
  (version "1.0")
  (name "FirmwareAudit.jl")
  (type "julia-package")
  (purpose "Firmware validation and security audit suite")

  (position-in-ecosystem
    (domain "security")
    (role "Automated firmware image analysis, vulnerability detection, and compliance auditing")
    (maturity "alpha"))

  (related-projects
    ((name . "hyperpolymath ecosystem")
     (relationship . part-of)
     (nature . "Julia packages for interdisciplinary computing"))))
