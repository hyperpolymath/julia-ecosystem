;; SPDX-License-Identifier: PMPL-1.0-or-later
;; ECOSYSTEM.scm for HardwareResilience.jl
;; Media Type: application/vnd.ecosystem+scm

(ecosystem
  (version "1.0")
  (name "HardwareResilience.jl")
  (type "julia-package")
  (purpose "Self-healing kernel guardian for hardware faults")

  (position-in-ecosystem
    (domain "systems-programming")
    (role "Detection, mitigation, and recovery from hardware faults at the kernel and driver level")
    (maturity "alpha"))

  (related-projects
    ((name . "hyperpolymath ecosystem")
     (relationship . part-of)
     (nature . "Julia packages for interdisciplinary computing"))))
