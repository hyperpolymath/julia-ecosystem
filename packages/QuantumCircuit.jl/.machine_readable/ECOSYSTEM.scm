;; SPDX-License-Identifier: PMPL-1.0-or-later
;; ECOSYSTEM.scm for QuantumCircuit.jl
;; Media Type: application/vnd.ecosystem+scm

(ecosystem
  (version "1.0")
  (name "QuantumCircuit.jl")
  (type "julia-package")
  (purpose "Quantum circuit simulation framework with backend abstraction")

  (position-in-ecosystem
    (domain "quantum-computing")
    (role "Composable quantum circuit simulation and gate library")
    (maturity "alpha"))

  (related-projects
    ((name . "hyperpolymath ecosystem")
     (relationship . part-of)
     (nature . "Julia packages for interdisciplinary computing"))))
