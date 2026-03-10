;; SPDX-License-Identifier: PMPL-1.0-or-later
;; STATE.scm for AcceleratorGate.jl

(define state
  '((metadata
     (project . "AcceleratorGate.jl")
     (version . "0.1.0")
     (updated . "2026-03-10")
     (maintainers . ("Jonathan D.A. Jewell <j.d.a.jewell@open.ac.uk>")))

    (project-context
     (description . "Shared coprocessor dispatch infrastructure for the hyperpolymath Julia ecosystem")
     (domain . "hardware-acceleration")
     (languages . ("Julia"))
     (primary-purpose . "Unified dispatch layer for GPU, FPGA, and other coprocessor backends"))

    (current-position
     (phase . "alpha")
     (overall-completion . 5))

    (blockers-and-issues
     (known-issues . ()))

    (critical-next-actions
     (immediate . ("Define core dispatch interface and backend trait system")))))
