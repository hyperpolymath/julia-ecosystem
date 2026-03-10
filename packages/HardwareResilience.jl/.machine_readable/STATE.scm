;; SPDX-License-Identifier: PMPL-1.0-or-later
;; STATE.scm for HardwareResilience.jl

(define state
  '((metadata
     (project . "HardwareResilience.jl")
     (version . "unreleased")
     (updated . "2026-03-10")
     (maintainers . ("Jonathan D.A. Jewell <j.d.a.jewell@open.ac.uk>")))

    (project-context
     (description . "Self-healing kernel guardian for hardware faults")
     (domain . "systems-programming")
     (languages . ("Julia"))
     (primary-purpose . "Detection, mitigation, and recovery from hardware faults at the kernel and driver level"))

    (current-position
     (phase . "skeleton")
     (overall-completion . 10))

    (blockers-and-issues
     (known-issues . ()))

    (critical-next-actions
     (immediate . ("Define fault detection interfaces and self-healing recovery strategies")))))
