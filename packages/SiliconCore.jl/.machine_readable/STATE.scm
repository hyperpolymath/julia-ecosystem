;; SPDX-License-Identifier: PMPL-1.0-or-later
;; STATE.scm for SiliconCore.jl

(define state
  '((metadata
     (project . "SiliconCore.jl")
     (version . "0.0.0-dev")
     (updated . "2026-03-10")
     (maintainers . ("Jonathan D.A. Jewell <j.d.a.jewell@open.ac.uk>")))

    (project-context
     (description . "Hardware capability detection and low-level vector operations")
     (domain . "hardware-acceleration")
     (languages . ("Julia"))
     (primary-purpose . "Detect hardware features and provide optimised vector operation primitives"))

    (current-position
     (phase . "skeleton")
     (overall-completion . 10))

    (blockers-and-issues
     (known-issues . ()))

    (critical-next-actions
     (immediate . ("Implement CPU feature detection and SIMD capability probing")))))
