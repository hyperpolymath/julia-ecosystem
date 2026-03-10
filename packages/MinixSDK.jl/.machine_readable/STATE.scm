;; SPDX-License-Identifier: PMPL-1.0-or-later
;; STATE.scm for MinixSDK.jl

(define state
  '((metadata
     (project . "MinixSDK.jl")
     (version . "0.1.0")
     (updated . "2026-03-10")
     (maintainers . ("Jonathan D.A. Jewell <j.d.a.jewell@open.ac.uk>")))

    (project-context
     (description . "Cross-compilation toolkit for lowering Julia functions to MINIX 3 microkernel services")
     (domain . "systems-programming")
     (languages . ("Julia"))
     (primary-purpose . "Bridge Julia high-level abstractions to MINIX 3 microkernel service targets"))

    (current-position
     (phase . "skeleton")
     (overall-completion . 15))

    (blockers-and-issues
     (known-issues . ()))

    (critical-next-actions
     (immediate . ("Define cross-compilation pipeline and MINIX 3 service interface")))))
