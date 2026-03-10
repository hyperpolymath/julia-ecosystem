;; SPDX-License-Identifier: PMPL-1.0-or-later
;; STATE.scm for FirmwareAudit.jl

(define state
  '((metadata
     (project . "FirmwareAudit.jl")
     (version . "unreleased")
     (updated . "2026-03-10")
     (maintainers . ("Jonathan D.A. Jewell <j.d.a.jewell@open.ac.uk>")))

    (project-context
     (description . "Firmware validation and security audit suite")
     (domain . "security")
     (languages . ("Julia"))
     (primary-purpose . "Automated firmware image analysis, vulnerability detection, and compliance auditing"))

    (current-position
     (phase . "skeleton")
     (overall-completion . 5))

    (blockers-and-issues
     (known-issues . ()))

    (critical-next-actions
     (immediate . ("Define firmware parsing interfaces and audit rule engine")))))
