;; SPDX-License-Identifier: PMPL-1.0-or-later
;; STATE.scm for Hyperpolymath.jl

(define state
  '((metadata
     (project . "Hyperpolymath.jl")
     (version . "0.1.0")
     (updated . "2026-03-10")
     (maintainers . ("Jonathan D.A. Jewell <j.d.a.jewell@open.ac.uk>")))

    (project-context
     (description . "Metapackage aggregating the hyperpolymath Julia ecosystem")
     (domain . "meta")
     (languages . ("Julia"))
     (primary-purpose . "Single entry point that re-exports all hyperpolymath Julia packages"))

    (current-position
     (phase . "alpha")
     (overall-completion . "metapackage"))

    (blockers-and-issues
     (known-issues . ()))

    (critical-next-actions
     (immediate . ("Keep dependency list in sync with ecosystem packages")))))
