;; SPDX-License-Identifier: PMPL-1.0-or-later
;; PLAYBOOK.scm - Operational runbook for Cladistics.jl

(define playbook
  `((version . "1.0.0")
    (procedures
      ((deploy . (("build" . "julia --project=. -e 'using Pkg; Pkg.build()'")
                  ("test" . "julia --project=. -e 'using Pkg; Pkg.test()'")
                  ("release" . "just release")))
       (rollback . ())
       (debug . (("repl" . "julia --project=.")
                 ("test-verbose" . "julia --project=. test/runtests.jl")))))
    (alerts . ())
    (contacts . ())))
