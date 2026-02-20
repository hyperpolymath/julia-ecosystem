;; SPDX-License-Identifier: PMPL-1.0-or-later
;; PLAYBOOK.scm - Operational runbook for Cliodynamics.jl

(define playbook
  `((version . "1.0.0")
    (procedures
      ((deploy . (("build" . "julia --project=. -e 'using Pkg; Pkg.instantiate()'")
                  ("test" . "julia --project=. -e 'using Pkg; Pkg.test()'")
                  ("release" . "julia --project=. -e 'using Pkg; Pkg.build()'")
                  ("docs" . "julia --project=docs -e 'using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate(); include(\"docs/make.jl\")'")))
       (rollback . ())
       (debug . (("check-deps" . "julia --project=. -e 'using Pkg; Pkg.status()'")
                 ("repl" . "julia --project=.")))))
    (alerts . ())
    (contacts . ())))
