;; SPDX-License-Identifier: PMPL-1.0-or-later
;; AGENTIC.scm for Exnovation.jl
;; Agent behavior directives

(define agentic-profile
  '((agent-guidelines
     (code-style
       (language . "Julia")
       (conventions . ("Follow Julia style guide" "Use descriptive names" "Type annotations for public API"))
       (patterns . ("Prefer immutable types" "Functional approach" "Explicit over implicit")))

     (testing-requirements
       (always-run-tests . #t)
       (minimum-coverage . 80)
       (test-command . "julia --project=. test/runtests.jl")
       (expect-passing . 32))

     (documentation-standards
       (api-docs . "Documenter.jl with @docs blocks")
       (examples-required . #t)
       (docstring-format . "Julia docstrings with type signatures"))

     (modification-protocol
       (read-state-first . #t)
       (test-after-changes . #t)
       (update-docs-if-api-changes . #t)
       (commit-message-format . "conventional-commits"))

     (safety-constraints
       (never-remove-tests . #t)
       (preserve-public-api . #t)
       (backwards-compatibility . "within major version")
       (require-validation-on-inputs . #t)))

    (autonomous-capabilities
     (allowed-actions
       "Implement missing functions from TODO comments"
       "Add tests for untested code paths"
       "Fix formatting and style issues"
       "Update documentation to match code"
       "Refactor without changing behavior"
       "Add input validation")

     (requires-approval
       "Change public API signatures"
       "Remove public functions"
       "Modify test assertions"
       "Change algorithm behavior"
       "Add new dependencies"))

    (tool-preferences
     (formatter . "Julia built-in formatter")
     (linter . "panic-attack")
     (docs-generator . "Documenter.jl")
     (test-runner . "Julia Test stdlib"))

    (learning-directives
     (domain-knowledge
       "Organizational change theory"
       "Sociotechnical systems"
       "Behavioral economics"
       "Legacy system modernization")

     (reference-materials
       "README.md for project overview"
       "docs/src/api.md for API structure"
       "test/runtests.jl for behavior examples"
       "STATE.scm for current status"))))
