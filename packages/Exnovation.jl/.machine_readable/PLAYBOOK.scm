;; SPDX-License-Identifier: PMPL-1.0-or-later
;; PLAYBOOK.scm for Exnovation.jl
;; Operational playbook for common scenarios

(define playbook
  '((common-operations
     (setup-dev-environment
       (steps
         "Clone repository to ~/Documents/hyperpolymath-repos/"
         "cd Exnovation.jl"
         "julia --project=. -e 'using Pkg; Pkg.instantiate()'"
         "julia --project=. test/runtests.jl")
       (expected-outcome . "32 tests passing"))

     (add-new-feature
       (steps
         "Read STATE.scm and docs/src/api.md"
         "Identify integration points in src/Exnovation.jl"
         "Implement function with docstring"
         "Add tests in test/runtests.jl"
         "Update docs/src/api.md if public API"
         "Run tests: julia --project=. test/runtests.jl"
         "Update STATE.scm completion percentage"
         "Commit with conventional-commits format")
       (expected-outcome . "All tests pass, docs updated"))

     (fix-bug
       (steps
         "Write failing test that reproduces bug"
         "Implement fix"
         "Verify test now passes"
         "Check no regressions: run full suite"
         "Update STATE.scm if blocker resolved"
         "Commit with fix: prefix")
       (expected-outcome . "Bug fixed, test added"))

     (release-new-version
       (steps
         "Update version in Project.toml"
         "Update CHANGELOG.md"
         "Update STATE.scm version and date"
         "Run full test suite"
         "Commit: chore: release v X.Y.Z"
         "Tag: git tag -a vX.Y.Z -m 'Release vX.Y.Z'"
         "Push: git push origin main --tags"
         "Push to mirrors: git push gitlab main --tags")
       (expected-outcome . "New version available on all forges")))

    (troubleshooting
     (tests-fail
       (diagnosis-steps
         "Check error message in output"
         "Isolate failing test"
         "Run single test with --verbose"
         "Check if data assumptions changed")
       (common-causes
         "DataFrame column type mismatch (String vs Symbol)"
         "Missing dependency"
         "API signature changed"
         "Test data format changed"))

     (docs-not-building
       (diagnosis-steps
         "Check docs/make.jl syntax"
         "Verify all @docs references exist"
         "Check Documenter.jl version compatibility")
       (common-causes
         "Missing docstring"
         "Incorrect @docs block path"
         "Documenter.jl version mismatch"))

     (ci-workflow-failing
       (diagnosis-steps
         "Check .github/workflows/ for syntax errors"
         "Verify secrets are configured"
         "Check action SHA pins are current"
         "Ensure permissions: read-all present")
       (common-causes
         "Missing workflow permissions"
         "Unpinned GitHub Actions"
         "Secret not configured")))

    (maintenance-schedule
     (weekly
       "Run panic-attack assail . --output scan.json"
       "Review open issues and PRs"
       "Check CI status")

     (monthly
       "Update dependencies"
       "Review and update STATE.scm"
       "Check for upstream Julia ecosystem changes")

     (quarterly
       "Review and update documentation"
       "Analyze test coverage"
       "Conduct security audit"
       "Update ECOSYSTEM.scm relationships"))

    (emergency-procedures
     (critical-bug-found
       (steps
         "Create issue with [CRITICAL] prefix"
         "Write failing test"
         "Implement fix on hotfix/ branch"
         "Fast-track review and merge"
         "Release patch version immediately"
         "Notify users via GitHub discussions"))

     (security-vulnerability
       (steps
         "DO NOT open public issue"
         "Email jonathan.jewell@open.ac.uk"
         "Create private security advisory"
         "Develop fix in private fork"
         "Coordinate disclosure with reporter"
         "Release security patch"
         "Publish advisory"))

     (breaking-change-needed
       (steps
         "Document why breaking change necessary"
         "Create RFC in discussions"
         "Gather community feedback"
         "Implement on feature/ branch"
         "Update migration guide"
         "Bump major version"
         "Announce with examples"))))
