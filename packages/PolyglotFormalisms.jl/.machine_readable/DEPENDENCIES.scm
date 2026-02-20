;; SPDX-License-Identifier: PMPL-1.0-or-later
;; DEPENDENCIES.scm - PolyglotFormalisms.jl dependency specification

(dependencies
  (version "1.0")
  (updated "2025-01-23")

  (runtime
    ((name . "Julia")
     (version . ">=1.9")
     (required . #t)
     (reason . "Core language runtime")
     (source . "https://julialang.org"))

    ((name . "Test")
     (version . "1.11.0")
     (required . #t)
     (reason . "Standard library for testing")
     (source . "Julia stdlib")
     (scope . "test")))

  (planned
    ((name . "Axiom.jl")
     (version . ">=0.1")
     (required . #f)
     (reason . "Formal verification with @prove macros")
     (source . "https://github.com/hyperpolymath/Axiom.jl")
     (status . "integration-planned-v0.4.0"))

    ((name . "SMTLib.jl")
     (version . ">=0.1")
     (required . #f)
     (reason . "SMT solver integration for cross-language equivalence proofs")
     (source . "https://github.com/hyperpolymath/SMTLib.jl")
     (status . "integration-planned-v0.4.0")))

  (build-tools
    ((name . "Julia Package Manager")
     (version . "builtin")
     (required . #t)
     (reason . "Dependency management and testing"))

    ((name . "GitHub Actions")
     (version . "latest")
     (required . #t)
     (reason . "CI/CD automation"))

    ((name . "julia-actions/setup-julia")
     (version . "v2@f2258781")
     (required . #t)
     (reason . "CI Julia installation")
     (sha-pinned . #t))

    ((name . "julia-actions/cache")
     (version . "v2@824c9ea9")
     (required . #t)
     (reason . "CI dependency caching")
     (sha-pinned . #t)))

  (security-tools
    ((name . "CodeQL")
     (version . "v4@a4784f2d")
     (required . #t)
     (reason . "Static security analysis")
     (sha-pinned . #t))

    ((name . "OpenSSF Scorecard")
     (version . "v2.4.0@62b2cac7")
     (required . #t)
     (reason . "Supply chain security scoring")
     (sha-pinned . #t)))

  (zero-npm-policy
    (enforced . #t)
    (rationale . "RSR policy: Julia projects use Pkg, not npm/Node.js")
    (alternatives . ("Julia Pkg" "Deno for JS interop if needed")))

  (supply-chain-security
    (all-actions-sha-pinned . #t)
    (minimal-dependencies . #t)
    (dependency-count . 1)
    (stdlib-only . #t)))
