; SPDX-License-Identifier: PMPL-1.0-or-later
;; guix.scm — GNU Guix package definition for julia-ecosystem
;; Usage: guix shell -f guix.scm

(use-modules (guix packages)
             (guix build-system gnu)
             (guix licenses))

(package
  (name "julia-ecosystem")
  (version "0.1.0")
  (source #f)
  (build-system gnu-build-system)
  (synopsis "julia-ecosystem")
  (description "julia-ecosystem — part of the hyperpolymath ecosystem.")
  (home-page "https://github.com/hyperpolymath/julia-ecosystem")
  (license ((@@ (guix licenses) license) "PMPL-1.0-or-later"
             "https://github.com/hyperpolymath/palimpsest-license")))
