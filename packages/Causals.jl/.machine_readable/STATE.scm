; SPDX-License-Identifier: PMPL-1.0-or-later
; Causals.jl - Project State
; Updated: 2026-02-12

(state
  (version "0.2.0")
  (status "alpha")

  (current-milestone
    (name "Core Implementation Complete")
    (completion 95)
    (description "All 19 SONNET-TASKS.md tasks completed"))

  (components
    (module "DempsterShafer" status: complete)
    (module "BradfordHill" status: complete)
    (module "CausalDAG" status: complete)
    (module "Granger" status: complete)
    (module "PropensityScore" status: complete)
    (module "DoCalculus" status: complete)
    (module "Counterfactuals" status: complete)
    (tests "105 passing" status: complete)
    (examples "2 working examples" status: complete)
    (documentation "10 pages" status: complete))

  (next-actions
    "Publish v0.2.0 release"
    "Add more comprehensive examples"
    "Expand test coverage for edge cases"))
