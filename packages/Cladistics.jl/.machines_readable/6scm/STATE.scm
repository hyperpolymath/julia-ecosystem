;; SPDX-License-Identifier: PMPL-1.0-or-later
;; STATE.scm - Project state tracking for Cladistics.jl
;; Media-Type: application/vnd.state+scm

(define-state Cladistics.jl
  (metadata
    (version "0.2.0")
    (schema-version "1.0.0")
    (created "2026-01-30")
    (updated "2026-02-12")
    (project "Cladistics.jl")
    (repo "hyperpolymath/Cladistics.jl"))

  (project-context
    (name "Cladistics.jl")
    (tagline "Julia phylogenetics library with distance methods, tree building, and parsimony analysis")
    (tech-stack ("Julia" "LinearAlgebra" "Graphs" "Clustering" "Distances" "DataFrames")))

  (current-position
    (phase "active-development")
    (overall-completion 85)
    (components
      ("distance-metrics" . 100)
      ("tree-building" . 100)
      ("parsimony" . 100)
      ("bootstrap-support" . 100)
      ("clade-identification" . 100)
      ("tree-comparison" . 100)
      ("newick-export" . 100)
      ("tree-rerooting" . 100)
      ("newick-parser" . 0))
    (working-features
      "Hamming distance calculation"
      "p-distance calculation"
      "Jukes-Cantor 69 distance"
      "Kimura 2-parameter distance"
      "UPGMA tree construction"
      "Neighbor-Joining tree construction"
      "Maximum parsimony search with stepwise addition"
      "Fitch algorithm for parsimony scoring"
      "Bootstrap support calculation"
      "Clade identification"
      "Robinson-Foulds distance"
      "Newick format export"
      "Tree rerooting on outgroup"))

  (route-to-mvp
    (milestones
      ((name "v0.1.0 - Core algorithms")
       (status "done")
       (completion 100)
       (items
         ("Distance matrix calculations" . done)
         ("UPGMA implementation" . done)
         ("Neighbor-Joining implementation" . done)
         ("Fitch parsimony scoring" . done)
         ("Bootstrap support" . done)
         ("Clade identification" . done)
         ("Robinson-Foulds distance" . done)
         ("Newick export" . done)))
      ((name "v0.2.0 - Maximum parsimony and rerooting")
       (status "done")
       (completion 100)
       (items
         ("Fix fitch_score return type bug" . done)
         ("Fix K2P operator precedence bug" . done)
         ("Implement tree rerooting" . done)
         ("Implement maximum_parsimony search" . done)
         ("Add parsimony tests" . done)))
      ((name "v0.3.0 - Newick I/O")
       (status "in-progress")
       (completion 50)
       (items
         ("Implement Newick parser" . in-progress)
         ("Add round-trip tests" . todo)))
      ((name "v1.0.0 - Package registration")
       (status "todo")
       (completion 0)
       (items
         ("Julia General registry submission" . todo)
         ("Full API documentation" . todo)
         ("Performance benchmarks" . todo)
         ("Tree visualization examples" . todo)))))

  (blockers-and-issues
    (critical ())
    (high ())
    (medium
      ("Newick parser not yet implemented"))
    (low
      ("Could add more distance metrics (e.g., LogDet)"
       "Could add maximum likelihood methods")))

  (critical-next-actions
    (immediate
      "Complete Newick parser implementation"
      "Add comprehensive tests for parser")
    (this-week
      "Update all documentation placeholders"
      "Verify all RSR template sections customized")
    (this-month
      "Prepare for Julia General registry submission"
      "Add performance benchmarks"
      "Write user guide with examples"))

  (session-history
    ((date "2026-02-12")
     (actions
       "Fixed fitch_score inconsistent return type"
       "Fixed K2P operator precedence bug"
       "Implemented root_tree with midpoint rerooting"
       "Implemented maximum_parsimony with stepwise addition"
       "Fixed SPDX license headers (AGPL -> PMPL)"
       "Customized STATE.scm for Cladistics.jl"))))

;; Helper functions
(define (get-completion-percentage state)
  (current-position 'overall-completion state))

(define (get-blockers state severity)
  (blockers-and-issues severity state))

(define (get-milestone state name)
  (find (lambda (m) (equal? (car m) name))
        (route-to-mvp 'milestones state)))
