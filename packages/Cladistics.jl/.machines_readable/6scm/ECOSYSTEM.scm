;; SPDX-License-Identifier: PMPL-1.0-or-later
;; ECOSYSTEM.scm - Ecosystem relationships for Cladistics.jl
;; Media-Type: application/vnd.ecosystem+scm

(ecosystem
  (version "1.0.0")
  (name "Cladistics.jl")
  (type "library")
  (purpose "Julia phylogenetics library for evolutionary tree reconstruction and analysis")

  (position-in-ecosystem
    "Cladistics.jl is a pure Julia implementation of phylogenetic tree construction "
    "and analysis methods, providing distance-based (UPGMA, Neighbor-Joining) and "
    "character-based (maximum parsimony) algorithms. It fills the gap for lightweight, "
    "fast phylogenetic inference in the Julia ecosystem, complementing the broader "
    "BioJulia ecosystem for computational biology.")

  (related-projects
    (dependency "Graphs.jl" "Graph data structures for tree representation")
    (dependency "Clustering.jl" "UPGMA clustering algorithm")
    (dependency "Distances.jl" "Distance metric calculations")
    (dependency "DataFrames.jl" "Tabular data handling")
    (dependency "LinearAlgebra" "Matrix operations")
    (dependency "Statistics" "Statistical computations")
    (related "PhyloNetworks.jl" "Phylogenetic networks and more advanced methods")
    (related "BioSequences.jl" "Biological sequence types")
    (related "PhyloTrees.jl" "Phylogenetic tree data structures")
    (inspiration "MEGA" "Molecular Evolutionary Genetics Analysis")
    (inspiration "RAxML" "Maximum likelihood phylogenetic inference")
    (inspiration "PAUP*" "Phylogenetic Analysis Using Parsimony"))

  (what-this-is
    "Cladistics.jl is a library for constructing and analyzing phylogenetic trees "
    "from molecular sequence data. It provides: "
    "(1) Four distance metrics: Hamming, p-distance, Jukes-Cantor 69, Kimura 2-parameter; "
    "(2) Tree building: UPGMA, Neighbor-Joining, maximum parsimony; "
    "(3) Tree analysis: bootstrap support, clade identification, Robinson-Foulds distance; "
    "(4) Tree manipulation: rerooting, Newick format I/O. "
    "Designed for bioinformatics workflows, evolutionary biology research, and "
    "educational use in computational phylogenetics courses.")

  (what-this-is-not
    "Cladistics.jl is not a maximum likelihood or Bayesian phylogenetic inference tool. "
    "For those methods, see PhyloNetworks.jl or external tools like RAxML-NG or MrBayes. "
    "It does not handle multiple sequence alignment - use external aligners or BioAlignments.jl. "
    "It is not a visualization library - use Phylo.jl or Makie.jl for tree plotting."))
