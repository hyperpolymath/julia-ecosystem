# Cladistics.jl

[![Project Topology](https://img.shields.io/badge/Project-Topology-9558B2)](TOPOLOGY.md)
[![Completion Status](https://img.shields.io/badge/Completion-75%25-yellow)](TOPOLOGY.md)

[![License](https://img.shields.io/badge/license-PMPL--1.0--or--later-blue.svg)](LICENSE)
[![Julia](https://img.shields.io/badge/julia-1.6+-purple.svg)](https://julialang.org)

A Julia package for phylogenetic analysis and cladistics - the study of evolutionary relationships among organisms.

## Overview

Cladistics is a method of biological classification that groups organisms based on their evolutionary ancestry and shared derived characteristics (synapomorphies). This package provides computational tools for reconstructing and analyzing phylogenetic trees from molecular sequence data and morphological characters.

## What is Cladistics?

Cladistics revolutionized biological classification by emphasizing evolutionary relationships over superficial similarities. Key concepts:

- **Clade**: A monophyletic group consisting of an ancestor and all its descendants
- **Synapomorphy**: A shared derived characteristic that defines a clade
- **Phylogenetic Tree**: A branching diagram showing evolutionary relationships
- **Parsimony**: The principle that the simplest evolutionary explanation (fewest changes) is preferred
- **Bootstrap Analysis**: Statistical method to assess confidence in tree topology

## Features

### Distance-Based Methods
- **Multiple Distance Metrics**:
  - Hamming distance (simple count of differences)
  - p-distance (proportion of differences)
  - Jukes-Cantor 1969 (corrects for multiple substitutions)
  - Kimura 2-parameter (distinguishes transitions vs transversions)
- **UPGMA**: Unweighted Pair Group Method with Arithmetic Mean (assumes molecular clock)
- **Neighbor-Joining**: Does not assume molecular clock, handles rate variation

### Character-Based Methods
- **Maximum Parsimony**: Find trees requiring fewest evolutionary changes
- **Fitch Algorithm**: Efficient parsimony score calculation
- **Parsimony-Informative Sites**: Identify characters useful for phylogenetic inference

### Tree Analysis
- **Bootstrap Support**: Assess confidence in tree topology (1000+ replicates)
- **Clade Identification**: Extract well-supported monophyletic groups
- **Robinson-Foulds Distance**: Compare tree topologies quantitatively
- **Tree Rooting**: Root unrooted trees using outgroup taxa
- **Newick Format**: Export trees in standard phylogenetic format

## Installation

```julia
using Pkg
Pkg.add("Cladistics")
```

## Quick Start

```julia
using Cladistics
using Plots

# 1. DNA sequence data (aligned)
sequences = [
    "ATCGATCGATCG",  # Species A
    "ATCGATCGATCG",  # Species B (same as A)
    "ATCGTTCGATCG",  # Species C
    "TTCGATCGATCG",  # Species D
    "TTCGTTCGATCC"   # Species E (outgroup)
]
taxa_names = ["Species_A", "Species_B", "Species_C", "Species_D", "Species_E"]

# 2. Calculate evolutionary distances
dmat = distance_matrix(sequences, method=:jc69)
println("Distance matrix:")
display(dmat)

# 3. Build phylogenetic tree using UPGMA
upgma_tree = upgma(dmat, taxa_names=taxa_names)
newick = tree_to_newick(upgma_tree)
println("\nUPGMA tree (Newick): $newick")

# 4. Build tree using Neighbor-Joining
nj_tree = neighbor_joining(dmat, taxa_names=taxa_names)

# 5. Compare tree topologies
rf_distance = tree_distance(upgma_tree, nj_tree)
println("Robinson-Foulds distance: $rf_distance")

# 6. Bootstrap analysis for confidence
support = bootstrap_support(sequences, replicates=1000, method=:nj)
println("\nBootstrap support values:")
for (clade, value) in support
    println("  Clade $(collect(clade)): $(round(value*100, digits=1))%")
end

# 7. Identify well-supported clades
clades = identify_clades(upgma_tree, 0.95)
println("\nWell-supported clades (>95%):")
for clade in clades
    println("  ", clade)
end
```

## Distance Methods Comparison

Different evolutionary models for different scenarios:

```julia
using Cladistics

sequences = ["ATCG", "ATCG", "TTCG", "TTCC"]

# Simple counting (good for very similar sequences)
hamming = distance_matrix(sequences, method=:hamming)

# Proportion of differences (normalized)
p_dist = distance_matrix(sequences, method=:p_distance)

# Jukes-Cantor (corrects for multiple substitutions at same site)
jc69 = distance_matrix(sequences, method=:jc69)

# Kimura 2-parameter (accounts for transition/transversion bias)
k2p = distance_matrix(sequences, method=:k2p)

println("Hamming distance [1,3]: ", hamming[1,3])
println("p-distance [1,3]: ", p_dist[1,3])
println("Jukes-Cantor [1,3]: ", jc69[1,3])
println("Kimura 2P [1,3]: ", k2p[1,3])
```

## Tree Construction Methods

### UPGMA: Simple but assumes molecular clock

```julia
using Cladistics

# UPGMA assumes constant evolutionary rate (molecular clock)
# Produces ultrametric tree (all tips equidistant from root)
dmat = [0.0 0.2 0.4;
        0.2 0.0 0.3;
        0.4 0.3 0.0]

tree = upgma(dmat, taxa_names=["Human", "Chimp", "Gorilla"])

# Good for:
# - Species with similar generation times
# - Recent divergences
# - Molecular clock hypothesis valid

# Not good for:
# - Ancient divergences
# - Variable evolutionary rates
# - Rapid radiations
```

### Neighbor-Joining: No molecular clock assumption

```julia
using Cladistics

# NJ handles rate variation across lineages
# More sophisticated than UPGMA
dmat = [0.0 0.5 0.8 1.0;
        0.5 0.0 0.6 0.9;
        0.8 0.6 0.0 0.4;
        1.0 0.9 0.4 0.0]

tree = neighbor_joining(dmat, taxa_names=["A", "B", "C", "D"])

# Good for:
# - Variable evolutionary rates
# - Large datasets
# - When molecular clock violated

# Produces minimum evolution tree
```

## Bootstrap Analysis

Assess confidence in your phylogenetic tree:

```julia
using Cladistics
using Random

Random.seed!(42)  # Reproducible results

# Real mitochondrial DNA sequences (example)
sequences = [
    "ATCGATCGATCGATCG",
    "ATCGATCGATCGATCG",  # Identical to first
    "ATCGTTCGATCGATCG",
    "TTCGATCGATCGATCG",
    "TTCGTTCGATCGTTCC"
]

# Perform 1000 bootstrap replicates
# Resamples alignment columns with replacement
support = bootstrap_support(sequences, replicates=1000, method=:nj)

# Interpret support values:
# >95%: Strong support (publish with confidence)
# 70-95%: Moderate support (mention uncertainty)
# <70%: Weak support (tree topology unreliable)

for (clade, value) in sort(collect(support), by=x->x[2], rev=true)
    taxa = collect(clade)
    percent = round(value * 100, digits=1)

    if percent >= 95
        confidence = "Strong"
    elseif percent >= 70
        confidence = "Moderate"
    else
        confidence = "Weak"
    end

    println("Clade $taxa: $percent% ($confidence)")
end
```

## Maximum Parsimony

Find trees requiring the fewest evolutionary changes:

```julia
using Cladistics

# Aligned DNA sequences
alignment = [
    "ATCG",
    "ATCG",
    "TTCG",
    "TTCC"
]

# Convert to character matrix
char_matrix = character_state_matrix(alignment)

# Find parsimony-informative sites
# (sites where at least 2 states each appear in ≥2 taxa)
informative_sites = parsimony_informative_sites(char_matrix)
println("Parsimony-informative sites: $informative_sites")

# Build tree
dmat = distance_matrix(alignment, method=:hamming)
tree = upgma(dmat, taxa_names=["Tax1", "Tax2", "Tax3", "Tax4"])

# Calculate parsimony score (total number of changes)
score = calculate_parsimony_score(tree, char_matrix)
println("Parsimony score: $score changes")

# Lower scores = more parsimonious (preferred)
```

## Real-World Example: Primate Phylogeny

```julia
using Cladistics

# Partial mitochondrial cytochrome b sequences (simplified)
primate_sequences = [
    "ATCGATCGATCGATCGATCG",  # Human
    "ATCGATCGATCGATCGATCG",  # Chimp (very similar to human)
    "ATCGATCGTTCGATCGATCG",  # Gorilla
    "ATCGTTCGTTCGATCGTTCG",  # Orangutan
    "TTCGTTCGTTCGTTCGTTCG"   # Lemur (outgroup)
]

taxa = ["Human", "Chimp", "Gorilla", "Orangutan", "Lemur"]

# Use Kimura 2-parameter (best for DNA)
dmat = distance_matrix(primate_sequences, method=:k2p)

# Build NJ tree (no molecular clock assumption)
tree = neighbor_joining(dmat, taxa_names=taxa)

# Root on lemur (outgroup)
rooted_tree = root_tree(tree, "Lemur")

# Bootstrap confidence
support = bootstrap_support(primate_sequences, replicates=100, method=:nj)

# Export for visualization
newick = tree_to_newick(rooted_tree)
println("Newick format: $newick")

# Can import into FigTree, iTOL, or other phylogenetic viewers
```

## Character Evolution Example

```julia
using Cladistics

# Morphological character matrix
# Characters: [wings, feathers, warm-blooded, lays eggs]
# 0 = absent, 1 = present
taxa_morphology = [
    "AAAA",  # Lizard (outgroup)
    "AABB",  # Crocodile
    "BBBB",  # Chicken
    "BBBB",  # Eagle
    "BAAB"   # Bat
]

taxa_names = ["Lizard", "Crocodile", "Chicken", "Eagle", "Bat"]

# Build tree based on characters
dmat = distance_matrix(taxa_morphology, method=:hamming)
tree = upgma(dmat, taxa_names=taxa_names)

# Analyze character evolution
char_matrix = character_state_matrix(taxa_morphology)
parsimony_score = calculate_parsimony_score(tree, char_matrix)

println("Minimum evolutionary changes: $parsimony_score")

# Low score = good fit between characters and tree topology
```

## Comparing Alternative Hypotheses

```julia
using Cladistics

sequences = ["ATCG", "ATCG", "TTCG", "GTCC"]

# Hypothesis 1: UPGMA tree (assumes molecular clock)
dmat = distance_matrix(sequences, method=:jc69)
tree1 = upgma(dmat, taxa_names=["A", "B", "C", "D"])

# Hypothesis 2: Neighbor-Joining tree (no clock)
tree2 = neighbor_joining(dmat, taxa_names=["A", "B", "C", "D"])

# Compare topologies
rf_dist = tree_distance(tree1, tree2)

if rf_dist == 0
    println("Trees have identical topology")
else
    println("Trees differ by $rf_dist bipartitions")
end

# Calculate parsimony scores for both
char_matrix = character_state_matrix(sequences)
score1 = calculate_parsimony_score(tree1, char_matrix)
score2 = calculate_parsimony_score(tree2, char_matrix)

println("UPGMA parsimony score: $score1")
println("NJ parsimony score: $score2")

# Prefer tree with lower parsimony score (fewer changes)
```

## Key Concepts Explained

### Molecular Clock
- **Assumption**: Evolutionary rate is constant across lineages
- **When valid**: Recent species, similar generation times
- **Methods**: UPGMA assumes molecular clock
- **When violated**: Use Neighbor-Joining instead

### Bootstrap Support
- **Purpose**: Assess confidence in tree branches
- **Method**: Resample alignment columns with replacement
- **Interpretation**:
  - >95%: Publish with confidence
  - 70-95%: Mention uncertainty
  - <70%: Weak support, collect more data

### Parsimony vs Distance
- **Parsimony**: Find tree requiring fewest character changes
  - Good: Morphological data, theoretical clarity
  - Bad: Computationally expensive (NP-hard)
- **Distance**: Build tree from pairwise distances
  - Good: Fast, scales to large datasets
  - Bad: Information loss from pairwise comparisons

## References

### Classic Papers
- Felsenstein, J. (1985). "Confidence limits on phylogenies: An approach using the bootstrap." *Evolution*, 39(4), 783-791.
- Saitou, N., & Nei, M. (1987). "The neighbor-joining method: A new method for reconstructing phylogenetic trees." *Molecular Biology and Evolution*, 4(4), 406-425.
- Fitch, W. M. (1971). "Toward defining the course of evolution: Minimum change for a specific tree topology." *Systematic Zoology*, 20(4), 406-416.

### Textbooks
- Felsenstein, J. (2004). *Inferring Phylogenies*. Sinauer Associates.
- Lemey, P., Salemi, M., & Vandamme, A. M. (2009). *The Phylogenetic Handbook: A Practical Approach to Phylogenetic Analysis and Hypothesis Testing*. Cambridge University Press.
- Hall, B. G. (2011). *Phylogenetic Trees Made Easy: A How-To Manual*. Sinauer Associates.

### Evolutionary Models
- Jukes, T. H., & Cantor, C. R. (1969). "Evolution of protein molecules." In *Mammalian Protein Metabolism*, pp. 21-132.
- Kimura, M. (1980). "A simple method for estimating evolutionary rates of base substitutions." *Journal of Molecular Evolution*, 16(2), 111-120.

## Citation

If you use this package in research, please cite:

```bibtex
@software{cladistics_jl,
  author = {Jewell, Jonathan D.A.},
  title = {Cladistics.jl: Phylogenetic Analysis in Julia},
  year = {2026},
  url = {https://github.com/hyperpolymath/Cladistics.jl}
}
```

## Related Projects

- [BioJulia](https://github.com/BioJulia) - Broader bioinformatics ecosystem
- [PhyloNetworks.jl](https://github.com/crsl4/PhyloNetworks.jl) - Phylogenetic networks
- [Phylo.jl](https://github.com/richardreeve/Phylo.jl) - Alternative phylogenetics package

## External Tools

Visualize Newick trees with:
- [FigTree](http://tree.bio.ed.ac.uk/software/figtree/)
- [iTOL](https://itol.embl.de/)
- [ETE Toolkit](http://etetoolkit.org/)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the Palimpsest License (PMPL-1.0-or-later). See [LICENSE](LICENSE) for details.
