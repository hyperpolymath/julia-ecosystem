# SPDX-License-Identifier: PMPL-1.0-or-later

"""
    Cladistics

A Julia package for phylogenetic analysis and cladistics - the study of evolutionary
relationships among organisms based on shared derived characteristics.

Cladistics is a method of biological classification that groups organisms based on
their evolutionary ancestry and shared derived traits (synapomorphies). This package
provides tools for:

- Computing evolutionary distances between taxa
- Constructing phylogenetic trees using multiple algorithms (UPGMA, Neighbor-Joining, Maximum Parsimony)
- Analyzing character states and evolutionary changes
- Identifying monophyletic clades (groups of organisms with a common ancestor)
- Assessing tree reliability through bootstrap analysis
- Comparing different phylogenetic hypotheses

# Key Concepts

- **Taxa**: Organisms or groups being classified
- **Characters**: Observable traits used for classification
- **Clade**: A monophyletic group (an ancestor and all its descendants)
- **Phylogenetic tree**: A diagram showing evolutionary relationships
- **Bootstrap support**: Statistical confidence in tree branches
- **Parsimony**: Principle that the simplest explanation (fewest evolutionary changes) is preferred

# Examples

```julia
using Cladistics

# Calculate distance matrix from sequence data
sequences = ["ATCG", "ATCG", "TTCG", "TTCC"]
dmat = distance_matrix(sequences, method=:hamming)

# Build phylogenetic tree using UPGMA
tree = upgma(dmat)

# Perform bootstrap analysis
support = bootstrap_support(sequences, replicates=1000)

# Identify well-supported clades
clades = identify_clades(tree, 0.95)
```
"""
module Cladistics

using LinearAlgebra
using Statistics
using Clustering
using Distances
using DataFrames
using Graphs

export distance_matrix, upgma, neighbor_joining, maximum_parsimony,
       bootstrap_support, tree_distance, identify_clades, character_state_matrix,
       PhylogeneticTree, TreeNode, calculate_parsimony_score, root_tree,
       tree_to_newick, parse_newick, parsimony_informative_sites

# Data structures

"""
    TreeNode

Represents a node in a phylogenetic tree.

# Fields
- `name::String`: Name of the taxon (for terminal nodes) or internal label
- `children::Vector{TreeNode}`: Child nodes
- `branch_length::Float64`: Length of branch leading to this node
- `support::Float64`: Bootstrap or other support value
- `parent::Union{Nothing,TreeNode}`: Parent node reference
"""
mutable struct TreeNode
    name::String
    children::Vector{TreeNode}
    branch_length::Float64
    support::Float64
    parent::Union{Nothing,TreeNode}
end

TreeNode(name::String) = TreeNode(name, TreeNode[], 0.0, 1.0, nothing)

"""
    PhylogeneticTree

Represents a complete phylogenetic tree.

# Fields
- `root::TreeNode`: Root node of the tree
- `taxa::Vector{String}`: List of taxa names
- `method::Symbol`: Method used to construct tree (:upgma, :nj, :parsimony)
"""
struct PhylogeneticTree
    root::TreeNode
    taxa::Vector{String}
    method::Symbol
end

# Distance matrix calculations

"""
    distance_matrix(sequences::Vector{String}; method=:hamming) -> Matrix{Float64}

Calculate pairwise evolutionary distances between sequences.

# Arguments
- `sequences`: Vector of aligned sequence strings (DNA, RNA, or protein)
- `method`: Distance metric to use
  - `:hamming` - Simple count of differences
  - `:jc69` - Jukes-Cantor 1969 correction
  - `:k2p` - Kimura 2-parameter model
  - `:p_distance` - Proportion of differences

# Returns
Symmetric matrix of pairwise distances

# Examples
```julia
seqs = ["ATCG", "ATCG", "TTCG", "TTCC"]
dmat = distance_matrix(seqs, method=:hamming)
```
"""
function distance_matrix(sequences::Vector{String}; method=:hamming)
    n = length(sequences)
    dmat = zeros(Float64, n, n)

    for i in 1:n
        for j in (i+1):n
            dist = if method == :hamming
                hamming_distance(sequences[i], sequences[j])
            elseif method == :jc69
                jukes_cantor_distance(sequences[i], sequences[j])
            elseif method == :k2p
                kimura_2p_distance(sequences[i], sequences[j])
            elseif method == :p_distance
                p_distance(sequences[i], sequences[j])
            else
                error("Unknown distance method: $method")
            end

            dmat[i, j] = dist
            dmat[j, i] = dist
        end
    end

    return dmat
end

function hamming_distance(seq1::String, seq2::String)
    count(i -> seq1[i] != seq2[i], 1:min(length(seq1), length(seq2)))
end

function p_distance(seq1::String, seq2::String)
    len = min(length(seq1), length(seq2))
    diffs = count(i -> seq1[i] != seq2[i], 1:len)
    return diffs / len
end

function jukes_cantor_distance(seq1::String, seq2::String)
    p = p_distance(seq1, seq2)
    p >= 0.75 && return Inf  # Saturation
    return -0.75 * log(1.0 - (4.0 * p / 3.0))
end

function kimura_2p_distance(seq1::String, seq2::String)
    len = min(length(seq1), length(seq2))
    transitions = 0
    transversions = 0

    for i in 1:len
        a, b = seq1[i], seq2[i]
        if a != b
            if (a in "AG" && b in "AG") || (a in "CT" && b in "CT")
                transitions += 1
            else
                transversions += 1
            end
        end
    end

    P = transitions / len  # Transition frequency
    Q = transversions / len  # Transversion frequency

    if (1.0 - 2.0 * P - Q) <= 0.0 || (1.0 - 2.0 * Q) <= 0.0
        return Inf
    end

    return -0.5 * log((1.0 - 2.0 * P - Q) * sqrt(1.0 - 2.0 * Q))
end

# UPGMA algorithm

"""
    upgma(dmat::Matrix{Float64}; taxa_names=nothing) -> PhylogeneticTree

Construct phylogenetic tree using UPGMA (Unweighted Pair Group Method with Arithmetic Mean).

UPGMA assumes a constant rate of evolution (molecular clock) and produces an ultrametric
tree where all tips are equidistant from the root.

# Arguments
- `dmat`: Symmetric distance matrix
- `taxa_names`: Optional vector of taxon names (defaults to numbered labels)

# Returns
`PhylogeneticTree` object with ultrametric topology

# Examples
```julia
dmat = [0.0 0.2 0.4; 0.2 0.0 0.3; 0.4 0.3 0.0]
tree = upgma(dmat, taxa_names=["A", "B", "C"])
```
"""
function upgma(dmat::Matrix{Float64}; taxa_names=nothing)
    n = size(dmat, 1)
    names = taxa_names === nothing ? ["T$i" for i in 1:n] : taxa_names

    # Initialize clusters
    clusters = [TreeNode(names[i]) for i in 1:n]
    cluster_sizes = ones(Int, n)
    active = trues(n)
    distances = copy(dmat)
    heights = zeros(n)

    while count(active) > 1
        # Find minimum distance between active clusters
        min_dist = Inf
        min_i, min_j = 0, 0

        for i in 1:n
            active[i] || continue
            for j in (i+1):n
                active[j] || continue
                if distances[i, j] < min_dist
                    min_dist = distances[i, j]
                    min_i, min_j = i, j
                end
            end
        end

        # Create new internal node
        new_height = min_dist / 2.0
        new_node = TreeNode("Node$(n - count(active) + 1)")

        # Set branch lengths
        clusters[min_i].branch_length = new_height - heights[min_i]
        clusters[min_j].branch_length = new_height - heights[min_j]

        # Add children
        push!(new_node.children, clusters[min_i], clusters[min_j])
        clusters[min_i].parent = new_node
        clusters[min_j].parent = new_node

        # Update distances using UPGMA formula
        new_size = cluster_sizes[min_i] + cluster_sizes[min_j]
        for k in 1:n
            active[k] || k == min_i || k == min_j || continue
            new_dist = (cluster_sizes[min_i] * distances[min_i, k] +
                       cluster_sizes[min_j] * distances[min_j, k]) / new_size
            distances[min_i, k] = new_dist
            distances[k, min_i] = new_dist
        end

        # Update state
        clusters[min_i] = new_node
        cluster_sizes[min_i] = new_size
        heights[min_i] = new_height
        active[min_j] = false
    end

    root_idx = findfirst(active)
    return PhylogeneticTree(clusters[root_idx], names, :upgma)
end

# Neighbor-Joining algorithm

"""
    neighbor_joining(dmat::Matrix{Float64}; taxa_names=nothing) -> PhylogeneticTree

Construct phylogenetic tree using Neighbor-Joining algorithm.

NJ does not assume a molecular clock and can handle varying rates of evolution.
It produces an unrooted tree that minimizes total branch length.

# Arguments
- `dmat`: Symmetric distance matrix
- `taxa_names`: Optional vector of taxon names

# Returns
`PhylogeneticTree` object with minimum evolution topology

# Examples
```julia
dmat = [0.0 0.2 0.4; 0.2 0.0 0.3; 0.4 0.3 0.0]
tree = neighbor_joining(dmat, taxa_names=["A", "B", "C"])
```
"""
function neighbor_joining(dmat::Matrix{Float64}; taxa_names=nothing)
    n = size(dmat, 1)
    names = taxa_names === nothing ? ["T$i" for i in 1:n] : taxa_names

    nodes = [TreeNode(names[i]) for i in 1:n]
    active = trues(n)
    distances = copy(dmat)

    while count(active) > 2
        n_active = count(active)

        # Calculate Q matrix
        Q = zeros(n, n)
        for i in 1:n
            active[i] || continue
            for j in (i+1):n
                active[j] || continue
                row_sum_i = sum(distances[i, k] for k in 1:n if active[k])
                row_sum_j = sum(distances[j, k] for k in 1:n if active[k])
                Q[i, j] = (n_active - 2) * distances[i, j] - row_sum_i - row_sum_j
                Q[j, i] = Q[i, j]
            end
        end

        # Find minimum Q value
        min_q = Inf
        min_i, min_j = 0, 0
        for i in 1:n
            active[i] || continue
            for j in (i+1):n
                active[j] || continue
                if Q[i, j] < min_q
                    min_q = Q[i, j]
                    min_i, min_j = i, j
                end
            end
        end

        # Calculate branch lengths
        row_sum_i = sum(distances[min_i, k] for k in 1:n if active[k])
        row_sum_j = sum(distances[min_j, k] for k in 1:n if active[k])

        branch_i = 0.5 * distances[min_i, min_j] +
                  (row_sum_i - row_sum_j) / (2 * (n_active - 2))
        branch_j = distances[min_i, min_j] - branch_i

        # Create new internal node
        new_node = TreeNode("Node$(n - count(active) + 1)")
        nodes[min_i].branch_length = max(0.0, branch_i)
        nodes[min_j].branch_length = max(0.0, branch_j)

        push!(new_node.children, nodes[min_i], nodes[min_j])
        nodes[min_i].parent = new_node
        nodes[min_j].parent = new_node

        # Update distances
        for k in 1:n
            active[k] || k == min_i || k == min_j || continue
            new_dist = 0.5 * (distances[min_i, k] + distances[min_j, k] -
                            distances[min_i, min_j])
            distances[min_i, k] = new_dist
            distances[k, min_i] = new_dist
        end

        nodes[min_i] = new_node
        active[min_j] = false
    end

    # Join last two nodes
    active_indices = findall(active)
    i, j = active_indices[1], active_indices[2]
    root = TreeNode("Root")
    nodes[i].branch_length = distances[i, j] / 2
    nodes[j].branch_length = distances[i, j] / 2
    push!(root.children, nodes[i], nodes[j])
    nodes[i].parent = root
    nodes[j].parent = root

    return PhylogeneticTree(root, names, :nj)
end

# Character state analysis

"""
    character_state_matrix(alignment::Vector{String}) -> Matrix{Char}

Convert sequence alignment to character state matrix.

# Arguments
- `alignment`: Vector of aligned sequences

# Returns
Matrix where rows are taxa and columns are character positions

# Examples
```julia
alignment = ["ATCG", "ATCG", "TTCG"]
char_matrix = character_state_matrix(alignment)
```
"""
function character_state_matrix(alignment::Vector{String})
    n_taxa = length(alignment)
    n_chars = length(alignment[1])

    matrix = Matrix{Char}(undef, n_taxa, n_chars)
    for i in 1:n_taxa
        for j in 1:n_chars
            matrix[i, j] = alignment[i][j]
        end
    end

    return matrix
end

"""
    parsimony_informative_sites(char_matrix::Matrix{Char}) -> Vector{Int}

Identify parsimony-informative character positions.

A site is parsimony-informative if at least two different states each occur
in at least two taxa.

# Examples
```julia
char_matrix = ['A' 'T'; 'A' 'T'; 'C' 'G'; 'C' 'G']
sites = parsimony_informative_sites(char_matrix)
```
"""
function parsimony_informative_sites(char_matrix::Matrix{Char})
    n_chars = size(char_matrix, 2)
    informative = Int[]

    for j in 1:n_chars
        states = Dict{Char, Int}()
        for i in 1:size(char_matrix, 1)
            char = char_matrix[i, j]
            states[char] = get(states, char, 0) + 1
        end

        # Count states that appear at least twice
        if count(x -> x >= 2, values(states)) >= 2
            push!(informative, j)
        end
    end

    return informative
end

# Maximum Parsimony

"""
    maximum_parsimony(sequences::Vector{String}; taxa_names=nothing) -> PhylogeneticTree

Construct a phylogenetic tree using maximum parsimony criterion.

Uses stepwise addition heuristic to find a parsimonious tree. Starts with 3 taxa
and iteratively adds remaining taxa at the position that minimizes parsimony score.

# Arguments
- `sequences`: Vector of aligned sequences (all same length)
- `taxa_names`: Optional vector of taxon names (defaults to "Taxon1", "Taxon2", ...)

# Returns
A `PhylogeneticTree` with `method = :parsimony`

# Examples
```julia
seqs = ["ATCG", "ATCG", "TTCG", "TTCC"]
tree = maximum_parsimony(seqs, taxa_names=["A", "B", "C", "D"])
```
"""
function maximum_parsimony(sequences::Vector{String}; taxa_names=nothing)
    n = length(sequences)
    n < 3 && error("Need at least 3 sequences for parsimony analysis")

    # Generate taxa names if not provided
    if taxa_names === nothing
        taxa_names = ["Taxon$i" for i in 1:n]
    end

    # Convert sequences to character matrix
    char_matrix = character_state_matrix(sequences)

    # Start with first 3 taxa (only one unrooted topology)
    # Create a simple star tree with 3 leaves
    root = TreeNode("Internal1")
    root.parent = nothing
    leaf1 = TreeNode(taxa_names[1])
    leaf1.parent = root
    leaf1.branch_length = 0.1
    leaf2 = TreeNode(taxa_names[2])
    leaf2.parent = root
    leaf2.branch_length = 0.1
    leaf3 = TreeNode(taxa_names[3])
    leaf3.parent = root
    leaf3.branch_length = 0.1
    root.children = [leaf1, leaf2, leaf3]

    current_tree = PhylogeneticTree(root, taxa_names[1:3], :parsimony)

    # Stepwise addition for remaining taxa
    for i in 4:n
        taxon_name = taxa_names[i]
        best_tree = nothing
        best_score = Inf

        # Try inserting new taxon on every branch
        branches = collect_branches(current_tree.root)

        for (parent_node, child_node) in branches
            # Create a copy of the tree
            tree_copy = deepcopy(current_tree)

            # Find the corresponding nodes in the copy
            parent_copy = find_node_by_name(tree_copy.root, parent_node.name)
            child_copy = parent_copy === nothing ? nothing : find_child_by_name(parent_copy, child_node.name)

            if parent_copy !== nothing && child_copy !== nothing
                # Insert new taxon between parent and child
                new_internal = TreeNode("Internal$(i-2)")
                new_internal.parent = parent_copy
                new_internal.branch_length = child_copy.branch_length / 2.0

                new_leaf = TreeNode(taxon_name)
                new_leaf.parent = new_internal
                new_leaf.branch_length = 0.1

                child_copy.parent = new_internal
                child_copy.branch_length = child_copy.branch_length / 2.0

                new_internal.children = [child_copy, new_leaf]

                # Replace child in parent's children list
                idx = findfirst(c -> c === child_copy, parent_copy.children)
                if idx !== nothing
                    parent_copy.children[idx] = new_internal
                end

                # Update tree taxa list - create new instance since PhylogeneticTree is immutable
                tree_copy = PhylogeneticTree(tree_copy.root, vcat(tree_copy.taxa, [taxon_name]), :parsimony)

                # Calculate parsimony score for this configuration
                cm_extended = char_matrix[1:length(tree_copy.taxa), :]
                score = calculate_parsimony_score(tree_copy, cm_extended)

                if score < best_score
                    best_score = score
                    best_tree = tree_copy
                end
            end
        end

        # Keep the best tree
        if best_tree !== nothing
            current_tree = best_tree
        end
    end

    return current_tree
end

# Helper function to collect all parent-child branch pairs
function collect_branches(node::TreeNode, parent=nothing, branches=[])
    if parent !== nothing
        push!(branches, (parent, node))
    end
    for child in node.children
        collect_branches(child, node, branches)
    end
    return branches
end

# Helper function to find a child node by name
function find_child_by_name(parent::TreeNode, name::String)
    for child in parent.children
        if child.name == name
            return child
        end
    end
    return nothing
end

"""
    calculate_parsimony_score(tree::PhylogeneticTree, char_matrix::Matrix{Char}) -> Int

Calculate the parsimony score (total number of character state changes) for a tree.

Lower scores indicate fewer evolutionary changes and are preferred under parsimony.

# Examples
```julia
tree = upgma(distance_matrix(seqs))
score = calculate_parsimony_score(tree, character_state_matrix(seqs))
```
"""
function calculate_parsimony_score(tree::PhylogeneticTree, char_matrix::Matrix{Char})
    # Fitch algorithm for each character
    total_score = 0

    for j in 1:size(char_matrix, 2)
        (_, score) = fitch_score(tree.root, char_matrix[:, j], tree.taxa)
        total_score += score
    end

    return total_score
end

function fitch_score(node::TreeNode, char_column::Vector{Char}, taxa::Vector{String})
    if isempty(node.children)
        # Terminal node - return (Set, 0)
        idx = findfirst(==(node.name), taxa)
        return (Set([char_column[idx]]), 0)
    end

    # Internal node - recursively compute child results
    child_results = [fitch_score(child, char_column, taxa) for child in node.children]

    # Extract sets and scores
    child_sets = [result[1] for result in child_results]
    child_score_sum = sum(result[2] for result in child_results)

    # Intersection of child sets
    intersection = reduce(intersect, child_sets)

    if !isempty(intersection)
        return (intersection, child_score_sum + 0)  # No change needed
    else
        return (reduce(union, child_sets), child_score_sum + 1)  # Change required
    end
end

# Bootstrap support

"""
    bootstrap_support(sequences::Vector{String}; replicates=1000, method=:upgma) -> Dict

Perform bootstrap analysis to assess confidence in tree topology.

# Arguments
- `sequences`: Aligned sequence data
- `replicates`: Number of bootstrap replicates (default: 1000)
- `method`: Tree construction method (:upgma or :nj)

# Returns
Dictionary mapping clades to bootstrap support values (0.0-1.0)

# Examples
```julia
support = bootstrap_support(sequences, replicates=100, method=:nj)
```
"""
function bootstrap_support(sequences::Vector{String}; replicates=1000, method=:upgma)
    n_sites = length(sequences[1])
    original_tree = if method == :upgma
        upgma(distance_matrix(sequences))
    else
        neighbor_joining(distance_matrix(sequences))
    end

    clade_counts = Dict{Set{String}, Int}()

    for _ in 1:replicates
        # Resample sites with replacement
        resampled_indices = rand(1:n_sites, n_sites)
        resampled_seqs = [join([seq[i] for i in resampled_indices]) for seq in sequences]

        # Build tree from resampled data
        boot_tree = if method == :upgma
            upgma(distance_matrix(resampled_seqs))
        else
            neighbor_joining(distance_matrix(resampled_seqs))
        end

        # Extract clades and increment counts
        clades = extract_clades(boot_tree.root)
        for clade in clades
            clade_counts[clade] = get(clade_counts, clade, 0) + 1
        end
    end

    # Convert counts to proportions
    support_values = Dict(clade => count/replicates for (clade, count) in clade_counts)

    return support_values
end

function extract_clades(node::TreeNode)
    clades = Set{Set{String}}()

    if isempty(node.children)
        return clades
    end

    # Get all descendant taxa
    descendants = get_descendants(node)
    if length(descendants) > 1
        push!(clades, Set(descendants))
    end

    # Recurse on children
    for child in node.children
        union!(clades, extract_clades(child))
    end

    return clades
end

function get_descendants(node::TreeNode)
    if isempty(node.children)
        return [node.name]
    end

    descendants = String[]
    for child in node.children
        append!(descendants, get_descendants(child))
    end

    return descendants
end

# Clade identification

"""
    identify_clades(tree::PhylogeneticTree, support_threshold::Float64=0.95) -> Vector{Set{String}}

Identify well-supported monophyletic clades in a phylogenetic tree.

# Arguments
- `tree`: Phylogenetic tree with bootstrap support values
- `support_threshold`: Minimum support value (default: 0.95)

# Returns
Vector of sets, each containing taxa names forming a well-supported clade

# Examples
```julia
clades = identify_clades(tree, 0.90)
```
"""
function identify_clades(tree::PhylogeneticTree, support_threshold::Float64=0.95)
    clades = Set{String}[]
    _identify_clades_recursive(tree.root, support_threshold, clades)
    return clades
end

function _identify_clades_recursive(node::TreeNode, threshold::Float64, clades::Vector{Set{String}})
    if isempty(node.children)
        return
    end

    if node.support >= threshold
        descendants = get_descendants(node)
        if length(descendants) > 1
            push!(clades, Set(descendants))
        end
    end

    for child in node.children
        _identify_clades_recursive(child, threshold, clades)
    end
end

# Tree comparison

"""
    tree_distance(tree1::PhylogeneticTree, tree2::PhylogeneticTree) -> Int

Calculate Robinson-Foulds distance between two phylogenetic trees.

The RF distance counts the number of clades (bipartitions) that differ between trees.
A distance of 0 indicates identical topologies.

# Examples
```julia
tree1 = upgma(dmat1)
tree2 = neighbor_joining(dmat2)
rf_distance = tree_distance(tree1, tree2)
```
"""
function tree_distance(tree1::PhylogeneticTree, tree2::PhylogeneticTree)
    clades1 = extract_clades(tree1.root)
    clades2 = extract_clades(tree2.root)

    # Symmetric difference
    unique_to_1 = setdiff(clades1, clades2)
    unique_to_2 = setdiff(clades2, clades1)

    return length(unique_to_1) + length(unique_to_2)
end

# Tree utilities

"""
    root_tree(tree::PhylogeneticTree, outgroup::String) -> PhylogeneticTree

Root an unrooted tree using the specified outgroup taxon.

# Examples
```julia
rooted = root_tree(tree, "Outgroup_species")
```
"""
function root_tree(tree::PhylogeneticTree, outgroup::String)
    # Find the outgroup node
    outgroup_node = find_node_by_name(tree.root, outgroup)
    outgroup_node === nothing && error("Outgroup '$outgroup' not found in tree")

    # Check if outgroup is a leaf
    !isempty(outgroup_node.children) && error("Outgroup must be a leaf node")

    # Get outgroup's parent
    old_parent = outgroup_node.parent
    old_parent === nothing && error("Outgroup has no parent - cannot reroot")

    # Edge case: outgroup is already a direct child of root
    if old_parent.parent === nothing
        # Already rooted, just rebalance branch lengths at root
        # Create a new root with balanced branches
        new_root = TreeNode("Root")
        new_root.parent = nothing

        # Split outgroup's branch length
        half_length = outgroup_node.branch_length / 2.0
        outgroup_copy = deepcopy(outgroup_node)
        outgroup_copy.branch_length = half_length
        outgroup_copy.parent = new_root

        # Find the other child of root (the ingroup)
        other_children = filter(c -> c !== outgroup_node, old_parent.children)
        if length(other_children) == 1
            ingroup_copy = deepcopy(other_children[1])
            ingroup_copy.parent = new_root
            ingroup_copy.branch_length += half_length
            new_root.children = [outgroup_copy, ingroup_copy]
        else
            # Multiple children - keep root structure but rebalance
            ingroup_root = TreeNode("Ingroup")
            ingroup_root.branch_length = half_length
            ingroup_root.parent = new_root
            ingroup_root.children = [deepcopy(c) for c in other_children]
            for c in ingroup_root.children
                c.parent = ingroup_root
            end
            new_root.children = [outgroup_copy, ingroup_root]
        end

        return PhylogeneticTree(new_root, tree.taxa, tree.method)
    end

    # General case: reroot at midpoint of outgroup's branch
    # Create new root node
    new_root = TreeNode("Root")
    new_root.parent = nothing

    # Split the outgroup's branch length
    half_length = outgroup_node.branch_length / 2.0

    # Create outgroup copy with half branch length
    outgroup_copy = deepcopy(outgroup_node)
    outgroup_copy.branch_length = half_length
    outgroup_copy.parent = new_root

    # Create ingroup side: old parent becomes child of new root
    # Remove outgroup from old parent's children
    old_parent_copy = deepcopy(old_parent)
    old_parent_copy.children = filter(c -> c.name != outgroup, old_parent_copy.children)

    # Fix parent references in the copied subtree
    function fix_parents(node::TreeNode)
        for child in node.children
            child.parent = node
            fix_parents(child)
        end
    end
    fix_parents(old_parent_copy)

    # Set ingroup branch length (other half of split)
    old_parent_copy.branch_length = half_length
    old_parent_copy.parent = new_root

    # Assemble new tree
    new_root.children = [outgroup_copy, old_parent_copy]

    return PhylogeneticTree(new_root, tree.taxa, tree.method)
end

function find_node_by_name(node::TreeNode, name::String)
    if node.name == name
        return node
    end

    for child in node.children
        result = find_node_by_name(child, name)
        result !== nothing && return result
    end

    return nothing
end

"""
    tree_to_newick(tree::PhylogeneticTree) -> String

Convert phylogenetic tree to Newick format string.

# Examples
```julia
newick_str = tree_to_newick(tree)
println(newick_str)  # ((A:0.1,B:0.2):0.3,C:0.4);
```
"""
function tree_to_newick(tree::PhylogeneticTree)
    return _node_to_newick(tree.root) * ";"
end

function _node_to_newick(node::TreeNode)
    if isempty(node.children)
        return node.name * ":" * string(node.branch_length)
    end

    child_strings = [_node_to_newick(child) for child in node.children]
    subtree = "(" * join(child_strings, ",") * ")"

    if node.name != "Root" && node.name != ""
        subtree *= node.name
    end

    if node.branch_length > 0
        subtree *= ":" * string(node.branch_length)
    end

    return subtree
end

"""
    parse_newick(newick_str::String) -> PhylogeneticTree

Parse a Newick format string into a `PhylogeneticTree`.

Newick format is a standard way of representing phylogenetic trees as nested parentheses
with branch lengths. This function supports:
- Named and unnamed leaf nodes
- Named and unnamed internal nodes
- Branch lengths in the format `:0.123`
- Nested parentheses for subtrees
- Trailing semicolon (optional)

# Format

The Newick format uses:
- Parentheses `()` to denote subtrees
- Commas `,` to separate siblings
- Colons `:` to specify branch lengths
- Semicolon `;` to mark the end of the tree (optional)

# Examples

```jldoctest
julia> using Cladistics

julia> tree = parse_newick("((A:0.1,B:0.2):0.3,C:0.4);");

julia> tree.taxa
3-element Vector{String}:
 "A"
 "B"
 "C"

julia> tree2 = parse_newick("(A,B,C);");  # No branch lengths

julia> length(tree2.taxa)
3
```

# See Also
- [`tree_to_newick`](@ref): Convert a tree to Newick format
- [`upgma`](@ref), [`neighbor_joining`](@ref): Tree construction methods
"""
function parse_newick(newick_str::String)
    # Remove trailing semicolon and whitespace
    s = String(strip(newick_str))
    if endswith(s, ';')
        s = s[1:end-1]
    end
    s = String(strip(s))

    # Track position in string
    pos = Ref(1)

    # Parse the tree recursively
    root, taxa = _parse_newick_node(s, pos, 0)

    return PhylogeneticTree(root, taxa, :newick)
end

# Helper function to parse a single node or subtree
function _parse_newick_node(s::String, pos::Ref{Int}, internal_counter::Int)
    taxa = String[]
    internal_counter_ref = Ref(internal_counter)

    # Skip whitespace
    while pos[] <= length(s) && s[pos[]] == ' '
        pos[] += 1
    end

    if pos[] > length(s)
        error("Unexpected end of Newick string")
    end

    # Check if this is a subtree (starts with '(')
    if s[pos[]] == '('
        pos[] += 1  # Skip '('

        # Parse children
        children = TreeNode[]

        while true
            # Skip whitespace
            while pos[] <= length(s) && s[pos[]] == ' '
                pos[] += 1
            end

            # Parse child
            child, child_taxa = _parse_newick_node(s, pos, internal_counter_ref[])
            push!(children, child)
            append!(taxa, child_taxa)

            internal_counter_ref[] += 1

            # Skip whitespace
            while pos[] <= length(s) && s[pos[]] == ' '
                pos[] += 1
            end

            # Check for comma (more children) or close paren (done with children)
            if pos[] <= length(s) && s[pos[]] == ','
                pos[] += 1  # Skip comma
            elseif pos[] <= length(s) && s[pos[]] == ')'
                pos[] += 1  # Skip ')'
                break
            else
                error("Expected ',' or ')' at position $(pos[])")
            end
        end

        # Create internal node
        node = TreeNode("Internal$(internal_counter_ref[])")
        node.children = children
        for child in children
            child.parent = node
        end

        # Parse optional name for internal node
        name_start = pos[]
        while pos[] <= length(s) && s[pos[]] != ':' && s[pos[]] != ',' && s[pos[]] != ')' && s[pos[]] != ';'
            pos[] += 1
        end

        if pos[] > name_start
            name = String(strip(s[name_start:pos[]-1]))
            if !isempty(name)
                node.name = name
            end
        end

        # Parse optional branch length
        if pos[] <= length(s) && s[pos[]] == ':'
            pos[] += 1  # Skip ':'
            length_start = pos[]

            while pos[] <= length(s) && (isdigit(s[pos[]]) || s[pos[]] == '.' || s[pos[]] == '-' || s[pos[]] == 'e' || s[pos[]] == 'E')
                pos[] += 1
            end

            length_str = s[length_start:pos[]-1]
            if !isempty(length_str)
                node.branch_length = parse(Float64, length_str)
            end
        end

        return node, taxa

    else
        # This is a leaf node - parse taxon name
        name_start = pos[]

        while pos[] <= length(s) && s[pos[]] != ':' && s[pos[]] != ',' && s[pos[]] != ')' && s[pos[]] != ';'
            pos[] += 1
        end

        name = String(strip(s[name_start:pos[]-1]))
        if isempty(name)
            error("Empty taxon name at position $(name_start)")
        end

        node = TreeNode(name)
        push!(taxa, name)

        # Parse optional branch length
        if pos[] <= length(s) && s[pos[]] == ':'
            pos[] += 1  # Skip ':'
            length_start = pos[]

            while pos[] <= length(s) && (isdigit(s[pos[]]) || s[pos[]] == '.' || s[pos[]] == '-' || s[pos[]] == 'e' || s[pos[]] == 'E')
                pos[] += 1
            end

            length_str = s[length_start:pos[]-1]
            if !isempty(length_str)
                node.branch_length = parse(Float64, length_str)
            end
        end

        return node, taxa
    end
end

end # module Cladistics
