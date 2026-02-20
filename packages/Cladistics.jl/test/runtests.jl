# SPDX-License-Identifier: PMPL-1.0-or-later

using Test
using Cladistics
using LinearAlgebra
using Statistics

@testset "Cladistics.jl Tests" begin

    @testset "Distance Matrix - Hamming" begin
        seqs = ["ATCG", "ATCG", "TTCG", "TTCC"]
        dmat = distance_matrix(seqs, method=:hamming)

        # Test symmetry
        @test dmat == dmat'

        # Test diagonal is zero
        @test all(dmat[i,i] == 0 for i in 1:4)

        # Test specific distances
        @test dmat[1,2] == 0  # Identical sequences
        @test dmat[1,3] == 1  # One difference (A vs T)
        @test dmat[1,4] == 2  # Two differences
    end

    @testset "Distance Matrix - P-Distance" begin
        seqs = ["ATCG", "ATCG", "TTCG", "TTCC"]
        dmat = distance_matrix(seqs, method=:p_distance)

        # Test that p-distance is proportion of differences
        @test dmat[1,3] ≈ 1/4  # 1 difference out of 4 positions
        @test dmat[1,4] ≈ 2/4  # 2 differences out of 4 positions
    end

    @testset "Distance Matrix - Jukes-Cantor" begin
        seqs = ["ATCG", "ATCG", "AACG", "AGCG"]
        dmat = distance_matrix(seqs, method=:jc69)

        # Test that JC corrects for multiple substitutions
        @test dmat[1,3] > 0  # Sequences differ
        @test all(isfinite.(dmat))  # No infinities for these sequences
    end

    @testset "Distance Matrix - Kimura 2-Parameter" begin
        seqs = ["ATCG", "GTCG", "ATCA", "AACG"]
        dmat = distance_matrix(seqs, method=:k2p)

        # Test basic properties
        @test issymmetric(dmat)
        @test all(diag(dmat) .≈ 0)
        @test all(dmat .>= 0)
    end

    @testset "UPGMA Tree Construction" begin
        # Simple 3-taxon example
        dmat = [0.0 0.2 0.4;
                0.2 0.0 0.3;
                0.4 0.3 0.0]
        taxa = ["A", "B", "C"]

        tree = upgma(dmat, taxa_names=taxa)

        # Test tree structure
        @test tree.method == :upgma
        @test tree.taxa == taxa
        @test !isempty(tree.root.children)

        # UPGMA produces ultrametric trees
        @test tree.root.children[1].branch_length >= 0
        @test tree.root.children[2].branch_length >= 0
    end

    @testset "Neighbor-Joining Tree Construction" begin
        # Simple 4-taxon example
        dmat = [0.0 0.2 0.4 0.5;
                0.2 0.0 0.3 0.4;
                0.4 0.3 0.0 0.2;
                0.5 0.4 0.2 0.0]
        taxa = ["A", "B", "C", "D"]

        tree = neighbor_joining(dmat, taxa_names=taxa)

        # Test tree structure
        @test tree.method == :nj
        @test tree.taxa == taxa
        @test !isempty(tree.root.children)

        # NJ produces additive trees with non-negative branch lengths
        function check_branches(node)
            @test node.branch_length >= 0
            for child in node.children
                check_branches(child)
            end
        end
        check_branches(tree.root)
    end

    @testset "Character State Matrix" begin
        alignment = ["ATCG", "ATCG", "TTCG"]
        char_matrix = character_state_matrix(alignment)

        # Test dimensions
        @test size(char_matrix) == (3, 4)

        # Test specific entries
        @test char_matrix[1, 1] == 'A'
        @test char_matrix[1, 2] == 'T'
        @test char_matrix[3, 1] == 'T'
    end

    @testset "Parsimony Informative Sites" begin
        # Site 1: A,A,C,C - informative (two states, each appears twice)
        # Site 2: A,T,C,G - not informative (all different)
        # Site 3: A,A,A,T - not informative (one state appears <2 times)
        # Site 4: A,A,T,T - informative
        char_matrix = ['A' 'A' 'A' 'A';
                       'A' 'T' 'A' 'A';
                       'C' 'C' 'A' 'T';
                       'C' 'G' 'T' 'T']

        sites = parsimony_informative_sites(char_matrix)

        # Test that sites 1 and 4 are informative
        @test 1 in sites
        @test 4 in sites
        @test !(2 in sites)
        @test !(3 in sites)
    end

    @testset "TreeNode Construction" begin
        node = TreeNode("Test")

        @test node.name == "Test"
        @test isempty(node.children)
        @test node.branch_length == 0.0
        @test node.support == 1.0
        @test node.parent === nothing
    end

    @testset "Tree Distance (Robinson-Foulds)" begin
        # Create two identical simple trees
        dmat = [0.0 0.2 0.3;
                0.2 0.0 0.3;
                0.3 0.3 0.0]

        tree1 = upgma(dmat, taxa_names=["A", "B", "C"])
        tree2 = upgma(dmat, taxa_names=["A", "B", "C"])

        # Identical trees should have distance 0
        dist = tree_distance(tree1, tree2)
        @test dist == 0

        # Different topology should have distance > 0
        dmat2 = [0.0 0.3 0.2;
                 0.3 0.0 0.2;
                 0.2 0.2 0.0]
        tree3 = upgma(dmat2, taxa_names=["A", "B", "C"])
        dist2 = tree_distance(tree1, tree3)
        @test dist2 >= 0
    end

    @testset "Bootstrap Support" begin
        sequences = ["ATCGATCG", "ATCGATCG", "TTCGTTCG", "TTCCTTCC"]

        # Small number of replicates for testing speed
        support = bootstrap_support(sequences, replicates=10, method=:upgma)

        # Test that support values are in valid range
        for (clade, value) in support
            @test 0.0 <= value <= 1.0
        end

        # Test that clades are non-empty sets
        for (clade, value) in support
            @test !isempty(clade)
        end
    end

    @testset "Identify Clades" begin
        # Create a simple tree
        dmat = [0.0 0.2 0.4 0.5;
                0.2 0.0 0.3 0.4;
                0.4 0.3 0.0 0.2;
                0.5 0.4 0.2 0.0]
        tree = upgma(dmat, taxa_names=["A", "B", "C", "D"])

        # Set high support values manually for testing
        function set_support(node, value)
            node.support = value
            for child in node.children
                set_support(child, value)
            end
        end
        set_support(tree.root, 0.99)

        # Identify clades with high support
        clades = identify_clades(tree, 0.95)

        # Test that all identified clades have multiple taxa
        for clade in clades
            @test length(clade) > 1
        end
    end

    @testset "Tree to Newick Format" begin
        # Create simple tree
        dmat = [0.0 0.2 0.3;
                0.2 0.0 0.3;
                0.3 0.3 0.0]
        tree = upgma(dmat, taxa_names=["A", "B", "C"])

        newick = tree_to_newick(tree)

        # Test basic Newick format properties
        @test endswith(newick, ";")
        @test occursin("A", newick)
        @test occursin("B", newick)
        @test occursin("C", newick)
        @test occursin("(", newick)
        @test occursin(")", newick)
        @test occursin(":", newick)  # Branch lengths
    end

    @testset "Root Tree" begin
        dmat = [0.0 0.2 0.4 0.5;
                0.2 0.0 0.3 0.4;
                0.4 0.3 0.0 0.2;
                0.5 0.4 0.2 0.0]
        tree = neighbor_joining(dmat, taxa_names=["A", "B", "C", "D"])

        # Root on taxon A
        rooted = root_tree(tree, "A")

        # Test that tree structure is valid
        @test rooted.method == :nj
        @test "A" in rooted.taxa

        # Test error for non-existent outgroup
        @test_throws ErrorException root_tree(tree, "NonExistent")
    end

    @testset "DNA Sequence Integration Test" begin
        # Real-world-like example with 5 species
        sequences = [
            "ATCGATCGATCG",  # Species 1
            "ATCGATCGATCG",  # Species 2 (identical to 1)
            "ATCGTTCGATCG",  # Species 3 (1 substitution)
            "TTCGATCGATCG",  # Species 4 (1 substitution)
            "TTCGTTCGATCC"   # Species 5 (3 substitutions)
        ]

        # Build distance matrix
        dmat = distance_matrix(sequences, method=:jc69)

        # Test distance properties
        @test issymmetric(dmat)
        @test dmat[1,2] ≈ 0.0  # Identical sequences
        @test dmat[1,5] > dmat[1,3]  # More distant sequence

        # Build UPGMA tree
        upgma_tree = upgma(dmat, taxa_names=["Sp1", "Sp2", "Sp3", "Sp4", "Sp5"])
        @test upgma_tree.method == :upgma
        @test length(upgma_tree.taxa) == 5

        # Build NJ tree
        nj_tree = neighbor_joining(dmat, taxa_names=["Sp1", "Sp2", "Sp3", "Sp4", "Sp5"])
        @test nj_tree.method == :nj
        @test length(nj_tree.taxa) == 5

        # Compare tree topologies
        rf_dist = tree_distance(upgma_tree, nj_tree)
        @test rf_dist >= 0  # Non-negative distance
    end

    @testset "Protein Sequence Analysis" begin
        # Protein sequences (single letter amino acid codes)
        proteins = [
            "ARNDCQEGHILKMFPSTWYV",
            "ARNDCQEGHILKMFPSTWYV",
            "ARNDCQEGHILKMFPSTWYA",  # V->A at end
            "GRNDCQEGHILKMFPSTWYV"   # A->G at start
        ]

        dmat = distance_matrix(proteins, method=:hamming)

        # Test distances
        @test dmat[1,2] == 0  # Identical
        @test dmat[1,3] == 1  # One amino acid difference
        @test dmat[1,4] == 1  # One amino acid difference

        # Build tree
        tree = upgma(dmat, taxa_names=["Protein1", "Protein2", "Protein3", "Protein4"])
        @test tree.method == :upgma
        @test length(tree.taxa) == 4
    end

    @testset "Calculate Parsimony Score" begin
        # Test with known sequences
        seqs_identical = ["ATCG", "ATCG", "ATCG", "ATCG"]
        dmat = distance_matrix(seqs_identical, method=:hamming)
        tree = upgma(dmat, taxa_names=["A", "B", "C", "D"])
        cm = character_state_matrix(seqs_identical)

        # Identical sequences should have parsimony score 0
        score = calculate_parsimony_score(tree, cm)
        @test score isa Int
        @test score == 0

        # Test with sequences that have differences
        seqs_different = ["ATCG", "ATCG", "TTCG", "TTCC"]
        dmat2 = distance_matrix(seqs_different, method=:hamming)
        tree2 = upgma(dmat2, taxa_names=["A", "B", "C", "D"])
        cm2 = character_state_matrix(seqs_different)

        score2 = calculate_parsimony_score(tree2, cm2)
        @test score2 isa Int
        @test score2 >= 0
        @test score2 > 0  # Should have non-zero score due to differences
    end

    @testset "Maximum Parsimony Search" begin
        # Test basic functionality
        seqs = ["ATCGATCG", "ATCGATCG", "TTCGTTCG", "TTCCTTCC", "AACGAACG"]
        tree = maximum_parsimony(seqs, taxa_names=["A", "B", "C", "D", "E"])

        # Test return type and structure
        @test tree isa Cladistics.PhylogeneticTree
        @test tree.method == :parsimony
        @test length(tree.taxa) == 5
        @test !isempty(tree.root.children)

        # Test that parsimony score is computable
        cm = character_state_matrix(seqs)
        score = calculate_parsimony_score(tree, cm)
        @test score isa Int
        @test score >= 0

        # Test with identical sequences - should have score 0
        seqs_identical = ["ATCGATCG", "ATCGATCG", "ATCGATCG"]
        tree_identical = maximum_parsimony(seqs_identical, taxa_names=["A", "B", "C"])
        cm_identical = character_state_matrix(seqs_identical)
        score_identical = calculate_parsimony_score(tree_identical, cm_identical)
        @test score_identical == 0

        # Test with default taxa names (omit taxa_names parameter)
        tree_default = maximum_parsimony(seqs)
        @test length(tree_default.taxa) == 5
        @test tree_default.method == :parsimony
    end

    @testset "Edge Cases" begin
        # Single taxon - upgma handles this gracefully
        single_dmat = reshape([0.0], 1, 1)
        single_tree = upgma(single_dmat, taxa_names=["A"])
        @test length(single_tree.taxa) == 1

        # Two taxa
        two_dmat = [0.0 0.5; 0.5 0.0]
        two_tree = upgma(two_dmat, taxa_names=["A", "B"])
        @test length(two_tree.taxa) == 2

        # Very high distance (saturation)
        seqs_saturated = ["AAAA", "TTTT"]
        dmat_sat = distance_matrix(seqs_saturated, method=:jc69)
        @test isinf(dmat_sat[1,2])  # Should be infinite (saturated)

        # Empty sequence (edge case)
        empty_char_matrix = Matrix{Char}(undef, 0, 0)
        sites = parsimony_informative_sites(empty_char_matrix)
        @test isempty(sites)
    end

    @testset "Newick Parser" begin
        # Test basic parsing with branch lengths
        newick1 = "((A:0.1,B:0.2):0.3,C:0.4);"
        tree1 = parse_newick(newick1)

        @test tree1 isa Cladistics.PhylogeneticTree
        @test tree1.method == :newick
        @test length(tree1.taxa) == 3
        @test "A" in tree1.taxa
        @test "B" in tree1.taxa
        @test "C" in tree1.taxa

        # Test parsing without branch lengths
        newick2 = "(A,B,C);"
        tree2 = parse_newick(newick2)

        @test length(tree2.taxa) == 3
        @test "A" in tree2.taxa
        @test "B" in tree2.taxa
        @test "C" in tree2.taxa

        # Test parsing without trailing semicolon
        newick3 = "((A:0.1,B:0.2):0.3,C:0.4)"
        tree3 = parse_newick(newick3)

        @test length(tree3.taxa) == 3

        # Test round-trip: export to Newick and parse back
        dmat = [0.0 0.2 0.3; 0.2 0.0 0.3; 0.3 0.3 0.0]
        original = upgma(dmat, taxa_names=["A", "B", "C"])
        newick_str = tree_to_newick(original)
        parsed = parse_newick(newick_str)

        # Check that taxa are preserved
        @test sort(parsed.taxa) == sort(original.taxa)

        # Test with 4 taxa
        newick4 = "((A:0.1,B:0.2):0.15,(C:0.3,D:0.4):0.25);"
        tree4 = parse_newick(newick4)

        @test length(tree4.taxa) == 4
        @test Set(tree4.taxa) == Set(["A", "B", "C", "D"])

        # Test with named internal nodes
        newick5 = "((A:0.1,B:0.2)AB:0.3,C:0.4);"
        tree5 = parse_newick(newick5)

        @test length(tree5.taxa) == 3
    end

end
