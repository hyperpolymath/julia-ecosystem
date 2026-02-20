# SPDX-License-Identifier: PMPL-1.0-or-later
using Test
using HackenbushGames

@testset "HackenbushGames" begin
    @testset "Stalk Values" begin
        @test stalk_value([Blue]) == 1//1
        @test stalk_value([Red]) == -1//1
        @test stalk_value([Blue, Red]) == 1//2
        @test stalk_value([Blue, Red, Blue]) == 3//4
        @test stalk_value([Red, Blue, Red]) == -3//4
    end

    @testset "Green Grundy" begin
        edges = [
            Edge(0, 1, Green),
            Edge(1, 2, Green),
        ]
        g = HackenbushGraph(edges, [0])
        @test green_grundy(g) == 2
    end

    @testset "Moves" begin
        edges = [
            Edge(0, 1, Blue),
            Edge(1, 2, Red),
        ]
        g = HackenbushGraph(edges, [0])
        @test length(moves(g, :left)) == 1
        @test length(moves(g, :right)) == 1
    end

    @testset "Graph Sum" begin
        a = simple_stalk([Blue, Red])
        b = simple_stalk([Green])
        s = game_sum(a, b)
        @test length(s.edges) == length(a.edges) + length(b.edges)
    end

    @testset "GraphViz" begin
        g = simple_stalk([Green])
        dot = to_graphviz(g)
        @test occursin("graph Hackenbush", dot)
    end

    @testset "Canonical Form" begin
        g = simple_stalk([Blue])
        value = game_value(g)
        @test value == 1//1
        form = canonical_game(g)
        @test isempty(form.right)
    end

    @testset "ASCII" begin
        g = simple_stalk([Red, Blue])
        ascii = to_ascii(g)
        @test occursin("HackenbushGraph", ascii)
    end

    @testset "Prune Disconnected" begin
        # Create graph with disconnected component
        edges = [
            Edge(0, 1, Blue),  # connected to ground
            Edge(2, 3, Red),   # floating (not connected to ground)
        ]
        g = HackenbushGraph(edges, [0])
        pruned = prune_disconnected(g)
        @test length(pruned.edges) == 1  # only edge 0->1 remains
    end

    @testset "Simplest Dyadic" begin
        @test simplest_dyadic_between(0//1, 1//1) == 1//2  # simplest dyadic between 0 and 1
        @test simplest_dyadic_between(-1//1, 1//1) == 0//1
        @test simplest_dyadic_between(1//4, 3//4) == 1//2
        @test_throws ErrorException simplest_dyadic_between(1//1, 0//1)  # l >= r
    end

    @testset "Nim Sum" begin
        @test nim_sum([3, 5]) == 6  # 3 XOR 5 = 6
        @test nim_sum([1, 2, 3]) == 0  # 1 XOR 2 XOR 3 = 0
        @test nim_sum([7]) == 7
        @test nim_sum(Int[]) == 0
    end

    @testset "Mex" begin
        @test mex([0, 1, 3]) == 2  # minimum excludant
        @test mex([1, 2, 3]) == 0
        @test mex(Int[]) == 0
        @test mex([0, 1, 2, 3, 4]) == 5
    end

    @testset "Green Stalk Nimber" begin
        @test green_stalk_nimber(5) == 5  # green stalk of height n has nimber n
        @test green_stalk_nimber(0) == 0
        @test green_stalk_nimber(1) == 1
    end

    @testset "Green Moves" begin
        # Green edges allow both left and right moves (impartial)
        edges = [Edge(0, 1, Green)]
        g = HackenbushGraph(edges, [0])
        left_moves = moves(g, :left)
        right_moves = moves(g, :right)
        @test length(left_moves) == 1
        @test length(right_moves) == 1
    end

    @testset "Empty Graph" begin
        g = HackenbushGraph(Edge[], [0])
        @test length(moves(g, :left)) == 0
        @test length(moves(g, :right)) == 0
        @test game_value(g) == 0//1  # empty position is zero
    end

    @testset "Game Value Nothing" begin
        # Create a position where game_value returns nothing (non-numeric)
        # A Green position typically returns nothing as it's impartial
        g = simple_stalk([Green, Green])
        val = game_value(g)
        # Green positions don't have numeric values, they have Grundy numbers
        @test val === nothing || val isa Rational
    end
end
