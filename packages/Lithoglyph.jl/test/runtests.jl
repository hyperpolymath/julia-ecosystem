# SPDX-License-Identifier: PMPL-1.0-or-later
using Test

# Lithoglyph depends on HTTP, JSON3, Libdl — load module directly if available
include(joinpath(@__DIR__, "..", "src", "Lithoglyph.jl"))
using .Lithoglyph

@testset "Lithoglyph.jl" begin

    @testset "Glyph construction" begin
        g = Glyph(:alpha, "test data", ["tag1", "tag2"], "sha256:abc123")
        @test g.id === :alpha
        @test g.data == "test data"
        @test g.tags == ["tag1", "tag2"]
        @test g.provenance == "sha256:abc123"
    end

    @testset "Glyph with empty fields" begin
        g = Glyph(:empty, "", String[], "")
        @test g.id === :empty
        @test g.data == ""
        @test isempty(g.tags)
        @test g.provenance == ""
    end

    @testset "Glyph is immutable" begin
        g = Glyph(:immut, "data", ["a"], "prov")
        @test !ismutable(g)
    end

    @testset "LithoglyphClient construction" begin
        c = LithoglyphClient("https://litho.example.com/api", "tok_secret")
        @test c.endpoint == "https://litho.example.com/api"
        @test c.token == "tok_secret"
    end

    @testset "LithoglyphClient is immutable" begin
        c = LithoglyphClient("http://localhost", "t")
        @test !ismutable(c)
    end

    @testset "register_glyph returns :success" begin
        c = LithoglyphClient("http://localhost", "test-token")
        g = Glyph(:reg_test, "payload", ["test"], "sha256:000")
        result = register_glyph(c, g)
        @test result === :success
        @test result isa Symbol
    end

    @testset "search_glyphs returns empty Glyph vector" begin
        c = LithoglyphClient("http://localhost", "test-token")
        results = search_glyphs(c, "nonexistent")
        @test results isa Vector{Glyph}
        @test isempty(results)
    end

    @testset "search_glyphs with empty query" begin
        c = LithoglyphClient("http://localhost", "test-token")
        results = search_glyphs(c, "")
        @test results isa Vector{Glyph}
        @test isempty(results)
    end

    @testset "litho_normalize returns String" begin
        result = Lithoglyph.litho_normalize("Hello World")
        @test result isa String
        # Stub just returns input unchanged
        @test result == "Hello World"
    end

    @testset "litho_normalize with empty string" begin
        result = Lithoglyph.litho_normalize("")
        @test result == ""
    end

    @testset "litho_normalize with unicode" begin
        result = Lithoglyph.litho_normalize("hieroglyph")
        @test result == "hieroglyph"
    end

end
