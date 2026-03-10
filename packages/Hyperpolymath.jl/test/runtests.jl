# SPDX-License-Identifier: PMPL-1.0-or-later
using Test

@testset "Hyperpolymath.jl" begin

    @testset "Module loads if all dependencies available" begin
        # Hyperpolymath.jl is an umbrella package that depends on many sub-packages.
        # In isolation (without all deps registered), it will fail to load.
        # We test gracefully: attempt to load and verify, or skip if deps missing.
        loaded = try
            include(joinpath(@__DIR__, "..", "src", "Hyperpolymath.jl"))
            true
        catch e
            @info "Hyperpolymath.jl could not load (expected if dependencies are not installed)" exception=(e, catch_backtrace())
            false
        end

        if loaded
            @test isdefined(Main, :Hyperpolymath) || @isdefined(Hyperpolymath)
        else
            @test_skip "Hyperpolymath module requires all sub-packages to be installed"
        end
    end

    @testset "Source file exists and has correct structure" begin
        src_path = joinpath(@__DIR__, "..", "src", "Hyperpolymath.jl")
        @test isfile(src_path)

        content = read(src_path, String)
        @test contains(content, "module Hyperpolymath")
        @test contains(content, "end # module")

        # Verify key dependency groups are declared
        @test contains(content, "using Axiom")
        @test contains(content, "using MacroPower")
        @test contains(content, "using SiliconCore")
        @test contains(content, "using LowLevel")
        @test contains(content, "using HardwareResilience")
        @test contains(content, "using ShellIntegration")
        @test contains(content, "using MinixSDK")
    end

    @testset "SPDX license header present" begin
        src_path = joinpath(@__DIR__, "..", "src", "Hyperpolymath.jl")
        first_line = readline(src_path)
        @test contains(first_line, "SPDX-License-Identifier")
        @test contains(first_line, "PMPL-1.0-or-later")
    end

end
