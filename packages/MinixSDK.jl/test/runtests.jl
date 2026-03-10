# SPDX-License-Identifier: PMPL-1.0-or-later
using Test

include(joinpath(@__DIR__, "..", "src", "MinixSDK.jl"))
using .MinixSDK

@testset "MinixSDK.jl" begin

    @testset "cross_compile_to_minix returns String" begin
        result = cross_compile_to_minix(identity)
        @test result isa String
        @test result == "minix_service.c"
    end

    @testset "cross_compile_to_minix with different functions" begin
        result1 = cross_compile_to_minix(sin)
        @test result1 == "minix_service.c"
        result2 = cross_compile_to_minix(() -> 42)
        @test result2 == "minix_service.c"
    end

    @testset "cross_compile_to_minix method signature" begin
        @test hasmethod(cross_compile_to_minix, Tuple{Function})
    end

    @testset "generate_minix_service returns C boilerplate" begin
        result = generate_minix_service("my_service", "do_something()")
        @test result isa String
        @test contains(result, "#include <minix/drivers.h>")
        @test contains(result, "Julia-generated logic")
    end

    @testset "generate_minix_service with empty name" begin
        result = generate_minix_service("", "")
        @test result isa String
        @test contains(result, "#include <minix/drivers.h>")
    end

    @testset "generate_minix_service with special characters" begin
        result = generate_minix_service("service_v2.0", "x = 1 + 2;")
        @test result isa String
        @test contains(result, "#include <minix/drivers.h>")
    end

    @testset "generate_minix_service method signature" begin
        @test hasmethod(generate_minix_service, Tuple{String, String})
    end

    @testset "exports are correct" begin
        @test isdefined(MinixSDK, :cross_compile_to_minix)
        @test isdefined(MinixSDK, :generate_minix_service)
    end

end
