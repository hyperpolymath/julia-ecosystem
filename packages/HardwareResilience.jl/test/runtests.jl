# SPDX-License-Identifier: PMPL-1.0-or-later
using Test

include(joinpath(@__DIR__, "..", "src", "HardwareResilience.jl"))
using .HardwareResilience

@testset "HardwareResilience.jl" begin

    @testset "KernelGuardian construction" begin
        g = KernelGuardian("test_guardian", :Healthy)
        @test g.name == "test_guardian"
        @test g.status === :Healthy
    end

    @testset "KernelGuardian is mutable" begin
        g = KernelGuardian("mut_test", :Healthy)
        @test ismutable(g)
        g.status = :Degraded
        @test g.status === :Degraded
        g.name = "renamed"
        @test g.name == "renamed"
    end

    @testset "KernelGuardian various status values" begin
        for status in [:Healthy, :Degraded, :Failed, :Recovering, :Unknown]
            g = KernelGuardian("status_test", status)
            @test g.status === status
        end
    end

    @testset "monitor_kernel returns op result on success" begin
        g = KernelGuardian("monitor_test", :Healthy)
        result = monitor_kernel(g, () -> 42)
        @test result == 42
    end

    @testset "monitor_kernel returns string from op" begin
        g = KernelGuardian("string_test", :Healthy)
        result = monitor_kernel(g, () -> "hello")
        @test result == "hello"
    end

    @testset "monitor_kernel returns nothing on error" begin
        g = KernelGuardian("error_test", :Healthy)
        result = monitor_kernel(g, () -> error("simulated failure"))
        @test result === nothing
    end

    @testset "monitor_kernel catches DomainError" begin
        g = KernelGuardian("domain_test", :Healthy)
        result = monitor_kernel(g, () -> throw(DomainError(-1, "negative")))
        @test result === nothing
    end

    @testset "monitor_kernel catches BoundsError" begin
        g = KernelGuardian("bounds_test", :Healthy)
        result = monitor_kernel(g, () -> [1, 2, 3][10])
        @test result === nothing
    end

    @testset "monitor_kernel with complex return type" begin
        g = KernelGuardian("complex_test", :Healthy)
        result = monitor_kernel(g, () -> Dict("a" => 1, "b" => 2))
        @test result isa Dict
        @test result["a"] == 1
    end

    @testset "monitor_kernel with no-op" begin
        g = KernelGuardian("noop_test", :Healthy)
        result = monitor_kernel(g, () -> nothing)
        @test result === nothing
    end

    @testset "exports are correct" begin
        @test isdefined(HardwareResilience, :KernelGuardian)
        @test isdefined(HardwareResilience, :monitor_kernel)
    end

end
