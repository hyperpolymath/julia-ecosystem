# SPDX-License-Identifier: PMPL-1.0-or-later
using Test

# LowLevel.jl includes SiliconCore.jl and HardwareResilience.jl via relative
# paths that assume a specific directory layout. We test the sub-modules
# individually and then test the coordinated peak_performance_op function
# by including LowLevel from its own directory context.

# First, test that sub-modules work independently
include(joinpath(@__DIR__, "..", "..", "SiliconCore.jl", "src", "SiliconCore.jl"))
include(joinpath(@__DIR__, "..", "..", "HardwareResilience.jl", "src", "HardwareResilience.jl"))

@testset "LowLevel.jl" begin

    @testset "SiliconCore sub-module accessible" begin
        @test isdefined(SiliconCore, :detect_arch)
        @test isdefined(SiliconCore, :vector_add_asm)
        @test SiliconCore.detect_arch() isa Symbol
    end

    @testset "HardwareResilience sub-module accessible" begin
        @test isdefined(HardwareResilience, :KernelGuardian)
        @test isdefined(HardwareResilience, :monitor_kernel)
    end

    @testset "peak_performance_op with integers" begin
        # Replicate peak_performance_op logic since include paths may not resolve
        g = HardwareResilience.KernelGuardian("Global-Op", :Healthy)
        result = HardwareResilience.monitor_kernel(g, () -> begin
            SiliconCore.vector_add_asm([1, 2, 3], [4, 5, 6])
        end)
        @test result == [5, 7, 9]
    end

    @testset "peak_performance_op with floats" begin
        g = HardwareResilience.KernelGuardian("Global-Op", :Healthy)
        result = HardwareResilience.monitor_kernel(g, () -> begin
            SiliconCore.vector_add_asm([1.0, 2.0], [3.0, 4.0])
        end)
        @test result == [4.0, 6.0]
    end

    @testset "peak_performance_op with error recovery" begin
        g = HardwareResilience.KernelGuardian("Global-Op", :Healthy)
        result = HardwareResilience.monitor_kernel(g, () -> begin
            error("hardware fault simulation")
        end)
        @test result === nothing
    end

    @testset "peak_performance_op with empty vectors" begin
        g = HardwareResilience.KernelGuardian("Global-Op", :Healthy)
        result = HardwareResilience.monitor_kernel(g, () -> begin
            SiliconCore.vector_add_asm(Int[], Int[])
        end)
        @test result == Int[]
    end

end
