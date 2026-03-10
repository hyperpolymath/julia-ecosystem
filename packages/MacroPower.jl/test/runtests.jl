# SPDX-License-Identifier: PMPL-1.0-or-later
using Test

include(joinpath(@__DIR__, "..", "src", "MacroPower.jl"))
using .MacroPower

@testset "MacroPower.jl" begin

    @testset "Trigger construction" begin
        t = Trigger("always_true", () -> true)
        @test t.name == "always_true"
        @test t.check isa Function
        @test t.check() === true
    end

    @testset "Action construction" begin
        a = Action("print_hello", () -> "hello")
        @test a.name == "print_hello"
        @test a.execute isa Function
        @test a.execute() == "hello"
    end

    @testset "Workflow construction" begin
        t = Trigger("t1", () -> true)
        a = Action("a1", () -> nothing)
        wf = Workflow("test_workflow", [t], [a])
        @test wf.name == "test_workflow"
        @test length(wf.triggers) == 1
        @test length(wf.actions) == 1
        @test wf.triggers[1].name == "t1"
        @test wf.actions[1].name == "a1"
    end

    @testset "Workflow with empty triggers and actions" begin
        wf = Workflow("empty", Trigger[], Action[])
        @test wf.name == "empty"
        @test isempty(wf.triggers)
        @test isempty(wf.actions)
    end

    @testset "Workflow with multiple triggers and actions" begin
        triggers = [
            Trigger("t1", () -> true),
            Trigger("t2", () -> false),
        ]
        actions = [
            Action("a1", () -> 1),
            Action("a2", () -> 2),
        ]
        wf = Workflow("multi", triggers, actions)
        @test length(wf.triggers) == 2
        @test length(wf.actions) == 2
    end

    @testset "@workflow macro" begin
        wf = @workflow "test_macro" begin end
        @test wf isa Workflow
        @test wf.name == "test_macro"
        # The stub macro creates empty triggers/actions
        @test isempty(wf.triggers)
        @test isempty(wf.actions)
    end

    @testset "run_workflow with no triggers" begin
        wf = Workflow("no_triggers", Trigger[], Action[])
        # Should complete without error
        result = run_workflow(wf)
        @test result === nothing
    end

    @testset "run_workflow executes actions when triggered" begin
        executed = Ref(false)
        t = Trigger("fire", () -> true)
        a = Action("set_flag", () -> (executed[] = true))
        wf = Workflow("exec_test", [t], [a])
        run_workflow(wf)
        @test executed[] === true
    end

    @testset "run_workflow skips actions when not triggered" begin
        executed = Ref(false)
        t = Trigger("no_fire", () -> false)
        a = Action("set_flag", () -> (executed[] = true))
        wf = Workflow("skip_test", [t], [a])
        run_workflow(wf)
        @test executed[] === false
    end

    @testset "types are immutable" begin
        @test !ismutable(Trigger("t", () -> true))
        @test !ismutable(Action("a", () -> nothing))
        @test !ismutable(Workflow("w", Trigger[], Action[]))
    end

end
