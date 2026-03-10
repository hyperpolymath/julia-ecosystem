# SPDX-License-Identifier: PMPL-1.0-or-later
using Test
using JuliaForChildren

@testset "JuliaForChildren.jl" begin

    # -----------------------------------------------------------------------
    # Missions
    # -----------------------------------------------------------------------
    @testset "Missions" begin
        @testset "Exercise" begin
            validator = (state) -> (true, "Well done!")
            ex = Exercise("ex1", "Draw a circle", "Draw one circle",
                          "draw_circle(50)", validator)
            @test ex.id == "ex1"
            @test ex.prompt == "Draw a circle"
            @test ex.goal_description == "Draw one circle"
            @test ex.starter_code == "draw_circle(50)"
            @test ex.validator isa Function
        end

        @testset "Mission" begin
            validator = (state) -> (true, "Passed")
            ex1 = Exercise("e1", "Do X", "Goal X", "code()", validator)
            ex2 = Exercise("e2", "Do Y", "Goal Y", "more_code()", validator)
            mission = Mission("m1", "First Mission", "Loops", [ex1, ex2])
            @test mission.id == "m1"
            @test mission.title == "First Mission"
            @test mission.concept == "Loops"
            @test length(mission.exercises) == 2
        end

        @testset "check_mission success" begin
            validator = (state) -> (true, "You drew enough circles!")
            ex = Exercise("check1", "Draw circles", "3 circles", "", validator)
            mission = Mission("cm1", "Circle Mission", "Drawing", [ex])
            result = check_mission(mission, 1)
            @test occursin("GREAT JOB", result)
            @test occursin("You drew enough circles!", result)
        end

        @testset "check_mission failure" begin
            validator = (state) -> (false, "Try drawing more circles")
            ex = Exercise("check2", "Draw circles", "5 circles", "", validator)
            mission = Mission("cm2", "Circle Mission", "Drawing", [ex])
            result = check_mission(mission, 1)
            @test occursin("Keep trying", result)
            @test occursin("Try drawing more circles", result)
        end

        @testset "ClassroomState (GLOBAL_STATE)" begin
            state = JuliaForChildren.Missions.GLOBAL_STATE
            @test state isa JuliaForChildren.Missions.ClassroomState
            @test hasproperty(state, :circles_drawn)
            @test hasproperty(state, :squares_drawn)
            @test hasproperty(state, :stars_drawn)
            @test hasproperty(state, :colors_used)
            @test hasproperty(state, :robot_x)
            @test hasproperty(state, :robot_y)
            @test hasproperty(state, :robot_path)
            @test hasproperty(state, :last_feedback)
        end

        @testset "reset_state!" begin
            state = JuliaForChildren.Missions.GLOBAL_STATE
            state.circles_drawn = 42
            state.robot_x = 100.0
            JuliaForChildren.Missions.reset_state!()
            @test state.circles_drawn == 0
            @test state.squares_drawn == 0
            @test state.stars_drawn == 0
            @test isempty(state.colors_used)
            @test state.robot_x == 0.0
            @test state.robot_y == 0.0
            @test isempty(state.robot_path)
            @test state.last_feedback == "Started!"
        end
    end

    # -----------------------------------------------------------------------
    # Minecraft (struct + constants only, no network)
    # -----------------------------------------------------------------------
    @testset "Minecraft" begin
        @testset "Blocks constants" begin
            @test Blocks.air == 0
            @test Blocks.stone == 1
            @test Blocks.grass == 2
            @test Blocks.dirt == 3
            @test Blocks.cobblestone == 4
            @test Blocks.wood == 5
            @test Blocks.water == 8
            @test Blocks.lava == 10
            @test Blocks.gold == 41
            @test Blocks.iron == 42
            @test Blocks.diamond == 57
            @test Blocks.tnt == 46
        end
    end

    # -----------------------------------------------------------------------
    # Automation
    # -----------------------------------------------------------------------
    @testset "Automation" begin
        @testset "when registers a rule" begin
            # Clear any previous rules
            empty!(JuliaForChildren.Automation.ACTIVE_RULES)

            result = when(:test_event, (args...) -> nothing)
            @test result isa String
            @test occursin("Magic Rule", result)
            @test length(JuliaForChildren.Automation.ACTIVE_RULES) == 1
        end

        @testset "trigger_event fires matching rules" begin
            empty!(JuliaForChildren.Automation.ACTIVE_RULES)
            fired = Ref(false)
            when(:button_press, (args...) -> (fired[] = true))
            JuliaForChildren.Automation.trigger_event(:button_press)
            @test fired[] == true
        end

        @testset "trigger_event ignores non-matching events" begin
            empty!(JuliaForChildren.Automation.ACTIVE_RULES)
            fired = Ref(false)
            when(:click, (args...) -> (fired[] = true))
            JuliaForChildren.Automation.trigger_event(:hover)
            @test fired[] == false
        end

        @testset "trigger_event with condition" begin
            empty!(JuliaForChildren.Automation.ACTIVE_RULES)
            result = Ref(0)
            when(:score, (x...) -> (result[] = x[1]);
                 condition=(x...) -> length(x) > 0 && x[1] > 50)
            JuliaForChildren.Automation.trigger_event(:score, 30)
            @test result[] == 0  # condition not met
            JuliaForChildren.Automation.trigger_event(:score, 75)
            @test result[] == 75  # condition met
        end

        @testset "MagicRule struct" begin
            rule = JuliaForChildren.Automation.MagicRule(
                :test, (args...) -> true, (args...) -> nothing
            )
            @test rule.event_type == :test
            @test rule.condition isa Function
            @test rule.action isa Function
        end
    end

    # -----------------------------------------------------------------------
    # KSP (placeholder functions, no network)
    # -----------------------------------------------------------------------
    @testset "KSP" begin
        @testset "connect_to_ksp" begin
            result = connect_to_ksp()
            @test result isa String
            @test occursin("KSP", result)
        end

        @testset "launch_rocket" begin
            result = launch_rocket()
            @test result isa String
            @test occursin("Lift off", result)
        end

        @testset "get_altitude" begin
            alt = get_altitude()
            @test alt isa Float64
            @test alt == 10000.0
        end
    end

    # -----------------------------------------------------------------------
    # GameBolt (placeholder functions, no network)
    # -----------------------------------------------------------------------
    @testset "GameBolt" begin
        @testset "connect_to_gamebolt" begin
            result = connect_to_gamebolt()
            @test result isa String
            @test occursin("GameBolt", result)
        end

        @testset "spawn_sprite" begin
            result = spawn_sprite("hero", 10, 20)
            @test result isa String
            @test occursin("hero", result)
            @test occursin("10", result)
            @test occursin("20", result)
        end

        @testset "move_sprite" begin
            result = move_sprite("enemy", 5, 5)
            @test result isa String
            @test occursin("enemy", result)
        end
    end

    # -----------------------------------------------------------------------
    # Accessibility
    # -----------------------------------------------------------------------
    @testset "Accessibility" begin
        @testset "speak does not error" begin
            # speak() only prints, so just verify it does not throw
            @test speak("Hello world") === nothing
        end

        @testset "high_contrast_mode" begin
            @test high_contrast_mode(true) === nothing
            @test high_contrast_mode(false) === nothing
        end

        @testset "describe_image" begin
            desc = describe_image("some_drawing")
            @test desc isa String
            @test !isempty(desc)
        end
    end

    # -----------------------------------------------------------------------
    # TimeTools
    # -----------------------------------------------------------------------
    @testset "TimeTools" begin
        @testset "set_alarm does not error" begin
            @test set_alarm("15:30", "Coding class!") === nothing
        end

        @testset "calendar_event does not error" begin
            @test calendar_event("Science Fair", "2026-05-15") === nothing
        end
    end

    # -----------------------------------------------------------------------
    # BibTools
    # -----------------------------------------------------------------------
    @testset "BibTools" begin
        @testset "Reference struct" begin
            ref = JuliaForChildren.BibTools.Reference("Julia Book", "Jeff Bezanson", 2023, "https://example.com")
            @test ref.title == "Julia Book"
            @test ref.author == "Jeff Bezanson"
            @test ref.year == 2023
            @test ref.link == "https://example.com"
        end

        @testset "add_reference" begin
            # Clear library
            empty!(JuliaForChildren.BibTools.MY_LIBRARY)
            add_reference("My Book", "Author A", 2025)
            @test length(JuliaForChildren.BibTools.MY_LIBRARY) == 1
            @test JuliaForChildren.BibTools.MY_LIBRARY[1].title == "My Book"
            @test JuliaForChildren.BibTools.MY_LIBRARY[1].link == ""  # default

            add_reference("Second Book", "Author B", 2024, "https://example.org")
            @test length(JuliaForChildren.BibTools.MY_LIBRARY) == 2
            @test JuliaForChildren.BibTools.MY_LIBRARY[2].link == "https://example.org"
        end

        @testset "export_to_zotero does not error" begin
            @test export_to_zotero("library") === nothing
            @test export_to_zotero("library.json") === nothing
        end
    end

    # -----------------------------------------------------------------------
    # Collaborate
    # -----------------------------------------------------------------------
    @testset "Collaborate" begin
        @testset "join_classroom" begin
            result = join_classroom("Room 101")
            @test result isa String
            @test occursin("connected", result)
        end

        @testset "say_hello does not error" begin
            @test say_hello("Hi everyone!") === nothing
        end
    end

    # -----------------------------------------------------------------------
    # LLMBuddy
    # -----------------------------------------------------------------------
    @testset "LLMBuddy" begin
        @testset "ask_buddy" begin
            answer = ask_buddy("How do I make a loop?")
            @test answer isa String
            @test !isempty(answer)
        end
    end

    # -----------------------------------------------------------------------
    # FactCheck
    # -----------------------------------------------------------------------
    @testset "FactCheck" begin
        @testset "is_true whale-fish" begin
            @test is_true("A whale is a fish") == false
            @test is_true("a WHALE is a FISH") == false  # case insensitive
        end

        @testset "is_true generic statements" begin
            @test is_true("The sky is blue") == true
            @test is_true("Julia is fast") == true
        end

        @testset "is_true empty string" begin
            @test is_true("") == true  # optimistic default
        end
    end

    # -----------------------------------------------------------------------
    # Robot (JulietRobot) - state tests without Luxor rendering
    # -----------------------------------------------------------------------
    @testset "JulietRobot state" begin
        @testset "JULIET initial state" begin
            juliet = JuliaForChildren.JulietRobot.JULIET
            @test juliet isa JuliaForChildren.JulietRobot.RobotState
            @test hasproperty(juliet, :x)
            @test hasproperty(juliet, :y)
            @test hasproperty(juliet, :angle)
            @test hasproperty(juliet, :drawing)
            @test hasproperty(juliet, :color)
        end

        @testset "pen_up and pen_down" begin
            pen_down()
            juliet = JuliaForChildren.JulietRobot.JULIET
            @test juliet.drawing == true

            pen_up()
            @test juliet.drawing == false

            pen_down()
            @test juliet.drawing == true
        end

        @testset "set_robot_color" begin
            set_robot_color(:red)
            @test JuliaForChildren.JulietRobot.JULIET.color == :red

            set_robot_color("green")
            @test JuliaForChildren.JulietRobot.JULIET.color == "green"
        end
    end

end # top-level testset
