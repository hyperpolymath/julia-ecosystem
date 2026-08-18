using JuliaForChildren

# Define the first mission: "The Snowman Challenge"
snowman_mission = Mission(
    "m1",
    "Building a Snowman",
    "Loops and Shapes",
    [
        Exercise(
            "e1",
            "Draw a snowman using exactly 3 circles!",
            "Goal: 3 circles on the canvas.",
            "Canvas()",
            (state) -> begin
                if state.circles_drawn == 3
                    return (true, "You built a perfect snowman!")
                elseif state.circles_drawn < 3
                    return (false, "Your snowman needs more snowballs! (Try drawing 3 circles)")
                else
                    return (false, "That's a lot of snow! Try exactly 3 circles.")
                end
            end
        )
    ]
)

# Simulation of a kid doing the mission:
Canvas()
draw_circle(100, y=100)
draw_circle(70, y=0)
draw_circle(40, y=-70)

# Check if they won!
println(check_mission(snowman_mission, 1))
