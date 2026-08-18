using JuliaForChildren

# Define Juliet's House Mission
house_mission = Mission(
    "m2",
    "Juliet's Dream Home",
    "Basic Geometry",
    [
        Exercise(
            "e1",
            "Help Juliet draw a square for the house body! (Move 100 units, turn 90 deg, 4 times)",
            "Goal: A closed square path.",
            "Canvas()",
            (state) -> begin
                if length(state.robot_path) >= 4 && isapprox(state.robot_x, 0.0, atol=1.0) && isapprox(state.robot_y, 0.0, atol=1.0)
                    return (true, "A solid foundation! Juliet has a house body.")
                else
                    return (false, "Juliet needs to finish her square! Did she return to (0,0)?")
                end
            end
        )
    ]
)

# Simulation of a kid solving the mission:
Canvas()

# Draw the house body
for _ in 1:4
    move_forward(100)
    turn_right(90)
end

# Check the mission result
println(check_mission(house_mission, 1))

# Now Juliet draws a roof just for fun
set_robot_color(:red)
turn_left(30)
move_forward(100)
turn_right(120)
move_forward(100)

finish_drawing()
println("Mission accomplished! 🤖🏠")
