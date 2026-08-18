using JuliaForChildren

Canvas(800, 800)

# Let's make Juliet draw a square spiral!
set_robot_color(:purple)

for i in 1:20
    move_forward(i * 10)
    turn_right(90)
end

# Now let's draw a flower with Juliet
pen_up()
move_forward(200)
pen_down()
set_robot_color("orange")

for _ in 1:36
    move_forward(50)
    turn_left(170)
end

finish_drawing()
println("Juliet finished her dance! 🤖💃")
