using JuliaForChildren

# BASIC EXAMPLE: Drawing a face
Canvas(400, 400)

# The Head
set_brush(:yellow)
draw_circle(100)

# The Eyes
set_brush(:black)
draw_circle(10, x=-40, y=-30)
draw_circle(10, x=40, y=-30)

# The Smile (made of little circles)
set_brush(:red)
for x_pos in -50:20:50
    # Use a little math to make a curve!
    y_pos = (x_pos^2 / 100) + 20
    draw_circle(5, x=x_pos, y=y_pos)
end

finish_drawing()
println("Hello, Julia! 😊")
