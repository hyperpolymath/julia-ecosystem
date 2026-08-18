using JuliaForChildren

# INTERMEDIATE EXAMPLE: A Magic Garden
# This script uses a loop to build art and a Magic Rule to log it.

Canvas(800, 600)

# Setup a Magic Rule to celebrate every flower (star)
when(:star_planted, (x, y) -> begin
    println("Garden growth at ($x, $y)!")
    minecraft_chat("A new flower grew in my garden!")
end)

# Function to draw a "flower"
function plant_flower(x, y, color)
    set_brush(color)
    draw_star(20, x=x, y=y)
    # Manually trigger our magic rule
    import .Automation: trigger_event
    trigger_event(:star_planted, x, y)
end

# Plant a row of flowers
colors = [:red, :blue, :green, :orange, :purple]
for i in 1:10
    x = -300 + (i * 60)
    y = 100 + (rand(-50:50)) # Add some random height
    plant_flower(x, y, rand(colors))
end

finish_drawing()
println("My garden is beautiful! 🌸")
