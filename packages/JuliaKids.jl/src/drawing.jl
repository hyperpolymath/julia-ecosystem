# SPDX-License-Identifier: PMPL-1.0-or-later
module Drawing

using Luxor
import ..Missions: GLOBAL_STATE

export Canvas, draw_circle, draw_square, draw_star, set_brush, finish_drawing

"""
    Canvas(width, height)

Creates a new drawing area for your art!
"""
function Canvas(w=500, h=500)
    Drawing(w, h, :png)
    origin() # Move (0,0) to the center of the screen
    background("white")
    
    # Reset tracking for a new drawing
    GLOBAL_STATE.circles_drawn = 0
    GLOBAL_STATE.squares_drawn = 0
    GLOBAL_STATE.stars_drawn = 0
    empty!(GLOBAL_STATE.colors_used)
    
    return "Canvas is ready! Let's draw!"
end

"""
    set_brush(color)

Changes the color of your pen. You can use names like :red, :blue, "gold", etc.
"""
function set_brush(c)
    setcolor(c)
    push!(GLOBAL_STATE.colors_used, c)
end

"""
    draw_circle(size; x=0, y=0, mode=:fill)

Draws a circle! 'size' is how big it is. 'x' and 'y' are where it goes.
"""
function draw_circle(s; x=0, y=0, mode=:fill)
    circle(Point(x, y), s, mode)
    GLOBAL_STATE.circles_drawn += 1
end

"""
    draw_square(size; x=0, y=0, mode=:fill)

Draws a square! 'size' is how long the sides are.
"""
function draw_square(s; x=0, y=0, mode=:fill)
    # Luxor rect uses top-left by default, we'll center it for kids
    rect(Point(x - s/2, y - s/2), s, s, mode)
    GLOBAL_STATE.squares_drawn += 1
end

"""
    draw_star(size; x=0, y=0, points=5, mode=:fill)

Draws a sparkly star!
"""
function draw_star(s; x=0, y=0, points=5, mode=:fill)
    star(Point(x, y), s, points, 0.5, 0, mode)
    GLOBAL_STATE.stars_drawn += 1
end

"""
    finish_drawing()

Tells Julia you are done with your masterpiece.
"""
function finish_drawing()
    finish()
    preview()
end

end # module
