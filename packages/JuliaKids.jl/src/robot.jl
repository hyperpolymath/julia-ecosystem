# SPDX-License-Identifier: PMPL-1.0-or-later
module JulietRobot

using Luxor
import ..Missions: GLOBAL_STATE

export move_forward, turn_left, turn_right, pen_up, pen_down, set_robot_color

mutable struct RobotState
    x::Float64
    y::Float64
    angle::Float64 # in degrees
    drawing::Bool
    color::Any
end

const JULIET = RobotState(0.0, 0.0, -90.0, true, :black) # Start at center, facing up

"""
    draw_juliet()

Draws a cool little robot icon at Juliet's current position.
"""
function draw_juliet()
    @layer begin
        translate(JULIET.x, JULIET.y)
        rotate(deg2rad(JULIET.angle + 90)) # Rotate robot to match heading
        
        # Robot Body
        setcolor("silver")
        rect(Point(-15, -15), 30, 30, :fill)
        setcolor("gray")
        rect(Point(-15, -15), 30, 30, :stroke)
        
        # Robot Head
        setcolor("silver")
        circle(Point(0, -20), 10, :fill)
        setcolor("gray")
        circle(Point(0, -20), 10, :stroke)
        
        # Glowing Eyes
        setcolor("cyan")
        circle(Point(-4, -22), 2, :fill)
        circle(Point(4, -22), 2, :fill)
        
        # Little Antenna
        setcolor("red")
        line(Point(0, -30), Point(0, -35), :stroke)
        circle(Point(0, -37), 2, :fill)
    end
end

"""
    move_forward(distance)

Tells Juliet to roll forward!
"""
function move_forward(d)
    rad = deg2rad(JULIET.angle)
    new_x = JULIET.x + d * cos(rad)
    new_y = JULIET.y + d * sin(rad)
    
    if JULIET.drawing
        setcolor(JULIET.color)
        line(Point(JULIET.x, JULIET.y), Point(new_x, new_y), :stroke)
    end
    
    JULIET.x = new_x
    JULIET.y = new_y
    
    # Track for missions
    GLOBAL_STATE.robot_x = JULIET.x
    GLOBAL_STATE.robot_y = JULIET.y
    push!(GLOBAL_STATE.robot_path, (JULIET.x, JULIET.y))
    
    draw_juliet()
end

"""
    turn_left(degrees)

Tells Juliet to spin to her left.
"""
function turn_left(deg)
    JULIET.angle -= deg
    draw_juliet()
end

"""
    turn_right(degrees)

Tells Juliet to spin to her right.
"""
function turn_right(deg)
    JULIET.angle += deg
    draw_juliet()
end

function pen_up()
    JULIET.drawing = false
end

function pen_down()
    JULIET.drawing = true
end

function set_robot_color(c)
    JULIET.color = c
end

end # module
