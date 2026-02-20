# SPDX-License-Identifier: PMPL-1.0-or-later
module Avatars

using Luxor

export make_avatar

"""
    make_avatar(name; style=:robot, color=:blue)

Generates a cool avatar picture for you!
"""
function make_avatar(name; style=:robot, color=:blue)
    Drawing(200, 200, "$name.png")
    background("white")
    origin()
    
    setcolor(color)
    if style == :robot
        rect(Point(-50, -50), 100, 100, :fill) # Head
        setcolor("white")
        circle(Point(-20, -20), 10, :fill) # Eye
        circle(Point(20, -20), 10, :fill) # Eye
        rect(Point(-30, 20), 60, 10, :fill) # Mouth
    else
        circle(Point(0, 0), 60, :fill) # Face
    end
    
    finish()
    return "Avatar created: $name.png 🖼️"
end

end # module
