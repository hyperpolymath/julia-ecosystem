# SPDX-License-Identifier: PMPL-1.0-or-later

"""
    JuliaForChildren

Educational Julia programming toolkit for children aged 7-14. Provides
simplified interfaces for drawing, robotics, Minecraft modding, KSP
mission planning, game development, and collaborative coding.

# Key Features
- Mission-based learning with progressive difficulty
- Turtle graphics and Luxor drawing integration
- Minecraft and KSP automation interfaces
- Accessibility-first design with screen reader support

# Example
```julia
using JuliaForChildren
mission = create_mission("Draw a Star", difficulty=:beginner)
```
"""
module JuliaForChildren

include("missions.jl")
include("drawing.jl")
include("interop.jl")
include("robot.jl")
include("minecraft.jl")
include("automation.jl")
include("ksp.jl")
include("gamebolt.jl")
include("accessibility.jl")
include("timetools.jl")
include("bibtools.jl")
include("avatars.jl")
include("collaborate.jl")
include("llm_buddy.jl")
include("factcheck.jl")

using .Drawing
using .Missions
using .SchoolTools
using .JulietRobot
using .Minecraft
using .Automation
using .KSP
using .GameBolt
using .Accessibility
using .TimeTools
using .BibTools
using .Avatars
using .Collaborate
using .LLMBuddy
using .FactCheck

# Re-export everything
export Canvas, draw_circle, draw_square, draw_star, set_brush, finish_drawing
export Mission, Exercise, check_mission
export save_spreadsheet, read_spreadsheet, make_report, make_slides
export move_forward, turn_left, turn_right, pen_up, pen_down, set_robot_color
export connect_to_minecraft, place_block, teleport_player, minecraft_chat, Blocks
export when, trigger_event
export connect_to_ksp, launch_rocket, get_altitude
export connect_to_gamebolt, spawn_sprite, move_sprite
export speak, high_contrast_mode, describe_image
export set_alarm, calendar_event
export add_reference, export_to_zotero
export make_avatar
export join_classroom, say_hello
export ask_buddy
export is_true

end # module JuliaForChildren
