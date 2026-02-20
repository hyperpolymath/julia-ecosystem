# SPDX-License-Identifier: PMPL-1.0-or-later
module Missions

using JSON3

export Mission, Exercise, load_mission, start_mission, check_mission

struct Exercise
    id::String
    prompt::String
    goal_description::String
    starter_code::String
    validator::Function # A function that checks the environment/state
end

struct Mission
    id::String
    title::String
    concept::String
    exercises::Vector{Exercise}
end

# A simple "State" tracker to see what the kid has done
mutable struct ClassroomState
    circles_drawn::Int
    squares_drawn::Int
    stars_drawn::Int
    colors_used::Set{Any}
    robot_x::Float64
    robot_y::Float64
    robot_path::Vector{Tuple{Float64, Float64}}
    last_feedback::String
end

const GLOBAL_STATE = ClassroomState(0, 0, 0, Set(), 0.0, 0.0, [], "Ready to start!")

function reset_state!()
    GLOBAL_STATE.circles_drawn = 0
    GLOBAL_STATE.squares_drawn = 0
    GLOBAL_STATE.stars_drawn = 0
    empty!(GLOBAL_STATE.colors_used)
    GLOBAL_STATE.robot_x = 0.0
    GLOBAL_STATE.robot_y = 0.0
    empty!(GLOBAL_STATE.robot_path)
    GLOBAL_STATE.last_feedback = "Started!"
end

"""
    check_mission(exercise_id)

Checks if the current work meets the goals of the mission.
"""
function check_mission(mission::Mission, exercise_idx::Int)
    ex = mission.exercises[exercise_idx]
    success, message = ex.validator(GLOBAL_STATE)
    
    if success
        return "🌟 GREAT JOB! 🌟
$message"
    else
        return "Keep trying! 💡
Hint: $message"
    end
end

# We will implement load_mission later to read from JSON
# For now, let's provide a way to build a mission in code

end # module
