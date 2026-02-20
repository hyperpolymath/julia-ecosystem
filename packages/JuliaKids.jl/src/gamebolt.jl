# SPDX-License-Identifier: PMPL-1.0-or-later
module GameBolt

# Interop with the GameBolt open-source game engine
export connect_to_gamebolt, spawn_sprite, move_sprite

function connect_to_gamebolt()
    return "GameBolt Engine ready! 🎮"
end

function spawn_sprite(name, x, y)
    return "Spawned $name at ($x, $y) 👾"
end

function move_sprite(name, x, y)
    return "Moving $name... 🏃"
end

end # module
