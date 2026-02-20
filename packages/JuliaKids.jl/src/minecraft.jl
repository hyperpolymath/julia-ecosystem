# SPDX-License-Identifier: PMPL-1.0-or-later
module Minecraft

using Sockets

export connect_to_minecraft, place_block, teleport_player, minecraft_chat, Blocks

# A small dictionary of friendly block names to Minecraft IDs
const Blocks = (
    air = 0,
    stone = 1,
    grass = 2,
    dirt = 3,
    cobblestone = 4,
    wood = 5,
    water = 8,
    lava = 10,
    gold = 41,
    iron = 42,
    diamond = 57,
    tnt = 46
)

mutable struct MinecraftConnection
    socket::Union{TCPSocket, Nothing}
end

const CURRENT_CONN = MinecraftConnection(nothing)

"""
    connect_to_minecraft(host="localhost", port=4711)

Connects Julia to a Minecraft server running the RaspberryJuice plugin.
"""
function connect_to_minecraft(host="localhost", port=4711)
    try
        CURRENT_CONN.socket = connect(host, port)
        return "Connected to Minecraft! ⛏️💎"
    catch e
        return "Could not find Minecraft. Make sure the server is running with the 'RaspberryJuice' plugin!"
    end
end

function send_cmd(cmd)
    if CURRENT_CONN.socket === nothing
        return "Not connected! Run connect_to_minecraft() first."
    end
    println(CURRENT_CONN.socket, cmd)
end

"""
    place_block(x, y, z, block_id)

Places a single block in the world. Use the 'Blocks' list for names!
"""
function place_block(x, y, z, id)
    send_cmd("world.setBlock($x,$y,$z,$id)")
end

"""
    teleport_player(x, y, z)

Zaps the player to a new location!
"""
function teleport_player(x, y, z)
    send_cmd("player.setPos($x,$y,$z)")
end

"""
    minecraft_chat(message)

Sends a message to the in-game chat.
"""
function minecraft_chat(msg)
    send_cmd("chat.post($msg)")
end

end # module
