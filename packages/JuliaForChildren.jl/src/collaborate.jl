# SPDX-License-Identifier: MPL-2.0
module Collaborate

export join_classroom, say_hello

"""
    join_classroom(room_name)

Connects to your friends in the classroom!
"""
function join_classroom(room_name)
    println("📡 Connecting to room: $room_name...")
    return "You are connected! 👋"
end

"""
    say_hello(msg)

Sends a message to everyone in the room.
"""
function say_hello(msg)
    println("[Chat]: $msg")
end

end # module
