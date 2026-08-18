using JuliaForChildren

# Let's build something in Minecraft!
println(connect_to_minecraft())

minecraft_chat("Julia is building a tower!")

# Build a gold tower 10 blocks high
for y in 1:10
    place_block(0, y, 0, Blocks.gold)
end

# Put a diamond on top!
place_block(0, 11, 0, Blocks.diamond)

# Teleport to the top to see the view
teleport_player(0, 15, 0)

minecraft_chat("Look at my tower! 💎")
println("Tower built! ⛏️")
