using JuliaForChildren
import .Automation: trigger_event

# 1. Setup some Magic Rules (Automation)
println(when(:homework_done, () -> begin
    println("School is out! Time for games! 🎮")
    minecraft_chat("Homework finished! Let's play!")
    launch_rocket() # KSP launch!
end))

println(when(:new_achievement, (badge) -> begin
    println("CONGRATS! You earned the $badge badge!")
    spawn_sprite("Trophy", 100, 100) # GameBolt spawn
end))

# 2. Simulate doing some work and triggering events
println("
--- Doing Homework ---")
make_report("math_homework", "1+1=2", "I did my math.")

# Trigger the "homework_done" event manually
# In the future, this will happen automatically when make_report finishes!
trigger_event(:homework_done)

# Trigger an achievement
trigger_event(:new_achievement, "Star-Builder")

println("
Automation is working! ⚙️✨")
