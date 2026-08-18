using JuliaForChildren
using DataFrames
import .Automation: trigger_event

# COMPLEX EXAMPLE: Interplanetary Data Mission
# Goal: Monitor a KSP rocket, log data to a report, and build a base in Minecraft.

# 1. Setup Mission Rules
when(:rocket_launched, () -> begin
    println("MISSION CONTROL: Ignition confirmed! 🚀")
    minecraft_chat("Rocket is going up!!")
end)

when(:high_altitude, (alt) -> begin
    println("MISSION CONTROL: Target altitude reached: $alt meters")
    # Build a beacon in Minecraft at that altitude
    place_block(0, Int(round(alt/100)), 0, Blocks.diamond)
    minecraft_chat("Beacon placed at altitude $alt!")
end)

# 2. Start the Mission
println(connect_to_minecraft())
println(connect_to_ksp())

# Simulate Launch
trigger_event(:rocket_launched)
launch_rocket()

# Track Telemetry over "Time"
telemetry = DataFrame(Time_s = Int[], Altitude_m = Float64[])

for t in 1:10
    current_alt = get_altitude() * (t/5)^2 # Simulate acceleration
    push!(telemetry, (t, current_alt))
    
    if current_alt > 5000.0
        trigger_event(:high_altitude, current_alt)
    end
end

# 3. Save the Mission Science
save_spreadsheet("rocket_data", telemetry)

report_text = """
Mission Report: The Great Julia Rocket
Total Flight Time: 10 seconds
Max Altitude Reached: $(maximum(telemetry.Altitude_m)) meters
Status: Success!
"""
make_report("science_report", "Rocket Science Notes", report_text)

# 4. Final Slide Deck for School
make_slides("rocket_presentation", [
    "My Space Mission",
    "How I used Julia",
    "The Data Results",
    "Minecraft Beacon Status"
])

println("
Space mission complete and homework is ready! 🌌🎓")
