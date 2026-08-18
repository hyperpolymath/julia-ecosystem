using JuliaForChildren
using DataFrames

# 1. Let's track some "Experiment" data
my_experiment = DataFrame(
    Day = [1, 2, 3],
    PlantHeight_cm = [2.5, 4.1, 6.8],
    Note = ["Sprouting!", "Growing fast", "Wow!"]
)

# Save it to Excel!
save_spreadsheet("my_plant_growth", my_experiment)

# 2. Make a report about it
report_text = """
I am growing a plant in my kitchen.
I use Julia to track how fast it grows.
It grew almost 3cm between Day 2 and Day 3!
"""
make_report("plant_report", "My Biology Project", report_text)

# 3. Create a presentation for class
make_slides("class_presentation", [
    "My Science Project",
    "How I used Julia",
    "The Results",
    "Thank You!"
])

println("Homework is finished! 📚✨")
