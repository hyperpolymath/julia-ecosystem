using PostDisciplinary
using UUIDs

# 1. Start the project
p = ResearchProject("Ecosystem Synergy Analysis")

# 2. Define entities from different 'worlds'
# (In a real script, these would come from the actual library outputs)

kid_learning = LinkedEntity(
    uuid4(), :JuliaForChildren, :badge_01, :badge, 
    Dict(:name => "Logic Lion", :learner => "Alice")
)

union_drive = LinkedEntity(
    uuid4(), :TradeUnionist, :site_north, :event, 
    Dict(:employer => "TechCorp", :type => "Organizing Drive")
)

journalist_leak = LinkedEntity(
    uuid4(), :InvestigativeJournalist, :claim_fraud, :claim, 
    Dict(:text => "TechCorp overvalued assets", :confidence => 0.95)
)

# 3. Link them together across boundaries!
# "Child's education motivates future worker leaders"
# "Investigation provides evidence for the union drive"

println(PostDisciplinary.add_link!(p, kid_learning, union_drive, :inspires))
println(PostDisciplinary.add_link!(p, journalist_leak, union_drive, :supports))

# 4. Generate the Synthesis Report
report = generate_synthesis(p)

println("
--- FINAL SYNTHESIS ---")
println(report.report_title)
display(report.entities)

println("
Boundaries successfully transcended. 🌌")
