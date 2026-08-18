using TradeUnionist
using Dates

# 1. Start a new drive
site = register_worksite("Acme Corp", "North Site", "Manufacturing", 250)
println("Organizing drive started at $(site.employer) ($(site.location))")

# 2. Add a potential supporter
worker = upsert_member(site.id, :worker_42, :supporter, :worker)

# 3. Log a house-visit conversation
conv = log_conversation(worker.id, ["Wages", "Safety"], 0.7, "Invite to committee meeting")
println("Logged conversation with $(worker.id): Sentiment $(conv.sentiment)")

# 4. Open a grievance for a safety issue
grievance = open_grievance(worker.id, "Oil leak on Line 4")
println("Grievance $(grievance.id) opened. Due: $(grievance.due_date)")

# 5. Bargaining Prep: Cost a wage increase
proposal = BargainingProposal(:p1, :wage_clause, "Catch up with inflation", 2500.0, :draft)
total_cost = cost_proposal(proposal, site.headcount_estimate)
println("Proposed wage increase total cost: \$$(total_cost)")

println("
Union power is building! ✊🛠️")
