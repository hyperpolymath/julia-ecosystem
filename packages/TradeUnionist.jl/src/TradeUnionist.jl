# SPDX-License-Identifier: PMPL-1.0-or-later
"""
    TradeUnionist

Trade union analytics, membership tracking, and labour economics toolkit. Covers
worksite registration, member management, organiser conversations, grievance
handling, collective bargaining analysis, spatial mapping, and union density metrics.

# Key Features
- Worksite and member record management with organiser conversation logging
- Grievance case tracking with status updates
- Contract clause comparison and proposal costing
- Spatial member mapping and geo-mashup analysis
- Union density, coverage, leadership ratio, and wage Gini coefficient metrics

# Example
```julia
using TradeUnionist
site = register_worksite("Factory A", 500)
metrics = UnionMetrics(site)
calc_density(metrics)
```
"""
module TradeUnionist

include("types.jl")
include("organizing.jl")
include("grievances.jl")
include("bargaining.jl")
include("mapping.jl")
include("metrics.jl")
include("planning.jl")
include("events.jl")
include("comms_bridge.jl")
include("branding.jl")

using .Types
using .Organizing
using .Grievances
using .Bargaining
using .Mapping
using .Metrics
using .Planning
using .Events
using .CommsBridge
using .Branding

# Re-export core operations
export Worksite, MemberRecord, OrganizerConversation, GrievanceCase, ContractClause, BargainingProposal, MobilizationPlan, GeoLocation
export register_worksite, upsert_member, log_conversation
export open_grievance, update_grievance_status
export compare_clauses, cost_proposal
export find_members_near, spatial_mashup
export UnionMetrics, calc_density, calc_coverage, calc_leadership_ratio, wage_gini_coefficient

# Re-export new strategic & comms operations
export UnionActivity, StrategicGoal, TacticalObjective, OperationalTask
export UnionEvent, StrikeVote, Rally, TownHall, CommitteeMeeting, create_event_template
export build_union_press_release, generate_member_alert
export UnionLeaderProfile, make_leader_card

end # module TradeUnionist
