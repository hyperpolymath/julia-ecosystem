# SPDX-License-Identifier: PMPL-1.0-or-later
module Strategy

using Dates
using DataFrames

export CommsPlan, add_milestone, brand_equity_valuation

struct CommsPlan
    id::Symbol
    name::String
    timeline::DataFrame # Date, Channel, Action, Owner
end

function CommsPlan(id::Symbol, name::String)
    return CommsPlan(id, name, DataFrame(Date=Date[], Channel=Symbol[], Action=String[], Owner=String[]))
end

function add_milestone(plan::CommsPlan, date::Date, channel::Symbol, action::String, owner::String)
    push!(plan.timeline, (date, channel, action, owner))
    sort!(plan.timeline, :Date)
    return "Milestone added: $action on $date"
end

"""
    brand_equity_valuation(revenue, brand_strength_index)

A simplified "Relief from Royalty" style valuation model to help Comms talk to Finance.
Translates reputation strength into estimated asset value.
"""
function brand_equity_valuation(annual_revenue::Float64, brand_strength_index::Float64)
    # brand_strength_index is 0.0 to 1.0
    # Implied royalty rate curve (simplified industry standard heuristic)
    royalty_rate = 0.01 + (0.04 * brand_strength_index) # 1% to 5% range
    
    brand_value = annual_revenue * royalty_rate * 5 # 5-year NPV multiplier approximation
    
    return (
        value = brand_value,
        royalty_rate_percent = royalty_rate * 100,
        explanation = "Estimated Brand Asset Value based on $(round(royalty_rate*100, digits=1))% implied royalty."
    )
end

end # module
