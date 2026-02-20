# SPDX-License-Identifier: PMPL-1.0-or-later
module Bargaining

using ..Types
using DataFrames

export compare_clauses, cost_proposal

function compare_clauses(c::ContractClause)
    return DataFrame(
        Section = [c.section],
        Current = [c.current_text],
        Proposed = [c.proposed_text],
        Priority = [c.priority]
    )
end

"""
    cost_proposal(proposal, unit_count)
Calculates the total cost impact of a bargaining proposal.
"""
function cost_proposal(p::BargainingProposal, headcount::Int)
    return p.cost_estimate * headcount
end

end # module
