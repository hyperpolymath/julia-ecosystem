# SPDX-License-Identifier: PMPL-1.0-or-later
module Branding

using PRComms
using ..Types

export UnionLeaderProfile, make_leader_card

struct UnionLeaderProfile
    name::String
    title::String
    local_number::String
    narratives::Vector{String}
end

"""
    make_leader_card(leader)
Uses PRComms asset engine to make professional cards for stewards/officers.
"""
function make_leader_card(l::UnionLeaderProfile)
    filename = "card_$(l.name)"
    return make_business_card(filename, l.name, l.title, "Local $(l.local_number)")
end

end # module
