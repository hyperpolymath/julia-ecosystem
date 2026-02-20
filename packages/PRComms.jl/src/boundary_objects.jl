# SPDX-License-Identifier: PMPL-1.0-or-later
module BoundaryObjects

using ..Types

export MessageHouse, SharedGlossary, create_message_house

"""
    MessageHouse
A classic boundary object used to align different departments on a single narrative.
"""
struct MessageHouse
    roof::String # The Umbrella Statement
    pillars::Vector{String} # Core arguments
    foundation::Vector{String} # Supporting facts/data
end

"""
    SharedGlossary
Aligns terminology between 'Comms speak' and 'Technical/Finance speak'.
"""
struct SharedGlossary
    terms::Dict{String, String} # Term => Definition
end

function create_message_house(roof, pillars, foundation)
    return MessageHouse(roof, pillars, foundation)
end

end # module
