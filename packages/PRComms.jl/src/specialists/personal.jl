# SPDX-License-Identifier: MPL-2.0
module PersonalPR

using ..Types

export ThoughtLeaderProfile, PresentationRecord, track_appearance

struct ThoughtLeaderProfile
    executive_name::String
    expertise_areas::Vector{String}
    key_narratives::Vector{String}
end

struct PresentationRecord
    event_name::String
    date::Any
    topic::String
    audience_reach::Int
end

function track_appearance(profile, event, date, topic, reach)
    return PresentationRecord(event, date, topic, reach)
end

end # module
