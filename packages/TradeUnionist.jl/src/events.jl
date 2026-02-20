# SPDX-License-Identifier: PMPL-1.0-or-later
module Events

using Dates
using ..Types

export UnionEvent, StrikeVote, Rally, TownHall, CommitteeMeeting, create_event_template

abstract type EventType end

struct StrikeVote <: EventType end
struct Rally <: EventType end
struct TownHall <: EventType end
struct CommitteeMeeting <: EventType end

struct UnionEvent
    type::EventType
    title::String
    date::DateTime
    location::String
    target_attendance::Int
    check_in_list::Vector{Symbol} # IDs of members
end

function create_event_template(type::EventType, title::String, date::DateTime)
    return UnionEvent(type, title, date, "TBD", 0, Symbol[])
end

end # module
