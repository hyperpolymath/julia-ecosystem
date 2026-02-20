# SPDX-License-Identifier: PMPL-1.0-or-later
module Types

using Dates

export Worksite, MemberRecord, OrganizerConversation, GrievanceCase, ContractClause, BargainingProposal, MobilizationPlan

struct GeoLocation
    lat::Float64
    lon::Float64
end

struct Worksite
    id::Symbol
    employer::String
    location_name::String
    unit::String
    headcount_estimate::Int
    geo::Union{GeoLocation, Nothing}
end

mutable struct MemberRecord
    id::Symbol
    worksite_id::Symbol
    status::Symbol # :member, :supporter, :non_member, :leader
    role::Symbol   # :worker, :steward, :organizer
    issues::Vector{String}
    last_contact_at::Union{DateTime, Nothing}
    home_geo::Union{GeoLocation, Nothing}
end

struct OrganizerConversation
    id::Symbol
    member_id::Symbol
    topic_tags::Vector{String}
    sentiment::Float64 # -1.0 to 1.0
    next_step::String
    timestamp::DateTime
end

mutable struct GrievanceCase
    id::Symbol
    member_id::Symbol
    filed_at::DateTime
    status::Symbol # :intake, :step1, :step2, :arbitration, :resolved
    evidence_refs::Vector{String}
    due_date::DateTime
end

struct ContractClause
    id::Symbol
    section::String
    current_text::String
    proposed_text::String
    priority::Symbol # :high, :medium, :low
end

struct BargainingProposal
    id::Symbol
    clause_id::Symbol
    rationale::String
    cost_estimate::Float64
    status::Symbol # :draft, :presented, :accepted, :rejected
end

struct MobilizationPlan
    id::Symbol
    action_type::Symbol # :strike_vote, :rally, :petition
    date::DateTime
    target_turnout::Int
    actual_turnout::Int
end

end # module
