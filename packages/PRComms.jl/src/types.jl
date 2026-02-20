# SPDX-License-Identifier: PMPL-1.0-or-later
module Types

using Dates

export MessagePillar, AudienceVariant, PressRelease, MediaContact, PitchRecord, Campaign, CrisisPlaybook

struct MessagePillar
    id::Symbol
    theme::String
    key_points::Vector{String}
    approved_by::String
    approved_at::DateTime
end

struct AudienceVariant
    id::Symbol
    pillar_id::Symbol
    audience::Symbol # :investors, :customers, :employees, :media
    channel::Symbol  # :twitter, :linkedin, :press_release, :internal
    tone::Symbol     # :formal, :empathetic, :bold
    body::String
end

mutable struct PressRelease
    id::Symbol
    title::String
    body::String
    status::Symbol # :draft, :review, :embargoed, :published
    embargo_at::Union{DateTime, Nothing}
    approved_at::Union{DateTime, Nothing}
end

struct MediaContact
    id::Symbol
    name::String
    outlet::String
    beat::Vector{Symbol}
    email::String
    last_contacted_at::Union{DateTime, Nothing}
end

struct PitchRecord
    id::Symbol
    contact_id::Symbol
    release_id::Symbol
    sent_at::DateTime
    response_status::Symbol # :sent, :opened, :responded, :declined
end

struct Campaign
    id::Symbol
    name::String
    goals::Vector{String}
    start_at::DateTime
    end_at::DateTime
end

struct CrisisPlaybook
    id::Symbol
    incident_type::Symbol
    severity::Int # 1-5
    holding_statements::Vector{String}
    escalation_tree::Vector{String}
end

end # module
