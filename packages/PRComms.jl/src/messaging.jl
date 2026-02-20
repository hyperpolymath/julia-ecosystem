# SPDX-License-Identifier: PMPL-1.0-or-later
module Messaging

using ..Types
using Dates
using Mustache

export create_pillar, generate_variant

"""
    create_pillar(theme, points; approver="Chief Comms")

Creates a high-integrity message pillar.
"""
function create_pillar(id::Symbol, theme::String, points::Vector{String}; approver="Chief Comms")
    return MessagePillar(id, theme, points, approver, now())
end

"""
    generate_variant(pillar::MessagePillar, audience::Symbol, channel::Symbol; tone=:formal)

Generates a targeted message variant based on a pillar.
"""
function generate_variant(pillar::MessagePillar, audience::Symbol, channel::Symbol; tone=:formal)
    # In a real system, this would use LLM or templates to transform points.
    # For now, we'll build a structured body.
    body = "Targeting $(audience) via $(channel):
"
    for pt in pillar.key_points
        body *= "- $pt
"
    end
    
    return AudienceVariant(
        gensym("var"),
        pillar.id,
        audience,
        channel,
        tone,
        body
    )
end

end # module
