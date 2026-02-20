# SPDX-License-Identifier: PMPL-1.0-or-later
module CommsBridge

using PRComms
using ..Types

export build_union_press_release, generate_member_alert

"""
    build_union_press_release(pillar, incident)
Uses PRComms to build a professional union statement about an incident (e.g. unfair labor practice).
"""
function build_union_press_release(pillar::MessagePillar, details::String)
    # Draft a release using the PRComms engine
    body = "FOR IMMEDIATE RELEASE

$(details)

Our stance:
"
    for pt in pillar.key_points
        body *= "• $pt
"
    end
    
    return draft_release(:union_stmt, "Union Statement: $(pillar.theme)", body)
end

"""
    generate_member_alert(pillar, channel)
Adapts a message pillar for rapid member communication (SMS/Social).
"""
function generate_member_alert(pillar::MessagePillar, channel::Symbol)
    return generate_variant(pillar, :employees, channel, tone=:bold)
end

end # module
