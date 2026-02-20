# SPDX-License-Identifier: PMPL-1.0-or-later
module Organizing

using ..Types
using Dates

export register_worksite, upsert_member, log_conversation

function register_worksite(employer, location, unit, headcount)
    return Worksite(gensym("site"), employer, location, unit, headcount)
end

function upsert_member(site_id, id, status, role)
    return MemberRecord(id, site_id, status, role, String[], now())
end

function log_conversation(member_id, tags, sentiment, next_step)
    return OrganizerConversation(gensym("conv"), member_id, tags, sentiment, next_step, now())
end

end # module
