# SPDX-License-Identifier: PMPL-1.0-or-later
module Grievances

using ..Types
using Dates

export open_grievance, update_grievance_status

function open_grievance(member_id, issue_desc, due_in_days=14)
    id = gensym("grievance")
    due = now() + Day(due_in_days)
    return GrievanceCase(id, member_id, now(), :intake, [issue_desc], due)
end

function update_grievance_status(case::GrievanceCase, new_status::Symbol)
    case.status = new_status
    return "Grievance $(case.id) moved to $new_status"
end

end # module
