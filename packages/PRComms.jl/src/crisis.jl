# SPDX-License-Identifier: PMPL-1.0-or-later
module Crisis

using ..Types

export activate_crisis_mode, issue_holding_statement

function activate_crisis_mode(playbook::CrisisPlaybook)
    println("🚨 CRISIS MODE ACTIVATED: $(playbook.incident_type) (Severity: $(playbook.severity))")
    println("ESCALATION TREE:")
    for contact in playbook.escalation_tree
        println("  -> $contact")
    end
    return "Operations shifted to Incident Management. 🛡️"
end

function issue_holding_statement(playbook::CrisisPlaybook, variant_idx::Int)
    statement = playbook.holding_statements[variant_idx]
    return "STMT: "$statement""
end

end # module
