# SPDX-License-Identifier: PMPL-1.0-or-later
module Automation

export MagicRule, when, check_rules!

struct MagicRule
    event_type::Symbol
    condition::Function
    action::Function
end

const ACTIVE_RULES = MagicRule[]

"""
    when(event, action)

Sets up a Magic Rule! 
Example: when(:file_saved, () -> println("Great work!"))
"""
function when(event_type::Symbol, action::Function; condition=(args...)->true)
    push!(ACTIVE_RULES, MagicRule(event_type, condition, action))
    return "Magic Rule added! ✨"
end

"""
    trigger_event(type, args...)

Checks all rules and runs the ones that match.
"""
function trigger_event(type::Symbol, args...)
    for rule in ACTIVE_RULES
        if rule.event_type == type && rule.condition(args...)
            rule.action(args...)
        end
    end
end

end # module
