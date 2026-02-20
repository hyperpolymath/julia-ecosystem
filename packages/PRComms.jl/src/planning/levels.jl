# SPDX-License-Identifier: PMPL-1.0-or-later
module PlanningLevels

export PRActivity, BusinessPR, StrategicPR, TacticalPR, OperationalPR

abstract type PRLevel end

struct BusinessPR <: PRLevel end    # Corporate alignment & value
struct StrategicPR <: PRLevel end   # Long-term reputation & goals
struct TacticalPR <: PRLevel end    # Campaign-specific execution
struct OperationalPR <: PRLevel end # Day-to-day tasks (posting, pitching)

struct PRActivity
    level::PRLevel
    function_area::Symbol # :HR, :Finance, :IT, :RD, :Personal
    objective::String
    status::Symbol
end

end # module
