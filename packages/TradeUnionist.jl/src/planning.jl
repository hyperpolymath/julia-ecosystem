# SPDX-License-Identifier: PMPL-1.0-or-later
module Planning

export UnionActivity, StrategicGoal, TacticalObjective, OperationalTask

abstract type PlanningLevel end

struct StrategicGoal <: PlanningLevel end   # e.g. "Win the contract", "50% density"
struct TacticalObjective <: PlanningLevel end # e.g. "Identify 10 new leaders", "Successful rally"
struct OperationalTask <: PlanningLevel end   # e.g. "Hand out 100 flyers", "1-on-1 meeting"

struct UnionActivity
    level::PlanningLevel
    function_area::Symbol # :Organizing, :Bargaining, :Grievance, :Political, :Admin
    description::String
    owner::String
    status::Symbol
end

end # module
