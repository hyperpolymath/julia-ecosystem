# SPDX-License-Identifier: PMPL-1.0-or-later
module ResearchTemplates

using ..PostDisciplinary
using ..Methodology
using ..MixedMethods

export ResearchTemplate, WickedProblem, StructuralForensics, ActionResearch, scaffold_project

abstract type ResearchTemplate end

struct WickedProblem <: ResearchTemplate end # Goal: Mapping complexity and values
struct StructuralForensics <: ResearchTemplate end # Goal: Historical/Causal modeling
struct ActionResearch <: ResearchTemplate end # Goal: Social change and organizing

"""
    scaffold_project(template, name)
Pre-configures a ResearchProject with the correct strategy and design for the goal.
"""
function scaffold_project(::WickedProblem, name)
    p = ResearchProject(name)
    execute_strategy(TransDisciplinary(), p)
    run_design(MixedDesign(), p)
    println("Project '$name' scaffolded for WICKED PROBLEM analysis. 🌪️")
    return p
end

function scaffold_project(::StructuralForensics, name)
    p = ResearchProject(name)
    execute_strategy(MultiDisciplinary(), p)
    run_design(QuantDesign(), p)
    println("Project '$name' scaffolded for STRUCTURAL FORENSICS. 🏛️🔎")
    return p
end

function scaffold_project(::ActionResearch, name)
    p = ResearchProject(name)
    execute_strategy(InterDisciplinary(), p)
    run_design(QualDesign(), p)
    println("Project '$name' scaffolded for ACTION RESEARCH. ✊🛠️")
    return p
end

end # module
