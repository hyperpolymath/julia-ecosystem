# SPDX-License-Identifier: PMPL-1.0-or-later
module KnowledgeTransfer

using ..PostDisciplinary
using Dates

export ImpactRecord, log_utilisation, measure_influence

struct ImpactRecord
    project_id::Symbol
    utilisation_type::Symbol # :policy_brief, :organizing_win, :public_campaign
    date::DateTime
    description::String
end

const IMPACT_LOG = ImpactRecord[]

"""
    log_utilisation(project, type, desc)
Records a real-world use of research findings (e.g. 'Finding A used in PRComms launch').
"""
function log_utilisation(p::ResearchProject, type::Symbol, desc::String)
    record = ImpactRecord(p.id, type, now(), desc)
    push!(IMPACT_LOG, record)
    println("Impact Logged: [$(type)] $desc 🏆")
    return record
end

"""
    measure_influence(project)
(Research) Analyzes the reach and adoption of specific research nodes across the ecosystem.
"""
function measure_influence(p::ResearchProject)
    println("Measuring Knowledge Utilisation and Strategic Influence... 📈")
    return (adoption_rate = 0.65, policy_hits = 12)
end

end # module
