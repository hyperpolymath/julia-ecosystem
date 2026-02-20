# SPDX-License-Identifier: PMPL-1.0-or-later
module PostDisciplinary

using Graphs
using MetaGraphsNext
using DataFrames
using UUIDs
using JSON3

include("consensus/raft.jl")
include("meta_analysis.jl")
include("synthesis/diffing.jl")
include("memetics.jl")
include("methodology.jl")
include("storage/verisim.jl")
include("storage/lithoglyph.jl")
include("synthesis/triangulation.jl")
include("synthesis/evolutionary.jl")
include("synthesis/experts.jl")
include("synthesis/hermeneutics.jl")
include("methodology/mixed_methods.jl")
include("impact/knowledge_transfer.jl")
include("synthesis/templates.jl")

using .RaftConsensus
using .MetaAnalysis
using .KnowledgeDiffing
using .Memetics
using .Methodology
using .VeriSimBridge
using .LithoglyphBridge
using .Triangulation
using .EvolutionarySynthesis
using .ExpertSynthesis
using .Hermeneutics
using .MixedMethods
using .KnowledgeTransfer
using .ResearchTemplates

export ResearchProject, LinkedEntity, @link, generate_synthesis, add_link!
export RaftNode, RaftState, request_vote, append_entries
export EffectSize, aggregate_effects, heterogeneity_q
export diff_knowledge_graphs, KnowledgeDiff
export Meme, Replicator, mutate, calculate_fitness
export ResearchStrategy, MultiDisciplinary, InterDisciplinary, TransDisciplinary, AntiDisciplinary, execute_strategy
export VeriSimClient, store_hexad, vql_query
export LithoglyphClient, store_glyph, find_symbol
export triangulate_findings, CorrelationReport
export evolutionary_link, AxiologicalAlignment
export ExpertRouter, GraphOfThought, route_to_discipline, evolve_thought!
export HermeneuticCircle, interpret_part, synthesize_context
export ResearchDesign, QuantDesign, QualDesign, MixedDesign, run_design
export ImpactRecord, log_utilisation, measure_influence
export ResearchTemplate, WickedProblem, StructuralForensics, ActionResearch, scaffold_project

"""
    LinkedEntity
A universal entity that can represent a Claim, Proof, Model, or Event 
from any .jl library in the ecosystem.
"""
struct LinkedEntity
    id::UUID
    source_library::Symbol
    original_id::Symbol
    kind::Symbol
    metadata::Dict{Symbol, Any}
end

"""
    ResearchProject
A meta-container for cross-disciplinary research.
"""
struct ResearchProject
    id::Symbol
    name::String
    graph::MetaGraph
end

function ResearchProject(name::String)
    g = MetaGraph(
        SimpleGraph(),
        label_type=UUID,
        vertex_data_type=LinkedEntity,
        edge_data_type=Dict{Symbol, Any},
        graph_data=name
    )
    return ResearchProject(gensym("project"), name, g)
end

"""
    @link project begin ... end
Syntactic sugar for connecting outputs from multiple libraries into a project graph.
"""
macro link(project, block)
    return quote
        println("Linking multidisciplinary nodes into $( $(esc(project)).name )...")
        $(esc(block))
    end
end

"""
    add_link!(project, entity1, entity2, relationship)
Explicitly links two multidisciplinary entities.
"""
function add_link!(p::ResearchProject, e1::LinkedEntity, e2::LinkedEntity, rel::Symbol)
    p.graph[e1.id] = e1
    p.graph[e2.id] = e2
    p.graph[e1.id, e2.id] = Dict(:relationship => rel)
    return "Link established: $(e1.kind) --($rel)--> $(e2.kind)"
end

"""
    generate_synthesis(project)
Produces a unified report summarizing the connections between disciplines.
"""
function generate_synthesis(p::ResearchProject)
    println("Synthesizing Knowledge Graph: $(p.name) 🧠")
    
    df = DataFrame(
        Origin = [v.source_library for v in values(p.graph.vertex_data)],
        Kind = [v.kind for v in values(p.graph.vertex_data)],
        Metadata = [v.metadata for v in values(p.graph.vertex_data)]
    )
    
    return (
        report_title = "Synthesis: $(p.name)",
        entities = df,
        graph_density = density(p.graph.graph)
    )
end

end # module
