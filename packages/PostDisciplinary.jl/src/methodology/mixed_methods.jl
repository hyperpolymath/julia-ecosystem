# SPDX-License-Identifier: MPL-2.0
module MixedMethods

using ..PostDisciplinary

export ResearchDesign, QuantDesign, QualDesign, MixedDesign, run_design

abstract type ResearchDesign end

struct QuantDesign <: ResearchDesign end # Focus: Data, Stats, Proofs
struct QualDesign  <: ResearchDesign end # Focus: Interpretation, Context
struct MixedDesign <: ResearchDesign end # Focus: Triangulation of both

"""
    run_design(design, project)
Configures the ecosystem to prioritize specific modules based on the research design.
"""
function run_design(::QuantDesign, p)
    println("Running QUANTITATIVE design: Prioritizing SMTLib, ZeroProb, and Cliometrics... 📊")
end

function run_design(::QualDesign, p)
    println("Running QUALITATIVE design: Prioritizing Hermeneutics and InvestigativeJournalist... 🔍")
end

function run_design(::MixedDesign, p)
    println("Running MIXED-METHODS design: Executing Triangulation loops... 🛰️")
end

end # module
