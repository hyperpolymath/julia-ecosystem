# SPDX-License-Identifier: MPL-2.0
module Exnovation

include("JSI.jl")
# (Other existing includes would be here)

using .JSI

# Re-export core types
export JustSustainabilityIndex, evaluate_jsi

"""
    ExnovationItem
A legacy practice, product, or routine being considered for phase-out.
"""
struct ExnovationItem
    id::Symbol
    description::String
    context::String
end

# (Simplified re-scaffold of the rest of the module logic for brevity)

end # module
