# SPDX-License-Identifier: PMPL-1.0-or-later
module ViableSystems

include("VSM.jl")
include("SSM.jl")
include("optimization.jl")
include("boundary.jl")

using .VSM
using .SSM
using .SystemOptimization
using .BoundaryObjects

# Re-export everything
export System1, System2, System3, System4, System5, ViableOrganization
export algedonic_alert, check_variety
export CATWOE, RootDefinition, analyze_problem
export simulated_annealing_optimize, genetic_algorithm_optimize
export SystemBoundaryObject, SharedModel, create_boundary_object

end # module
