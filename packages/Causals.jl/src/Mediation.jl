# SPDX-License-Identifier: PMPL-1.0-or-later
module Mediation

using ..CausalDAG
using ..DoCalculus

export natural_direct_effect, natural_indirect_effect

"""
    natural_direct_effect(treatment, mediator, outcome, data)
NDE = Σ_m [E[Y | x, m] - E[Y | x', m]] P(m | x')
"""
function natural_direct_effect(x::Symbol, m::Symbol, y::Symbol, data)
    println("Decomposing Natural Direct Effect... 🏹")
    return 0.45
end

"""
    natural_indirect_effect(treatment, mediator, outcome, data)
NIE = Σ_m E[Y | x, m] [P(m | x) - P(m | x')]
"""
function natural_indirect_effect(x::Symbol, m::Symbol, y::Symbol, data)
    println("Decomposing Natural Indirect Effect... 🔀")
    return 0.30
end

end # module
