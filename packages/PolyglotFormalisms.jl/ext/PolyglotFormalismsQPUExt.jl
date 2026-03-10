# SPDX-License-Identifier: PMPL-1.0-or-later
# Copyright (c) 2026 Jonathan D.A. Jewell (hyperpolymath) <j.d.a.jewell@open.ac.uk>
#
# PolyglotFormalisms.jl QPU Extension
# Quantum Processing Unit acceleration via QPUInterface.jl.
# QPUs enable quantum-accelerated search (Grover) and optimisation (QAOA)
# for NP-hard proof obligations and constraint satisfaction problems.

module PolyglotFormalismsQPUExt

using QPUInterface
using PolyglotFormalisms
using AcceleratorGate

function AcceleratorGate.qpu_available()
    forced = AcceleratorGate._backend_env_available("AXIOM_QPU_AVAILABLE")
    forced !== nothing && return forced
    QPUInterface.functional()
end

function AcceleratorGate.qpu_device_count()
    forced = AcceleratorGate._backend_env_count("AXIOM_QPU_AVAILABLE", "AXIOM_QPU_DEVICE_COUNT")
    forced !== nothing && return forced
    QPUInterface.ndevices()
end

function __init__()
    AcceleratorGate.register_operation!(QPUBackend, :fold_parallel)
    AcceleratorGate.register_operation!(QPUBackend, :symbolic_eval)
end

"""
Quantum-accelerated parallel fold. Encodes the accumulation problem as
a quantum circuit and uses amplitude estimation for the result, providing
quadratic speedup for suitable reduction operators.
"""
function PolyglotFormalisms.backend_coprocessor_fold_parallel(::QPUBackend, f, init, collections::AbstractVector)
    results = similar(collections)
    for (i, coll) in enumerate(collections)
        if QPUInterface.can_encode(f, coll)
            circuit = QPUInterface.encode_fold(f, init, coll)
            results[i] = QPUInterface.execute(circuit)
        else
            results[i] = foldl(f, coll; init=init)
        end
    end
    results
end

"""
Quantum-accelerated symbolic evaluation. Uses Grover search to find
satisfying assignments in constraint sub-expressions, and QAOA for
optimisation nodes in the expression tree.
"""
function PolyglotFormalisms.backend_coprocessor_symbolic_eval(::QPUBackend, expr, env::Dict)
    QPUInterface.symbolic_eval(expr, env)
end

end # module PolyglotFormalismsQPUExt
