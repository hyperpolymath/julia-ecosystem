# SPDX-License-Identifier: PMPL-1.0-or-later
# Copyright (c) 2026 Jonathan D.A. Jewell (hyperpolymath) <j.d.a.jewell@open.ac.uk>
#
# Cladistics.jl Backend Abstraction
# Uses AcceleratorGate.jl for shared coprocessor dispatch infrastructure.

using AcceleratorGate
using AcceleratorGate: generate_self_healing_hooks

# ============================================================================
# Domain-Specific Operation Hooks -- Cladistics
# ============================================================================

# Julia fallback implementations
backend_distance_matrix(::JuliaBackend, args...) = nothing
backend_neighbor_join(::JuliaBackend, args...) = nothing
backend_parsimony_score(::JuliaBackend, args...) = nothing
backend_bootstrap_replicate(::JuliaBackend, args...) = nothing
backend_tree_search(::JuliaBackend, args...) = nothing

# Coprocessor extension hooks (concrete extensions overload these)
function backend_coprocessor_distance_matrix end
function backend_coprocessor_neighbor_join end
function backend_coprocessor_parsimony_score end
function backend_coprocessor_bootstrap_replicate end
function backend_coprocessor_tree_search end

# Self-healing fallback generation for coprocessor hooks
generate_self_healing_hooks(@__MODULE__, [
    (:backend_coprocessor_distance_matrix, :backend_distance_matrix),
    (:backend_coprocessor_neighbor_join, :backend_neighbor_join),
    (:backend_coprocessor_parsimony_score, :backend_parsimony_score),
    (:backend_coprocessor_bootstrap_replicate, :backend_bootstrap_replicate),
    (:backend_coprocessor_tree_search, :backend_tree_search),
])

# Dispatch: CoprocessorBackend -> coprocessor hook
backend_distance_matrix(b::CoprocessorBackend, args...) = backend_coprocessor_distance_matrix(b, args...)
backend_neighbor_join(b::CoprocessorBackend, args...) = backend_coprocessor_neighbor_join(b, args...)
backend_parsimony_score(b::CoprocessorBackend, args...) = backend_coprocessor_parsimony_score(b, args...)
backend_bootstrap_replicate(b::CoprocessorBackend, args...) = backend_coprocessor_bootstrap_replicate(b, args...)
backend_tree_search(b::CoprocessorBackend, args...) = backend_coprocessor_tree_search(b, args...)

# Dispatch: GPUBackend -> Julia fallback (extensions override)
backend_distance_matrix(b::GPUBackend, args...) = backend_distance_matrix(JuliaBackend(), args...)
backend_neighbor_join(b::GPUBackend, args...) = backend_neighbor_join(JuliaBackend(), args...)
backend_parsimony_score(b::GPUBackend, args...) = backend_parsimony_score(JuliaBackend(), args...)
backend_bootstrap_replicate(b::GPUBackend, args...) = backend_bootstrap_replicate(JuliaBackend(), args...)
backend_tree_search(b::GPUBackend, args...) = backend_tree_search(JuliaBackend(), args...)
