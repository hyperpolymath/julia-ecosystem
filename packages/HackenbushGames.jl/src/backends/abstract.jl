# SPDX-License-Identifier: PMPL-1.0-or-later
# Copyright (c) 2026 Jonathan D.A. Jewell (hyperpolymath) <j.d.a.jewell@open.ac.uk>
#
# HackenbushGames.jl Backend Abstraction
# Uses AcceleratorGate.jl for shared coprocessor dispatch infrastructure.

using AcceleratorGate
using AcceleratorGate: generate_self_healing_hooks

# ============================================================================
# Domain-Specific Operation Hooks -- HackenbushGames
# ============================================================================

# Julia fallback implementations
backend_game_tree_eval(::JuliaBackend, args...) = nothing
backend_grundy_number(::JuliaBackend, args...) = nothing
backend_minimax_search(::JuliaBackend, args...) = nothing
backend_position_hash(::JuliaBackend, args...) = nothing
backend_move_gen(::JuliaBackend, args...) = nothing

# Coprocessor extension hooks (concrete extensions overload these)
function backend_coprocessor_game_tree_eval end
function backend_coprocessor_grundy_number end
function backend_coprocessor_minimax_search end
function backend_coprocessor_position_hash end
function backend_coprocessor_move_gen end

# Self-healing fallback generation for coprocessor hooks
generate_self_healing_hooks(@__MODULE__, [
    (:backend_coprocessor_game_tree_eval, :backend_game_tree_eval),
    (:backend_coprocessor_grundy_number, :backend_grundy_number),
    (:backend_coprocessor_minimax_search, :backend_minimax_search),
    (:backend_coprocessor_position_hash, :backend_position_hash),
    (:backend_coprocessor_move_gen, :backend_move_gen),
])

# Dispatch: CoprocessorBackend -> coprocessor hook
backend_game_tree_eval(b::CoprocessorBackend, args...) = backend_coprocessor_game_tree_eval(b, args...)
backend_grundy_number(b::CoprocessorBackend, args...) = backend_coprocessor_grundy_number(b, args...)
backend_minimax_search(b::CoprocessorBackend, args...) = backend_coprocessor_minimax_search(b, args...)
backend_position_hash(b::CoprocessorBackend, args...) = backend_coprocessor_position_hash(b, args...)
backend_move_gen(b::CoprocessorBackend, args...) = backend_coprocessor_move_gen(b, args...)

# Dispatch: GPUBackend -> Julia fallback (extensions override)
backend_game_tree_eval(b::GPUBackend, args...) = backend_game_tree_eval(JuliaBackend(), args...)
backend_grundy_number(b::GPUBackend, args...) = backend_grundy_number(JuliaBackend(), args...)
backend_minimax_search(b::GPUBackend, args...) = backend_minimax_search(JuliaBackend(), args...)
backend_position_hash(b::GPUBackend, args...) = backend_position_hash(JuliaBackend(), args...)
backend_move_gen(b::GPUBackend, args...) = backend_move_gen(JuliaBackend(), args...)
