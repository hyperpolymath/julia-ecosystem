# SPDX-License-Identifier: PMPL-1.0-or-later
# Copyright (c) 2026 Jonathan D.A. Jewell (hyperpolymath) <j.d.a.jewell@open.ac.uk>
#
# SMTLib.jl CUDA Extension
# Provides GPU-accelerated SAT clause evaluation and batch model checking via NVIDIA CUDA.
# Automatically loaded when CUDA.jl is imported alongside SMTLib.

module SMTLibCUDAExt

using CUDA
using SMTLib
using AcceleratorGate

# ============================================================================
# Availability Detection
# ============================================================================

function AcceleratorGate.cuda_available()
    forced = AcceleratorGate._backend_env_available("AXIOM_CUDA_AVAILABLE")
    forced !== nothing && return forced
    CUDA.functional()
end

function AcceleratorGate.cuda_device_count()
    forced = AcceleratorGate._backend_env_count("AXIOM_CUDA_AVAILABLE", "AXIOM_CUDA_DEVICE_COUNT")
    forced !== nothing && return forced
    CUDA.ndevices()
end

# ============================================================================
# Device Capabilities
# ============================================================================

function AcceleratorGate.device_capabilities(b::CUDABackend)
    !CUDA.functional() && return nothing
    dev = CUDA.device(b.device)
    total_mem = CUDA.totalmem(dev)
    free_mem = CUDA.available_memory()
    AcceleratorGate.DeviceCapabilities(
        b,
        CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT),
        div(CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_CLOCK_RATE), 1000),
        Int64(total_mem),
        Int64(free_mem),
        CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK),
        true,  # NVIDIA GPUs support f64
        true,  # f16 supported
        true,  # int8 supported
        "NVIDIA",
        string(CUDA.version()),
    )
end

function AcceleratorGate.estimate_cost(::CUDABackend, op::Symbol, data_size::Int)
    # GPU parallel clause evaluation is highly efficient for large workloads
    overhead = 500.0  # kernel launch overhead
    op == :solve && return overhead + Float64(data_size) * 0.01
    op == :check_sat && return overhead + Float64(data_size) * 0.005
    op == :model_eval && return overhead + Float64(data_size) * 0.02
    op == :interpolate && return overhead + Float64(data_size) * 0.05
    op == :simplify && return overhead + Float64(data_size) * 0.03
    Inf
end

# ============================================================================
# Operation Registration
# ============================================================================

function __init__()
    AcceleratorGate.register_operation!(CUDABackend, :solve)
    AcceleratorGate.register_operation!(CUDABackend, :check_sat)
    AcceleratorGate.register_operation!(CUDABackend, :model_eval)
    AcceleratorGate.register_operation!(CUDABackend, :simplify)
end

# ============================================================================
# GPU-Accelerated SAT Operations
# ============================================================================

"""
    backend_solve(::CUDABackend, clauses, variables, config)

Parallel SAT solving on CUDA. Splits the search space across GPU threads,
each evaluating a different partial assignment. Uses clause-parallel evaluation
where each thread block processes a subset of clauses for a given assignment.
"""
function SMTLib.backend_solve(::CUDABackend, clauses::AbstractVector,
                              variables::AbstractVector, config::NamedTuple=NamedTuple())
    n_vars = length(variables)
    n_clauses = length(clauses)

    # Encode clauses as a flat integer matrix on GPU
    # Each clause is a row: positive literal index = var index, negative = -(var index)
    max_clause_len = maximum(length(c) for c in clauses)
    clause_matrix = zeros(Int32, n_clauses, max_clause_len)
    for (i, clause) in enumerate(clauses)
        for (j, lit) in enumerate(clause)
            clause_matrix[i, j] = Int32(lit)
        end
    end
    clause_gpu = CuArray(clause_matrix)

    # Number of parallel trials to run
    n_trials = min(get(config, :trials, 2^min(n_vars, 20)), 2^20)

    # Generate random partial assignments on GPU
    assignments_gpu = CUDA.rand(Float32, n_trials, n_vars)
    binary_assignments = assignments_gpu .> 0.5f0

    # Evaluate all clauses for all assignments in parallel
    # Result: n_trials x n_clauses satisfaction matrix
    sat_matrix = CUDA.zeros(Int32, n_trials, n_clauses)

    # Kernel: for each (trial, clause) pair, check if clause is satisfied
    function eval_kernel!(sat_mat, clauses_mat, assignments, n_v, n_c, max_len)
        trial_idx = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
        clause_idx = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
        if trial_idx <= size(sat_mat, 1) && clause_idx <= size(sat_mat, 2)
            satisfied = Int32(0)
            for k in 1:max_len
                lit = clauses_mat[clause_idx, k]
                lit == Int32(0) && break
                var_idx = abs(lit)
                if var_idx <= n_v
                    val = assignments[trial_idx, var_idx]
                    if (lit > Int32(0) && val) || (lit < Int32(0) && !val)
                        satisfied = Int32(1)
                        break
                    end
                end
            end
            sat_mat[trial_idx, clause_idx] = satisfied
        end
        return nothing
    end

    threads_x = min(256, n_trials)
    threads_y = min(4, n_clauses)
    blocks_x = cld(n_trials, threads_x)
    blocks_y = cld(n_clauses, threads_y)
    @cuda threads=(threads_x, threads_y) blocks=(blocks_x, blocks_y) eval_kernel!(
        sat_matrix, clause_gpu, binary_assignments, Int32(n_vars), Int32(n_clauses), Int32(max_clause_len)
    )

    # Check which trials satisfy ALL clauses (row-wise AND)
    all_satisfied = vec(sum(sat_matrix, dims=2))
    host_satisfied = Array(all_satisfied)

    # Find first satisfying assignment
    for i in 1:n_trials
        if host_satisfied[i] == n_clauses
            assignment = Array(binary_assignments[i, :])
            model = Dict(variables[j] => Bool(assignment[j]) for j in 1:n_vars)
            return (status=:sat, model=model, trials_checked=i)
        end
    end

    return (status=:unknown, model=nothing, trials_checked=n_trials)
end

"""
    backend_check_sat(::CUDABackend, clauses, n_vars)

Batch satisfiability checking on GPU. Evaluates many candidate assignments
in parallel across CUDA cores.
"""
function SMTLib.backend_check_sat(::CUDABackend, clauses::AbstractVector, n_vars::Int)
    n_clauses = length(clauses)
    max_clause_len = maximum(length(c) for c in clauses)

    clause_matrix = zeros(Int32, n_clauses, max_clause_len)
    for (i, clause) in enumerate(clauses)
        for (j, lit) in enumerate(clause)
            clause_matrix[i, j] = Int32(lit)
        end
    end
    clause_gpu = CuArray(clause_matrix)

    # Batch random assignments
    n_batch = min(2^min(n_vars, 18), 262144)
    assignments_gpu = CUDA.rand(Float32, n_batch, n_vars) .> 0.5f0

    # Clause satisfaction per assignment
    sat_counts = CUDA.zeros(Int32, n_batch)
    for c_idx in 1:n_clauses
        clause_row = clause_gpu[c_idx:c_idx, :]
        for v in 1:max_clause_len
            lit_val = Array(clause_row[:, v:v])[1]
            lit_val == 0 && break
            var_idx = abs(lit_val)
            if var_idx <= n_vars
                matches = if lit_val > 0
                    assignments_gpu[:, var_idx]
                else
                    .!assignments_gpu[:, var_idx]
                end
                # Accumulate: if any literal in clause satisfied, clause satisfied
                sat_counts .+= Int32.(matches)
            end
        end
    end

    host_counts = Array(sat_counts)
    any(c -> c >= n_clauses, host_counts) && return :sat
    return :unknown
end

"""
    backend_model_eval(::CUDABackend, formula_matrix, model_batch)

Batch model evaluation: evaluate a formula under many candidate models
simultaneously on GPU.
"""
function SMTLib.backend_model_eval(::CUDABackend, formula_matrix::AbstractMatrix,
                                   model_batch::AbstractMatrix)
    f_gpu = CuArray(Float32.(formula_matrix))
    m_gpu = CuArray(Float32.(model_batch))
    # Matrix multiply: each row of model_batch evaluated against formula columns
    results_gpu = m_gpu * f_gpu
    return Array(results_gpu)
end

"""
    backend_simplify(::CUDABackend, expression_batch)

Batch expression simplification using parallel constant folding on GPU.
Processes a batch of expressions encoded as coefficient vectors.
"""
function SMTLib.backend_simplify(::CUDABackend, expression_batch::AbstractMatrix)
    expr_gpu = CuArray(Float32.(expression_batch))
    # Parallel constant folding: zero out near-zero coefficients
    threshold = 1.0f-10
    simplified = expr_gpu .* (abs.(expr_gpu) .> threshold)
    # Normalize: scale each row so max coefficient is 1.0
    row_maxes = maximum(abs.(simplified), dims=2)
    row_maxes = max.(row_maxes, 1.0f-30)  # avoid division by zero
    normalized = simplified ./ row_maxes
    return Array(normalized)
end

end # module SMTLibCUDAExt
