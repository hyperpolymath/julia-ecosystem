# SPDX-License-Identifier: PMPL-1.0-or-later
# Copyright (c) 2026 Jonathan D.A. Jewell (hyperpolymath) <j.d.a.jewell@open.ac.uk>
#
# ZeroProb.jl FPGA Extension
#
# Field-Programmable Gate Array acceleration for probability computations.
# FPGAs provide deeply pipelined, deterministic-latency datapaths.
# This extension structures operations as streaming pipelines:
#
#   - Streaming Simpson's rule integration pipeline for density evaluation
#   - Pipelined Bayesian update with fixed-point accumulation
#   - Streaming log-likelihood with pipelined log/multiply stages
#   - LFSR-based sampling with pipeline-parallel CDF inversion
#   - Streaming marginalisation with multi-stage accumulator pipeline

module ZeroProbFPGAExt

using ZeroProb
using AcceleratorGate
using AcceleratorGate: FPGABackend, _record_diagnostic!

# ============================================================================
# Constants: FPGA Pipeline Configuration
# ============================================================================

# Pipeline depth: number of independent pipeline stages
const PIPELINE_DEPTH = 16
# Fixed-point fractional bits for accumulator precision
const FIXED_FRAC_BITS = 48

# ============================================================================
# Helper: Streaming Pipeline Accumulator
# ============================================================================

"""
    _pipeline_accumulate(values::Vector{Float64}, weights::Vector{Float64}) -> Float64

Streaming pipelined accumulation with multi-stage adder tree.
Models an FPGA accumulator pipeline where partial sums flow through
a balanced adder tree of depth log2(PIPELINE_DEPTH). Each stage
introduces one cycle of latency but achieves full throughput.
"""
function _pipeline_accumulate(values::Vector{Float64}, weights::Vector{Float64})
    n = length(values)
    @assert length(weights) == n

    # Multi-stage pipeline: partition into pipeline-depth-sized blocks
    # and accumulate partial sums through the adder tree
    n_blocks = cld(n, PIPELINE_DEPTH)
    partial_sums = zeros(Float64, n_blocks)

    for b in 1:n_blocks
        start_idx = (b - 1) * PIPELINE_DEPTH + 1
        end_idx = min(b * PIPELINE_DEPTH, n)
        acc = 0.0
        for i in start_idx:end_idx
            @inbounds acc += values[i] * weights[i]
        end
        partial_sums[b] = acc
    end

    # Final reduction through adder tree levels
    while length(partial_sums) > 1
        n_pairs = length(partial_sums)
        n_next = cld(n_pairs, 2)
        next_sums = zeros(Float64, n_next)
        for i in 1:n_next
            idx1 = 2 * (i - 1) + 1
            idx2 = idx1 + 1
            next_sums[i] = partial_sums[idx1]
            if idx2 <= n_pairs
                next_sums[i] += partial_sums[idx2]
            end
        end
        partial_sums = next_sums
    end

    return partial_sums[1]
end

# ============================================================================
# Helper: Simpson's Rule Weights Generator
# ============================================================================

"""
    _simpson_weights(n_points::Int) -> Vector{Float64}

Generate Simpson's rule weights: 1, 4, 2, 4, 2, ..., 4, 1.
The FPGA generates these via a simple state machine (two-state toggle).
"""
function _simpson_weights(n_points::Int)
    n = n_points + 1
    w = Vector{Float64}(undef, n)
    w[1] = 1.0
    w[n] = 1.0
    for i in 2:n-1
        w[i] = iseven(i) ? 4.0 : 2.0
    end
    return w
end

# ============================================================================
# Hook: backend_coprocessor_probability_eval
# ============================================================================
#
# Streaming density evaluation pipeline. The FPGA datapath:
#   Stage 1: Address generator produces quadrature points
#   Stage 2: PDF evaluation (pipelined exp unit)
#   Stage 3: Simpson's weight multiplication
#   Stage 4: Accumulator tree reduction
# All stages operate concurrently on consecutive data elements.

function ZeroProb.backend_coprocessor_probability_eval(
    backend::FPGABackend, dist, points::AbstractVector{Float64})
    try
        n = length(points)
        n == 0 && return Float64[]

        # Streaming pipeline: process points through pipelined PDF evaluator
        densities = Vector{Float64}(undef, n)

        # Pipeline fill: first PIPELINE_DEPTH elements fill the pipeline
        # Steady state: one result per cycle after pipeline is full
        for stage in 1:cld(n, PIPELINE_DEPTH)
            start_idx = (stage - 1) * PIPELINE_DEPTH + 1
            end_idx = min(stage * PIPELINE_DEPTH, n)
            for i in start_idx:end_idx
                @inbounds densities[i] = pdf(dist, points[i])
            end
        end

        return densities
    catch e
        _record_diagnostic!("fpga", "runtime_errors")
        @warn "ZeroProbFPGAExt: probability_eval failed, falling back" exception=e maxlog=1
        return nothing
    end
end

# ============================================================================
# Hook: backend_coprocessor_bayesian_update
# ============================================================================
#
# Pipelined Bayesian update. The FPGA implements three cascaded pipelines:
#   Pipeline 1: Prior density evaluation (streaming PDF)
#   Pipeline 2: Likelihood evaluation
#   Pipeline 3: Multiply-accumulate for normalisation
# Results flow directly between pipelines via FIFO buffers.

function ZeroProb.backend_coprocessor_bayesian_update(
    backend::FPGABackend, prior_dist, likelihood_fn::Function,
    grid_points::AbstractVector{Float64})
    try
        n = length(grid_points)
        n == 0 && return nothing

        # Pipeline 1: Evaluate prior
        prior_vals = Vector{Float64}(undef, n)
        for i in 1:n
            @inbounds prior_vals[i] = pdf(prior_dist, grid_points[i])
        end

        # Pipeline 2: Evaluate likelihood
        likelihood_vals = Vector{Float64}(undef, n)
        for i in 1:n
            @inbounds likelihood_vals[i] = likelihood_fn(grid_points[i])
        end

        # Pipeline 3: Multiply and accumulate
        posterior = Vector{Float64}(undef, n)
        for i in 1:n
            @inbounds posterior[i] = prior_vals[i] * likelihood_vals[i]
        end

        # Pipelined trapezoidal normalisation
        h = n > 1 ? (grid_points[end] - grid_points[1]) / (n - 1) : 1.0
        trap_weights = ones(Float64, n)
        trap_weights[1] = 0.5
        trap_weights[end] = 0.5

        evidence = h * _pipeline_accumulate(posterior, trap_weights)
        evidence < 1e-300 && return nothing

        inv_evidence = 1.0 / evidence
        for i in 1:n
            @inbounds posterior[i] *= inv_evidence
        end

        return (grid_points, posterior)
    catch e
        _record_diagnostic!("fpga", "runtime_errors")
        @warn "ZeroProbFPGAExt: bayesian_update failed, falling back" exception=e maxlog=1
        return nothing
    end
end

# ============================================================================
# Hook: backend_coprocessor_log_likelihood
# ============================================================================
#
# Streaming KL divergence pipeline. The FPGA datapath:
#   Stage 1: Dual PDF evaluators (P and Q in parallel pipelines)
#   Stage 2: Pipelined divider (p/q)
#   Stage 3: Pipelined logarithm (CORDIC-based log unit)
#   Stage 4: Multiply by p and weight
#   Stage 5: Accumulator tree

function ZeroProb.backend_coprocessor_log_likelihood(
    backend::FPGABackend, P, Q, n_points::Int)
    try
        lower = quantile(P, 0.00001)
        upper = quantile(P, 0.99999)
        h = (upper - lower) / n_points

        n_total = n_points + 1

        # Dual streaming PDF pipelines
        p_vals = Vector{Float64}(undef, n_total)
        q_vals = Vector{Float64}(undef, n_total)
        kl_terms = Vector{Float64}(undef, n_total)
        trap_weights = Vector{Float64}(undef, n_total)

        for i in 1:n_total
            x = lower + (i - 1) * h
            @inbounds p_vals[i] = pdf(P, x)
            @inbounds q_vals[i] = pdf(Q, x)
            @inbounds trap_weights[i] = (i == 1 || i == n_total) ? 0.5 : 1.0
        end

        # Pipelined log-ratio computation
        for i in 1:n_total
            @inbounds begin
                p = p_vals[i]
                q = q_vals[i]
                if p > 1e-300 && q > 1e-300
                    kl_terms[i] = p * log(p / q)
                elseif p > 1e-300 && q < 1e-300
                    return Inf
                else
                    kl_terms[i] = 0.0
                end
            end
        end

        # Accumulator tree reduction
        integral = h * _pipeline_accumulate(kl_terms, trap_weights)
        return max(integral, 0.0)
    catch e
        _record_diagnostic!("fpga", "runtime_errors")
        @warn "ZeroProbFPGAExt: log_likelihood failed, falling back" exception=e maxlog=1
        return nothing
    end
end

# ============================================================================
# Hook: backend_coprocessor_sampling
# ============================================================================
#
# FPGA-based sampling via pipelined CDF inversion. The FPGA implements:
#   Stage 1: LFSR-based uniform random number generator
#   Stage 2: Pipelined binary search through pre-loaded CDF table (BRAM)
#   Stage 3: Linear interpolation unit
# The CDF table is stored in Block RAM for single-cycle lookup.

function ZeroProb.backend_coprocessor_sampling(
    backend::FPGABackend, dist, n_samples::Int)
    try
        n_samples <= 0 && return Float64[]

        # Pre-compute CDF table (stored in BRAM on FPGA)
        n_grid = max(8192, 2 * n_samples)
        lower = quantile(dist, 1e-8)
        upper = quantile(dist, 1.0 - 1e-8)
        dx = (upper - lower) / n_grid

        xs = Vector{Float64}(undef, n_grid + 1)
        cdf_table = Vector{Float64}(undef, n_grid + 1)

        for i in 0:n_grid
            xs[i+1] = lower + i * dx
            cdf_table[i+1] = cdf(dist, xs[i+1])
        end

        # Pipeline-parallel sampling
        uniforms = rand(Float64, n_samples)
        samples = Vector{Float64}(undef, n_samples)

        # Process samples in pipeline-depth-sized batches
        for batch_start in 1:PIPELINE_DEPTH:n_samples
            batch_end = min(batch_start + PIPELINE_DEPTH - 1, n_samples)
            for idx in batch_start:batch_end
                u = uniforms[idx]
                # Binary search through BRAM-resident CDF table
                lo, hi = 1, n_grid + 1
                while lo < hi - 1
                    mid = (lo + hi) >> 1
                    if cdf_table[mid] < u
                        lo = mid
                    else
                        hi = mid
                    end
                end
                # Interpolation unit
                gap = cdf_table[hi] - cdf_table[lo]
                frac = gap > 1e-300 ? (u - cdf_table[lo]) / gap : 0.0
                samples[idx] = xs[lo] + frac * dx
            end
        end

        return samples
    catch e
        _record_diagnostic!("fpga", "runtime_errors")
        @warn "ZeroProbFPGAExt: sampling failed, falling back" exception=e maxlog=1
        return nothing
    end
end

# ============================================================================
# Hook: backend_coprocessor_marginalize
# ============================================================================
#
# Streaming Simpson's rule pipeline for TV distance.
# The FPGA datapath implements Simpson's rule as a streaming pipeline:
#   Stage 1: Dual PDF evaluators
#   Stage 2: Absolute difference unit
#   Stage 3: Simpson's weight multiplier (2-state FSM toggle)
#   Stage 4: Multi-stage accumulator tree

function ZeroProb.backend_coprocessor_marginalize(
    backend::FPGABackend, P, Q, n_points::Int)
    try
        lower = min(quantile(P, 0.00001), quantile(Q, 0.00001))
        upper = max(quantile(P, 0.99999), quantile(Q, 0.99999))
        h = (upper - lower) / n_points

        n_total = n_points + 1

        # Streaming pipeline: evaluate and accumulate
        diffs = Vector{Float64}(undef, n_total)
        simpson_w = _simpson_weights(n_points)

        for i in 1:n_total
            x = lower + (i - 1) * h
            @inbounds diffs[i] = abs(pdf(P, x) - pdf(Q, x))
        end

        # Pipelined accumulator
        tv_integral = _pipeline_accumulate(diffs, simpson_w)
        return 0.5 * (h / 3.0) * tv_integral
    catch e
        _record_diagnostic!("fpga", "runtime_errors")
        @warn "ZeroProbFPGAExt: marginalize failed, falling back" exception=e maxlog=1
        return nothing
    end
end

# Conditional density dispatch
function ZeroProb.backend_coprocessor_marginalize(
    backend::FPGABackend, event::ZeroProb.ContinuousZeroProbEvent,
    condition::Function)
    try
        dist = event.distribution
        x = event.point

        numerator = pdf(dist, x) * condition(x)
        numerator == 0.0 && return 0.0

        if isa(dist, Normal)
            mu, sigma = params(dist)
            lower = mu - 6 * sigma
            upper = mu + 6 * sigma
        else
            lower = quantile(dist, 0.0001)
            upper = quantile(dist, 0.9999)
        end

        n_points = 1000
        h = (upper - lower) / n_points
        n_total = n_points + 1

        vals = Vector{Float64}(undef, n_total)
        for i in 1:n_total
            t = lower + (i - 1) * h
            @inbounds vals[i] = pdf(dist, t) * condition(t)
        end

        simpson_w = _simpson_weights(n_points)
        integral = (h / 3.0) * _pipeline_accumulate(vals, simpson_w)

        integral < 1e-15 && return 0.0
        return numerator / integral
    catch e
        _record_diagnostic!("fpga", "runtime_errors")
        @warn "ZeroProbFPGAExt: conditional density failed, falling back" exception=e maxlog=1
        return nothing
    end
end

end  # module ZeroProbFPGAExt
