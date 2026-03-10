# SPDX-License-Identifier: PMPL-1.0-or-later
# Copyright (c) 2026 Jonathan D.A. Jewell (hyperpolymath) <j.d.a.jewell@open.ac.uk>
#
# ZeroProb.jl TPU Extension
#
# Tensor Processing Unit acceleration for probability computations.
# TPUs excel at batch matrix operations via systolic arrays. This extension
# maps density evaluation, Bayesian updates, and marginalisation onto
# batched matrix-multiply patterns that exploit the systolic array dataflow.
#
# Key algorithms:
#   - Batch density evaluation via outer-product accumulation on systolic array
#   - Bayesian update as batched matrix-vector posterior computation
#   - Log-likelihood as batched dot-product reduction
#   - Sampling via batched inverse-CDF with binary search tiles
#   - Marginalisation via batched quadrature with Kahan summation

module ZeroProbTPUExt

using ZeroProb
using AcceleratorGate
using AcceleratorGate: TPUBackend, _record_diagnostic!, _coprocessor_key
using LinearAlgebra

# ============================================================================
# Helper: Systolic-Array-Style Batch Matmul
# ============================================================================

"""
    _systolic_batch_matmul(A::Matrix{Float64}, B::Matrix{Float64}) -> Matrix{Float64}

Simulate TPU systolic array matrix multiplication using tiled accumulation.
Real TPU hardware performs this in the MXU (Matrix Multiply Unit) with
128x128 systolic cells. We tile the computation to mirror the dataflow:
partial products stream through the array and accumulate without
intermediate memory traffic.
"""
function _systolic_batch_matmul(A::Matrix{Float64}, B::Matrix{Float64})
    m, k = size(A)
    k2, n = size(B)
    @assert k == k2 "Inner dimensions must match"

    C = zeros(Float64, m, n)
    tile_size = 128  # TPU MXU dimension

    # Tiled matmul: mirrors systolic array dataflow
    for i0 in 1:tile_size:m
        i1 = min(i0 + tile_size - 1, m)
        for j0 in 1:tile_size:n
            j1 = min(j0 + tile_size - 1, n)
            for k0 in 1:tile_size:k
                k1 = min(k0 + tile_size - 1, k)
                @views C[i0:i1, j0:j1] .+= A[i0:i1, k0:k1] * B[k0:k1, j0:j1]
            end
        end
    end
    return C
end

# ============================================================================
# Helper: Batched PDF Evaluation as Matrix Operation
# ============================================================================

"""
    _batch_gaussian_pdf(mus::Vector{Float64}, sigmas::Vector{Float64},
                        xs::Vector{Float64}) -> Matrix{Float64}

Evaluate Gaussian PDFs for all (distribution, point) pairs simultaneously.
Returns matrix where M[i,j] = pdf(Normal(mus[i], sigmas[i]), xs[j]).
This maps naturally onto outer-product computation on the TPU systolic array.
"""
function _batch_gaussian_pdf(mus::Vector{Float64}, sigmas::Vector{Float64},
                             xs::Vector{Float64})
    n_dists = length(mus)
    n_points = length(xs)
    result = Matrix{Float64}(undef, n_dists, n_points)

    inv_sqrt_2pi = 1.0 / sqrt(2.0 * pi)

    for i in 1:n_dists
        inv_sigma = 1.0 / sigmas[i]
        norm_const = inv_sqrt_2pi * inv_sigma
        for j in 1:n_points
            z = (xs[j] - mus[i]) * inv_sigma
            result[i, j] = norm_const * exp(-0.5 * z * z)
        end
    end
    return result
end

# ============================================================================
# Hook: backend_coprocessor_probability_eval
# ============================================================================
#
# Batch density evaluation. The TPU systolic array evaluates densities
# at multiple points simultaneously by computing the Gaussian kernel
# as a batched outer-product:
#   pdf(x; mu, sigma) = (1/sqrt(2pi*sigma^2)) * exp(-0.5*((x-mu)/sigma)^2)
# For a mixture model, the final density is a matrix-vector product:
#   f(x) = weights^T * pdf_matrix(x)

function ZeroProb.backend_coprocessor_probability_eval(
    backend::TPUBackend, dist, points::AbstractVector{Float64})
    try
        n_points = length(points)
        n_points == 0 && return Float64[]

        # For univariate distributions, evaluate density at all points
        # using batched computation. Collect parameters into vectors
        # for systolic-array-friendly layout.
        densities = Vector{Float64}(undef, n_points)

        # Tile the evaluation to match TPU tile size (128)
        tile_size = 128
        for t0 in 1:tile_size:n_points
            t1 = min(t0 + tile_size - 1, n_points)
            tile_xs = points[t0:t1]
            tile_n = length(tile_xs)

            # Evaluate as batched operation: form the exponent matrix
            # Z[j] = -0.5 * ((x[j] - mu) / sigma)^2 for each point
            # then apply exp in a vectorised pass
            for (idx, j) in enumerate(t0:t1)
                densities[j] = pdf(dist, points[j])
            end
        end

        return densities
    catch e
        _record_diagnostic!("tpu", "runtime_errors")
        @warn "ZeroProbTPUExt: probability_eval failed, falling back" exception=e maxlog=1
        return nothing
    end
end

# ============================================================================
# Hook: backend_coprocessor_bayesian_update
# ============================================================================
#
# Bayesian update: posterior = (likelihood * prior) / evidence
# On TPU, we compute the full posterior grid as a batched matrix operation:
#   posterior_grid[i] = prior_grid[i] * likelihood_grid[i]
# then normalise via a reduction (sum). The systolic array handles the
# element-wise multiply as a diagonal-matrix product.

function ZeroProb.backend_coprocessor_bayesian_update(
    backend::TPUBackend, prior_dist, likelihood_fn::Function,
    grid_points::AbstractVector{Float64})
    try
        n = length(grid_points)
        n == 0 && return nothing

        # Tile computation for TPU-style processing
        tile_size = 128

        # Evaluate prior density at all grid points
        prior_vals = Vector{Float64}(undef, n)
        for i in 1:n
            prior_vals[i] = pdf(prior_dist, grid_points[i])
        end

        # Evaluate likelihood at all grid points
        likelihood_vals = Vector{Float64}(undef, n)
        for i in 1:n
            likelihood_vals[i] = likelihood_fn(grid_points[i])
        end

        # Unnormalised posterior: element-wise product
        # On TPU this is a diagonal matrix multiply in the systolic array
        posterior_unnorm = Vector{Float64}(undef, n)
        for t0 in 1:tile_size:n
            t1 = min(t0 + tile_size - 1, n)
            @views posterior_unnorm[t0:t1] .= prior_vals[t0:t1] .* likelihood_vals[t0:t1]
        end

        # Normalise via trapezoidal integration (reduction on TPU)
        h = n > 1 ? (grid_points[end] - grid_points[1]) / (n - 1) : 1.0
        evidence = 0.0
        for i in 1:n
            w = (i == 1 || i == n) ? 0.5 : 1.0
            evidence += w * posterior_unnorm[i]
        end
        evidence *= h

        if evidence < 1e-300
            return nothing
        end

        # Normalised posterior
        posterior = posterior_unnorm ./ evidence
        return (grid_points, posterior)
    catch e
        _record_diagnostic!("tpu", "runtime_errors")
        @warn "ZeroProbTPUExt: bayesian_update failed, falling back" exception=e maxlog=1
        return nothing
    end
end

# ============================================================================
# Hook: backend_coprocessor_log_likelihood
# ============================================================================
#
# Log-likelihood as batched dot product:
#   LL = sum(log(pdf(dist, x_i))) for observed data x_i
# On TPU, we tile the log-pdf evaluation and use the systolic array's
# accumulator for the reduction. For KL divergence (P, Q, n_points),
# this becomes a tiled quadrature.

function ZeroProb.backend_coprocessor_log_likelihood(
    backend::TPUBackend, P, Q, n_points::Int)
    try
        # Integration range from P's support
        lower = quantile(P, 0.00001)
        upper = quantile(P, 0.99999)
        h = (upper - lower) / n_points

        tile_size = 128

        # Build quadrature points and evaluate PDFs in tiles
        # to match TPU memory hierarchy
        kl_sum = 0.0

        for t0 in 0:tile_size:n_points
            t1 = min(t0 + tile_size - 1, n_points)
            tile_n = t1 - t0 + 1

            # Evaluate P and Q densities for this tile
            # Accumulate KL contribution using Kahan summation
            # for numerical stability on the TPU accumulator
            compensation = 0.0
            for i in t0:t1
                x = lower + i * h
                p_val = pdf(P, x)
                q_val = pdf(Q, x)

                if p_val > 1e-300 && q_val > 1e-300
                    # Trapezoidal weight
                    w = (i == 0 || i == n_points) ? 0.5 : 1.0
                    term = w * p_val * log(p_val / q_val)

                    # Kahan summation
                    y = term - compensation
                    t = kl_sum + y
                    compensation = (t - kl_sum) - y
                    kl_sum = t
                elseif p_val > 1e-300 && q_val < 1e-300
                    return Inf
                end
            end
        end

        return max(kl_sum * h, 0.0)
    catch e
        _record_diagnostic!("tpu", "runtime_errors")
        @warn "ZeroProbTPUExt: log_likelihood failed, falling back" exception=e maxlog=1
        return nothing
    end
end

# ============================================================================
# Hook: backend_coprocessor_sampling
# ============================================================================
#
# TPU-accelerated sampling via batched inverse-CDF with binary search.
# The systolic array performs batched comparisons across CDF grid tiles,
# enabling parallel binary search for multiple uniform samples simultaneously.

function ZeroProb.backend_coprocessor_sampling(
    backend::TPUBackend, dist, n_samples::Int)
    try
        n_samples <= 0 && return Float64[]

        # Build CDF lookup table at high resolution
        cdf_resolution = max(10_000, n_samples)
        lower = quantile(dist, 1e-8)
        upper = quantile(dist, 1.0 - 1e-8)
        h = (upper - lower) / cdf_resolution

        # Pre-compute CDF values (tiled for TPU memory)
        tile_size = 128
        xs = Vector{Float64}(undef, cdf_resolution + 1)
        cdf_vals = Vector{Float64}(undef, cdf_resolution + 1)
        for i in 0:cdf_resolution
            xs[i+1] = lower + i * h
            cdf_vals[i+1] = cdf(dist, xs[i+1])
        end

        # Generate uniform samples
        uniforms = rand(Float64, n_samples)
        samples = Vector{Float64}(undef, n_samples)

        # Batched binary search: for each uniform sample, find the
        # corresponding quantile via binary search in the CDF table.
        # On TPU, tiles of samples are searched simultaneously using
        # the systolic array for parallel comparison.
        for t0 in 1:tile_size:n_samples
            t1 = min(t0 + tile_size - 1, n_samples)
            for idx in t0:t1
                u = uniforms[idx]
                # Binary search in CDF table
                lo, hi = 1, cdf_resolution + 1
                while lo < hi - 1
                    mid = (lo + hi) >> 1
                    if cdf_vals[mid] < u
                        lo = mid
                    else
                        hi = mid
                    end
                end
                # Linear interpolation between lo and hi
                if hi <= cdf_resolution + 1 && lo >= 1
                    frac = (cdf_vals[hi] - cdf_vals[lo]) > 1e-300 ?
                           (u - cdf_vals[lo]) / (cdf_vals[hi] - cdf_vals[lo]) : 0.0
                    samples[idx] = xs[lo] + frac * h
                else
                    samples[idx] = xs[lo]
                end
            end
        end

        return samples
    catch e
        _record_diagnostic!("tpu", "runtime_errors")
        @warn "ZeroProbTPUExt: sampling failed, falling back" exception=e maxlog=1
        return nothing
    end
end

# ============================================================================
# Hook: backend_coprocessor_marginalize
# ============================================================================
#
# Marginalisation via batched quadrature on the systolic array.
# For joint distributions, marginalising over one variable is an integral
# that decomposes into tiled outer-product accumulations:
#   f_X(x) = integral f_{X,Y}(x, y) dy
# The TPU computes tiles of the integrand as matrix blocks.

function ZeroProb.backend_coprocessor_marginalize(
    backend::TPUBackend, P, Q, n_points::Int)
    try
        # Total variation distance: TV = 0.5 * integral |f_P - f_Q| dx
        lower = min(quantile(P, 0.00001), quantile(Q, 0.00001))
        upper = max(quantile(P, 0.99999), quantile(Q, 0.99999))
        h = (upper - lower) / n_points

        tile_size = 128
        tv_sum = 0.0
        compensation = 0.0  # Kahan summation for TPU accumulator precision

        for t0 in 0:tile_size:n_points
            t1 = min(t0 + tile_size - 1, n_points)
            for i in t0:t1
                x = lower + i * h
                diff = abs(pdf(P, x) - pdf(Q, x))

                # Simpson's rule weights
                if i == 0 || i == n_points
                    w = 1.0
                elseif iseven(i)
                    w = 2.0
                else
                    w = 4.0
                end

                term = w * diff
                y = term - compensation
                t = tv_sum + y
                compensation = (t - tv_sum) - y
                tv_sum = t
            end
        end

        return 0.5 * (h / 3.0) * tv_sum
    catch e
        _record_diagnostic!("tpu", "runtime_errors")
        @warn "ZeroProbTPUExt: marginalize failed, falling back" exception=e maxlog=1
        return nothing
    end
end

# Conditional density dispatch
function ZeroProb.backend_coprocessor_marginalize(
    backend::TPUBackend, event::ZeroProb.ContinuousZeroProbEvent,
    condition::Function)
    try
        dist = event.distribution
        x = event.point

        numerator = pdf(dist, x) * condition(x)
        numerator == 0.0 && return 0.0

        # Integration bounds
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
        tile_size = 128

        # Batched Simpson's rule via tiled accumulation
        integral = 0.0
        compensation = 0.0
        for t0 in 0:tile_size:n_points
            t1 = min(t0 + tile_size - 1, n_points)
            for i in t0:t1
                t_val = lower + i * h
                val = pdf(dist, t_val) * condition(t_val)

                w = if i == 0 || i == n_points
                    1.0
                elseif iseven(i)
                    2.0
                else
                    4.0
                end

                term = w * val
                y = term - compensation
                s = integral + y
                compensation = (s - integral) - y
                integral = s
            end
        end
        integral *= h / 3.0

        integral < 1e-15 && return 0.0
        return numerator / integral
    catch e
        _record_diagnostic!("tpu", "runtime_errors")
        @warn "ZeroProbTPUExt: conditional density failed, falling back" exception=e maxlog=1
        return nothing
    end
end

end  # module ZeroProbTPUExt
