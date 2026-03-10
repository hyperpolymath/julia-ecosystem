# SPDX-License-Identifier: PMPL-1.0-or-later
# Copyright (c) 2026 Jonathan D.A. Jewell (hyperpolymath) <j.d.a.jewell@open.ac.uk>
#
# ZeroProb.jl CUDA Extension
#
# GPU-accelerated kernels for zero-probability measure computation on NVIDIA GPUs.
# Uses KernelAbstractions.jl for portable kernel definitions and CUDA.jl for
# device management and memory transfers.
#
# Provides GPU implementations for:
#   - Hausdorff dimension estimation (Monte Carlo box-counting)
#   - KL divergence (parallel quadrature)
#   - Fisher information (parallel Monte Carlo score computation)
#   - Total variation distance (parallel Simpson's rule)
#   - Conditional density (parallel normalising-denominator integration)

module ZeroProbCUDAExt

using CUDA
using KernelAbstractions
using KernelAbstractions: @kernel, @index, @Const
using Distributions
using ZeroProb
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
# Kernel Definitions (KernelAbstractions.jl — portable across GPU backends)
# ============================================================================

# ---------------------------------------------------------------------------
# (a) Hausdorff Dimension — Monte Carlo Box-Counting
# ---------------------------------------------------------------------------
#
# Each thread processes one random sample point. It evaluates the set-indicator
# function and, if the point lies inside the set, computes the integer box
# index at the current scale and atomically marks the box as occupied in the
# output bitmap.
#
# We flatten the d-dimensional box index to a 1D hash so that we can use a
# fixed-size boolean array rather than a dictionary.

@kernel function box_count_kernel!(
    counts,           # Int32[n_scales] — number of occupied boxes per scale
    occupied,         # Int32[max_boxes, n_scales] — occupancy bitmap (0/1)
    samples,          # Float64[dim, n_samples] — random points in [0,1]^dim
    membership,       # Int32[n_samples] — 1 if point is in set, 0 otherwise
    scales,           # Float64[n_scales]
    @Const(dim),
    @Const(max_boxes_per_dim)
)
    idx = @index(Global)
    n_samples = size(samples, 2)
    n_scales = length(scales)

    if idx <= n_samples && membership[idx] == Int32(1)
        for s in 1:n_scales
            eps = scales[s]
            # Compute a 1D hash from the d-dimensional box index using
            # mixed-radix encoding: hash = sum_k( floor(x_k/eps) * max_boxes^k )
            hash = Int32(0)
            stride = Int32(1)
            valid = true
            for k in 1:dim
                box_idx = floor(Int32, samples[k, idx] / eps)
                if box_idx < Int32(0) || box_idx >= max_boxes_per_dim
                    valid = false
                    break
                end
                hash += box_idx * stride
                stride *= max_boxes_per_dim
            end
            if valid
                # Clamp hash into the occupied array range
                flat_idx = hash % size(occupied, 1) + Int32(1)
                # Mark occupied (atomic write — races are benign: worst case
                # two threads both write 1 to the same slot)
                @inbounds occupied[flat_idx, s] = Int32(1)
            end
        end
    end
end

@kernel function reduce_occupied_kernel!(counts, occupied, @Const(n_slots))
    s = @index(Global)  # scale index
    if s <= size(counts, 1)
        total = Int32(0)
        for i in 1:n_slots
            @inbounds total += occupied[i, s]
        end
        @inbounds counts[s] = total
    end
end

# ---------------------------------------------------------------------------
# (b) KL Divergence — Parallel Quadrature
# ---------------------------------------------------------------------------
#
# Evaluates p(x)*log(p(x)/q(x)) at each quadrature point in parallel.

@kernel function kl_divergence_kernel!(result, P_vals, Q_vals, weights, @Const(n))
    idx = @index(Global)
    if idx <= n
        @inbounds begin
            p = P_vals[idx]
            q = Q_vals[idx]
            if p > 1e-300 && q > 1e-300
                result[idx] = weights[idx] * p * log(p / q)
            else
                result[idx] = 0.0
            end
        end
    end
end

# ---------------------------------------------------------------------------
# (c) Fisher Information — Parallel Score Computation
# ---------------------------------------------------------------------------
#
# For each sample, computes the score function (derivative of log-likelihood)
# via central finite differences and squares it. The mean of squared scores
# gives the Fisher information estimate.

@kernel function fisher_score_kernel!(
    scores_sq,       # Float64[n_samples] — output: score(x)^2
    samples,         # Float64[n_samples]
    logpdf_plus,     # Float64[n_samples] — log f(x; theta + delta)
    logpdf_minus,    # Float64[n_samples] — log f(x; theta - delta)
    @Const(two_delta)
)
    idx = @index(Global)
    if idx <= length(samples)
        @inbounds begin
            score = (logpdf_plus[idx] - logpdf_minus[idx]) / two_delta
            scores_sq[idx] = score * score
        end
    end
end

# ---------------------------------------------------------------------------
# (d) Total Variation Distance — Parallel Simpson's Rule
# ---------------------------------------------------------------------------
#
# Evaluates |f_P(x) - f_Q(x)| at each quadrature point, applies Simpson's
# rule weights, then reduces on CPU. TV = 0.5 * integral.

@kernel function tv_distance_kernel!(result, P_vals, Q_vals, weights, @Const(n))
    idx = @index(Global)
    if idx <= n
        @inbounds begin
            result[idx] = weights[idx] * abs(P_vals[idx] - Q_vals[idx])
        end
    end
end

# ---------------------------------------------------------------------------
# (e) Conditional Density — Parallel Simpson's Rule for Normalising Denominator
# ---------------------------------------------------------------------------
#
# Evaluates f(t) * condition(t) at each quadrature point with Simpson's
# weights. The sum gives the normalising integral; the conditional density
# is then numerator / integral.

@kernel function cond_density_kernel!(result, pdf_vals, cond_vals, weights, @Const(n))
    idx = @index(Global)
    if idx <= n
        @inbounds begin
            result[idx] = weights[idx] * pdf_vals[idx] * cond_vals[idx]
        end
    end
end

# ============================================================================
# Helper: Build Simpson's Rule Weights on GPU
# ============================================================================

"""
    _simpson_weights_gpu(n_points::Int) -> CuArray{Float64,1}

Build Simpson's rule weight vector of length `n_points + 1` on the GPU.
Weights are: 1, 4, 2, 4, 2, ..., 4, 1  (scaled by h/3 by the caller).
"""
function _simpson_weights_gpu(n_points::Int)
    w = ones(Float64, n_points + 1)
    for i in 2:n_points
        w[i] = iseven(i) ? 4.0 : 2.0
    end
    CuArray(w)
end

"""
    _trapezoidal_weights_gpu(n_points::Int) -> CuArray{Float64,1}

Build trapezoidal rule weight vector of length `n_points + 1` on the GPU.
Weights: 0.5, 1, 1, ..., 1, 0.5
"""
function _trapezoidal_weights_gpu(n_points::Int)
    w = ones(Float64, n_points + 1)
    w[1] = 0.5
    w[end] = 0.5
    CuArray(w)
end

# ============================================================================
# Helper: Evaluate PDF at Quadrature Points on GPU
# ============================================================================

"""
    _eval_pdf_on_gpu(dist::Distribution, xs::Vector{Float64}) -> CuArray{Float64,1}

Evaluate the PDF of `dist` at all points in `xs`, transfer to GPU.
(PDF evaluation uses Distributions.jl on CPU, then bulk-transfers.)
"""
function _eval_pdf_on_gpu(dist::Distribution, xs::Vector{Float64})
    vals = pdf.(Ref(dist), xs)
    CuArray(vals)
end

"""
    _eval_logpdf_on_gpu(dist::Distribution, xs::Vector{Float64}) -> CuArray{Float64,1}

Evaluate log-PDF of `dist` at all points in `xs`, transfer to GPU.
"""
function _eval_logpdf_on_gpu(dist::Distribution, xs::Vector{Float64})
    vals = logpdf.(Ref(dist), xs)
    CuArray(vals)
end

# ============================================================================
# Backend Hook: backend_sampling — Hausdorff Dimension (box-counting)
# ============================================================================
#
# Signature from measures.jl hausdorff_dimension():
#   backend_sampling(backend, set_indicator, dim, n_boxes)
#
# Returns: Float64 (estimated box-counting dimension), or nothing on failure.

function ZeroProb.backend_sampling(backend::CUDABackend,
                                   set_indicator::Function, dim::Int, n_boxes::Int)
    try
        # Default scales: logarithmically spaced from 0.01 to 1.0
        scales_cpu = exp.(range(log(0.01), log(1.0), length=20))
        n_scales = length(scales_cpu)

        # Generate random points in [0,1]^dim on CPU, evaluate set membership,
        # then transfer to GPU for box-counting.
        samples_cpu = rand(dim, n_boxes)
        membership_cpu = Int32[set_indicator(samples_cpu[:, i]) ? Int32(1) : Int32(0)
                               for i in 1:n_boxes]

        # Determine occupancy grid sizing
        # max_boxes_per_dim is chosen so that the total flat hash space fits
        # in reasonable GPU memory.  At the finest scale (0.01) in d dimensions,
        # there are up to 100^d boxes — we cap the per-dimension count.
        max_boxes_per_dim = min(100, floor(Int, 1.0 / minimum(scales_cpu)) + 1)
        # Total slots in occupancy bitmap (capped for memory)
        n_slots = min(max_boxes_per_dim ^ min(dim, 3), 1_000_000)

        # Transfer to GPU
        samples_gpu = CuArray(samples_cpu)
        membership_gpu = CuArray(membership_cpu)
        scales_gpu = CuArray(scales_cpu)
        occupied_gpu = CUDA.zeros(Int32, n_slots, n_scales)
        counts_gpu = CUDA.zeros(Int32, n_scales)

        # Launch box-counting kernel
        ka_backend = CUDABackendKA()
        kernel_box = box_count_kernel!(ka_backend, 256)
        kernel_box(counts_gpu, occupied_gpu, samples_gpu, membership_gpu,
                   scales_gpu, dim, Int32(max_boxes_per_dim);
                   ndrange=n_boxes)

        # Reduce occupancy bitmap to counts
        kernel_reduce = reduce_occupied_kernel!(ka_backend, 256)
        kernel_reduce(counts_gpu, occupied_gpu, Int32(n_slots);
                      ndrange=n_scales)

        KernelAbstractions.synchronize(ka_backend)

        # Transfer counts back to CPU
        counts = Array(counts_gpu)

        # Linear regression: log(N) vs log(1/eps)
        log_inv_scales = Float64[]
        log_counts = Float64[]
        for (i, eps) in enumerate(scales_cpu)
            c = counts[i]
            if c > 0
                push!(log_inv_scales, log(1.0 / eps))
                push!(log_counts, log(Float64(c)))
            end
        end

        if length(log_inv_scales) < 2
            return 0.0
        end

        n = length(log_inv_scales)
        sx  = sum(log_inv_scales)
        sy  = sum(log_counts)
        sxy = sum(log_inv_scales .* log_counts)
        sx2 = sum(log_inv_scales .^ 2)

        denom = n * sx2 - sx^2
        if abs(denom) < 1e-15
            return 0.0
        end

        slope = (n * sxy - sx * sy) / denom
        return slope
    catch e
        AcceleratorGate._record_diagnostic!("cuda", "runtime_errors")
        @warn "ZeroProbCUDAExt: box-counting kernel failed, falling back" exception=e maxlog=1
        return nothing
    end
end

# ============================================================================
# Backend Hook: backend_sampling — Fisher Information (Monte Carlo scores)
# ============================================================================
#
# Signature from measures.jl fisher_information():
#   backend_sampling(backend, dist, param, n_samples, delta)
#
# Returns: Float64 (estimated Fisher information), or nothing on failure.

function ZeroProb.backend_sampling(backend::CUDABackend,
                                   dist::Distribution, param::Symbol,
                                   n_samples::Int, delta::Float64)
    try
        # Generate samples on CPU (Distributions.jl requires CPU)
        samples_cpu = rand(dist, n_samples)

        # Build perturbed distributions for finite-difference score
        if param == :mean
            if isa(dist, Normal)
                mu, sigma = params(dist)
                dist_plus  = Normal(mu + delta, sigma)
                dist_minus = Normal(mu - delta, sigma)
            else
                return nothing  # Unsupported distribution type
            end
        elseif param == :std
            if isa(dist, Normal)
                mu, sigma = params(dist)
                dist_plus  = Normal(mu, sigma + delta)
                dist_minus = Normal(mu, sigma - delta)
            else
                return nothing
            end
        else
            return nothing  # Unknown parameter
        end

        # Evaluate log-PDFs under perturbed distributions on CPU, then transfer
        logpdf_plus_cpu  = logpdf.(Ref(dist_plus), samples_cpu)
        logpdf_minus_cpu = logpdf.(Ref(dist_minus), samples_cpu)

        samples_gpu      = CuArray(samples_cpu)
        logpdf_plus_gpu  = CuArray(logpdf_plus_cpu)
        logpdf_minus_gpu = CuArray(logpdf_minus_cpu)
        scores_sq_gpu    = CUDA.zeros(Float64, n_samples)

        # Launch score-squaring kernel
        ka_backend = CUDABackendKA()
        kernel = fisher_score_kernel!(ka_backend, 256)
        kernel(scores_sq_gpu, samples_gpu, logpdf_plus_gpu, logpdf_minus_gpu,
               2.0 * delta; ndrange=n_samples)
        KernelAbstractions.synchronize(ka_backend)

        # Reduce: mean of squared scores
        scores_sq_cpu = Array(scores_sq_gpu)
        fisher_info = sum(scores_sq_cpu) / n_samples
        return fisher_info
    catch e
        AcceleratorGate._record_diagnostic!("cuda", "runtime_errors")
        @warn "ZeroProbCUDAExt: Fisher information kernel failed, falling back" exception=e maxlog=1
        return nothing
    end
end

# ============================================================================
# Backend Hook: backend_log_likelihood — KL Divergence
# ============================================================================
#
# Signature from measures.jl kl_divergence():
#   backend_log_likelihood(backend, P, Q, n_points)
#
# Returns: Float64 (KL divergence), or nothing on failure.

function ZeroProb.backend_log_likelihood(backend::CUDABackend,
                                         P::Distribution, Q::Distribution,
                                         n_points::Int)
    try
        # Integration range from P's support
        lower = quantile(P, 0.00001)
        upper = quantile(P, 0.99999)
        h = (upper - lower) / n_points

        # Build quadrature points on CPU
        xs = [lower + i * h for i in 0:n_points]

        # Evaluate PDFs on CPU, transfer to GPU
        P_vals_gpu = _eval_pdf_on_gpu(P, xs)
        Q_vals_gpu = _eval_pdf_on_gpu(Q, xs)

        # Check for support mismatch (P > 0 where Q = 0)
        P_vals_cpu = pdf.(Ref(P), xs)
        Q_vals_cpu = pdf.(Ref(Q), xs)
        for i in eachindex(xs)
            if P_vals_cpu[i] > 1e-300 && Q_vals_cpu[i] < 1e-300
                return Inf
            end
        end

        # Trapezoidal weights on GPU
        weights_gpu = _trapezoidal_weights_gpu(n_points)
        result_gpu  = CUDA.zeros(Float64, n_points + 1)

        # Launch KL kernel
        ka_backend = CUDABackendKA()
        kernel = kl_divergence_kernel!(ka_backend, 256)
        kernel(result_gpu, P_vals_gpu, Q_vals_gpu, weights_gpu,
               Int32(n_points + 1); ndrange=n_points + 1)
        KernelAbstractions.synchronize(ka_backend)

        # Reduce on CPU
        result_cpu = Array(result_gpu)
        integral = h * sum(result_cpu)
        return max(integral, 0.0)
    catch e
        AcceleratorGate._record_diagnostic!("cuda", "runtime_errors")
        @warn "ZeroProbCUDAExt: KL divergence kernel failed, falling back" exception=e maxlog=1
        return nothing
    end
end

# ============================================================================
# Backend Hook: backend_marginalize — Total Variation Distance
# ============================================================================
#
# Signature from measures.jl total_variation_distance():
#   backend_marginalize(backend, P, Q, n_points)
#
# Returns: Float64 (TV distance), or nothing on failure.

function ZeroProb.backend_marginalize(backend::CUDABackend,
                                      P::Distribution, Q::Distribution,
                                      n_points::Int)
    try
        # Integration range from both distributions
        lower = min(quantile(P, 0.00001), quantile(Q, 0.00001))
        upper = max(quantile(P, 0.99999), quantile(Q, 0.99999))
        h = (upper - lower) / n_points

        # Quadrature points
        xs = [lower + i * h for i in 0:n_points]

        # Evaluate PDFs and transfer to GPU
        P_vals_gpu = _eval_pdf_on_gpu(P, xs)
        Q_vals_gpu = _eval_pdf_on_gpu(Q, xs)

        # Simpson's rule weights on GPU
        weights_gpu = _simpson_weights_gpu(n_points)
        result_gpu  = CUDA.zeros(Float64, n_points + 1)

        # Launch TV kernel
        ka_backend = CUDABackendKA()
        kernel = tv_distance_kernel!(ka_backend, 256)
        kernel(result_gpu, P_vals_gpu, Q_vals_gpu, weights_gpu,
               Int32(n_points + 1); ndrange=n_points + 1)
        KernelAbstractions.synchronize(ka_backend)

        # Reduce on CPU — Simpson's rule: multiply sum by h/3
        result_cpu = Array(result_gpu)
        integral = (h / 3.0) * sum(result_cpu)

        # TV = 0.5 * integral |f_P - f_Q|
        return 0.5 * integral
    catch e
        AcceleratorGate._record_diagnostic!("cuda", "runtime_errors")
        @warn "ZeroProbCUDAExt: TV distance kernel failed, falling back" exception=e maxlog=1
        return nothing
    end
end

# ============================================================================
# Backend Hook: backend_marginalize — Conditional Density
# ============================================================================
#
# Signature from measures.jl conditional_density():
#   backend_marginalize(backend, event, condition)
#
# Returns: Float64 (conditional density value), or nothing on failure.

function ZeroProb.backend_marginalize(backend::CUDABackend,
                                      event::ZeroProb.ContinuousZeroProbEvent,
                                      condition::Function)
    try
        dist = event.distribution
        x = event.point

        # Numerator: f(x) * condition(x)
        numerator = pdf(dist, x) * condition(x)
        if numerator == 0.0
            return 0.0
        end

        # Determine integration bounds
        if isa(dist, Normal)
            mu, sigma = params(dist)
            lower = mu - 6 * sigma
            upper = mu + 6 * sigma
        else
            lower = quantile(dist, 0.0001)
            upper = quantile(dist, 0.9999)
        end

        # Simpson's rule with 1000 points
        n_points = 1000
        h = (upper - lower) / n_points

        # Quadrature points
        xs = [lower + i * h for i in 0:n_points]

        # Evaluate on CPU (condition is an arbitrary Julia function)
        pdf_vals_cpu  = pdf.(Ref(dist), xs)
        cond_vals_cpu = Float64[condition(t) for t in xs]

        # Transfer to GPU
        pdf_vals_gpu  = CuArray(pdf_vals_cpu)
        cond_vals_gpu = CuArray(cond_vals_cpu)
        weights_gpu   = _simpson_weights_gpu(n_points)
        result_gpu    = CUDA.zeros(Float64, n_points + 1)

        # Launch conditional density kernel
        ka_backend = CUDABackendKA()
        kernel = cond_density_kernel!(ka_backend, 256)
        kernel(result_gpu, pdf_vals_gpu, cond_vals_gpu, weights_gpu,
               Int32(n_points + 1); ndrange=n_points + 1)
        KernelAbstractions.synchronize(ka_backend)

        # Reduce: Simpson's rule
        result_cpu = Array(result_gpu)
        integral = (h / 3.0) * sum(result_cpu)

        if integral < 1e-15
            return 0.0
        end

        return numerator / integral
    catch e
        AcceleratorGate._record_diagnostic!("cuda", "runtime_errors")
        @warn "ZeroProbCUDAExt: conditional density kernel failed, falling back" exception=e maxlog=1
        return nothing
    end
end

# ============================================================================
# Utility: CUDA KernelAbstractions Backend Alias
# ============================================================================

"""
    CUDABackendKA() -> KernelAbstractions.Backend

Return the KernelAbstractions backend corresponding to CUDA.
Uses the CUDABackend from KernelAbstractions (provided by CUDA.jl extension).
"""
CUDABackendKA() = KernelAbstractions.get_backend(CUDA.zeros(1))

end  # module ZeroProbCUDAExt
