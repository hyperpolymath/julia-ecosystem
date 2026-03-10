# SPDX-License-Identifier: PMPL-1.0-or-later
# Copyright (c) 2026 Jonathan D.A. Jewell (hyperpolymath) <j.d.a.jewell@open.ac.uk>
#
# ZeroProb.jl Metal Extension
#
# GPU-accelerated kernels for zero-probability measure computation on Apple GPUs.
# Uses KernelAbstractions.jl for portable kernel definitions and Metal.jl for
# device management and memory transfers.
#
# Provides the same five GPU operations as the CUDA extension:
#   - Hausdorff dimension estimation (Monte Carlo box-counting)
#   - KL divergence (parallel quadrature)
#   - Fisher information (parallel Monte Carlo score computation)
#   - Total variation distance (parallel Simpson's rule)
#   - Conditional density (parallel normalising-denominator integration)

module ZeroProbMetalExt

using Metal
using KernelAbstractions
using KernelAbstractions: @kernel, @index, @Const
using Distributions
using ZeroProb
using AcceleratorGate

# ============================================================================
# Availability Detection
# ============================================================================

function AcceleratorGate.metal_available()
    forced = AcceleratorGate._backend_env_available("AXIOM_METAL_AVAILABLE")
    forced !== nothing && return forced
    Metal.functional()
end

function AcceleratorGate.metal_device_count()
    forced = AcceleratorGate._backend_env_count("AXIOM_METAL_AVAILABLE", "AXIOM_METAL_DEVICE_COUNT")
    forced !== nothing && return forced
    Metal.functional() ? 1 : 0
end

# ============================================================================
# Kernel Definitions (KernelAbstractions.jl — identical to CUDA extension)
# ============================================================================

@kernel function box_count_kernel!(
    counts, occupied, samples, membership, scales,
    @Const(dim), @Const(max_boxes_per_dim)
)
    idx = @index(Global)
    n_samples = size(samples, 2)
    n_scales = length(scales)

    if idx <= n_samples && membership[idx] == Int32(1)
        for s in 1:n_scales
            eps = scales[s]
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
                flat_idx = hash % size(occupied, 1) + Int32(1)
                @inbounds occupied[flat_idx, s] = Int32(1)
            end
        end
    end
end

@kernel function reduce_occupied_kernel!(counts, occupied, @Const(n_slots))
    s = @index(Global)
    if s <= size(counts, 1)
        total = Int32(0)
        for i in 1:n_slots
            @inbounds total += occupied[i, s]
        end
        @inbounds counts[s] = total
    end
end

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

@kernel function fisher_score_kernel!(scores_sq, samples, logpdf_plus, logpdf_minus, @Const(two_delta))
    idx = @index(Global)
    if idx <= length(samples)
        @inbounds begin
            score = (logpdf_plus[idx] - logpdf_minus[idx]) / two_delta
            scores_sq[idx] = score * score
        end
    end
end

@kernel function tv_distance_kernel!(result, P_vals, Q_vals, weights, @Const(n))
    idx = @index(Global)
    if idx <= n
        @inbounds begin
            result[idx] = weights[idx] * abs(P_vals[idx] - Q_vals[idx])
        end
    end
end

@kernel function cond_density_kernel!(result, pdf_vals, cond_vals, weights, @Const(n))
    idx = @index(Global)
    if idx <= n
        @inbounds begin
            result[idx] = weights[idx] * pdf_vals[idx] * cond_vals[idx]
        end
    end
end

# ============================================================================
# Helpers
# ============================================================================

function _simpson_weights_mtl(n_points::Int)
    w = ones(Float64, n_points + 1)
    for i in 2:n_points
        w[i] = iseven(i) ? 4.0 : 2.0
    end
    MtlArray(w)
end

function _trapezoidal_weights_mtl(n_points::Int)
    w = ones(Float64, n_points + 1)
    w[1] = 0.5
    w[end] = 0.5
    MtlArray(w)
end

function _eval_pdf_on_mtl(dist::Distribution, xs::Vector{Float64})
    MtlArray(pdf.(Ref(dist), xs))
end

# KernelAbstractions backend for Metal — obtain via get_backend on a device array
MetalBackendKA() = KernelAbstractions.get_backend(MtlArray(zeros(Float32, 1)))

# ============================================================================
# Backend Hook: backend_sampling — Hausdorff Dimension (box-counting)
# ============================================================================

function ZeroProb.backend_sampling(backend::AcceleratorGate.MetalBackend,
                                   set_indicator::Function, dim::Int, n_boxes::Int)
    try
        scales_cpu = exp.(range(log(0.01), log(1.0), length=20))
        n_scales = length(scales_cpu)

        samples_cpu = rand(dim, n_boxes)
        membership_cpu = Int32[set_indicator(samples_cpu[:, i]) ? Int32(1) : Int32(0)
                               for i in 1:n_boxes]

        max_boxes_per_dim = min(100, floor(Int, 1.0 / minimum(scales_cpu)) + 1)
        n_slots = min(max_boxes_per_dim ^ min(dim, 3), 1_000_000)

        samples_gpu    = MtlArray(samples_cpu)
        membership_gpu = MtlArray(membership_cpu)
        scales_gpu     = MtlArray(scales_cpu)
        occupied_gpu   = Metal.zeros(Int32, n_slots, n_scales)
        counts_gpu     = Metal.zeros(Int32, n_scales)

        ka = MetalBackendKA()
        box_count_kernel!(ka, 256)(counts_gpu, occupied_gpu, samples_gpu, membership_gpu,
                                   scales_gpu, dim, Int32(max_boxes_per_dim);
                                   ndrange=n_boxes)
        reduce_occupied_kernel!(ka, 256)(counts_gpu, occupied_gpu, Int32(n_slots);
                                         ndrange=n_scales)
        KernelAbstractions.synchronize(ka)

        counts = Array(counts_gpu)

        log_inv = Float64[]
        log_cnt = Float64[]
        for (i, eps) in enumerate(scales_cpu)
            c = counts[i]
            if c > 0
                push!(log_inv, log(1.0 / eps))
                push!(log_cnt, log(Float64(c)))
            end
        end
        length(log_inv) < 2 && return 0.0

        n = length(log_inv)
        sx = sum(log_inv); sy = sum(log_cnt)
        sxy = sum(log_inv .* log_cnt); sx2 = sum(log_inv .^ 2)
        denom = n * sx2 - sx^2
        abs(denom) < 1e-15 && return 0.0
        return (n * sxy - sx * sy) / denom
    catch e
        AcceleratorGate._record_diagnostic!("metal", "runtime_errors")
        @warn "ZeroProbMetalExt: box-counting kernel failed, falling back" exception=e maxlog=1
        return nothing
    end
end

# ============================================================================
# Backend Hook: backend_sampling — Fisher Information
# ============================================================================

function ZeroProb.backend_sampling(backend::AcceleratorGate.MetalBackend,
                                   dist::Distribution, param::Symbol,
                                   n_samples::Int, delta::Float64)
    try
        samples_cpu = rand(dist, n_samples)

        if param == :mean
            isa(dist, Normal) || return nothing
            mu, sigma = params(dist)
            dist_plus  = Normal(mu + delta, sigma)
            dist_minus = Normal(mu - delta, sigma)
        elseif param == :std
            isa(dist, Normal) || return nothing
            mu, sigma = params(dist)
            dist_plus  = Normal(mu, sigma + delta)
            dist_minus = Normal(mu, sigma - delta)
        else
            return nothing
        end

        logpdf_plus_cpu  = logpdf.(Ref(dist_plus), samples_cpu)
        logpdf_minus_cpu = logpdf.(Ref(dist_minus), samples_cpu)

        samples_gpu      = MtlArray(samples_cpu)
        logpdf_plus_gpu  = MtlArray(logpdf_plus_cpu)
        logpdf_minus_gpu = MtlArray(logpdf_minus_cpu)
        scores_sq_gpu    = Metal.zeros(Float64, n_samples)

        ka = MetalBackendKA()
        fisher_score_kernel!(ka, 256)(scores_sq_gpu, samples_gpu,
                                      logpdf_plus_gpu, logpdf_minus_gpu,
                                      2.0 * delta; ndrange=n_samples)
        KernelAbstractions.synchronize(ka)

        return sum(Array(scores_sq_gpu)) / n_samples
    catch e
        AcceleratorGate._record_diagnostic!("metal", "runtime_errors")
        @warn "ZeroProbMetalExt: Fisher information kernel failed, falling back" exception=e maxlog=1
        return nothing
    end
end

# ============================================================================
# Backend Hook: backend_log_likelihood — KL Divergence
# ============================================================================

function ZeroProb.backend_log_likelihood(backend::AcceleratorGate.MetalBackend,
                                         P::Distribution, Q::Distribution,
                                         n_points::Int)
    try
        lower = quantile(P, 0.00001)
        upper = quantile(P, 0.99999)
        h = (upper - lower) / n_points
        xs = [lower + i * h for i in 0:n_points]

        P_cpu = pdf.(Ref(P), xs)
        Q_cpu = pdf.(Ref(Q), xs)
        for i in eachindex(xs)
            P_cpu[i] > 1e-300 && Q_cpu[i] < 1e-300 && return Inf
        end

        P_gpu = MtlArray(P_cpu)
        Q_gpu = MtlArray(Q_cpu)
        weights_gpu = _trapezoidal_weights_mtl(n_points)
        result_gpu  = Metal.zeros(Float64, n_points + 1)

        ka = MetalBackendKA()
        kl_divergence_kernel!(ka, 256)(result_gpu, P_gpu, Q_gpu, weights_gpu,
                                       Int32(n_points + 1); ndrange=n_points + 1)
        KernelAbstractions.synchronize(ka)

        return max(h * sum(Array(result_gpu)), 0.0)
    catch e
        AcceleratorGate._record_diagnostic!("metal", "runtime_errors")
        @warn "ZeroProbMetalExt: KL divergence kernel failed, falling back" exception=e maxlog=1
        return nothing
    end
end

# ============================================================================
# Backend Hook: backend_marginalize — Total Variation Distance
# ============================================================================

function ZeroProb.backend_marginalize(backend::AcceleratorGate.MetalBackend,
                                      P::Distribution, Q::Distribution,
                                      n_points::Int)
    try
        lower = min(quantile(P, 0.00001), quantile(Q, 0.00001))
        upper = max(quantile(P, 0.99999), quantile(Q, 0.99999))
        h = (upper - lower) / n_points
        xs = [lower + i * h for i in 0:n_points]

        P_gpu = _eval_pdf_on_mtl(P, xs)
        Q_gpu = _eval_pdf_on_mtl(Q, xs)
        weights_gpu = _simpson_weights_mtl(n_points)
        result_gpu  = Metal.zeros(Float64, n_points + 1)

        ka = MetalBackendKA()
        tv_distance_kernel!(ka, 256)(result_gpu, P_gpu, Q_gpu, weights_gpu,
                                     Int32(n_points + 1); ndrange=n_points + 1)
        KernelAbstractions.synchronize(ka)

        return 0.5 * (h / 3.0) * sum(Array(result_gpu))
    catch e
        AcceleratorGate._record_diagnostic!("metal", "runtime_errors")
        @warn "ZeroProbMetalExt: TV distance kernel failed, falling back" exception=e maxlog=1
        return nothing
    end
end

# ============================================================================
# Backend Hook: backend_marginalize — Conditional Density
# ============================================================================

function ZeroProb.backend_marginalize(backend::AcceleratorGate.MetalBackend,
                                      event::ZeroProb.ContinuousZeroProbEvent,
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
        xs = [lower + i * h for i in 0:n_points]

        pdf_vals_cpu  = pdf.(Ref(dist), xs)
        cond_vals_cpu = Float64[condition(t) for t in xs]

        pdf_gpu  = MtlArray(pdf_vals_cpu)
        cond_gpu = MtlArray(cond_vals_cpu)
        weights_gpu = _simpson_weights_mtl(n_points)
        result_gpu  = Metal.zeros(Float64, n_points + 1)

        ka = MetalBackendKA()
        cond_density_kernel!(ka, 256)(result_gpu, pdf_gpu, cond_gpu, weights_gpu,
                                      Int32(n_points + 1); ndrange=n_points + 1)
        KernelAbstractions.synchronize(ka)

        integral = (h / 3.0) * sum(Array(result_gpu))
        integral < 1e-15 && return 0.0
        return numerator / integral
    catch e
        AcceleratorGate._record_diagnostic!("metal", "runtime_errors")
        @warn "ZeroProbMetalExt: conditional density kernel failed, falling back" exception=e maxlog=1
        return nothing
    end
end

end  # module ZeroProbMetalExt
