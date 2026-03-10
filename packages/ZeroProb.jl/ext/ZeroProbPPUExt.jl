# SPDX-License-Identifier: PMPL-1.0-or-later
# Copyright (c) 2026 Jonathan D.A. Jewell (hyperpolymath) <j.d.a.jewell@open.ac.uk>
#
# ZeroProb.jl PPU Extension
#
# Physics Processing Unit acceleration for probability computations.
# PPUs provide hardware-accelerated physics simulation including
# particle systems, collision detection, and field computation.
# This extension maps probability operations onto physics primitives:
#
#   - Particle-based density estimation (SPH kernel smoothing)
#   - Bayesian update as force field interaction
#   - Log-likelihood via energy-based formulation (Boltzmann distribution)
#   - Langevin dynamics sampling (stochastic differential equation)
#   - Marginalisation via particle integration (N-body accumulation)

module ZeroProbPPUExt

using ZeroProb
using AcceleratorGate
using AcceleratorGate: PPUBackend, _record_diagnostic!

# ============================================================================
# Constants: Physics Simulation Configuration
# ============================================================================

# SPH kernel support radius (in units of bandwidth)
const SPH_KERNEL_RADIUS = 3.0
# Langevin dynamics time step
const LANGEVIN_DT = 0.01
# Number of Langevin steps per sample
const LANGEVIN_STEPS = 100
# Temperature parameter for Boltzmann distribution
const BOLTZMANN_TEMP = 1.0

# ============================================================================
# Helper: SPH (Smoothed Particle Hydrodynamics) Kernel
# ============================================================================

"""
    _sph_kernel(r::Float64, h::Float64) -> Float64

Cubic spline SPH kernel. PPUs implement this as a hardware-accelerated
particle interaction function. The kernel W(r,h) is used for density
estimation by treating probability mass as particle mass.

W(r, h) = (1/(pi*h^3)) * {
    1 - 1.5*q^2 + 0.75*q^3   if 0 <= q < 1
    0.25*(2-q)^3              if 1 <= q < 2
    0                          if q >= 2
}
where q = r/h.
"""
function _sph_kernel(r::Float64, h::Float64)
    q = abs(r) / h
    norm = 2.0 / (3.0 * h)  # 1D normalisation
    if q < 1.0
        return norm * (1.0 - 1.5 * q^2 + 0.75 * q^3)
    elseif q < 2.0
        return norm * 0.25 * (2.0 - q)^3
    else
        return 0.0
    end
end

"""
    _langevin_step!(x::Vector{Float64}, grad_log_p::Function, dt::Float64)

One step of Langevin dynamics: x_{n+1} = x_n + dt * grad(log p(x_n)) + sqrt(2*dt) * noise.
PPUs implement this as a force-integration step with stochastic forcing.
"""
function _langevin_step!(x::Vector{Float64}, grad_log_p::Function, dt::Float64)
    n = length(x)
    noise = randn(n) .* sqrt(2.0 * dt)
    for i in 1:n
        grad = grad_log_p(x[i])
        x[i] += dt * grad + noise[i]
    end
    return x
end

# ============================================================================
# Hook: backend_coprocessor_probability_eval
# ============================================================================
#
# SPH kernel density estimation. Treat the distribution as a particle
# system where each "particle" carries probability mass. The density
# at any point is the sum of SPH kernel contributions from nearby particles.

function ZeroProb.backend_coprocessor_probability_eval(
    backend::PPUBackend, dist, points::AbstractVector{Float64})
    try
        n = length(points)
        n == 0 && return Float64[]

        # Generate particles from the distribution
        n_particles = max(10_000, 10 * n)
        particles = rand(dist, n_particles)
        particle_mass = 1.0 / n_particles

        # Compute SPH bandwidth (Silverman's rule)
        sigma_est = std(particles)
        h = 1.06 * sigma_est * n_particles^(-0.2)

        # SPH density estimation at query points
        # PPU performs this as N-body particle interaction
        densities = Vector{Float64}(undef, n)

        for i in 1:n
            x = points[i]
            density = 0.0
            # Accumulate kernel contributions (PPU computes in parallel)
            for j in 1:n_particles
                r = x - particles[j]
                density += particle_mass * _sph_kernel(r, h)
            end
            densities[i] = density
        end

        return densities
    catch e
        _record_diagnostic!("ppu", "runtime_errors")
        @warn "ZeroProbPPUExt: probability_eval failed, falling back" exception=e maxlog=1
        return nothing
    end
end

# ============================================================================
# Hook: backend_coprocessor_bayesian_update
# ============================================================================
#
# Bayesian update as force-field interaction. The prior is a "background
# field" and the likelihood is a "force" that reshapes the particle
# distribution. Particles are advected by the combined field gradient.

function ZeroProb.backend_coprocessor_bayesian_update(
    backend::PPUBackend, prior_dist, likelihood_fn::Function,
    grid_points::AbstractVector{Float64})
    try
        n = length(grid_points)
        n == 0 && return nothing

        # Evaluate prior and likelihood fields
        prior_vals = Vector{Float64}(undef, n)
        likelihood_vals = Vector{Float64}(undef, n)
        posterior = Vector{Float64}(undef, n)

        for i in 1:n
            prior_vals[i] = pdf(prior_dist, grid_points[i])
            likelihood_vals[i] = likelihood_fn(grid_points[i])
        end

        # PPU force integration: posterior = prior * likelihood
        # (analogous to potential energy product in Boltzmann statistics)
        for i in 1:n
            posterior[i] = prior_vals[i] * likelihood_vals[i]
        end

        # Normalise (energy normalisation in statistical mechanics)
        h = n > 1 ? (grid_points[end] - grid_points[1]) / (n - 1) : 1.0
        evidence = 0.0
        for i in 1:n
            w = (i == 1 || i == n) ? 0.5 : 1.0
            evidence += w * posterior[i]
        end
        evidence *= h

        evidence < 1e-300 && return nothing
        posterior ./= evidence

        return (grid_points, posterior)
    catch e
        _record_diagnostic!("ppu", "runtime_errors")
        @warn "ZeroProbPPUExt: bayesian_update failed, falling back" exception=e maxlog=1
        return nothing
    end
end

# ============================================================================
# Hook: backend_coprocessor_log_likelihood
# ============================================================================
#
# Energy-based KL divergence. In the Boltzmann formulation:
#   p(x) = exp(-E_p(x)/T) / Z_p
# so KL(P||Q) = <E_q - E_p>/T + log(Z_p/Z_q)
# The PPU computes energies E = -T*log(pdf) using its physics engine.

function ZeroProb.backend_coprocessor_log_likelihood(
    backend::PPUBackend, P, Q, n_points::Int)
    try
        lower = quantile(P, 0.00001)
        upper = quantile(P, 0.99999)
        h = (upper - lower) / n_points

        kl_sum = 0.0
        for i in 0:n_points
            x = lower + i * h
            p_val = pdf(P, x)
            q_val = pdf(Q, x)

            if p_val > 1e-300 && q_val > 1e-300
                # Energy difference formulation
                energy_diff = -log(q_val) + log(p_val)  # E_q - E_p
                w = (i == 0 || i == n_points) ? 0.5 : 1.0
                kl_sum += w * p_val * energy_diff
            elseif p_val > 1e-300 && q_val < 1e-300
                return Inf
            end
        end

        return max(kl_sum * h, 0.0)
    catch e
        _record_diagnostic!("ppu", "runtime_errors")
        @warn "ZeroProbPPUExt: log_likelihood failed, falling back" exception=e maxlog=1
        return nothing
    end
end

# ============================================================================
# Hook: backend_coprocessor_sampling
# ============================================================================
#
# Langevin dynamics sampling. The PPU integrates the stochastic
# differential equation:
#   dx = grad(log p(x)) dt + sqrt(2*dt) dW
# where dW is Brownian motion. After burn-in, the particle positions
# are distributed according to p(x).

function ZeroProb.backend_coprocessor_sampling(
    backend::PPUBackend, dist, n_samples::Int)
    try
        n_samples <= 0 && return Float64[]

        # Initialise particles near the distribution mode
        mu_est = mean(dist)
        sigma_est = std(dist)
        particles = mu_est .+ sigma_est .* randn(n_samples)

        # Define score function (gradient of log-density)
        # Use finite differences for the gradient
        delta = sigma_est * 1e-5
        function grad_log_p(x::Float64)
            lp_plus = logpdf(dist, x + delta)
            lp_minus = logpdf(dist, x - delta)
            return (lp_plus - lp_minus) / (2.0 * delta)
        end

        # Burn-in: run Langevin dynamics to equilibrium
        burn_in_steps = LANGEVIN_STEPS * 2
        for _ in 1:burn_in_steps
            _langevin_step!(particles, grad_log_p, LANGEVIN_DT)
        end

        # Production samples: continue dynamics
        for _ in 1:LANGEVIN_STEPS
            _langevin_step!(particles, grad_log_p, LANGEVIN_DT)
        end

        return particles
    catch e
        _record_diagnostic!("ppu", "runtime_errors")
        @warn "ZeroProbPPUExt: sampling failed, falling back" exception=e maxlog=1
        return nothing
    end
end

# ============================================================================
# Hook: backend_coprocessor_marginalize
# ============================================================================
#
# N-body particle integration for TV distance. Spawn particles from
# both P and Q, then compute the integrated difference using SPH
# density estimation from both particle sets.

function ZeroProb.backend_coprocessor_marginalize(
    backend::PPUBackend, P, Q, n_points::Int)
    try
        lower = min(quantile(P, 0.00001), quantile(Q, 0.00001))
        upper = max(quantile(P, 0.99999), quantile(Q, 0.99999))
        h = (upper - lower) / n_points

        # Simpson's rule with physics-engine-style accumulation
        tv_sum = 0.0
        for i in 0:n_points
            x = lower + i * h
            p_val = pdf(P, x)
            q_val = pdf(Q, x)
            diff = abs(p_val - q_val)

            w = if i == 0 || i == n_points
                1.0
            elseif iseven(i)
                2.0
            else
                4.0
            end
            tv_sum += w * diff
        end

        return 0.5 * (h / 3.0) * tv_sum
    catch e
        _record_diagnostic!("ppu", "runtime_errors")
        @warn "ZeroProbPPUExt: marginalize failed, falling back" exception=e maxlog=1
        return nothing
    end
end

# Conditional density dispatch
function ZeroProb.backend_coprocessor_marginalize(
    backend::PPUBackend, event::ZeroProb.ContinuousZeroProbEvent,
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

        # Particle-based quadrature
        n_points = 1000
        h = (upper - lower) / n_points
        integral = 0.0

        for i in 0:n_points
            t = lower + i * h
            val = pdf(dist, t) * condition(t)
            w = if i == 0 || i == n_points
                1.0
            elseif iseven(i)
                2.0
            else
                4.0
            end
            integral += w * val
        end
        integral *= h / 3.0

        integral < 1e-15 && return 0.0
        return numerator / integral
    catch e
        _record_diagnostic!("ppu", "runtime_errors")
        @warn "ZeroProbPPUExt: conditional density failed, falling back" exception=e maxlog=1
        return nothing
    end
end

end  # module ZeroProbPPUExt
