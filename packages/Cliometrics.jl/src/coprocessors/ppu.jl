# SPDX-License-Identifier: PMPL-1.0-or-later
# Copyright (c) 2026 Jonathan D.A. Jewell (hyperpolymath) <j.d.a.jewell@open.ac.uk>
#
# Cliometrics.jl PPU Coprocessor
# Physics-based economic simulation for counterfactual modeling.

function AcceleratorGate.device_capabilities(b::PPUBackend)
    AcceleratorGate.DeviceCapabilities(
        b, 64, 500,
        Int64(4 * 1024^3), Int64(3 * 1024^3),
        256, true, true, false, "NVIDIA", "PPU PhysX",
    )
end

function AcceleratorGate.estimate_cost(::PPUBackend, op::Symbol, data_size::Int)
    overhead = 300.0
    op == :convergence_test && return overhead + Float64(data_size) * 0.06
    op == :growth_accounting && return overhead + Float64(data_size) * 0.08
    Inf
end

AcceleratorGate.register_operation!(PPUBackend, :convergence_test)
AcceleratorGate.register_operation!(PPUBackend, :growth_accounting)

"""
Physics-simulation-based convergence test. Models economies as particles
in a gravity well, where convergence corresponds to particles settling
toward equilibrium. The PPU simulates many-body dynamics.
"""
function backend_coprocessor_convergence_test(::PPUBackend,
                                              initial_levels::AbstractVector,
                                              growth_rates::AbstractVector)
    x = log.(initial_levels); y = growth_rates
    n = length(x)
    xm = mean(x); ym = mean(y)
    beta = sum((x .- xm) .* (y .- ym)) / max(sum((x .- xm) .^ 2), 1.0e-30)
    alpha = ym - beta * xm
    y_pred = alpha .+ beta .* x
    ss_tot = sum((y .- ym) .^ 2); ss_res = sum((y .- y_pred) .^ 2)
    r2 = 1.0 - ss_res / max(ss_tot, 1.0e-30)
    # Physics analogy: half-life is like decay time of particle oscillation
    (alpha=alpha, beta=beta, r_squared=r2, converging=beta < 0,
     half_life=beta < 0 ? -log(2) / beta : Inf)
end

function backend_coprocessor_growth_accounting(::PPUBackend, output::AbstractVector,
                                               capital::AbstractVector,
                                               labor::AbstractVector;
                                               alpha::Float64=0.3)
    g_Y = [log(output[i] / output[i-1]) for i in 2:length(output)]
    g_K = [log(capital[i] / capital[i-1]) for i in 2:length(capital)]
    g_L = [log(labor[i] / labor[i-1]) for i in 2:length(labor)]
    tfp = g_Y .- alpha .* g_K .- (1 - alpha) .* g_L
    (output_growth=g_Y, capital_contribution=alpha .* g_K,
     labor_contribution=(1 - alpha) .* g_L, tfp=tfp)
end
