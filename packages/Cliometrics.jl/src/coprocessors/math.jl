# SPDX-License-Identifier: PMPL-1.0-or-later
# Copyright (c) 2026 Jonathan D.A. Jewell (hyperpolymath) <j.d.a.jewell@open.ac.uk>
#
# Cliometrics.jl Math Coprocessor
# Arbitrary precision for long-run growth calculations.

function AcceleratorGate.device_capabilities(b::MathBackend)
    AcceleratorGate.DeviceCapabilities(
        b, 4, 1000,
        Int64(8 * 1024^3), Int64(6 * 1024^3),
        1, true, false, false, "Software", "BigFloat/Rational",
    )
end

function AcceleratorGate.estimate_cost(::MathBackend, op::Symbol, data_size::Int)
    overhead = 20.0
    op == :regression && return overhead + Float64(data_size) * 0.5
    op == :growth_accounting && return overhead + Float64(data_size) * 0.3
    op == :convergence_test && return overhead + Float64(data_size) * 0.4
    Inf
end

AcceleratorGate.register_operation!(MathBackend, :regression)
AcceleratorGate.register_operation!(MathBackend, :growth_accounting)
AcceleratorGate.register_operation!(MathBackend, :convergence_test)

"""
Arbitrary-precision OLS regression using BigFloat.
Eliminates floating-point accumulation errors in long historical series
(e.g., 1000+ year Maddison Project data).
"""
function backend_coprocessor_regression(::MathBackend, X::AbstractMatrix, y::AbstractVector)
    Xb = BigFloat.(X); yb = BigFloat.(y)
    beta = (Xb' * Xb) \ (Xb' * yb)
    residuals = yb .- Xb * beta
    ss_res = sum(residuals .^ 2)
    ym = sum(yb) / length(yb)
    ss_tot = sum((yb .- ym) .^ 2)
    r2 = BigFloat(1) - ss_res / max(ss_tot, BigFloat(1e-100))
    (coefficients=Float64.(beta), r_squared=Float64(r2), residuals=Float64.(residuals))
end

"""
Arbitrary-precision growth accounting for very long time series.
Prevents catastrophic cancellation in log-ratio calculations.
"""
function backend_coprocessor_growth_accounting(::MathBackend, output::AbstractVector,
                                               capital::AbstractVector,
                                               labor::AbstractVector;
                                               alpha::Float64=0.3)
    ob = BigFloat.(output); kb = BigFloat.(capital); lb = BigFloat.(labor)
    ab = BigFloat(alpha)
    g_Y = [log(ob[i] / ob[i-1]) for i in 2:length(ob)]
    g_K = [log(kb[i] / kb[i-1]) for i in 2:length(kb)]
    g_L = [log(lb[i] / lb[i-1]) for i in 2:length(lb)]
    tfp = g_Y .- ab .* g_K .- (BigFloat(1) - ab) .* g_L
    (output_growth=Float64.(g_Y), capital_contribution=Float64.(ab .* g_K),
     labor_contribution=Float64.((BigFloat(1) - ab) .* g_L), tfp=Float64.(tfp))
end

"""
Arbitrary-precision convergence analysis avoiding floating-point bias
in cross-country regression.
"""
function backend_coprocessor_convergence_test(::MathBackend,
                                              initial_levels::AbstractVector,
                                              growth_rates::AbstractVector)
    x = BigFloat.(log.(initial_levels)); y = BigFloat.(growth_rates)
    n = length(x)
    xm = sum(x) / n; ym = sum(y) / n
    beta = sum((x .- xm) .* (y .- ym)) / max(sum((x .- xm) .^ 2), BigFloat(1e-100))
    alpha = ym - beta * xm
    y_pred = alpha .+ beta .* x
    ss_tot = sum((y .- ym) .^ 2); ss_res = sum((y .- y_pred) .^ 2)
    r2 = BigFloat(1) - ss_res / max(ss_tot, BigFloat(1e-100))
    (alpha=Float64(alpha), beta=Float64(beta), r_squared=Float64(r2),
     converging=beta < 0, half_life=beta < 0 ? Float64(-log(BigFloat(2)) / beta) : Inf)
end
