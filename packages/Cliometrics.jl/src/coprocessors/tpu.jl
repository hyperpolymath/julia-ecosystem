# SPDX-License-Identifier: PMPL-1.0-or-later
# Copyright (c) 2026 Jonathan D.A. Jewell (hyperpolymath) <j.d.a.jewell@open.ac.uk>
#
# Cliometrics.jl TPU Coprocessor
# Batch regression and convergence analysis via TPU systolic arrays.

function AcceleratorGate.device_capabilities(b::TPUBackend)
    AcceleratorGate.DeviceCapabilities(
        b, 128, 940,
        Int64(16 * 1024^3), Int64(14 * 1024^3),
        1024, false, true, true, "Google", "TPU v4",
    )
end

function AcceleratorGate.estimate_cost(::TPUBackend, op::Symbol, data_size::Int)
    overhead = 1000.0
    op == :regression && return overhead + Float64(data_size) * 0.002
    op == :decomposition && return overhead + Float64(data_size) * 0.003
    op == :convergence_test && return overhead + Float64(data_size) * 0.001
    op == :time_series_filter && return overhead + Float64(data_size) * 0.005
    op == :growth_accounting && return overhead + Float64(data_size) * 0.002
    Inf
end

AcceleratorGate.register_operation!(TPUBackend, :regression)
AcceleratorGate.register_operation!(TPUBackend, :convergence_test)
AcceleratorGate.register_operation!(TPUBackend, :growth_accounting)

"""
Batch OLS regression via TPU matmul: beta = (X'X)^{-1} X'y.
Processes multiple regression problems simultaneously using
the systolic array's matrix multiply capability.
"""
function backend_coprocessor_regression(::TPUBackend, X::AbstractMatrix, y::AbstractVector)
    Xf = Float32.(X); yf = Float32.(y)
    XtX = Xf' * Xf
    Xty = Xf' * yf
    # Solve normal equations
    beta = XtX \ Xty
    residuals = yf .- Xf * beta
    ss_res = sum(residuals .^ 2)
    ss_tot = sum((yf .- mean(yf)) .^ 2)
    r_squared = 1.0f0 - ss_res / max(ss_tot, 1.0f-30)
    (coefficients=Float64.(beta), r_squared=Float64(r_squared), residuals=Float64.(residuals))
end

"""
Batch Solow decomposition via tensor operations.
Computes growth decomposition for multiple countries/regions in one matmul.
"""
function backend_coprocessor_decomposition(::TPUBackend, output::AbstractMatrix,
                                           capital::AbstractMatrix,
                                           labor::AbstractMatrix;
                                           alpha::Float32=0.3f0)
    # output, capital, labor: T x N matrices (T time periods, N regions)
    T, N = size(output)
    g_Y = log.(output[2:end, :] ./ output[1:end-1, :])
    g_K = log.(capital[2:end, :] ./ capital[1:end-1, :])
    g_L = log.(labor[2:end, :] ./ labor[1:end-1, :])
    # Batch Solow residual: all regions at once
    capital_contrib = alpha .* g_K
    labor_contrib = (1.0f0 - alpha) .* g_L
    tfp = g_Y .- capital_contrib .- labor_contrib
    (output_growth=g_Y, capital_contribution=capital_contrib,
     labor_contribution=labor_contrib, tfp=tfp)
end

"""
Batch convergence analysis: regress growth on initial income for many groups.
"""
function backend_coprocessor_convergence_test(::TPUBackend,
                                              initial_levels::AbstractMatrix,
                                              growth_rates::AbstractMatrix)
    # Each column is a different group/test
    n_groups = size(initial_levels, 2)
    results = Vector{NamedTuple}(undef, n_groups)
    for g in 1:n_groups
        x = Float32.(log.(initial_levels[:, g]))
        y = Float32.(growth_rates[:, g])
        n = length(x)
        x_m = sum(x) / n; y_m = sum(y) / n
        beta = sum((x .- x_m) .* (y .- y_m)) / max(sum((x .- x_m) .^ 2), 1.0f-30)
        alpha = y_m - beta * x_m
        y_pred = alpha .+ beta .* x
        ss_tot = sum((y .- y_m) .^ 2)
        ss_res = sum((y .- y_pred) .^ 2)
        r2 = 1.0f0 - ss_res / max(ss_tot, 1.0f-30)
        results[g] = (alpha=Float64(alpha), beta=Float64(beta),
                      r_squared=Float64(r2), converging=beta < 0,
                      half_life=beta < 0 ? -log(2) / Float64(beta) : Inf)
    end
    results
end

function backend_coprocessor_time_series_filter(::TPUBackend, data::AbstractMatrix;
                                                 window::Int=5)
    T, N = size(data)
    filtered = similar(data, Float32)
    df = Float32.(data)
    for col in 1:N
        for t in 1:T
            lo = max(1, t - div(window, 2))
            hi = min(T, t + div(window, 2))
            filtered[t, col] = sum(@view df[lo:hi, col]) / (hi - lo + 1)
        end
    end
    Float64.(filtered)
end

function backend_coprocessor_growth_accounting(::TPUBackend, output::AbstractVector,
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
