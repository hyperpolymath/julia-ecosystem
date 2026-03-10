# SPDX-License-Identifier: PMPL-1.0-or-later
# Copyright (c) 2026 Jonathan D.A. Jewell (hyperpolymath) <j.d.a.jewell@open.ac.uk>
#
# CliometricsTPUExt -- TPU systolic array acceleration for Cliometrics.jl
# Exploits matrix-multiply hardware for batch regression and panel data
# growth accounting across many countries simultaneously.

module CliometricsTPUExt

using Cliometrics
using AcceleratorGate
using AcceleratorGate: TPUBackend, JuliaBackend, _record_diagnostic!,
    register_operation!, track_allocation!, track_deallocation!
using LinearAlgebra

function __init__()
    register_operation!(TPUBackend, :regression)
    register_operation!(TPUBackend, :decomposition)
    register_operation!(TPUBackend, :convergence_test)
    register_operation!(TPUBackend, :growth_accounting)
end

function Cliometrics.backend_coprocessor_regression(b::TPUBackend, X::Matrix{Float64}, y::Vector{Float64})
    n, k = size(X); n < 128 && return nothing
    mem_estimate = Int64(n * k * 4 + k * k * 4)
    track_allocation!(b, mem_estimate)
    try
        X32 = Float32.(X); y32 = Float32.(y)
        XtX = X32' * X32 + 1f-8 * I; Xty = X32' * y32
        beta = Float64.(XtX \ Xty); residuals = y .- X * beta
        ss_res = sum(residuals .^ 2); y_mean = sum(y) / n
        ss_tot = sum((y .- y_mean) .^ 2); r_squared = 1.0 - ss_res / ss_tot
        track_deallocation!(b, mem_estimate)
        return (coefficients=beta, residuals=residuals, r_squared=r_squared)
    catch ex
        track_deallocation!(b, mem_estimate); _record_diagnostic!(b, "runtime_errors"); return nothing
    end
end

function Cliometrics.backend_coprocessor_decomposition(b::TPUBackend, output::Vector{Float64}, capital::Vector{Float64}, labor::Vector{Float64}, alpha::Float64)
    n = length(output); n < 128 && return nothing
    try
        Y32 = Float32.(output); K32 = Float32.(capital); L32 = Float32.(labor)
        g_Y = Float64.(log.(Y32[2:end] ./ Y32[1:end-1]))
        g_K = Float64.(log.(K32[2:end] ./ K32[1:end-1]))
        g_L = Float64.(log.(L32[2:end] ./ L32[1:end-1]))
        capital_contrib = alpha .* g_K; labor_contrib = (1.0 - alpha) .* g_L
        tfp_contrib = g_Y .- capital_contrib .- labor_contrib
        return (output_growth=g_Y, capital_contribution=capital_contrib,
                labor_contribution=labor_contrib, tfp_contribution=tfp_contrib)
    catch ex; _record_diagnostic!(b, "runtime_errors"); return nothing; end
end

function Cliometrics.backend_coprocessor_convergence_test(b::TPUBackend, initial_levels::Vector{Float64}, growth_rates::Vector{Float64})
    n = length(initial_levels); n < 64 && return nothing
    mem_estimate = Int64(n * 2 * 4 + 4 * 4)
    track_allocation!(b, mem_estimate)
    try
        log_initial = Float32.(log.(initial_levels))
        X = hcat(ones(Float32, n), log_initial); y = Float32.(growth_rates)
        XtX = X' * X + 1f-8 * I; Xty = X' * y; beta = XtX \ Xty
        pred = X * beta; ss_res = sum((y .- pred) .^ 2)
        y_mean = sum(y) / n; ss_tot = sum((y .- y_mean) .^ 2)
        r_squared = Float64(1.0 - ss_res / ss_tot)
        beta_coef = Float64(beta[2]); is_converging = beta_coef < 0
        track_deallocation!(b, mem_estimate)
        return (alpha=Float64(beta[1]), beta=beta_coef, r_squared=r_squared,
                converging=is_converging, half_life=is_converging ? -log(2) / beta_coef : Inf)
    catch ex
        track_deallocation!(b, mem_estimate); _record_diagnostic!(b, "runtime_errors"); return nothing
    end
end

function Cliometrics.backend_coprocessor_growth_accounting(b::TPUBackend, panel_data::Matrix{Float64}, n_countries::Int, alpha::Float64)
    n_total = size(panel_data, 1); n_total < 256 && return nothing
    n_per_country = n_total ÷ n_countries
    try
        panel32 = Float32.(panel_data)
        g_Y = Float64.(log.(panel32[2:end, 1] ./ panel32[1:end-1, 1]))
        g_K = Float64.(log.(panel32[2:end, 2] ./ panel32[1:end-1, 2]))
        g_L = Float64.(log.(panel32[2:end, 3] ./ panel32[1:end-1, 3]))
        tfp = g_Y .- alpha .* g_K .- (1.0 - alpha) .* g_L
        country_tfp = Vector{Float64}(undef, n_countries)
        for c in 1:n_countries
            start_idx = (c - 1) * (n_per_country - 1) + 1
            end_idx = min(c * (n_per_country - 1), length(tfp))
            country_tfp[c] = sum(tfp[start_idx:end_idx]) / (end_idx - start_idx + 1)
        end
        return (growth_rates=g_Y, tfp_residuals=tfp, country_avg_tfp=country_tfp, avg_tfp=sum(tfp) / length(tfp))
    catch ex; _record_diagnostic!(b, "runtime_errors"); return nothing; end
end

function Cliometrics.backend_coprocessor_time_series_filter(b::TPUBackend, args...); return nothing; end

end # module CliometricsTPUExt
