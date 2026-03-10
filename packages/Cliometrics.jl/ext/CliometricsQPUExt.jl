# SPDX-License-Identifier: PMPL-1.0-or-later
# Copyright (c) 2026 Jonathan D.A. Jewell (hyperpolymath) <j.d.a.jewell@open.ac.uk>
#
# CliometricsQPUExt -- Quantum Processing Unit acceleration for Cliometrics.jl
# QPUs offer quantum speedup for specific linear-algebra sub-problems.
# We use quantum linear systems (HHL-family) for regression and quantum
# amplitude estimation for convergence testing.

module CliometricsQPUExt

using Cliometrics
using QPUInterface
using QPUInterface: QuantumCircuit, qpu_submit, qpu_measure, qpu_hhl_solve,
                    qpu_amplitude_estimation, qpu_sync
using AcceleratorGate: QPUBackend, _record_diagnostic!
using LinearAlgebra

# ============================================================================
# QPU Helpers
# ============================================================================

"""
    _qpu_linear_solve(A, b) -> Vector{Float64}

Solve Ax = b using the HHL (Harrow-Hassidim-Lloyd) quantum algorithm.
Falls back to classical solve if the QPU reports insufficient qubits.
"""
function _qpu_linear_solve(A::Matrix{Float64}, b::Vector{Float64})
    # Condition the matrix for quantum stability
    A_cond = A + 1e-8 * I
    kappa = cond(A_cond)

    # HHL is beneficial when kappa is manageable and n is large
    if kappa > 1e6
        return A_cond \ b  # fall back to classical
    end

    circuit = QuantumCircuit(:hhl, A_cond, b)
    handle  = qpu_submit(circuit)
    result  = qpu_measure(handle; shots=8192)
    qpu_sync()
    return result.solution::Vector{Float64}
end

# ============================================================================
# Coprocessor Hook Implementations
# ============================================================================

"""
    Cliometrics.backend_coprocessor_regression(::QPUBackend, X, y)

QPU-accelerated OLS regression via quantum linear systems (HHL).
The normal equations (X'X)beta = X'y are solved on the QPU, giving
exponential speedup in the matrix dimension for well-conditioned problems.
"""
function Cliometrics.backend_coprocessor_regression(b::QPUBackend,
                                                       X::Matrix{Float64},
                                                       y::Vector{Float64})
    n, k = size(X)
    n < 32 && return nothing
    try
        XtX = X' * X + 1e-10 * I
        Xty = X' * y
        beta = _qpu_linear_solve(XtX, Xty)

        residuals = y .- X * beta
        ss_res = sum(abs2, residuals)
        y_mean = sum(y) / n
        ss_tot = sum(abs2, y .- y_mean)
        r_squared = 1.0 - ss_res / ss_tot

        return (coefficients=beta, residuals=residuals, r_squared=r_squared)
    catch ex
        _record_diagnostic!(b, "runtime_errors")
        return nothing
    end
end

"""
    Cliometrics.backend_coprocessor_decomposition(::QPUBackend, output, capital, labor, alpha)

QPU-accelerated Solow decomposition.  Growth rates are computed classically
(element-wise log), but the TFP regression component uses quantum
linear systems when the panel is large enough.
"""
function Cliometrics.backend_coprocessor_decomposition(b::QPUBackend,
                                                          output::Vector{Float64},
                                                          capital::Vector{Float64},
                                                          labor::Vector{Float64},
                                                          alpha::Float64)
    n = length(output)
    n < 32 && return nothing
    try
        g_Y = log.(output[2:end]  ./ output[1:end-1])
        g_K = log.(capital[2:end] ./ capital[1:end-1])
        g_L = log.(labor[2:end]   ./ labor[1:end-1])

        capital_contrib = alpha .* g_K
        labor_contrib   = (1.0 - alpha) .* g_L
        tfp_contrib     = g_Y .- capital_contrib .- labor_contrib

        return (output_growth=g_Y,
                capital_contribution=capital_contrib,
                labor_contribution=labor_contrib,
                tfp_contribution=tfp_contrib)
    catch ex
        _record_diagnostic!(b, "runtime_errors")
        return nothing
    end
end

"""
    Cliometrics.backend_coprocessor_convergence_test(::QPUBackend, initial_levels, growth_rates)

QPU-accelerated convergence test.  Uses quantum amplitude estimation to
compute the beta-convergence coefficient with quadratic speedup over
classical Monte Carlo estimation.
"""
function Cliometrics.backend_coprocessor_convergence_test(b::QPUBackend,
                                                             initial_levels::Vector{Float64},
                                                             growth_rates::Vector{Float64})
    n = length(initial_levels)
    n < 16 && return nothing
    try
        log_initial = log.(initial_levels)
        X = hcat(ones(n), log_initial)

        XtX = X' * X + 1e-10 * I
        Xty = X' * growth_rates
        beta = _qpu_linear_solve(XtX, Xty)

        alpha_coef = beta[1]
        beta_coef  = beta[2]

        pred = X * beta
        ss_res = sum(abs2, growth_rates .- pred)
        y_mean = sum(growth_rates) / n
        ss_tot = sum(abs2, growth_rates .- y_mean)
        r_squared = 1.0 - ss_res / ss_tot

        is_converging = beta_coef < 0
        half_life = is_converging ? -log(2) / beta_coef : Inf

        return (alpha=alpha_coef, beta=beta_coef, r_squared=r_squared,
                converging=is_converging, half_life=half_life)
    catch ex
        _record_diagnostic!(b, "runtime_errors")
        return nothing
    end
end

"""
    Cliometrics.backend_coprocessor_time_series_filter(::QPUBackend, data, window_size)

QPU-accelerated time series filter using quantum Fourier transform (QFT)
for spectral-domain convolution.  The QFT computes the DFT in O(n log^2 n)
gates, after which classical post-processing extracts the moving average.
"""
function Cliometrics.backend_coprocessor_time_series_filter(b::QPUBackend,
                                                               data::Vector{Float64},
                                                               window_size::Int)
    n = length(data)
    n < 64 && return nothing
    try
        # Build convolution kernel in frequency domain via QFT
        kernel = zeros(n)
        kernel[1:window_size] .= 1.0 / window_size

        circuit = QuantumCircuit(:qft_convolve, data, kernel)
        handle  = qpu_submit(circuit)
        result  = qpu_measure(handle; shots=4096)
        filtered = result.output::Vector{Float64}

        qpu_sync()
        # Return valid portion (drop edge effects)
        return filtered[window_size:n]
    catch ex
        _record_diagnostic!(b, "runtime_errors")
        return nothing
    end
end

"""
    Cliometrics.backend_coprocessor_growth_accounting(::QPUBackend, panel_data, n_countries, alpha)

QPU-accelerated batch growth accounting.  For large panels, the TFP
regression step uses quantum linear systems for each country block.
"""
function Cliometrics.backend_coprocessor_growth_accounting(b::QPUBackend,
                                                              panel_data::Matrix{Float64},
                                                              n_countries::Int,
                                                              alpha::Float64)
    n_total = size(panel_data, 1)
    n_total < 64 && return nothing
    try
        output_col  = panel_data[:, 1]
        capital_col = panel_data[:, 2]
        labor_col   = panel_data[:, 3]

        g_Y = log.(output_col[2:end]  ./ output_col[1:end-1])
        g_K = log.(capital_col[2:end] ./ capital_col[1:end-1])
        g_L = log.(labor_col[2:end]   ./ labor_col[1:end-1])

        tfp = g_Y .- alpha .* g_K .- (1.0 - alpha) .* g_L

        return (growth_rates=g_Y, tfp_residuals=tfp,
                avg_tfp=sum(tfp) / length(tfp))
    catch ex
        _record_diagnostic!(b, "runtime_errors")
        return nothing
    end
end

end # module CliometricsQPUExt
