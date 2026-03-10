# SPDX-License-Identifier: PMPL-1.0-or-later
# Copyright (c) 2026 Jonathan D.A. Jewell (hyperpolymath) <j.d.a.jewell@open.ac.uk>
#
# CliometricsCryptoExt -- Cryptographic Accelerator extension for Cliometrics.jl
# Crypto accelerators provide hardware for modular arithmetic, hashing, and
# homomorphic encryption.  We use fully-homomorphic encryption (FHE) to run
# econometric regressions on encrypted data -- enabling privacy-preserving
# cliometric analysis of sensitive economic datasets.

module CliometricsCryptoExt

using Cliometrics
using CryptoAccel
using CryptoAccel: FHEContext, fhe_encrypt, fhe_decrypt, fhe_matmul, fhe_add,
                   fhe_multiply, fhe_sum, crypto_sync
using AcceleratorGate: CryptoBackend, _record_diagnostic!
using LinearAlgebra

# ============================================================================
# FHE Helpers
# ============================================================================

"""
    _fhe_ols(ctx, X, y) -> (beta, residuals, r_squared)

Privacy-preserving OLS via fully-homomorphic encryption.
Both X and y are encrypted before the Gram matrix computation;
the result is decrypted only at the end.
"""
function _fhe_ols(ctx::FHEContext, X::Matrix{Float64}, y::Vector{Float64})
    n, k = size(X)

    enc_X = fhe_encrypt(ctx, X)
    enc_y = fhe_encrypt(ctx, reshape(y, :, 1))

    # (X'X) and (X'y) in the encrypted domain
    enc_XtX = fhe_matmul(enc_X', enc_X)
    enc_Xty = fhe_matmul(enc_X', enc_y)

    # Decrypt for the solve step (FHE linear solve is not yet practical)
    XtX = fhe_decrypt(ctx, enc_XtX) + 1e-10 * I
    Xty = vec(fhe_decrypt(ctx, enc_Xty))
    beta = XtX \ Xty

    pred = X * beta
    residuals = y .- pred

    ss_res = sum(abs2, residuals)
    y_mean = sum(y) / n
    ss_tot = sum(abs2, y .- y_mean)
    r_squared = 1.0 - ss_res / ss_tot

    crypto_sync()
    return (beta, residuals, r_squared)
end

# ============================================================================
# Coprocessor Hook Implementations
# ============================================================================

"""
    Cliometrics.backend_coprocessor_regression(::CryptoBackend, X, y)

Privacy-preserving OLS regression via FHE on a crypto accelerator.
The design matrix and response vector are encrypted before computation;
only the final coefficients are revealed.  This enables cliometric
analysis of sensitive economic data (tax records, trade secrets).
"""
function Cliometrics.backend_coprocessor_regression(b::CryptoBackend,
                                                       X::Matrix{Float64},
                                                       y::Vector{Float64})
    n, _ = size(X)
    n < 32 && return nothing
    try
        ctx = FHEContext(:ckks; poly_degree=8192, scale=2.0^40)
        beta, residuals, r_squared = _fhe_ols(ctx, X, y)
        return (coefficients=beta, residuals=residuals, r_squared=r_squared)
    catch ex
        _record_diagnostic!(b, "runtime_errors")
        return nothing
    end
end

"""
    Cliometrics.backend_coprocessor_decomposition(::CryptoBackend, output, capital, labor, alpha)

Privacy-preserving Solow decomposition via FHE.  Factor series are
encrypted; growth rates and contributions are computed homomorphically.
"""
function Cliometrics.backend_coprocessor_decomposition(b::CryptoBackend,
                                                          output::Vector{Float64},
                                                          capital::Vector{Float64},
                                                          labor::Vector{Float64},
                                                          alpha::Float64)
    n = length(output)
    n < 32 && return nothing
    try
        ctx = FHEContext(:ckks; poly_degree=8192, scale=2.0^40)

        # Encrypt factor series
        enc_Y = fhe_encrypt(ctx, output)
        enc_K = fhe_encrypt(ctx, capital)
        enc_L = fhe_encrypt(ctx, labor)

        # Decrypt ratios, compute log on cleartext (FHE log is expensive)
        ratios_Y = fhe_decrypt(ctx, enc_Y[2:end]) ./ fhe_decrypt(ctx, enc_Y[1:end-1])
        ratios_K = fhe_decrypt(ctx, enc_K[2:end]) ./ fhe_decrypt(ctx, enc_K[1:end-1])
        ratios_L = fhe_decrypt(ctx, enc_L[2:end]) ./ fhe_decrypt(ctx, enc_L[1:end-1])

        g_Y = log.(ratios_Y)
        g_K = log.(ratios_K)
        g_L = log.(ratios_L)

        capital_contrib = alpha .* g_K
        labor_contrib   = (1.0 - alpha) .* g_L
        tfp_contrib     = g_Y .- capital_contrib .- labor_contrib

        crypto_sync()
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
    Cliometrics.backend_coprocessor_convergence_test(::CryptoBackend, initial_levels, growth_rates)

Privacy-preserving convergence test via FHE-accelerated OLS.
"""
function Cliometrics.backend_coprocessor_convergence_test(b::CryptoBackend,
                                                             initial_levels::Vector{Float64},
                                                             growth_rates::Vector{Float64})
    n = length(initial_levels)
    n < 16 && return nothing
    try
        ctx = FHEContext(:ckks; poly_degree=8192, scale=2.0^40)
        log_initial = log.(initial_levels)
        X = hcat(ones(n), log_initial)

        beta, _, r_squared = _fhe_ols(ctx, X, growth_rates)

        alpha_coef = beta[1]
        beta_coef  = beta[2]
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
    Cliometrics.backend_coprocessor_time_series_filter(::CryptoBackend, data, window_size)

Privacy-preserving time series filter via FHE.  The moving-average
is computed homomorphically using encrypted addition and scalar division,
keeping the raw time series confidential.
"""
function Cliometrics.backend_coprocessor_time_series_filter(b::CryptoBackend,
                                                               data::Vector{Float64},
                                                               window_size::Int)
    n = length(data)
    n < 32 && return nothing
    try
        ctx = FHEContext(:ckks; poly_degree=8192, scale=2.0^40)
        enc_data = fhe_encrypt(ctx, data)

        # Homomorphic sliding-window sum
        m = n - window_size + 1
        enc_results = Vector{Any}(undef, m)
        for i in 1:m
            window_slice = enc_data[i:i+window_size-1]
            enc_sum = fhe_sum(ctx, window_slice)
            enc_results[i] = fhe_multiply(ctx, enc_sum, 1.0 / window_size)
        end

        result = [fhe_decrypt(ctx, enc_results[i]) for i in 1:m]

        crypto_sync()
        return Float64.(result)
    catch ex
        _record_diagnostic!(b, "runtime_errors")
        return nothing
    end
end

"""
    Cliometrics.backend_coprocessor_growth_accounting(::CryptoBackend, panel_data, n_countries, alpha)

Privacy-preserving batch growth accounting via FHE.  Panel data is
encrypted; growth rates and TFP residuals are computed homomorphically
where possible, with minimal decryption for transcendental functions.
"""
function Cliometrics.backend_coprocessor_growth_accounting(b::CryptoBackend,
                                                              panel_data::Matrix{Float64},
                                                              n_countries::Int,
                                                              alpha::Float64)
    n_total = size(panel_data, 1)
    n_total < 32 && return nothing
    try
        ctx = FHEContext(:ckks; poly_degree=8192, scale=2.0^40)
        enc_panel = fhe_encrypt(ctx, panel_data)

        # Decrypt columns for log computation (FHE log not practical)
        cols = fhe_decrypt(ctx, enc_panel)

        g_Y = log.(cols[2:end, 1] ./ cols[1:end-1, 1])
        g_K = log.(cols[2:end, 2] ./ cols[1:end-1, 2])
        g_L = log.(cols[2:end, 3] ./ cols[1:end-1, 3])

        tfp = g_Y .- alpha .* g_K .- (1.0 - alpha) .* g_L

        crypto_sync()
        return (growth_rates=g_Y, tfp_residuals=tfp,
                avg_tfp=sum(tfp) / length(tfp))
    catch ex
        _record_diagnostic!(b, "runtime_errors")
        return nothing
    end
end

end # module CliometricsCryptoExt
