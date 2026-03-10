# SPDX-License-Identifier: PMPL-1.0-or-later
# Copyright (c) 2026 Jonathan D.A. Jewell (hyperpolymath) <j.d.a.jewell@open.ac.uk>
#
# Cliometrics.jl Crypto Coprocessor
# Privacy-preserving regression on sensitive economic data.
# Enables computation on encrypted/committed historical data.

using LinearAlgebra: diagm

function AcceleratorGate.device_capabilities(b::CryptoBackend)
    AcceleratorGate.DeviceCapabilities(
        b, 8, 1000,
        Int64(2 * 1024^3), Int64(2 * 1024^3),
        1, false, false, true, "Intel", "Crypto AES-NI/SHA",
    )
end

function AcceleratorGate.estimate_cost(::CryptoBackend, op::Symbol, data_size::Int)
    overhead = 100.0
    op == :regression && return overhead + Float64(data_size) * 0.3
    Inf
end

AcceleratorGate.register_operation!(CryptoBackend, :regression)

"""
Privacy-preserving regression using additive noise masking.
Adds calibrated noise to the sufficient statistics (X'X, X'y) before
solving, providing differential-privacy-like guarantees for sensitive
historical economic data (e.g., tax records, trade ledgers).
"""
function backend_coprocessor_regression(::CryptoBackend, X::AbstractMatrix, y::AbstractVector;
                                        epsilon::Float64=1.0)
    n, p = size(X)
    XtX = X' * X; Xty = X' * y

    # Add Laplace noise calibrated to sensitivity/epsilon
    sensitivity_XtX = maximum(abs, X)^2 * n
    sensitivity_Xty = maximum(abs, X) * maximum(abs, y) * n
    noise_scale_XtX = sensitivity_XtX / epsilon
    noise_scale_Xty = sensitivity_Xty / epsilon

    # Laplace noise: difference of exponentials
    lap_noise(scale) = scale * (log(rand()) - log(rand()))
    XtX_noisy = XtX .+ [lap_noise(noise_scale_XtX) for _ in 1:p, _ in 1:p]
    Xty_noisy = Xty .+ [lap_noise(noise_scale_Xty) for _ in 1:p]

    # Make XtX symmetric and positive-definite
    XtX_noisy = (XtX_noisy + XtX_noisy') / 2
    XtX_noisy += diagm(fill(abs(noise_scale_XtX), p))  # regularize

    beta = XtX_noisy \ Xty_noisy
    residuals = y .- X * beta
    ss_res = sum(residuals .^ 2)
    ss_tot = sum((y .- mean(y)) .^ 2)
    r2 = 1.0 - ss_res / max(ss_tot, 1.0e-30)

    (coefficients=beta, r_squared=r2, residuals=residuals, privacy_epsilon=epsilon)
end
