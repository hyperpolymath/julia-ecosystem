# SPDX-License-Identifier: PMPL-1.0-or-later
# Copyright (c) 2026 Jonathan D.A. Jewell (hyperpolymath) <j.d.a.jewell@open.ac.uk>
#
# PolyglotFormalisms.jl PPU Coprocessor
# Physics-simulation-based parallel evaluation.

function AcceleratorGate.device_capabilities(b::PPUBackend)
    AcceleratorGate.DeviceCapabilities(
        b, 64, 500,
        Int64(4 * 1024^3), Int64(3 * 1024^3),
        256, true, true, false, "NVIDIA", "PPU PhysX",
    )
end

function AcceleratorGate.estimate_cost(::PPUBackend, op::Symbol, data_size::Int)
    overhead = 300.0
    op == :map_parallel && return overhead + Float64(data_size) * 0.05
    op == :fold_parallel && return overhead + Float64(data_size) * 0.06
    Inf
end

AcceleratorGate.register_operation!(PPUBackend, :map_parallel)
AcceleratorGate.register_operation!(PPUBackend, :fold_parallel)

function backend_coprocessor_map_parallel(::PPUBackend, f, collections::AbstractVector)
    [map(f, coll) for coll in collections]
end

function backend_coprocessor_fold_parallel(::PPUBackend, f, init, collections::AbstractVector)
    results = similar(collections, typeof(init))
    for (i, coll) in enumerate(collections)
        results[i] = foldl(f, coll; init=init)
    end
    results
end
