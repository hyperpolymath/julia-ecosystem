# SPDX-License-Identifier: PMPL-1.0-or-later
# Copyright (c) 2026 Jonathan D.A. Jewell (hyperpolymath) <j.d.a.jewell@open.ac.uk>
#
# LowLevel.jl — Meta-Orchestrator for the subdivided Metal Layer.
# Coordinates SiliconCore (hardware detection) and HardwareResilience
# (fault-tolerant kernel execution) as proper package dependencies.

"""
    LowLevel

Low-level system programming utilities bridging SiliconCore (hardware detection)
and HardwareResilience (fault-tolerant execution). Serves as the meta-orchestrator
for the Metal Layer, coordinating CPU feature detection with supervised kernel
execution under a guardian monitor.

# Key Features
- Re-exports `CpuFeatures`, `detect_cpu_features`, and `has_feature` from SiliconCore
- Re-exports `KernelGuardian` and `monitor_kernel` from HardwareResilience
- `peak_performance_op` for guardian-monitored vector operations

# Example
```julia
using LowLevel
result = peak_performance_op([1.0, 2.0], [3.0, 4.0])  # [4.0, 6.0]
```
"""
module LowLevel

using SiliconCore
using SiliconCore: CpuFeatures, detect_cpu_features, has_feature, detect_arch, vector_add_asm
using HardwareResilience
using HardwareResilience: KernelGuardian, monitor_kernel

export SiliconCore, HardwareResilience

# Re-export key types and functions for convenience
export CpuFeatures, detect_cpu_features, has_feature
export detect_arch, vector_add_asm
export KernelGuardian, monitor_kernel

"""
    peak_performance_op(a, b)

Coordinated operation using the subdivided Metal Layer stack.
Wraps `SiliconCore.vector_add_asm` in a `HardwareResilience.KernelGuardian`
for fault-tolerant execution with automatic health monitoring.

# Arguments
- `a`: First operand (array or broadcastable)
- `b`: Second operand (array or broadcastable)

# Returns
The element-wise sum `a .+ b`, computed under kernel guardian monitoring.
"""
function peak_performance_op(a, b)
    g = KernelGuardian("Global-Op", :Healthy)
    return monitor_kernel(g, () -> begin
        vector_add_asm(a, b)
    end)
end

export peak_performance_op

end # module
