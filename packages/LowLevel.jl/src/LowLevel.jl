# SPDX-License-Identifier: PMPL-1.0-or-later
module LowLevel

# LowLevel.jl is now the Meta-Orchestrator for the subdivided Metal Layer.

include("SiliconCore.jl/src/SiliconCore.jl")
include("HardwareResilience.jl/src/HardwareResilience.jl")
# (Other includes follow as implemented)

export SiliconCore, HardwareResilience

"""
    peak_performance_op(a, b)
Example of a coordinated operation using the subdivided stack.
"""
function peak_performance_op(a, b)
    g = HardwareResilience.KernelGuardian("Global-Op", :Healthy)
    return HardwareResilience.monitor_kernel(g, () -> begin
        SiliconCore.vector_add_asm(a, b)
    end)
end

end # module
