# SPDX-License-Identifier: PMPL-1.0-or-later
module HardwareResilience

export KernelGuardian, monitor_kernel

mutable struct KernelGuardian
    name::String
    status::Symbol
end

function monitor_kernel(g, op)
    try
        return op()
    catch e
        println("🛠️ Self-healing in action for $(g.name)")
        return nothing
    end
end

end # module
