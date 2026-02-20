# SPDX-License-Identifier: PMPL-1.0-or-later
module Hardware

export Architecture, X86_64, ARM64, detect_arch, select_kernel

abstract type Architecture end
struct X86_64 <: Architecture end
struct ARM64 <: Architecture end
struct UnknownArch <: Architecture end

"""
    detect_arch()
Returns the CPU architecture of the current machine.
"""
function detect_arch()
    if Sys.ARCH === :x86_64
        return X86_64()
    elseif Sys.ARCH === :aarch64 || Sys.ARCH === :arm64
        return ARM64()
    else
        return UnknownArch()
    end
end

"""
    select_kernel(x86_func, arm_func, fallback_func)
Dispatches to the correct low-level kernel based on hardware.
"""
function select_kernel(x86_func::Function, arm_func::Function, fallback_func::Function)
    arch = detect_arch()
    if arch isa X86_64
        return x86_func
    elseif arch isa ARM64
        return arm_func
    else
        return fallback_func
    end
end

end # module
