# SPDX-License-Identifier: PMPL-1.0-or-later
# Copyright (c) 2026 Jonathan D.A. Jewell (hyperpolymath) <j.d.a.jewell@open.ac.uk>
#
# ProvenCrypto AMDGPU Extension (compatibility shim)
#
# This extension provides the AMDGPU.jl package trigger entry point.
# The actual GPU kernel implementations for AMD GPUs live in
# ProvenCryptoROCmExt.jl, which is also triggered by AMDGPU.jl.
#
# This extension ONLY provides the backend creation helper and availability
# check. It does NOT redefine the NTT/lattice/sampling methods -- those are
# all defined in ProvenCryptoROCmExt to avoid method redefinition conflicts.

module ProvenCryptoAMDGPUExt

using AMDGPU
using ..ProvenCrypto

# ============================================================================
# Backend availability and creation (alias for ROCm)
# ============================================================================

# Note: ProvenCryptoROCmExt also sets rocm_available() = true, which is fine
# since both are triggered by the same AMDGPU weakdep.
ProvenCrypto.rocm_available() = AMDGPU.functional()

"""
    create_amdgpu_backend() -> ProvenCrypto.ROCmBackend

Create a ROCm backend for AMD GPUs. This is an alias entry point;
the actual GPU kernels are provided by ProvenCryptoROCmExt.
"""
function ProvenCrypto.create_amdgpu_backend()
    device_id = 0
    has_matrix = false
    gcn_arch = "unknown"
    try
        agent = AMDGPU.get_default_agent()
        gcn_arch = string(AMDGPU.device_id(agent))
        has_matrix = occursin(r"gfx9[0-9]{2}", gcn_arch)
    catch
        # Fallback if device query fails
    end
    return ProvenCrypto.ROCmBackend(device_id, has_matrix, gcn_arch)
end

# All backend_ntt_transform, backend_ntt_inverse_transform,
# backend_lattice_multiply, backend_polynomial_multiply, and
# backend_sampling methods for ROCmBackend are defined in
# ProvenCryptoROCmExt.jl to avoid duplicate method definitions.

end # module ProvenCryptoAMDGPUExt
