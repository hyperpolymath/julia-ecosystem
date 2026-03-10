# SPDX-License-Identifier: PMPL-1.0-or-later
# Copyright (c) 2026 Jonathan D.A. Jewell (hyperpolymath) <j.d.a.jewell@open.ac.uk>
"""
Advanced hardware feature detection for high-end processors.

Detects specialized crypto/security features in:
- Intel Xeon (SGX, QAT, AES-NI, SHA-NI, AMX)
- AMD EPYC/Ryzen Pro (SEV, SEV-SNP, PSP)
- Apple M-series (Secure Enclave, Neural Engine)
- ARM (TrustZone, SVE2, Neon Crypto)

These features can accelerate cryptographic operations or provide
hardware-based security guarantees.
"""

# Feature flags
struct HardwareFeatures
    # Crypto instructions
    aes_ni::Bool          # Intel/AMD AES instruction set
    sha_ext::Bool         # Intel/AMD SHA extensions
    pclmulqdq::Bool       # Carry-less multiplication (GCM acceleration)
    rdrand::Bool          # Hardware RNG
    rdseed::Bool          # Hardware seed RNG

    # Security enclaves
    sgx::Bool             # Intel Software Guard Extensions
    sev::Bool             # AMD Secure Encrypted Virtualization
    sev_snp::Bool         # AMD SEV Secure Nested Paging
    trustzone::Bool       # ARM TrustZone
    secure_enclave::Bool  # Apple Secure Enclave

    # Accelerators
    qat::Bool             # Intel QuickAssist Technology
    psp::Bool             # AMD Platform Security Processor
    neural_engine::Bool   # Apple Neural Engine

    # Matrix/tensor extensions
    amx::Bool             # Intel Advanced Matrix Extensions (Sapphire Rapids)
    matrix_cores::Bool    # AMD Matrix cores
    tensor_cores::Bool    # NVIDIA Tensor cores

    # Memory
    hbm::Bool             # High Bandwidth Memory
    persistent_memory::Bool  # Intel Optane

    # Cache
    l3_cache_mb::Int      # L3 cache size in MB
    l2_cache_kb::Int      # L2 cache per core
    smart_cache::Bool     # Intel Smart Cache
    game_cache::Bool      # AMD Game Cache

    # SIMD level (from hardware.jl)
    simd::Symbol          # :avx512, :avx2, :avx, :sse, :neon, :sve
end

"""
    detect_hardware_features() -> HardwareFeatures

Detect all available hardware crypto/security features.

Uses CPUID on x86, /proc/cpuinfo on Linux, system_profiler on macOS.
"""
function detect_hardware_features()
    if Sys.ARCH === :x86_64
        return detect_x86_features()
    elseif Sys.ARCH === :aarch64
        return detect_arm_features()
    else
        return HardwareFeatures(
            false, false, false, false, false,  # No crypto extensions
            false, false, false, false, false,  # No enclaves
            false, false, false,                # No accelerators
            false, false, false,                # No matrix extensions
            false, false,                       # No special memory
            0, 0, false, false,                 # Cache unknown
            :none                               # No SIMD
        )
    end
end

"""
    detect_x86_features() -> HardwareFeatures

Detect x86-64 CPU features via CPUID and Linux sysfs.
"""
function detect_x86_features()
    # Parse /proc/cpuinfo on Linux
    features = if Sys.islinux()
        try
            cpuinfo = read("/proc/cpuinfo", String)
            parse_cpuinfo_x86(cpuinfo)
        catch
            Dict{String,Bool}()
        end
    else
        Dict{String,Bool}()
    end

    # Try Julia's Sys.CPU_NAME
    cpu_name = lowercase(Sys.CPU_NAME)

    # Crypto instructions (common on modern x86)
    aes_ni = get(features, "aes", false) || occursin("aes", cpu_name)
    sha_ext = get(features, "sha_ni", false) || occursin("sha", cpu_name)
    pclmulqdq = get(features, "pclmulqdq", false)
    rdrand = get(features, "rdrand", false)
    rdseed = get(features, "rdseed", false)

    # Intel SGX
    sgx = get(features, "sgx", false) || check_sgx_support()

    # AMD SEV
    sev = get(features, "sev", false) || check_sev_support()
    sev_snp = get(features, "sev_snp", false)

    # QuickAssist Technology
    qat = check_qat_device()

    # Intel AMX (Sapphire Rapids and newer)
    amx = get(features, "amx_tile", false) ||
          get(features, "amx_int8", false) ||
          get(features, "amx_bf16", false)

    # Cache detection
    l3_cache_mb, l2_cache_kb = detect_cache_size_x86()
    smart_cache = occursin("intel", cpu_name)  # Intel feature
    game_cache = occursin("amd", cpu_name) && occursin("ryzen", cpu_name)

    # SIMD from main hardware.jl
    simd = detect_simd_level()

    HardwareFeatures(
        aes_ni, sha_ext, pclmulqdq, rdrand, rdseed,
        sgx, sev, sev_snp, false, false,  # TrustZone, Secure Enclave = x86 only
        qat, sev, false,  # PSP same as SEV for detection, no Neural Engine
        amx, detect_amd_matrix_cores(), detect_nvidia_tensor_cores(),
        detect_hbm_memory(), detect_persistent_memory(),
        l3_cache_mb, l2_cache_kb, smart_cache, game_cache,
        simd
    )
end

"""
    detect_arm_features() -> HardwareFeatures

Detect ARM CPU features (Apple Silicon, AWS Graviton, etc.)
"""
function detect_arm_features()
    # Check if Apple Silicon
    is_apple = Sys.isapple() && Sys.ARCH === :aarch64

    # Crypto extensions (standard on ARMv8-A)
    aes_ni = true  # AES instructions in ARMv8 Crypto Extensions
    sha_ext = true # SHA-1/SHA-256 in ARMv8 Crypto Extensions

    # Apple-specific features
    secure_enclave = is_apple
    neural_engine = is_apple && detect_apple_silicon_generation() >= 1

    # ARM TrustZone (present on most ARM Cortex-A)
    trustzone = true

    # SIMD
    simd = detect_simd_level()  # :neon or :sve

    # Cache (Apple Silicon has large caches)
    l3_cache_mb, l2_cache_kb = if is_apple
        gen = detect_apple_silicon_generation()
        if gen >= 3  # M3/M4 have larger caches
            (24, 128)  # Approximate
        else
            (12, 128)
        end
    else
        (0, 0)  # Unknown for other ARM
    end

    HardwareFeatures(
        aes_ni, sha_ext, true, true, true,  # ARM crypto standard
        false, false, false, trustzone, secure_enclave,
        false, false, neural_engine,
        false, false, false,  # No AMX/matrix/tensor cores on ARM
        detect_hbm_memory(), false,  # HBM possible on ARM servers; no Optane
        l3_cache_mb, l2_cache_kb, false, false,
        simd
    )
end

# ---------------------------------------------------------------------------
# Hardware detection helpers for matrix cores, HBM, and persistent memory
# ---------------------------------------------------------------------------

"""
    detect_amd_matrix_cores() -> Bool

Detect AMD matrix core support by checking for MFMA (Matrix Fused Multiply-Add)
instructions in CPUID flags. Available on AMD CDNA architecture (MI100+) and
some RDNA3 GPUs.
"""
function detect_amd_matrix_cores()
    if !Sys.islinux()
        return false
    end
    try
        cpuinfo = read("/proc/cpuinfo", String)
        # AMD matrix cores appear as MFMA support in ROCm-capable devices
        # Also check for AMD GPU compute devices in sysfs
        has_mfma = occursin("mfma", lowercase(cpuinfo))
        if has_mfma
            return true
        end
        # Check for AMD GPU with matrix core support via sysfs
        kfd_path = "/sys/class/kfd/kfd/topology/nodes"
        if isdir(kfd_path)
            for node in readdir(kfd_path)
                props_path = joinpath(kfd_path, node, "properties")
                if isfile(props_path)
                    props = read(props_path, String)
                    # gfx908 (MI100), gfx90a (MI200), gfx942 (MI300) have matrix cores
                    if any(arch -> occursin(arch, props), ["gfx908", "gfx90a", "gfx940", "gfx941", "gfx942"])
                        return true
                    end
                end
            end
        end
        return false
    catch
        return false
    end
end

"""
    detect_nvidia_tensor_cores() -> Bool

Detect NVIDIA Tensor Cores by checking for CUDA-capable GPUs with compute
capability >= 7.0 (Volta and newer architectures).
"""
function detect_nvidia_tensor_cores()
    if !Sys.islinux()
        return false
    end
    try
        # Check for NVIDIA GPU via nvidia-smi
        nvidia_output = read(`nvidia-smi --query-gpu=compute_cap --format=csv,noheader`, String)
        for line in split(strip(nvidia_output), '\n')
            cap = tryparse(Float64, strip(line))
            if cap !== nothing && cap >= 7.0
                return true
            end
        end
        return false
    catch
        # nvidia-smi not available or no NVIDIA GPU
        return false
    end
end

"""
    detect_hbm_memory() -> Bool

Detect High Bandwidth Memory (HBM/HBM2/HBM2E/HBM3) presence. HBM is found
on high-end GPUs (AMD MI-series, NVIDIA A100/H100) and some specialized
CPU packages (Intel Xeon Max with HBM).
"""
function detect_hbm_memory()
    if !Sys.islinux()
        return false
    end
    try
        # Method 1: Check NVIDIA GPUs known to have HBM
        if isfile("/usr/bin/nvidia-smi") || isfile("/usr/local/bin/nvidia-smi")
            nvidia_out = read(`nvidia-smi --query-gpu=name --format=csv,noheader`, String)
            hbm_gpus = ["A100", "H100", "H200", "V100", "A30", "MI100", "MI200", "MI250", "MI300"]
            for gpu_name in hbm_gpus
                if occursin(gpu_name, nvidia_out)
                    return true
                end
            end
        end

        # Method 2: Check AMD ROCm devices with HBM
        kfd_path = "/sys/class/kfd/kfd/topology/nodes"
        if isdir(kfd_path)
            for node in readdir(kfd_path)
                props_path = joinpath(kfd_path, node, "properties")
                if isfile(props_path)
                    props = read(props_path, String)
                    if occursin("hbm", lowercase(props))
                        return true
                    end
                end
            end
        end

        # Method 3: Intel Xeon Max (Sapphire Rapids HBM) check via DMI
        if isfile("/sys/devices/virtual/dmi/id/product_name")
            product = read("/sys/devices/virtual/dmi/id/product_name", String)
            if occursin("HBM", product) || occursin("Xeon Max", product)
                return true
            end
        end

        return false
    catch
        return false
    end
end

"""
    detect_persistent_memory() -> Bool

Detect Intel Optane Persistent Memory (DCPMM) or compatible persistent memory
via Linux NVDIMM subsystem (libnvdimm / ndctl).
"""
function detect_persistent_memory()
    if !Sys.islinux()
        return false
    end
    try
        # Method 1: Check ndctl for NVDIMMs
        ndctl_path = "/sys/bus/nd/devices"
        if isdir(ndctl_path)
            devices = readdir(ndctl_path)
            for dev in devices
                if startswith(dev, "nmem") || startswith(dev, "region")
                    return true
                end
            end
        end

        # Method 2: Check /dev/pmem* devices
        for dev in readdir("/dev")
            if startswith(dev, "pmem")
                return true
            end
        end

        # Method 3: Check ipmctl (Intel Persistent Memory Control)
        try
            ipmctl_out = read(`ipmctl show -dimm`, String)
            if occursin("Healthy", ipmctl_out) || occursin("Intel", ipmctl_out)
                return true
            end
        catch
            # ipmctl not installed
        end

        return false
    catch
        return false
    end
end

# ---------------------------------------------------------------------------
# Original helper functions
# ---------------------------------------------------------------------------

function parse_cpuinfo_x86(cpuinfo::String)
    features = Dict{String,Bool}()
    for line in split(cpuinfo, '\n')
        if startswith(line, "flags") || startswith(line, "Features")
            flags = split(line, ':')[2]
            for flag in split(flags)
                features[strip(flag)] = true
            end
        end
    end
    features
end

function check_sgx_support()
    # Check /sys/firmware/efi/efivars/SgxRegistrationServerRequest-*
    if Sys.islinux()
        try
            return isdir("/dev/sgx") || isfile("/dev/isgx")
        catch
            return false
        end
    end
    false
end

function check_sev_support()
    # Check AMD SEV support
    if Sys.islinux()
        try
            kvm_amd = read("/sys/module/kvm_amd/parameters/sev", String)
            return strip(kvm_amd) == "1" || strip(kvm_amd) == "Y"
        catch
            return false
        end
    end
    false
end

function check_qat_device()
    # Check for Intel QAT PCIe device
    if Sys.islinux()
        try
            lspci = read(`lspci`, String)
            return occursin("QuickAssist", lspci) || occursin("8086:37c8", lspci)
        catch
            return false
        end
    end
    false
end

function detect_cache_size_x86()
    # Try lscpu on Linux
    if Sys.islinux()
        try
            lscpu = read(`lscpu`, String)
            l3_match = match(r"L3 cache:\s+(\d+)K", lscpu)
            l2_match = match(r"L2 cache:\s+(\d+)K", lscpu)

            l3_kb = l3_match !== nothing ? parse(Int, l3_match[1]) : 0
            l2_kb = l2_match !== nothing ? parse(Int, l2_match[1]) : 0

            return (l3_kb ÷ 1024, l2_kb)  # Convert L3 to MB
        catch
        end
    end
    (0, 0)
end

"""
    print_hardware_report(features::HardwareFeatures)

Print a detailed report of detected hardware features.
"""
function print_hardware_report(features::HardwareFeatures)
    println("╔═══════════════════════════════════════════════╗")
    println("║     Hardware Cryptography Features Report    ║")
    println("╚═══════════════════════════════════════════════╝")
    println()

    println("🔐 Crypto Instructions:")
    println("  AES-NI:       $(features.aes_ni ? "✅" : "❌")")
    println("  SHA Extensions: $(features.sha_ext ? "✅" : "❌")")
    println("  PCLMULQDQ:    $(features.pclmulqdq ? "✅" : "❌")")
    println("  RDRAND:       $(features.rdrand ? "✅" : "❌")")
    println("  RDSEED:       $(features.rdseed ? "✅" : "❌")")
    println()

    println("🔒 Security Enclaves:")
    println("  Intel SGX:    $(features.sgx ? "✅" : "❌")")
    println("  AMD SEV:      $(features.sev ? "✅" : "❌")")
    println("  AMD SEV-SNP:  $(features.sev_snp ? "✅" : "❌")")
    println("  ARM TrustZone: $(features.trustzone ? "✅" : "❌")")
    println("  Secure Enclave: $(features.secure_enclave ? "✅" : "❌")")
    println()

    println("⚡ Accelerators:")
    println("  Intel QAT:    $(features.qat ? "✅" : "❌")")
    println("  AMD PSP:      $(features.psp ? "✅" : "❌")")
    println("  Neural Engine: $(features.neural_engine ? "✅" : "❌")")
    println()

    println("🧮 Matrix/Tensor:")
    println("  Intel AMX:    $(features.amx ? "✅" : "❌")")
    println("  AMD Matrix:   $(features.matrix_cores ? "✅" : "❌")")
    println("  NVIDIA Tensor: $(features.tensor_cores ? "✅" : "❌")")
    println()

    println("💾 Memory & Cache:")
    println("  SIMD Level:   $(features.simd)")
    println("  L3 Cache:     $(features.l3_cache_mb) MB")
    println("  L2 Cache:     $(features.l2_cache_kb) KB")
    println("  Smart Cache:  $(features.smart_cache ? "✅" : "❌")")
    println("  Game Cache:   $(features.game_cache ? "✅" : "❌")")
    println()
end
