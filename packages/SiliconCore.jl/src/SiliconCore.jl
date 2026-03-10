# SPDX-License-Identifier: PMPL-1.0-or-later
# Copyright (c) 2026 Jonathan D.A. Jewell (hyperpolymath) <j.d.a.jewell@open.ac.uk>
#
# SiliconCore.jl — Cross-platform hardware detection for CPU features
# Supports Linux (/proc/cpuinfo + sysfs), macOS (sysctl), Windows (WMI/env),
# FreeBSD/BSD (sysctl), and architecture-specific detection for x86_64,
# aarch64/ARM, RISC-V, PowerPC, and MIPS.
#
# Provides structured CPU feature information for dispatch decisions in the
# Metal Layer stack (LowLevel.jl, AcceleratorGate.jl, etc.)

"""
    SiliconCore

Cross-platform CPU feature detection and hardware capability analysis. Probes
Linux, macOS, Windows, and BSD systems for SIMD instruction sets, cache hierarchy,
core topology, and platform classification across x86_64, aarch64, RISC-V, and
PowerPC architectures.

# Key Features
- Structured `CpuFeatures` with SIMD flags, cache sizes, and core counts
- Architecture-specific detection: AVX/AVX-512/AMX (x86), NEON/SVE/SME (ARM), RVV (RISC-V)
- Platform classification (desktop, server, mobile, embedded)
- Never throws -- returns partial information on detection failure

# Example
```julia
using SiliconCore
features = detect_cpu_features()
has_feature(features, :avx2)
```
"""
module SiliconCore

export detect_arch, vector_add_asm
export CpuFeatures, detect_cpu_features, has_feature
export detect_os, detect_platform

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

"""
    CpuFeatures

Structured representation of CPU capabilities detected at runtime.
Cross-platform: works on Linux, macOS, Windows, FreeBSD, and other BSDs.

# Fields — x86_64 SIMD
- `arch::Symbol`: CPU architecture (:x86_64, :aarch64, :arm, :riscv64, :powerpc64le, etc.)
- `vendor::String`: CPU vendor string (e.g. "GenuineIntel", "AuthenticAMD", "Apple")
- `model_name::String`: Full CPU model string
- `has_sse::Bool`: SSE instruction set (x86)
- `has_sse2::Bool`: SSE2 instruction set (x86)
- `has_avx::Bool`: AVX instruction set (x86)
- `has_avx2::Bool`: AVX2 instruction set (x86)
- `has_avx512f::Bool`: AVX-512 Foundation (x86)
- `has_avx512vl::Bool`: AVX-512 Vector Length Extensions (x86)
- `has_avx512bw::Bool`: AVX-512 Byte and Word Instructions (x86)
- `has_amx::Bool`: Advanced Matrix Extensions (x86, Intel)
- `has_aesni::Bool`: AES-NI hardware encryption (x86)

# Fields — ARM SIMD/Vector
- `has_neon::Bool`: NEON SIMD (ARM)
- `has_sve::Bool`: Scalable Vector Extension (ARM)
- `has_sve2::Bool`: Scalable Vector Extension 2 (ARM)
- `has_sme::Bool`: Scalable Matrix Extension (ARM)

# Fields — RISC-V Vector
- `has_rvv::Bool`: RISC-V Vector extension (any version)
- `has_rvv_1_0::Bool`: RISC-V Vector 1.0

# Fields — PowerPC Vector
- `has_altivec::Bool`: PowerPC AltiVec / VMX
- `has_vsx::Bool`: PowerPC VSX (Vector Scalar Extension)

# Fields — Cache hierarchy
- `cache_line_size::Int`: L1 cache line size in bytes (0 if unknown)
- `l1_cache_size::Int`: L1 data cache size in bytes (0 if unknown)
- `l2_cache_size::Int`: L2 cache size in bytes (0 if unknown)
- `l3_cache_size::Int`: L3 cache size in bytes (0 if unknown)

# Fields — Topology
- `num_cores::Int`: Number of physical CPU cores
- `num_threads::Int`: Number of hardware threads (includes SMT/HT)

# Fields — Platform
- `os::Symbol`: Detected OS (:linux, :macos, :windows, :freebsd, :openbsd, :netbsd, :minix, :unknown)
- `platform::Symbol`: Platform class (:desktop, :server, :mobile, :embedded, :unknown)
"""
struct CpuFeatures
    arch::Symbol
    vendor::String
    model_name::String
    # x86_64 SIMD
    has_sse::Bool
    has_sse2::Bool
    has_avx::Bool
    has_avx2::Bool
    has_avx512f::Bool
    has_avx512vl::Bool
    has_avx512bw::Bool
    has_amx::Bool
    has_aesni::Bool
    # ARM SIMD/Vector
    has_neon::Bool
    has_sve::Bool
    has_sve2::Bool
    has_sme::Bool
    # RISC-V Vector
    has_rvv::Bool
    has_rvv_1_0::Bool
    # PowerPC Vector
    has_altivec::Bool
    has_vsx::Bool
    # Cache hierarchy
    cache_line_size::Int
    l1_cache_size::Int
    l2_cache_size::Int
    l3_cache_size::Int
    # Topology
    num_cores::Int
    num_threads::Int
    # Platform
    os::Symbol
    platform::Symbol
end

# ---------------------------------------------------------------------------
# OS and Platform Detection
# ---------------------------------------------------------------------------

"""
    detect_os() -> Symbol

Detect the running operating system. Returns one of:
:linux, :macos, :windows, :freebsd, :openbsd, :netbsd, :minix, :android, :unknown

Never throws — returns :unknown on failure.
"""
function detect_os()::Symbol
    # Check Android before Linux (Android is Linux-based)
    if Sys.islinux()
        if haskey(ENV, "ANDROID_ROOT") || isfile("/system/build.prop")
            return :android
        end
        return :linux
    end
    Sys.isapple() && return :macos
    Sys.iswindows() && return :windows
    if Sys.isbsd()
        # Distinguish BSD variants
        try
            uname = strip(read(`uname -s`, String))
            uname == "FreeBSD" && return :freebsd
            uname == "OpenBSD" && return :openbsd
            uname == "NetBSD" && return :netbsd
        catch
            return :freebsd  # Default BSD assumption
        end
    end
    # Check for MINIX and other exotic systems
    try
        uname = strip(read(`uname -s`, String))
        lowercase(uname) == "minix" && return :minix
    catch end
    return :unknown
end

"""
    detect_platform() -> Symbol

Heuristic platform classification. Returns one of:
:desktop, :server, :mobile, :embedded, :unknown

Uses core count, OS, architecture, and environment hints.
"""
function detect_platform()::Symbol
    os = detect_os()

    # Mobile platforms
    os === :android && return :mobile
    if os === :macos && Sys.ARCH === :aarch64
        # Could be macOS on Apple Silicon or iOS via Catalyst
        # Check for mobile-specific indicators
        try
            model = strip(read(`sysctl -n hw.model`, String))
            # iPhone/iPad model identifiers start with "iPhone" or "iPad"
            if startswith(model, "iPhone") || startswith(model, "iPad") || startswith(model, "iPod")
                return :mobile
            end
        catch end
    end

    # Embedded: typically low core count ARM/RISC-V without desktop indicators
    if Sys.ARCH in (:arm, :armv7l) && Sys.CPU_THREADS <= 4
        if !haskey(ENV, "DISPLAY") && !haskey(ENV, "WAYLAND_DISPLAY")
            return :embedded
        end
    end

    # Server vs Desktop heuristic
    nthreads = Sys.CPU_THREADS
    if Sys.islinux()
        # Detect virtualisation or cloud (common server indicator)
        try
            dmi = read("/sys/class/dmi/id/chassis_type", String) |> strip
            chassis = tryparse(Int, dmi)
            if chassis !== nothing
                # Chassis types: 17=Rack Mount Server, 23=Rack Mount Chassis,
                # 25=Multi-system Chassis, 28=Main Server Chassis
                chassis in (17, 23, 25, 28) && return :server
                # 3=Desktop, 4=Low Profile Desktop, 6=Mini Tower, 7=Tower,
                # 8=Portable, 9=Laptop, 10=Notebook, 14=Sub Notebook
                chassis in (3, 4, 6, 7, 8, 9, 10, 14) && return :desktop
            end
        catch end
    end

    # Fallback: high thread count suggests server
    nthreads >= 32 && return :server

    return :desktop
end

# ---------------------------------------------------------------------------
# Internal: Safe command execution
# ---------------------------------------------------------------------------

"""
    _safe_read_cmd(cmd; default="") -> String

Run a command and return its stripped output, or `default` on any failure.
"""
function _safe_read_cmd(cmd; default::String="")::String
    try
        return strip(read(cmd, String))
    catch
        return default
    end
end

"""
    _safe_parse_int(s::String; default::Int=0) -> Int

Parse an integer from a string, returning `default` on failure.
"""
function _safe_parse_int(s::String; default::Int=0)::Int
    v = tryparse(Int, strip(s))
    return v === nothing ? default : v
end

# ---------------------------------------------------------------------------
# Internal: /proc/cpuinfo parsing (Linux)
# ---------------------------------------------------------------------------

"""
    _parse_cpuinfo_field(cpuinfo::String, field::String) -> String

Extract the first occurrence of a field value from /proc/cpuinfo text.
Returns an empty string if the field is not found.
"""
function _parse_cpuinfo_field(cpuinfo::String, field::String)::String
    for line in eachline(IOBuffer(cpuinfo))
        stripped = strip(line)
        if startswith(stripped, field)
            parts = split(stripped, ':', limit=2)
            length(parts) == 2 && return strip(parts[2])
        end
    end
    return ""
end

"""
    _parse_cpuinfo_flags(cpuinfo::String) -> Set{String}

Extract the CPU flags/features set from /proc/cpuinfo.
On x86, this is the "flags" line; on ARM, it is "Features".
On RISC-V, it is "isa"; on PowerPC, "cpu" line may contain AltiVec info.
"""
function _parse_cpuinfo_flags(cpuinfo::String)::Set{String}
    flags_str = _parse_cpuinfo_field(cpuinfo, "flags")
    if isempty(flags_str)
        flags_str = _parse_cpuinfo_field(cpuinfo, "Features")
    end
    if isempty(flags_str)
        # RISC-V uses "isa" field with extensions like rv64imafdc_v
        flags_str = _parse_cpuinfo_field(cpuinfo, "isa")
    end
    isempty(flags_str) && return Set{String}()
    return Set(split(flags_str))
end

"""
    _count_physical_cores(cpuinfo::String) -> Int

Count distinct physical cores from /proc/cpuinfo by tracking unique
(physical id, core id) pairs. Falls back to counting "processor" entries
if physical topology info is unavailable.
"""
function _count_physical_cores(cpuinfo::String)::Int
    physical_ids = Int[]
    core_ids = Int[]
    current_phys = -1
    current_core = -1
    processor_count = 0

    for line in eachline(IOBuffer(cpuinfo))
        stripped = strip(line)
        if startswith(stripped, "processor")
            processor_count += 1
            current_phys = -1
            current_core = -1
        elseif startswith(stripped, "physical id")
            parts = split(stripped, ':', limit=2)
            if length(parts) == 2
                current_phys = something(tryparse(Int, strip(parts[2])), -1)
            end
        elseif startswith(stripped, "core id")
            parts = split(stripped, ':', limit=2)
            if length(parts) == 2
                current_core = something(tryparse(Int, strip(parts[2])), -1)
            end
        end
        if current_phys >= 0 && current_core >= 0
            push!(physical_ids, current_phys)
            push!(core_ids, current_core)
            current_phys = -1
            current_core = -1
        end
    end

    if !isempty(physical_ids)
        unique_cores = Set(zip(physical_ids, core_ids))
        return length(unique_cores)
    end

    return max(processor_count, 1)
end

# ---------------------------------------------------------------------------
# Internal: /sys cache hierarchy parsing (Linux)
# ---------------------------------------------------------------------------

"""
    _read_sys_file(path::String) -> String

Safely read a single-line sysfs file. Returns empty string on failure.
"""
function _read_sys_file(path::String)::String
    try
        return strip(read(path, String))
    catch
        return ""
    end
end

"""
    _parse_cache_size(size_str::String) -> Int

Parse a cache size string like "32K", "256K", "8192K", "16M" into bytes.
Returns 0 if parsing fails.
"""
function _parse_cache_size(size_str::String)::Int
    isempty(size_str) && return 0
    size_str = strip(size_str)

    multiplier = 1
    if endswith(size_str, "K")
        multiplier = 1024
        size_str = size_str[1:end-1]
    elseif endswith(size_str, "M")
        multiplier = 1024 * 1024
        size_str = size_str[1:end-1]
    elseif endswith(size_str, "G")
        multiplier = 1024 * 1024 * 1024
        size_str = size_str[1:end-1]
    end

    val = tryparse(Int, strip(size_str))
    val === nothing && return 0
    return val * multiplier
end

"""
    _detect_cache_hierarchy_linux() -> (cache_line::Int, l1::Int, l2::Int, l3::Int)

Read cache sizes from /sys/devices/system/cpu/cpu0/cache/.
Each index directory (index0, index1, ...) has `level`, `type`, and `size` files.
"""
function _detect_cache_hierarchy_linux()
    cache_line = 0
    l1_size = 0
    l2_size = 0
    l3_size = 0

    base = "/sys/devices/system/cpu/cpu0/cache"
    isdir(base) || return (cache_line, l1_size, l2_size, l3_size)

    for entry in readdir(base)
        idx_path = joinpath(base, entry)
        isdir(idx_path) || continue

        level_str = _read_sys_file(joinpath(idx_path, "level"))
        type_str = _read_sys_file(joinpath(idx_path, "type"))
        size_str = _read_sys_file(joinpath(idx_path, "size"))
        line_str = _read_sys_file(joinpath(idx_path, "coherency_line_size"))

        level = tryparse(Int, level_str)
        level === nothing && continue

        size_bytes = _parse_cache_size(size_str)

        if level == 1 && lowercase(type_str) == "data"
            l1_size = size_bytes
            line_val = tryparse(Int, line_str)
            if line_val !== nothing
                cache_line = line_val
            end
        elseif level == 2
            l2_size = size_bytes
        elseif level == 3
            l3_size = size_bytes
        end
    end

    return (cache_line, l1_size, l2_size, l3_size)
end

# ---------------------------------------------------------------------------
# Internal: Feature flag extraction from various sources
# ---------------------------------------------------------------------------

"""
    _extract_features(flags::Set{String}, arch::Symbol) -> NamedTuple

Given a set of feature flag strings and the CPU architecture, return a
NamedTuple with all boolean feature fields.
"""
function _extract_features(flags::Set{String}, arch::Symbol)
    # x86_64 features
    has_sse = "sse" in flags || "SSE" in flags
    has_sse2 = "sse2" in flags || "SSE2" in flags
    has_avx = "avx" in flags || "AVX1.0" in flags || "AVX" in flags
    has_avx2 = "avx2" in flags || "AVX2" in flags
    has_avx512f = "avx512f" in flags || "AVX512F" in flags
    has_avx512vl = "avx512vl" in flags || "AVX512VL" in flags
    has_avx512bw = "avx512bw" in flags || "AVX512BW" in flags
    has_amx = "amx_tile" in flags || "amx_bf16" in flags || "amx_int8" in flags ||
              "AMX_TILE" in flags || "AMX_BF16" in flags || "AMX_INT8" in flags
    has_aesni = "aes" in flags || "AES" in flags || "AESNI" in flags

    # ARM features
    has_neon = "neon" in flags || "asimd" in flags || "NEON" in flags
    has_sve = "sve" in flags || "SVE" in flags
    has_sve2 = "sve2" in flags || "SVE2" in flags
    has_sme = "sme" in flags || "SME" in flags

    # RISC-V features — flags may be individual extension letters or full ISA strings
    has_rvv = false
    has_rvv_1_0 = false
    if arch in (:riscv64, :riscv32)
        for f in flags
            # RISC-V ISA strings like "rv64imafdc_v" or "rv64gcv1p0"
            if occursin("_v", f) || occursin("v1p0", f) || f == "v" || f == "V"
                has_rvv = true
            end
            if occursin("v1p0", f)
                has_rvv_1_0 = true
            end
        end
    end

    # PowerPC features
    has_altivec = "altivec" in flags || "AltiVec" in flags || "ALTIVEC" in flags ||
                  "vmx" in flags || "VMX" in flags
    has_vsx = "vsx" in flags || "VSX" in flags

    return (;
        has_sse, has_sse2, has_avx, has_avx2,
        has_avx512f, has_avx512vl, has_avx512bw,
        has_amx, has_aesni,
        has_neon, has_sve, has_sve2, has_sme,
        has_rvv, has_rvv_1_0,
        has_altivec, has_vsx,
    )
end

"""
    _default_features() -> NamedTuple

Return a NamedTuple with all feature flags set to false.
"""
function _default_features()
    return (;
        has_sse=false, has_sse2=false, has_avx=false, has_avx2=false,
        has_avx512f=false, has_avx512vl=false, has_avx512bw=false,
        has_amx=false, has_aesni=false,
        has_neon=false, has_sve=false, has_sve2=false, has_sme=false,
        has_rvv=false, has_rvv_1_0=false,
        has_altivec=false, has_vsx=false,
    )
end

# ---------------------------------------------------------------------------
# Platform-specific backends
# ---------------------------------------------------------------------------

"""
    _detect_linux() -> CpuFeatures

Detect CPU features on Linux via /proc/cpuinfo and /sys/devices/system/cpu/.
"""
function _detect_linux()::CpuFeatures
    arch = Sys.ARCH
    num_threads = Sys.CPU_THREADS

    cpuinfo = ""
    try
        cpuinfo = read("/proc/cpuinfo", String)
    catch
        # /proc not available (container, restricted env)
        return _make_minimal_features(arch, num_threads, :linux)
    end

    vendor = _parse_cpuinfo_field(cpuinfo, "vendor_id")
    model_name = _parse_cpuinfo_field(cpuinfo, "model name")
    if isempty(model_name)
        model_name = _parse_cpuinfo_field(cpuinfo, "Hardware")
        isempty(model_name) && (model_name = _parse_cpuinfo_field(cpuinfo, "Model"))
    end
    if isempty(model_name)
        # RISC-V uses "uarch" or has no standard model name field
        model_name = _parse_cpuinfo_field(cpuinfo, "uarch")
    end

    flags = _parse_cpuinfo_flags(cpuinfo)
    num_cores = _count_physical_cores(cpuinfo)

    # ARM on Linux: NEON is mandatory on aarch64 but may not be listed in flags
    if arch === :aarch64 && !("asimd" in flags) && !("neon" in flags)
        push!(flags, "asimd")  # aarch64 always has ASIMD (NEON equivalent)
    end

    feat = _extract_features(flags, arch)
    cache_line, l1, l2, l3 = _detect_cache_hierarchy_linux()

    return CpuFeatures(
        arch, vendor, model_name,
        feat.has_sse, feat.has_sse2, feat.has_avx, feat.has_avx2,
        feat.has_avx512f, feat.has_avx512vl, feat.has_avx512bw,
        feat.has_amx, feat.has_aesni,
        feat.has_neon, feat.has_sve, feat.has_sve2, feat.has_sme,
        feat.has_rvv, feat.has_rvv_1_0,
        feat.has_altivec, feat.has_vsx,
        cache_line, l1, l2, l3,
        num_cores, num_threads,
        :linux, detect_platform(),
    )
end

"""
    _detect_macos() -> CpuFeatures

Detect CPU features on macOS via `sysctl` commands.
Handles both Intel Macs and Apple Silicon (aarch64).
"""
function _detect_macos()::CpuFeatures
    arch = Sys.ARCH
    num_threads = Sys.CPU_THREADS

    # Brand / model
    brand = _safe_read_cmd(`sysctl -n machdep.cpu.brand_string`; default="")
    if isempty(brand)
        # Apple Silicon may not have brand_string via machdep.cpu
        brand = _safe_read_cmd(`sysctl -n hw.model`; default="unknown")
    end

    # Vendor
    vendor = if arch === :aarch64
        "Apple"
    else
        _safe_read_cmd(`sysctl -n machdep.cpu.vendor`; default="unknown")
    end

    # Feature flags — Intel Macs expose these via sysctl
    features_str = _safe_read_cmd(`sysctl -n machdep.cpu.features`; default="")
    leaf7_str = _safe_read_cmd(`sysctl -n machdep.cpu.leaf7_features`; default="")
    ext_features = _safe_read_cmd(`sysctl -n machdep.cpu.extfeatures`; default="")

    combined = features_str * " " * leaf7_str * " " * ext_features
    flags = Set(split(strip(combined)))
    # Remove empty strings from the set
    delete!(flags, "")

    # Apple Silicon (aarch64) always has NEON and may have additional features
    if arch === :aarch64
        push!(flags, "NEON")
        # Apple Silicon M1+ typically supports various ARM features;
        # sysctl hw.optional.* can reveal them
        _sysctl_bool("hw.optional.neon") && push!(flags, "NEON")
        _sysctl_bool("hw.optional.arm.FEAT_SVE") && push!(flags, "SVE")
        _sysctl_bool("hw.optional.arm.FEAT_SVE2") && push!(flags, "SVE2")
        _sysctl_bool("hw.optional.arm.FEAT_SME") && push!(flags, "SME")
    end

    feat = _extract_features(flags, arch)

    # Core counts
    num_cores = _safe_parse_int(_safe_read_cmd(`sysctl -n hw.physicalcpu`; default="1"); default=num_threads)
    nthreads_detected = _safe_parse_int(_safe_read_cmd(`sysctl -n hw.logicalcpu`; default="$num_threads"); default=num_threads)

    # Cache sizes (in bytes)
    l1 = _safe_parse_int(_safe_read_cmd(`sysctl -n hw.l1dcachesize`; default="0"))
    l2 = _safe_parse_int(_safe_read_cmd(`sysctl -n hw.l2cachesize`; default="0"))
    l3_str = _safe_read_cmd(`sysctl -n hw.l3cachesize`; default="0")
    l3 = _safe_parse_int(l3_str)
    cache_line = _safe_parse_int(_safe_read_cmd(`sysctl -n hw.cachelinesize`; default="0"))

    return CpuFeatures(
        arch, vendor, brand,
        feat.has_sse, feat.has_sse2, feat.has_avx, feat.has_avx2,
        feat.has_avx512f, feat.has_avx512vl, feat.has_avx512bw,
        feat.has_amx, feat.has_aesni,
        feat.has_neon, feat.has_sve, feat.has_sve2, feat.has_sme,
        feat.has_rvv, feat.has_rvv_1_0,
        feat.has_altivec, feat.has_vsx,
        cache_line, l1, l2, l3,
        num_cores, nthreads_detected,
        :macos, detect_platform(),
    )
end

"""
    _sysctl_bool(key::String) -> Bool

Check a macOS/BSD sysctl boolean key (returns 1 or 0). Returns false on failure.
"""
function _sysctl_bool(key::String)::Bool
    try
        val = strip(read(`sysctl -n $key`, String))
        return val == "1"
    catch
        return false
    end
end

"""
    _detect_windows() -> CpuFeatures

Detect CPU features on Windows via environment variables and PowerShell/WMI.
"""
function _detect_windows()::CpuFeatures
    arch = Sys.ARCH
    num_threads = Sys.CPU_THREADS

    # Basic info from environment
    proc_id = get(ENV, "PROCESSOR_IDENTIFIER", "unknown")
    proc_arch = get(ENV, "PROCESSOR_ARCHITECTURE", "")
    nproc = _safe_parse_int(get(ENV, "NUMBER_OF_PROCESSORS", "1"); default=num_threads)

    vendor = ""
    model_name = proc_id
    num_cores = nproc
    l1 = 0
    l2 = 0
    l3 = 0
    cache_line = 0

    # Attempt detailed detection via PowerShell + WMI
    try
        ps_cmd = `powershell -NoProfile -Command "Get-CimInstance -ClassName Win32_Processor | Select-Object Name,Manufacturer,NumberOfCores,NumberOfLogicalProcessors,L2CacheSize,L3CacheSize | ConvertTo-Json"`
        json_str = strip(read(ps_cmd, String))
        if !isempty(json_str)
            info = _parse_windows_wmi_json(json_str)
            if haskey(info, "Name")
                model_name = info["Name"]
            end
            if haskey(info, "Manufacturer")
                vendor = info["Manufacturer"]
            end
            if haskey(info, "NumberOfCores")
                nc = tryparse(Int, string(info["NumberOfCores"]))
                nc !== nothing && (num_cores = nc)
            end
            if haskey(info, "NumberOfLogicalProcessors")
                nt = tryparse(Int, string(info["NumberOfLogicalProcessors"]))
                nt !== nothing && (nproc = nt)
            end
            if haskey(info, "L2CacheSize")
                l2v = tryparse(Int, string(info["L2CacheSize"]))
                l2v !== nothing && (l2 = l2v * 1024)  # WMI reports in KB
            end
            if haskey(info, "L3CacheSize")
                l3v = tryparse(Int, string(info["L3CacheSize"]))
                l3v !== nothing && (l3 = l3v * 1024)  # WMI reports in KB
            end
        end
    catch
        # PowerShell not available or failed; use basic env info
    end

    # Map vendor from PROCESSOR_IDENTIFIER if not set by WMI
    if isempty(vendor)
        if occursin("Intel", proc_id)
            vendor = "GenuineIntel"
        elseif occursin("AMD", proc_id)
            vendor = "AuthenticAMD"
        elseif occursin("ARM", proc_id) || occursin("Qualcomm", proc_id)
            vendor = "ARM"
        else
            vendor = "unknown"
        end
    end

    # Feature detection on Windows — infer from architecture and CPUID
    # Julia's LLVM backend can tell us some features via Base.Sys
    flags = Set{String}()

    # On x86_64 Windows, try to detect features via PowerShell CPUID or registry
    if arch === :x86_64
        try
            # Check for AVX/AVX2 via a PowerShell one-liner
            avx_check = strip(read(`powershell -NoProfile -Command "[System.Environment]::Is64BitProcess"`, String))
            # Basic x86_64 always has SSE/SSE2
            push!(flags, "SSE")
            push!(flags, "SSE2")
            # Try to detect AVX support from OS support for XSAVE
            _detect_windows_x86_features!(flags)
        catch end
    elseif arch === :aarch64
        push!(flags, "NEON")  # ARM64 Windows always has NEON
    end

    feat = _extract_features(flags, arch)

    return CpuFeatures(
        arch, vendor, model_name,
        feat.has_sse, feat.has_sse2, feat.has_avx, feat.has_avx2,
        feat.has_avx512f, feat.has_avx512vl, feat.has_avx512bw,
        feat.has_amx, feat.has_aesni,
        feat.has_neon, feat.has_sve, feat.has_sve2, feat.has_sme,
        feat.has_rvv, feat.has_rvv_1_0,
        feat.has_altivec, feat.has_vsx,
        cache_line, l1, l2, l3,
        num_cores, nproc,
        :windows, detect_platform(),
    )
end

"""
    _parse_windows_wmi_json(json_str::String) -> Dict{String,Any}

Parse JSON output from PowerShell WMI query. Uses JSON stdlib if available,
falls back to basic regex extraction.
"""
function _parse_windows_wmi_json(json_str::String)::Dict{String,Any}
    result = Dict{String,Any}()
    # Lightweight regex-based JSON field extraction (avoids JSON dependency)
    for key in ["Name", "Manufacturer", "NumberOfCores",
                 "NumberOfLogicalProcessors", "L2CacheSize", "L3CacheSize"]
        m = match(Regex("\"$key\"\\s*:\\s*\"?([^,\"\\}]+)\"?"), json_str)
        if m !== nothing
            result[key] = strip(m.captures[1])
        end
    end
    return result
end

"""
    _detect_windows_x86_features!(flags::Set{String})

Attempt to detect x86_64 feature flags on Windows via registry or PowerShell.
Mutates the `flags` set in place. Best-effort; never throws.
"""
function _detect_windows_x86_features!(flags::Set{String})
    try
        # Use PowerShell to read the ProcessorFeatureBits from WMI
        # or check the registry for feature support
        ps_script = """
        \$p = Get-CimInstance Win32_Processor | Select-Object -First 1
        \$out = @()
        if (\$p.Name -match 'AVX') { \$out += 'AVX' }
        if (\$p.Name -match 'AVX2') { \$out += 'AVX2' }
        \$out -join ' '
        """
        result = strip(read(`powershell -NoProfile -Command $ps_script`, String))
        for f in split(result)
            push!(flags, strip(f))
        end
    catch end
end

"""
    _detect_bsd() -> CpuFeatures

Detect CPU features on FreeBSD, OpenBSD, NetBSD, or other BSDs via sysctl.
"""
function _detect_bsd()::CpuFeatures
    arch = Sys.ARCH
    num_threads = Sys.CPU_THREADS
    os = detect_os()

    model_name = _safe_read_cmd(`sysctl -n hw.model`; default="unknown")
    vendor = "unknown"

    # Detect vendor from model string
    if occursin("Intel", model_name)
        vendor = "GenuineIntel"
    elseif occursin("AMD", model_name)
        vendor = "AuthenticAMD"
    elseif occursin("ARM", model_name) || occursin("Apple", model_name)
        vendor = "ARM"
    end

    # Core count
    ncpus = _safe_parse_int(_safe_read_cmd(`sysctl -n hw.ncpu`; default="$num_threads"); default=num_threads)

    # Cache sizes — BSD sysctl names vary by variant
    # FreeBSD typically doesn't expose L1/L2/L3 directly via sysctl,
    # but some versions do via dev.cpu.0.cache.*
    cache_line = 0
    l1 = 0
    l2 = 0
    l3 = 0

    # Try FreeBSD-style cache detection
    try
        # FreeBSD dmesg parsing is another option, but sysctl is safer
        l1 = _safe_parse_int(_safe_read_cmd(`sysctl -n hw.cacheconfig`; default="0"))
    catch end

    # Feature flags — FreeBSD exposes them via dmesg or sysctl
    flags = Set{String}()
    try
        # FreeBSD: /var/run/dmesg.boot often has CPU features
        if isfile("/var/run/dmesg.boot")
            dmesg = read("/var/run/dmesg.boot", String)
            for line in eachline(IOBuffer(dmesg))
                if occursin("Features", line) || occursin("features", line)
                    # Extract text after the < > angle brackets (FreeBSD format)
                    m = match(r"<(.+)>", line)
                    if m !== nothing
                        for f in split(m.captures[1], ',')
                            push!(flags, strip(f))
                        end
                    end
                end
            end
        end
    catch end

    # ARM64 on BSDs always has NEON
    if arch === :aarch64
        push!(flags, "NEON")
    end

    feat = _extract_features(flags, arch)

    return CpuFeatures(
        arch, vendor, model_name,
        feat.has_sse, feat.has_sse2, feat.has_avx, feat.has_avx2,
        feat.has_avx512f, feat.has_avx512vl, feat.has_avx512bw,
        feat.has_amx, feat.has_aesni,
        feat.has_neon, feat.has_sve, feat.has_sve2, feat.has_sme,
        feat.has_rvv, feat.has_rvv_1_0,
        feat.has_altivec, feat.has_vsx,
        cache_line, l1, l2, l3,
        ncpus, num_threads,
        os, detect_platform(),
    )
end

"""
    _make_minimal_features(arch::Symbol, num_threads::Int, os::Symbol) -> CpuFeatures

Construct a minimal CpuFeatures with only architecture, thread count, and OS.
Used as a fallback when detailed detection fails.
"""
function _make_minimal_features(arch::Symbol, num_threads::Int, os::Symbol)::CpuFeatures
    # Even in minimal mode, infer guaranteed features from architecture
    has_neon = arch === :aarch64  # aarch64 always has NEON/ASIMD
    has_sse = arch === :x86_64    # x86_64 always has SSE
    has_sse2 = arch === :x86_64   # x86_64 always has SSE2

    return CpuFeatures(
        arch, "unknown", "unknown",
        has_sse, has_sse2, false, false, false, false, false, false, false,
        has_neon, false, false, false,
        false, false,
        false, false,
        0, 0, 0, 0,
        num_threads, num_threads,
        os, detect_platform(),
    )
end

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

"""
    detect_arch() -> Symbol

Return the CPU architecture as a Symbol (e.g. :x86_64, :aarch64, :riscv64).
Uses `Sys.ARCH`.
"""
function detect_arch()::Symbol
    return Sys.ARCH
end

"""
    detect_cpu_features() -> CpuFeatures

Detect CPU features using platform-appropriate methods:

- **Linux**: /proc/cpuinfo + /sys/devices/system/cpu/
- **macOS**: sysctl (supports both Intel and Apple Silicon)
- **Windows**: Environment variables + PowerShell WMI
- **FreeBSD/BSD**: sysctl + dmesg
- **Other**: Minimal detection from Sys.ARCH and Sys.CPU_THREADS

Architecture-specific feature detection:
- **x86_64**: SSE, SSE2, AVX, AVX2, AVX-512, AMX, AES-NI
- **aarch64/ARM**: NEON, SVE, SVE2, SME
- **RISC-V**: RVV (Vector extension)
- **PowerPC**: AltiVec, VSX

Never throws — returns partial information on detection failure.

# Examples
```julia
features = detect_cpu_features()
features.arch          # :x86_64
features.has_avx2      # true
features.l1_cache_size # 32768 (32K)
features.num_cores     # 6
features.os            # :linux
features.platform      # :desktop
```
"""
function detect_cpu_features()::CpuFeatures
    os = detect_os()

    try
        if os === :linux || os === :android
            return _detect_linux()
        elseif os === :macos
            return _detect_macos()
        elseif os === :windows
            return _detect_windows()
        elseif os in (:freebsd, :openbsd, :netbsd)
            return _detect_bsd()
        else
            # Unknown OS — attempt Linux-style detection first (many Unices have /proc)
            if isfile("/proc/cpuinfo")
                return _detect_linux()
            end
            return _make_minimal_features(Sys.ARCH, Sys.CPU_THREADS, os)
        end
    catch e
        # Ultimate fallback: never let detection crash the caller
        @warn "SiliconCore: CPU feature detection failed" exception=(e, catch_backtrace())
        return _make_minimal_features(Sys.ARCH, Sys.CPU_THREADS, os)
    end
end

"""
    has_feature(features::CpuFeatures, name::Symbol) -> Bool

Convenience function to query a CPU feature by name.

# Supported feature names
`:sse`, `:sse2`, `:avx`, `:avx2`, `:avx512f`, `:avx512vl`, `:avx512bw`,
`:amx`, `:aesni`, `:neon`, `:sve`, `:sve2`, `:sme`, `:rvv`, `:rvv_1_0`,
`:altivec`, `:vsx`

# Examples
```julia
features = detect_cpu_features()
has_feature(features, :avx2)    # true on modern x86_64
has_feature(features, :neon)    # true on ARM
has_feature(features, :rvv)     # true on RISC-V with vector ext
```
"""
function has_feature(features::CpuFeatures, name::Symbol)::Bool
    name === :sse && return features.has_sse
    name === :sse2 && return features.has_sse2
    name === :avx && return features.has_avx
    name === :avx2 && return features.has_avx2
    name === :avx512f && return features.has_avx512f
    name === :avx512vl && return features.has_avx512vl
    name === :avx512bw && return features.has_avx512bw
    name === :amx && return features.has_amx
    name === :aesni && return features.has_aesni
    name === :neon && return features.has_neon
    name === :sve && return features.has_sve
    name === :sve2 && return features.has_sve2
    name === :sme && return features.has_sme
    name === :rvv && return features.has_rvv
    name === :rvv_1_0 && return features.has_rvv_1_0
    name === :altivec && return features.has_altivec
    name === :vsx && return features.has_vsx
    error("Unknown CPU feature: :$name. Supported: :sse, :sse2, :avx, :avx2, " *
          ":avx512f, :avx512vl, :avx512bw, :amx, :aesni, :neon, :sve, :sve2, " *
          ":sme, :rvv, :rvv_1_0, :altivec, :vsx")
end

"""
    vector_add_asm(a, b)

Element-wise vector addition. Currently uses Julia's broadcast `.+` operator,
which automatically dispatches to SIMD instructions when available through
Julia's LLVM backend.

Future: may use explicit SIMD intrinsics via SiliconCore's feature detection
for fine-grained control over instruction selection.
"""
function vector_add_asm(a, b)
    return a .+ b
end

# ---------------------------------------------------------------------------
# Pretty-printing
# ---------------------------------------------------------------------------

function Base.show(io::IO, f::CpuFeatures)
    print(io, "CpuFeatures(")
    print(io, f.arch, "/", f.os, ", \"", f.model_name, "\"")
    # Show active SIMD features
    simd = String[]
    f.has_sse && push!(simd, "SSE")
    f.has_sse2 && push!(simd, "SSE2")
    f.has_avx && push!(simd, "AVX")
    f.has_avx2 && push!(simd, "AVX2")
    f.has_avx512f && push!(simd, "AVX-512F")
    f.has_avx512vl && push!(simd, "AVX-512VL")
    f.has_avx512bw && push!(simd, "AVX-512BW")
    f.has_amx && push!(simd, "AMX")
    f.has_aesni && push!(simd, "AES-NI")
    f.has_neon && push!(simd, "NEON")
    f.has_sve && push!(simd, "SVE")
    f.has_sve2 && push!(simd, "SVE2")
    f.has_sme && push!(simd, "SME")
    f.has_rvv && push!(simd, "RVV")
    f.has_rvv_1_0 && push!(simd, "RVV1.0")
    f.has_altivec && push!(simd, "AltiVec")
    f.has_vsx && push!(simd, "VSX")
    if !isempty(simd)
        print(io, ", ", join(simd, "+"))
    end
    print(io, ", ", f.num_cores, "c/", f.num_threads, "t")
    if f.l1_cache_size > 0
        print(io, ", L1=", div(f.l1_cache_size, 1024), "K")
    end
    if f.l2_cache_size > 0
        print(io, ", L2=", div(f.l2_cache_size, 1024), "K")
    end
    if f.l3_cache_size > 0
        l3_mb = div(f.l3_cache_size, 1024 * 1024)
        if l3_mb > 0
            print(io, ", L3=", l3_mb, "M")
        else
            print(io, ", L3=", div(f.l3_cache_size, 1024), "K")
        end
    end
    print(io, ", ", f.platform, ")")
end

end # module
