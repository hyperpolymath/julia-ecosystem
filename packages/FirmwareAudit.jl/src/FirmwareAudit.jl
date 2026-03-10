# SPDX-License-Identifier: PMPL-1.0-or-later
# Copyright (c) 2026 Jonathan D.A. Jewell (hyperpolymath) <j.d.a.jewell@open.ac.uk>
#
# FirmwareAudit.jl - Firmware image auditing and vulnerability scanning.
#
# Performs entropy analysis, string extraction, header parsing, hash verification,
# and known-vulnerability matching on firmware binary images.

"""
    FirmwareAudit

Firmware image auditing and vulnerability scanning. Performs entropy analysis,
string extraction, header format identification, hash verification, and
known-CVE matching against an embedded vendor vulnerability database.

# Key Features
- SHA-256 hash computation and verification
- Header parsing for UEFI, U-Boot, Android boot, Coreboot, and ELF formats
- Shannon entropy analysis to detect encrypted/compressed/padding regions
- Hardcoded credential and outdated library detection
- Known CVE lookup for Qualcomm, Intel, AMD, MediaTek, and Broadcom firmware

# Example
```julia
using FirmwareAudit
result = audit_firmware("firmware.bin"; vendor="intel")
println.(result.findings)
```
"""
module FirmwareAudit

using SHA
using Dates

export FirmwareImage, AuditResult, FirmwareHeader, EntropyProfile
export audit_firmware, verify_hash, list_known_vulnerabilities
export analyse_entropy, extract_strings, parse_firmware_header

# ============================================================================
# Types
# ============================================================================

"""
    FirmwareImage(path, hash, vendor, version)

Represents a firmware image on disk with its SHA-256 hash, vendor name,
and firmware version string.
"""
struct FirmwareImage
    path::String
    hash::String
    vendor::String
    version::String
end

"""
    AuditResult(image, findings, timestamp)

The result of auditing a firmware image. Contains a list of findings
(vulnerability descriptions, warnings, or informational notes) and the
timestamp when the audit was performed.
"""
struct AuditResult
    image::FirmwareImage
    findings::Vector{String}
    timestamp::DateTime
end

"""
    FirmwareHeader

Parsed header information extracted from the first bytes of a firmware image.
Identifies the firmware format and embedded metadata.
"""
struct FirmwareHeader
    magic::Vector{UInt8}
    format::Symbol          # :uefi, :uboot, :coreboot, :android_boot, :raw, :unknown
    header_size::Int
    image_size::UInt64
    checksum::UInt32
    entry_point::UInt64
    description::String
end

"""
    EntropyProfile

Shannon entropy analysis of a firmware image, computed over fixed-size blocks.
High-entropy regions suggest compression or encryption; low-entropy regions
suggest padding or uninitialized data.
"""
struct EntropyProfile
    overall_entropy::Float64
    block_entropies::Vector{Float64}
    block_size::Int
    high_entropy_regions::Vector{UnitRange{Int}}   # likely encrypted/compressed
    low_entropy_regions::Vector{UnitRange{Int}}     # likely padding/empty
end

# ============================================================================
# Known firmware magic bytes for format identification
# ============================================================================

const FIRMWARE_SIGNATURES = Dict{Vector{UInt8}, Symbol}(
    # UEFI: starts with "_FVH" (Firmware Volume Header) or ZZ signature
    UInt8[0x5F, 0x46, 0x56, 0x48] => :uefi,
    # U-Boot legacy image magic: 0x27051956
    UInt8[0x27, 0x05, 0x19, 0x56] => :uboot,
    # Android boot image magic: "ANDROID!"
    UInt8.(collect("ANDROID!")) => :android_boot,
    # Coreboot: "LBIO" (coreboot table)
    UInt8.(collect("LBIO")) => :coreboot,
    # Intel ME: "$FPT" (Flash Partition Table)
    UInt8.(collect("\$FPT")) => :intel_me,
    # Qualcomm: "MELF" or "ELF" for modem images
    UInt8[0x7F, 0x45, 0x4C, 0x46] => :elf,
)

# Known vulnerability patterns in firmware strings
const VULN_STRING_PATTERNS = [
    (r"password\s*=\s*\S+", "CRITICAL: Hardcoded password found"),
    (r"(api[_-]?key|secret[_-]?key)\s*=\s*\S+", "CRITICAL: Hardcoded API/secret key"),
    (r"BEGIN (RSA |DSA |EC )?PRIVATE KEY", "CRITICAL: Embedded private key"),
    (r"telnet(d|_server)", "HIGH: Telnet daemon present (unencrypted protocol)"),
    (r"(dropbear|openssh).*0\.[0-9]", "HIGH: Outdated SSH implementation"),
    (r"busybox\s+v?1\.[012][0-9]\.", "MEDIUM: Outdated BusyBox version"),
    (r"openssl[/ ]0\.", "HIGH: OpenSSL 0.x (known vulnerabilities)"),
    (r"openssl[/ ]1\.0\.", "MEDIUM: OpenSSL 1.0.x (deprecated)"),
    (r"curl[/ ][0-6]\.", "MEDIUM: Outdated curl version"),
    (r"debug_?mode\s*=\s*(1|true|on)", "HIGH: Debug mode enabled in production"),
    (r"root:[\$\w]", "MEDIUM: Root password hash in image"),
    (r"JTAG|jtag_enable", "MEDIUM: JTAG debug interface references"),
]

# Known vendor CVEs (sample database - in production this would query NVD)
const VENDOR_CVES = Dict{String, Vector{String}}(
    "qualcomm" => [
        "CVE-2023-33107 (Adreno GPU use-after-free)",
        "CVE-2023-33063 (DSP service memory corruption)",
        "CVE-2024-21473 (WLAN firmware buffer overflow)",
    ],
    "mediatek" => [
        "CVE-2023-32837 (modem memory corruption)",
        "CVE-2024-20069 (baseband RCE)",
    ],
    "intel" => [
        "CVE-2023-22655 (BIOS privilege escalation)",
        "CVE-2023-28746 (Register File Data Sampling / RFDS)",
        "CVE-2024-21801 (UEFI firmware race condition)",
    ],
    "amd" => [
        "CVE-2023-20569 (Inception / Return Address Predictor)",
        "CVE-2023-20592 (INVD instruction data leak)",
    ],
    "broadcom" => [
        "CVE-2023-44487 (WiFi firmware stack overflow)",
    ],
    "uefi" => [
        "CVE-2024-0762 (Phoenix SecureCore UEFI buffer overflow)",
        "CVE-2023-45229..45235 (PixieFail: UEFI PXE vulnerabilities)",
    ],
)

# ============================================================================
# Core audit function
# ============================================================================

"""
    audit_firmware(path::String; vendor="unknown", version="0.0.0") -> AuditResult

Perform a comprehensive audit of a firmware image file. The audit includes:

1. **Hash computation**: SHA-256 digest for integrity verification
2. **Header parsing**: Identify firmware format from magic bytes
3. **Entropy analysis**: Detect encrypted/compressed/padding regions
4. **String extraction**: Search for hardcoded credentials and vulnerable library versions
5. **Known CVE matching**: Cross-reference vendor against known vulnerability database

# Arguments
- `path`: path to the firmware binary file
- `vendor`: firmware vendor name for CVE lookup (default: "unknown")
- `version`: firmware version string (default: "0.0.0")

# Returns
An `AuditResult` containing the image metadata, a vector of finding strings,
and the audit timestamp.
"""
function audit_firmware(path::String; vendor::String="unknown", version::String="0.0.0")
    isfile(path) || throw(ArgumentError("Firmware image not found: $path"))

    # Compute SHA-256 hash
    hash = open(path, "r") do io
        bytes2hex(sha256(io))
    end

    image = FirmwareImage(path, hash, vendor, version)
    findings = String[]

    filesize_bytes = filesize(path)

    # Basic size checks
    if filesize_bytes == 0
        push!(findings, "CRITICAL: Firmware image is empty (0 bytes)")
        return AuditResult(image, findings, now())
    end

    if filesize_bytes > 256 * 1024 * 1024
        push!(findings, "INFO: Large firmware image ($(div(filesize_bytes, 1024*1024)) MiB)")
    end

    if filesize_bytes < 256
        push!(findings, "WARNING: Unusually small firmware image ($(filesize_bytes) bytes)")
    end

    # Read the firmware data
    data = read(path)

    # Parse header
    header = parse_firmware_header(data)
    push!(findings, "INFO: Detected format: $(header.format) ($(header.description))")

    if header.format == :unknown
        push!(findings, "WARNING: Unrecognised firmware format - magic bytes: $(bytes2hex(header.magic))")
    end

    # Entropy analysis
    entropy = analyse_entropy(data; block_size=4096)
    push!(findings, "INFO: Overall Shannon entropy: $(round(entropy.overall_entropy; digits=4)) bits/byte")

    if entropy.overall_entropy > 7.9
        push!(findings, "WARNING: Very high entropy ($(round(entropy.overall_entropy; digits=2))) - " *
              "image may be fully encrypted or compressed")
    end

    if !isempty(entropy.high_entropy_regions)
        push!(findings, "INFO: $(length(entropy.high_entropy_regions)) high-entropy region(s) " *
              "detected (likely encrypted or compressed)")
    end

    if !isempty(entropy.low_entropy_regions)
        n_low = length(entropy.low_entropy_regions)
        total_low = sum(length(r) for r in entropy.low_entropy_regions)
        pct = round(100.0 * total_low / filesize_bytes; digits=1)
        push!(findings, "INFO: $(n_low) low-entropy region(s) ($(pct)% of image, likely padding)")
    end

    # String extraction and vulnerability pattern matching
    strings = extract_strings(data; min_length=6)
    for (pattern, description) in VULN_STRING_PATTERNS
        for s in strings
            if occursin(pattern, s)
                push!(findings, description)
                break  # one finding per pattern
            end
        end
    end

    # Version string detection
    version_patterns = [
        r"Linux version (\d+\.\d+\.\d+)",
        r"U-Boot (\d{4}\.\d{2})",
        r"BusyBox v(\d+\.\d+\.\d+)",
    ]
    for vp in version_patterns
        for s in strings
            m = match(vp, s)
            if m !== nothing
                push!(findings, "INFO: Detected embedded software: $(m.match)")
                break
            end
        end
    end

    # Known CVE lookup
    vendor_lower = lowercase(vendor)
    if haskey(VENDOR_CVES, vendor_lower)
        cves = VENDOR_CVES[vendor_lower]
        push!(findings, "WARNING: $(length(cves)) known CVE(s) for vendor '$(vendor)':")
        for cve in cves
            push!(findings, "  - $cve")
        end
    end

    return AuditResult(image, findings, now())
end

# ============================================================================
# Entropy analysis
# ============================================================================

"""
    analyse_entropy(data::Vector{UInt8}; block_size=4096) -> EntropyProfile

Compute Shannon entropy of `data` both overall and per fixed-size block.
Classifies regions as high-entropy (> 7.5 bits/byte, likely encrypted/compressed)
or low-entropy (< 1.0 bit/byte, likely padding/empty).

# Arguments
- `data`: raw binary data to analyse
- `block_size`: size of each analysis block in bytes (default: 4096)

# Returns
An `EntropyProfile` with overall and per-block entropy values plus classified regions.
"""
function analyse_entropy(data::Vector{UInt8}; block_size::Int=4096)
    overall = _shannon_entropy(data)

    n_blocks = cld(length(data), block_size)
    block_entropies = Vector{Float64}(undef, n_blocks)

    for i in 1:n_blocks
        start_idx = (i - 1) * block_size + 1
        end_idx = min(i * block_size, length(data))
        block_entropies[i] = _shannon_entropy(@view data[start_idx:end_idx])
    end

    # Classify regions
    high_entropy = UnitRange{Int}[]
    low_entropy = UnitRange{Int}[]

    i = 1
    while i <= n_blocks
        if block_entropies[i] > 7.5
            start_block = i
            while i <= n_blocks && block_entropies[i] > 7.5
                i += 1
            end
            byte_start = (start_block - 1) * block_size + 1
            byte_end = min((i - 1) * block_size, length(data))
            push!(high_entropy, byte_start:byte_end)
        elseif block_entropies[i] < 1.0
            start_block = i
            while i <= n_blocks && block_entropies[i] < 1.0
                i += 1
            end
            byte_start = (start_block - 1) * block_size + 1
            byte_end = min((i - 1) * block_size, length(data))
            push!(low_entropy, byte_start:byte_end)
        else
            i += 1
        end
    end

    return EntropyProfile(overall, block_entropies, block_size, high_entropy, low_entropy)
end

"""
    _shannon_entropy(data) -> Float64

Compute Shannon entropy in bits per byte for the given data.
Returns a value between 0.0 (completely uniform) and 8.0 (maximally random).
"""
function _shannon_entropy(data::AbstractVector{UInt8})
    isempty(data) && return 0.0

    counts = zeros(Int, 256)
    for byte in data
        counts[byte + 1] += 1
    end

    n = length(data)
    entropy = 0.0
    for c in counts
        c == 0 && continue
        p = c / n
        entropy -= p * log2(p)
    end
    return entropy
end

# ============================================================================
# String extraction
# ============================================================================

"""
    extract_strings(data::Vector{UInt8}; min_length=6) -> Vector{String}

Extract printable ASCII strings from binary data. Similar to the Unix
`strings` utility. Only sequences of `min_length` or more consecutive
printable characters (0x20-0x7E) are returned.

# Arguments
- `data`: raw binary data
- `min_length`: minimum string length to extract (default: 6)

# Returns
A vector of extracted strings.
"""
function extract_strings(data::Vector{UInt8}; min_length::Int=6)
    results = String[]
    current = UInt8[]

    for byte in data
        if 0x20 <= byte <= 0x7E
            push!(current, byte)
        else
            if length(current) >= min_length
                push!(results, String(copy(current)))
            end
            empty!(current)
        end
    end

    # Handle trailing string
    if length(current) >= min_length
        push!(results, String(copy(current)))
    end

    return results
end

# ============================================================================
# Header parsing
# ============================================================================

"""
    parse_firmware_header(data::Vector{UInt8}) -> FirmwareHeader

Parse the first bytes of a firmware image to identify its format and
extract header metadata. Recognises UEFI, U-Boot, Android boot images,
Coreboot, Intel ME, and ELF formats.

# Returns
A `FirmwareHeader` with the detected format and any available metadata.
"""
function parse_firmware_header(data::Vector{UInt8})
    length(data) < 8 && return FirmwareHeader(
        UInt8[], :unknown, 0, UInt64(length(data)), UInt32(0), UInt64(0),
        "Image too small to identify"
    )

    magic = data[1:min(8, length(data))]

    # Try matching known signatures
    for (sig, fmt) in FIRMWARE_SIGNATURES
        sig_len = length(sig)
        if length(data) >= sig_len && data[1:sig_len] == sig
            return _parse_specific_header(data, magic, fmt)
        end
    end

    # Check for UEFI capsule (can start at various offsets)
    uefi_guid_offset = _find_uefi_volume(data)
    if uefi_guid_offset > 0
        return FirmwareHeader(
            magic, :uefi, 0, UInt64(length(data)), UInt32(0), UInt64(0),
            "UEFI firmware volume found at offset $(uefi_guid_offset)"
        )
    end

    return FirmwareHeader(
        magic, :unknown, 0, UInt64(length(data)), UInt32(0), UInt64(0),
        "Unknown firmware format"
    )
end

"""Parse format-specific header fields."""
function _parse_specific_header(data::Vector{UInt8}, magic::Vector{UInt8}, fmt::Symbol)
    if fmt == :uboot && length(data) >= 64
        # U-Boot legacy image header (64 bytes)
        # Bytes 4-7: header CRC, 8-11: timestamp, 12-15: data size,
        # 16-19: load address, 20-23: entry point
        ih_size = _read_be_u32(data, 13)
        ih_load = _read_be_u32(data, 17)
        ih_ep = _read_be_u32(data, 21)
        ih_dcrc = _read_be_u32(data, 5)
        # Image name at offset 32, up to 32 bytes
        name_end = min(64, length(data))
        name_bytes = data[33:name_end]
        null_pos = findfirst(==(0x00), name_bytes)
        name = null_pos !== nothing ? String(name_bytes[1:null_pos-1]) : String(name_bytes)
        return FirmwareHeader(
            magic, :uboot, 64, UInt64(ih_size), ih_dcrc, UInt64(ih_ep),
            "U-Boot legacy image: '$name'"
        )
    elseif fmt == :elf && length(data) >= 64
        # ELF header: class at offset 4 (1=32-bit, 2=64-bit)
        is_64 = data[5] == 0x02
        desc = is_64 ? "ELF 64-bit" : "ELF 32-bit"
        return FirmwareHeader(
            magic, :elf, is_64 ? 64 : 52, UInt64(length(data)), UInt32(0), UInt64(0),
            "$desc firmware/modem image"
        )
    elseif fmt == :android_boot && length(data) >= 48
        # Android boot header: kernel size at offset 8 (little-endian u32)
        kernel_size = _read_le_u32(data, 9)
        return FirmwareHeader(
            magic, :android_boot, 2048, UInt64(length(data)), UInt32(0), UInt64(0),
            "Android boot image (kernel size: $(kernel_size) bytes)"
        )
    else
        desc = fmt == :uefi ? "UEFI firmware volume" :
               fmt == :coreboot ? "Coreboot firmware table" :
               fmt == :intel_me ? "Intel Management Engine partition" :
               "Firmware image"
        return FirmwareHeader(
            magic, fmt, 0, UInt64(length(data)), UInt32(0), UInt64(0), desc
        )
    end
end

"""Read a big-endian UInt32 from data at the given 1-based offset."""
function _read_be_u32(data::Vector{UInt8}, offset::Int)
    return (UInt32(data[offset]) << 24) | (UInt32(data[offset+1]) << 16) |
           (UInt32(data[offset+2]) << 8) | UInt32(data[offset+3])
end

"""Read a little-endian UInt32 from data at the given 1-based offset."""
function _read_le_u32(data::Vector{UInt8}, offset::Int)
    return UInt32(data[offset]) | (UInt32(data[offset+1]) << 8) |
           (UInt32(data[offset+2]) << 16) | (UInt32(data[offset+3]) << 24)
end

"""Search for UEFI firmware volume GUID signature in the first 1MB of data."""
function _find_uefi_volume(data::Vector{UInt8})
    # UEFI Firmware Volume signature: "_FVH" at offset 40 in the volume header
    target = UInt8[0x5F, 0x46, 0x56, 0x48]
    limit = min(length(data) - 3, 1024 * 1024)
    for i in 1:limit
        if data[i] == target[1] && data[i:min(i+3, end)] == target
            return i
        end
    end
    return 0
end

# ============================================================================
# Hash verification
# ============================================================================

"""
    verify_hash(image::FirmwareImage, expected::String) -> Bool

Verify that the firmware image's stored hash matches the expected SHA-256 hash.
Comparison is case-insensitive.
"""
function verify_hash(image::FirmwareImage, expected::String)
    return lowercase(image.hash) == lowercase(expected)
end

# ============================================================================
# Known vulnerability lookup
# ============================================================================

"""
    list_known_vulnerabilities(vendor::String) -> Vector{String}

Return a list of known vulnerability identifiers for the given vendor.
Cross-references against an embedded CVE database covering major firmware
vendors (Qualcomm, MediaTek, Intel, AMD, Broadcom, UEFI).

# Arguments
- `vendor`: the vendor name (case-insensitive)

# Returns
A vector of CVE description strings. Returns an empty vector if the vendor
is not in the database.
"""
function list_known_vulnerabilities(vendor::String)
    vendor_lower = lowercase(strip(vendor))
    return get(VENDOR_CVES, vendor_lower, String[])
end

end # module FirmwareAudit
