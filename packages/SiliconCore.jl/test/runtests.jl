# SPDX-License-Identifier: PMPL-1.0-or-later
# Copyright (c) 2026 Jonathan D.A. Jewell (hyperpolymath) <j.d.a.jewell@open.ac.uk>
#
# Tests for SiliconCore.jl — cross-platform CPU feature detection

using Test

include(joinpath(@__DIR__, "..", "src", "SiliconCore.jl"))
using .SiliconCore

# Import internal functions for unit testing (they are not exported)
import .SiliconCore:
    _parse_cpuinfo_field, _parse_cpuinfo_flags, _count_physical_cores,
    _parse_cache_size, _extract_features, _default_features,
    _make_minimal_features, _parse_windows_wmi_json

@testset "SiliconCore.jl" begin

    # -----------------------------------------------------------------------
    # Basic API tests (run on any platform)
    # -----------------------------------------------------------------------

    @testset "detect_arch returns a Symbol" begin
        arch = detect_arch()
        @test arch isa Symbol
        @test arch === Sys.ARCH
    end

    @testset "detect_arch is consistent" begin
        @test detect_arch() === detect_arch()
    end

    @testset "detect_os returns a valid Symbol" begin
        os = detect_os()
        @test os isa Symbol
        @test os in (:linux, :macos, :windows, :freebsd, :openbsd, :netbsd,
                     :minix, :android, :unknown)
    end

    @testset "detect_platform returns a valid Symbol" begin
        plat = detect_platform()
        @test plat isa Symbol
        @test plat in (:desktop, :server, :mobile, :embedded, :unknown)
    end

    @testset "detect_cpu_features never throws" begin
        features = @test_nowarn detect_cpu_features()
        @test features isa CpuFeatures
    end

    @testset "detect_cpu_features returns correct arch" begin
        features = detect_cpu_features()
        @test features.arch === Sys.ARCH
    end

    @testset "detect_cpu_features has valid thread count" begin
        features = detect_cpu_features()
        @test features.num_threads >= 1
        @test features.num_cores >= 1
        @test features.num_cores <= features.num_threads
    end

    @testset "detect_cpu_features has os and platform fields" begin
        features = detect_cpu_features()
        @test features.os isa Symbol
        @test features.platform isa Symbol
        @test features.os !== :unknown || !Sys.islinux()  # On Linux we should detect it
    end

    @testset "detect_cpu_features architecture-specific guarantees" begin
        features = detect_cpu_features()
        if features.arch === :x86_64
            # x86_64 always has SSE and SSE2
            @test features.has_sse
            @test features.has_sse2
        elseif features.arch === :aarch64
            # aarch64 always has NEON/ASIMD
            @test features.has_neon
        end
    end

    @testset "has_feature queries all supported features" begin
        features = detect_cpu_features()
        # All supported feature names should work without error
        for name in [:sse, :sse2, :avx, :avx2, :avx512f, :avx512vl, :avx512bw,
                     :amx, :aesni, :neon, :sve, :sve2, :sme,
                     :rvv, :rvv_1_0, :altivec, :vsx]
            result = has_feature(features, name)
            @test result isa Bool
        end
    end

    @testset "has_feature throws on unknown feature" begin
        features = detect_cpu_features()
        @test_throws ErrorException has_feature(features, :nonexistent_feature)
    end

    @testset "exports are correct" begin
        @test isdefined(SiliconCore, :detect_arch)
        @test isdefined(SiliconCore, :vector_add_asm)
        @test isdefined(SiliconCore, :CpuFeatures)
        @test isdefined(SiliconCore, :detect_cpu_features)
        @test isdefined(SiliconCore, :has_feature)
        @test isdefined(SiliconCore, :detect_os)
        @test isdefined(SiliconCore, :detect_platform)
    end

    # -----------------------------------------------------------------------
    # vector_add_asm tests
    # -----------------------------------------------------------------------

    @testset "vector_add_asm with integers" begin
        result = vector_add_asm([1, 2, 3], [4, 5, 6])
        @test result == [5, 7, 9]
        @test result isa Vector{Int}
    end

    @testset "vector_add_asm with floats" begin
        result = vector_add_asm([1.0, 2.0], [3.0, 4.0])
        @test result == [4.0, 6.0]
        @test result isa Vector{Float64}
    end

    @testset "vector_add_asm with scalars" begin
        result = vector_add_asm(5, 10)
        @test result == 15
        @test result isa Int
    end

    @testset "vector_add_asm preserves element type" begin
        result = vector_add_asm(Float32[1.0, 2.0], Float32[3.0, 4.0])
        @test result isa Vector{Float32}
        @test result == Float32[4.0, 6.0]
    end

    @testset "vector_add_asm with empty vectors" begin
        result = vector_add_asm(Int[], Int[])
        @test result == Int[]
        @test isempty(result)
    end

    @testset "vector_add_asm with single element" begin
        result = vector_add_asm([42], [58])
        @test result == [100]
    end

    # -----------------------------------------------------------------------
    # Internal: /proc/cpuinfo parsing (mock data)
    # -----------------------------------------------------------------------

    @testset "cpuinfo parsing — x86_64 Intel" begin
        cpuinfo = """
        processor\t: 0
        vendor_id\t: GenuineIntel
        cpu family\t: 6
        model\t\t: 158
        model name\t: Intel(R) Core(TM) i7-8700K CPU @ 3.70GHz
        physical id\t: 0
        core id\t\t: 0
        cpu cores\t: 6
        flags\t\t: fpu vme sse sse2 avx avx2 avx512f avx512vl avx512bw aes amx_tile

        processor\t: 1
        vendor_id\t: GenuineIntel
        cpu family\t: 6
        model\t\t: 158
        model name\t: Intel(R) Core(TM) i7-8700K CPU @ 3.70GHz
        physical id\t: 0
        core id\t\t: 1
        cpu cores\t: 6
        flags\t\t: fpu vme sse sse2 avx avx2 avx512f avx512vl avx512bw aes amx_tile
        """

        @test _parse_cpuinfo_field(cpuinfo, "vendor_id") == "GenuineIntel"
        @test _parse_cpuinfo_field(cpuinfo, "model name") == "Intel(R) Core(TM) i7-8700K CPU @ 3.70GHz"
        @test _parse_cpuinfo_field(cpuinfo, "nonexistent") == ""

        flags = _parse_cpuinfo_flags(cpuinfo)
        @test "sse" in flags
        @test "sse2" in flags
        @test "avx" in flags
        @test "avx2" in flags
        @test "avx512f" in flags
        @test "aes" in flags
        @test "amx_tile" in flags

        @test _count_physical_cores(cpuinfo) == 2  # 2 unique (phys_id, core_id) pairs
    end

    @testset "cpuinfo parsing — AMD" begin
        cpuinfo = """
        processor\t: 0
        vendor_id\t: AuthenticAMD
        model name\t: AMD Ryzen 9 5950X 16-Core Processor
        flags\t\t: fpu vme sse sse2 ssse3 avx avx2 aes
        """

        @test _parse_cpuinfo_field(cpuinfo, "vendor_id") == "AuthenticAMD"
        flags = _parse_cpuinfo_flags(cpuinfo)
        @test "avx2" in flags
        @test !("avx512f" in flags)
    end

    @testset "cpuinfo parsing — ARM aarch64" begin
        cpuinfo = """
        processor\t: 0
        BogoMIPS\t: 48.00
        Features\t: fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics sve sve2
        CPU implementer\t: 0x41
        CPU architecture: 8

        Hardware\t: Generic ARM Board
        """

        flags = _parse_cpuinfo_flags(cpuinfo)
        @test "asimd" in flags
        @test "sve" in flags
        @test "sve2" in flags
        @test "aes" in flags

        @test _parse_cpuinfo_field(cpuinfo, "Hardware") == "Generic ARM Board"
    end

    @testset "cpuinfo parsing — RISC-V" begin
        cpuinfo = """
        processor\t: 0
        hart\t\t: 0
        isa\t\t: rv64imafdc_v
        mmu\t\t: sv39
        uarch\t\t: sifive,u74-mc
        """

        flags = _parse_cpuinfo_flags(cpuinfo)
        # The ISA string should be in the flags set
        @test !isempty(flags)

        feat = _extract_features(flags, :riscv64)
        @test feat.has_rvv  # "_v" present in ISA string
    end

    @testset "cpuinfo parsing — RISC-V v1.0" begin
        cpuinfo = """
        processor\t: 0
        hart\t\t: 0
        isa\t\t: rv64gcv1p0_zba_zbb_zbc
        """

        flags = _parse_cpuinfo_flags(cpuinfo)
        feat = _extract_features(flags, :riscv64)
        @test feat.has_rvv
        @test feat.has_rvv_1_0  # "v1p0" present
    end

    @testset "cpuinfo parsing — PowerPC" begin
        cpuinfo = """
        processor\t: 0
        cpu\t\t: POWER9 (raw), altivec supported
        clock\t\t: 3800.000000MHz
        revision\t: 2.2 (pvr 004e 1202)
        platform\t: PowerNV
        model\t\t: IBM,9009-22A
        machine\t\t: PowerNV
        firmware\t: OPAL
        """

        # PowerPC flags are often in the "cpu" field, not separate flags line
        # The _parse_cpuinfo_flags won't find them in "flags" or "Features",
        # so we test _extract_features with manually constructed flags
        flags = Set(["altivec", "vsx"])
        feat = _extract_features(flags, :powerpc64le)
        @test feat.has_altivec
        @test feat.has_vsx
    end

    @testset "cpuinfo parsing — physical core count with SMT" begin
        # 4 cores, 8 threads (SMT/HT)
        cpuinfo = join(["""
        processor\t: $i
        physical id\t: 0
        core id\t\t: $(i % 4)
        """ for i in 0:7], "\n")

        @test _count_physical_cores(cpuinfo) == 4
    end

    @testset "cpuinfo parsing — no topology info" begin
        cpuinfo = """
        processor\t: 0
        processor\t: 1
        processor\t: 2
        processor\t: 3
        """
        @test _count_physical_cores(cpuinfo) == 4
    end

    @testset "cpuinfo parsing — empty input" begin
        @test _parse_cpuinfo_field("", "flags") == ""
        @test isempty(_parse_cpuinfo_flags(""))
        @test _count_physical_cores("") == 1
    end

    # -----------------------------------------------------------------------
    # Internal: Cache size parsing
    # -----------------------------------------------------------------------

    @testset "cache size parsing" begin
        @test _parse_cache_size("32K") == 32 * 1024
        @test _parse_cache_size("256K") == 256 * 1024
        @test _parse_cache_size("8192K") == 8192 * 1024
        @test _parse_cache_size("16M") == 16 * 1024 * 1024
        @test _parse_cache_size("1G") == 1024 * 1024 * 1024
        @test _parse_cache_size("64") == 64  # No suffix = bytes
        @test _parse_cache_size("") == 0
        @test _parse_cache_size("invalid") == 0
        @test _parse_cache_size("  32K  ") == 32 * 1024  # Whitespace handling
    end

    # -----------------------------------------------------------------------
    # Internal: Feature extraction
    # -----------------------------------------------------------------------

    @testset "feature extraction — x86_64 flags" begin
        flags = Set(["sse", "sse2", "avx", "avx2", "avx512f", "avx512vl",
                     "avx512bw", "aes", "amx_tile"])
        feat = _extract_features(flags, :x86_64)
        @test feat.has_sse
        @test feat.has_sse2
        @test feat.has_avx
        @test feat.has_avx2
        @test feat.has_avx512f
        @test feat.has_avx512vl
        @test feat.has_avx512bw
        @test feat.has_aesni
        @test feat.has_amx
        @test !feat.has_neon
        @test !feat.has_sve
        @test !feat.has_rvv
        @test !feat.has_altivec
    end

    @testset "feature extraction — macOS uppercase flags" begin
        # macOS sysctl returns uppercase feature names
        flags = Set(["SSE", "SSE2", "AVX1.0", "AVX2", "AES", "NEON"])
        feat = _extract_features(flags, :x86_64)
        @test feat.has_sse
        @test feat.has_sse2
        @test feat.has_avx   # "AVX1.0" maps to AVX
        @test feat.has_avx2
        @test feat.has_aesni
    end

    @testset "feature extraction — ARM flags" begin
        flags = Set(["asimd", "sve", "sve2", "sme", "aes"])
        feat = _extract_features(flags, :aarch64)
        @test feat.has_neon  # "asimd" maps to NEON
        @test feat.has_sve
        @test feat.has_sve2
        @test feat.has_sme
        @test feat.has_aesni  # ARM AES
    end

    @testset "feature extraction — empty flags" begin
        feat = _extract_features(Set{String}(), :x86_64)
        @test !feat.has_sse
        @test !feat.has_avx
        @test !feat.has_neon
    end

    @testset "default features are all false" begin
        feat = _default_features()
        @test !feat.has_sse
        @test !feat.has_sse2
        @test !feat.has_avx
        @test !feat.has_neon
        @test !feat.has_sve
        @test !feat.has_sve2
        @test !feat.has_sme
        @test !feat.has_rvv
        @test !feat.has_rvv_1_0
        @test !feat.has_altivec
        @test !feat.has_vsx
    end

    # -----------------------------------------------------------------------
    # Internal: Minimal features fallback
    # -----------------------------------------------------------------------

    @testset "minimal features — x86_64 guarantees SSE/SSE2" begin
        f = _make_minimal_features(:x86_64, 8, :unknown)
        @test f.arch === :x86_64
        @test f.has_sse
        @test f.has_sse2
        @test !f.has_avx  # Cannot guarantee without detection
        @test f.num_threads == 8
        @test f.os === :unknown
    end

    @testset "minimal features — aarch64 guarantees NEON" begin
        f = _make_minimal_features(:aarch64, 4, :linux)
        @test f.arch === :aarch64
        @test f.has_neon
        @test !f.has_sse
        @test f.num_threads == 4
        @test f.os === :linux
    end

    @testset "minimal features — unknown arch" begin
        f = _make_minimal_features(:mips64, 2, :linux)
        @test f.arch === :mips64
        @test !f.has_sse
        @test !f.has_neon
        @test f.vendor == "unknown"
        @test f.model_name == "unknown"
    end

    # -----------------------------------------------------------------------
    # Internal: Windows WMI JSON parsing
    # -----------------------------------------------------------------------

    @testset "Windows WMI JSON parsing" begin
        json = """
        {
            "Name": "Intel(R) Core(TM) i9-12900K",
            "Manufacturer": "GenuineIntel",
            "NumberOfCores": 16,
            "NumberOfLogicalProcessors": 24,
            "L2CacheSize": 14336,
            "L3CacheSize": 30720
        }
        """
        parsed = _parse_windows_wmi_json(json)
        @test parsed["Name"] == "Intel(R) Core(TM) i9-12900K"
        @test parsed["Manufacturer"] == "GenuineIntel"
        @test parsed["NumberOfCores"] == "16"
        @test parsed["NumberOfLogicalProcessors"] == "24"
        @test parsed["L2CacheSize"] == "14336"
        @test parsed["L3CacheSize"] == "30720"
    end

    @testset "Windows WMI JSON parsing — empty input" begin
        parsed = _parse_windows_wmi_json("")
        @test isempty(parsed)
    end

    # -----------------------------------------------------------------------
    # Pretty-printing
    # -----------------------------------------------------------------------

    @testset "show method works without error" begin
        features = detect_cpu_features()
        buf = IOBuffer()
        @test_nowarn show(buf, features)
        output = String(take!(buf))
        @test startswith(output, "CpuFeatures(")
        @test endswith(output, ")")
        # Should include os and platform info
        @test occursin(string(features.os), output)
        @test occursin(string(features.platform), output)
    end

    @testset "show method includes arch" begin
        features = detect_cpu_features()
        buf = IOBuffer()
        show(buf, features)
        output = String(take!(buf))
        @test occursin(string(features.arch), output)
    end

    # -----------------------------------------------------------------------
    # CpuFeatures struct field completeness
    # -----------------------------------------------------------------------

    @testset "CpuFeatures has all expected fields" begin
        names = fieldnames(CpuFeatures)
        # Original fields
        @test :arch in names
        @test :vendor in names
        @test :model_name in names
        @test :has_sse in names
        @test :has_sse2 in names
        @test :has_avx in names
        @test :has_avx2 in names
        @test :has_avx512f in names
        @test :has_avx512vl in names
        @test :has_avx512bw in names
        @test :has_amx in names
        @test :has_aesni in names
        @test :has_neon in names
        @test :has_sve in names
        # New fields
        @test :has_sve2 in names
        @test :has_sme in names
        @test :has_rvv in names
        @test :has_rvv_1_0 in names
        @test :has_altivec in names
        @test :has_vsx in names
        @test :os in names
        @test :platform in names
    end

end
