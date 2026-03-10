# SPDX-License-Identifier: PMPL-1.0-or-later
# Copyright (c) 2026 Jonathan D.A. Jewell (hyperpolymath) <j.d.a.jewell@open.ac.uk>

using Test
using FirmwareAudit
using SHA
using Dates

@testset "FirmwareAudit.jl" begin

    # ── Type construction ──────────────────────────────────────────────
    @testset "FirmwareImage construction" begin
        img = FirmwareImage("/tmp/test.bin", "abc123", "TestVendor", "1.0.0")
        @test img.path == "/tmp/test.bin"
        @test img.hash == "abc123"
        @test img.vendor == "TestVendor"
        @test img.version == "1.0.0"
    end

    @testset "AuditResult construction" begin
        img = FirmwareImage("/tmp/test.bin", "abc123", "TestVendor", "1.0.0")
        ts = now()
        result = AuditResult(img, ["finding1", "finding2"], ts)
        @test result.image === img
        @test length(result.findings) == 2
        @test result.timestamp == ts
    end

    # ── Hash verification ──────────────────────────────────────────────
    @testset "verify_hash" begin
        img = FirmwareImage("/tmp/test.bin", "aaBBcc1234", "V", "1.0")
        @test verify_hash(img, "aabbcc1234") == true
        @test verify_hash(img, "AABBCC1234") == true
        @test verify_hash(img, "different") == false
        @test verify_hash(img, "") == false
    end

    # ── Firmware auditing ──────────────────────────────────────────────
    @testset "audit_firmware with real file" begin
        # Create a temporary firmware image with known content
        tmpfile = tempname()
        content = UInt8[0x7f, 0x45, 0x4c, 0x46, 0x01, 0x01, 0x01, 0x00]  # ELF-like header
        write(tmpfile, content)

        result = audit_firmware(tmpfile; vendor="TestCorp", version="2.0.1")
        @test result.image.path == tmpfile
        @test result.image.vendor == "TestCorp"
        @test result.image.version == "2.0.1"
        @test result.image.hash == bytes2hex(sha256(content))
        @test result.timestamp isa DateTime
        @test !isempty(result.findings)

        rm(tmpfile)
    end

    @testset "audit_firmware missing file" begin
        @test_throws ArgumentError audit_firmware("/nonexistent/firmware.bin")
    end

    @testset "audit_firmware empty file" begin
        tmpfile = tempname()
        write(tmpfile, UInt8[])

        result = audit_firmware(tmpfile)
        @test any(f -> occursin("empty", f), result.findings)

        rm(tmpfile)
    end

    @testset "audit_firmware default vendor/version" begin
        tmpfile = tempname()
        write(tmpfile, UInt8[0x01, 0x02, 0x03])

        result = audit_firmware(tmpfile)
        @test result.image.vendor == "unknown"
        @test result.image.version == "0.0.0"

        rm(tmpfile)
    end

    @testset "audit_firmware hash correctness" begin
        tmpfile = tempname()
        data = rand(UInt8, 1024)
        write(tmpfile, data)

        result = audit_firmware(tmpfile)
        expected_hash = bytes2hex(sha256(data))
        @test result.image.hash == expected_hash
        @test verify_hash(result.image, expected_hash) == true

        rm(tmpfile)
    end

    # ── Known vulnerabilities ──────────────────────────────────────────
    @testset "list_known_vulnerabilities" begin
        vulns = list_known_vulnerabilities("TestVendor")
        @test vulns isa Vector{String}
    end

    # ── Point-to-point integration ─────────────────────────────────────
    @testset "Point-to-point: audit → verify round-trip" begin
        tmpfile = tempname()
        data = UInt8[0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE]
        write(tmpfile, data)

        # Audit the firmware
        result = audit_firmware(tmpfile; vendor="Acme", version="3.1.4")

        # Verify hash matches independently computed hash
        expected = bytes2hex(sha256(data))
        @test verify_hash(result.image, expected)

        # Verify wrong hash fails
        @test !verify_hash(result.image, "0000000000000000")

        rm(tmpfile)
    end

    @testset "End-to-end: multi-image audit pipeline" begin
        images = String[]
        results = AuditResult[]

        # Create multiple firmware images
        for i in 1:5
            tmpfile = tempname()
            data = rand(UInt8, 256 * i)
            write(tmpfile, data)
            push!(images, tmpfile)

            result = audit_firmware(tmpfile; vendor="Vendor$i", version="$i.0.0")
            push!(results, result)
        end

        # Verify all results are distinct
        hashes = [r.image.hash for r in results]
        @test length(unique(hashes)) == 5

        # Verify all timestamps are valid
        @test all(r -> r.timestamp <= now(), results)

        # Verify vendor/version propagation
        for (i, r) in enumerate(results)
            @test r.image.vendor == "Vendor$i"
            @test r.image.version == "$i.0.0"
        end

        # Clean up
        foreach(rm, images)
    end

    # ── Benchmarks ─────────────────────────────────────────────────────
    @testset "Performance: audit 1MB firmware" begin
        tmpfile = tempname()
        data = rand(UInt8, 1024 * 1024)
        write(tmpfile, data)

        t = @elapsed begin
            for _ in 1:10
                audit_firmware(tmpfile)
            end
        end
        @test t < 30.0  # 10 audits of 1MB should complete within 30s

        rm(tmpfile)
    end
end
