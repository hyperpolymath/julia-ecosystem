using JuliaPackageSpitter
using Test

@testset "JuliaPackageSpitter.jl" begin
    # Test scaffolding a dummy package
    tmp_dir = mktempdir()
    spec = PackageSpec(
        "TestSpit",
        "A dummy package for testing",
        ["Author One"],
        :standard,
        false
    )

    result = generate_package(spec, tmp_dir)
    @test contains(result, "scaffolded successfully")
    @test isdir(joinpath(tmp_dir, "src"))
    @test isfile(joinpath(tmp_dir, "Project.toml"))
    @test isfile(joinpath(tmp_dir, "SONNET-TASKS.md"))
    @test isfile(joinpath(tmp_dir, "README.adoc"))

    # Check Project.toml content
    project_content = read(joinpath(tmp_dir, "Project.toml"), String)
    @test contains(project_content, "name = \"TestSpit\"")
    @test contains(project_content, "Author One")

    @testset "PackageSpec Field Validation" begin
        # Verify all fields are stored correctly
        spec = PackageSpec("MyPkg", "A description", ["Alice", "Bob"], :strict, true)
        @test spec.name == "MyPkg"
        @test spec.domain_summary == "A description"
        @test spec.authors == ["Alice", "Bob"]
        @test spec.ci_profile == :strict
        @test spec.ffi_enabled == true

        # Minimal CI profile
        spec_min = PackageSpec("MinPkg", "Minimal", ["Dev"], :minimal, false)
        @test spec_min.ci_profile == :minimal
        @test spec_min.ffi_enabled == false
    end

    @testset "FFI-Enabled Package Structure" begin
        tmp_ffi = mktempdir()
        spec_ffi = PackageSpec(
            "FFIPackage",
            "Package with FFI support",
            ["Jonathan D.A. Jewell"],
            :standard,
            true
        )

        result = generate_package(spec_ffi, tmp_ffi)
        @test contains(result, "scaffolded successfully")

        # FFI-specific directories should be created
        @test isdir(joinpath(tmp_ffi, "ffi", "zig"))
        @test isdir(joinpath(tmp_ffi, "src", "abi"))

        # Standard directories should also exist
        @test isdir(joinpath(tmp_ffi, "src"))
        @test isdir(joinpath(tmp_ffi, "test"))
        @test isdir(joinpath(tmp_ffi, "docs"))
        @test isdir(joinpath(tmp_ffi, "scripts"))
        @test isdir(joinpath(tmp_ffi, "contractiles"))
        @test isdir(joinpath(tmp_ffi, ".github", "workflows"))
        @test isdir(joinpath(tmp_ffi, ".machine_readable"))
    end

    @testset "FFI-Disabled Package Structure" begin
        tmp_no_ffi = mktempdir()
        spec_no_ffi = PackageSpec(
            "NoFFIPkg",
            "Package without FFI",
            ["Author"],
            :standard,
            false
        )

        generate_package(spec_no_ffi, tmp_no_ffi)

        # FFI directories should NOT be created
        @test !isdir(joinpath(tmp_no_ffi, "ffi", "zig"))
        @test !isdir(joinpath(tmp_no_ffi, "src", "abi"))

        # Standard directories should still exist
        @test isdir(joinpath(tmp_no_ffi, "src"))
        @test isdir(joinpath(tmp_no_ffi, "test"))
    end

    @testset "Multiple Authors in Project.toml" begin
        tmp_multi = mktempdir()
        spec_multi = PackageSpec(
            "MultiAuthor",
            "Multi-author package",
            ["Alice Smith", "Bob Jones", "Carol White"],
            :standard,
            false
        )

        generate_package(spec_multi, tmp_multi)
        content = read(joinpath(tmp_multi, "Project.toml"), String)
        @test contains(content, "Alice Smith")
        @test contains(content, "Bob Jones")
        @test contains(content, "Carol White")
        @test contains(content, "name = \"MultiAuthor\"")
    end

    @testset "Single Author Package" begin
        tmp_single = mktempdir()
        spec_single = PackageSpec(
            "SingleAuthor",
            "Solo developer",
            ["Solo Dev"],
            :minimal,
            false
        )

        generate_package(spec_single, tmp_single)
        content = read(joinpath(tmp_single, "Project.toml"), String)
        @test contains(content, "Solo Dev")
    end

    @testset "Generated Source File Content" begin
        tmp_src = mktempdir()
        spec_src = PackageSpec(
            "ContentCheck",
            "Verifying generated source file",
            ["Test Author"],
            :standard,
            false
        )

        generate_package(spec_src, tmp_src)

        # Main module file should exist and contain module declaration
        src_path = joinpath(tmp_src, "src", "ContentCheck.jl")
        @test isfile(src_path)
        src_content = read(src_path, String)
        @test contains(src_content, "module ContentCheck")
        @test contains(src_content, "SPDX-License-Identifier: PMPL-1.0-or-later")
        @test contains(src_content, "Verifying generated source file")
    end

    @testset "Project.toml Required Fields" begin
        tmp_toml = mktempdir()
        spec_toml = PackageSpec(
            "TomlCheck",
            "Checking TOML fields",
            ["Dev"],
            :standard,
            false
        )

        generate_package(spec_toml, tmp_toml)
        content = read(joinpath(tmp_toml, "Project.toml"), String)

        # Required fields
        @test contains(content, "name = \"TomlCheck\"")
        @test contains(content, "uuid = \"")
        @test contains(content, "version = \"0.1.0\"")
        @test contains(content, "[deps]")
        @test contains(content, "[compat]")
        @test contains(content, "julia = \"1.10\"")
    end

    @testset "Edge Case - Empty Authors List" begin
        tmp_empty_auth = mktempdir()
        spec_empty = PackageSpec(
            "EmptyAuthors",
            "No authors listed",
            String[],
            :standard,
            false
        )

        # Should still generate without error
        result = generate_package(spec_empty, tmp_empty_auth)
        @test contains(result, "scaffolded successfully")
        @test isfile(joinpath(tmp_empty_auth, "Project.toml"))
    end

    @testset "Special Characters in Domain Summary" begin
        tmp_special = mktempdir()
        spec_special = PackageSpec(
            "SpecialChars",
            "Handles <angles>, \"quotes\", & ampersands",
            ["Dev"],
            :standard,
            false
        )

        result = generate_package(spec_special, tmp_special)
        @test contains(result, "scaffolded successfully")

        # Source file should contain the summary
        src_content = read(joinpath(tmp_special, "src", "SpecialChars.jl"), String)
        @test contains(src_content, "module SpecialChars")
    end

    @testset "Idempotent Generation (Overwrite)" begin
        tmp_idem = mktempdir()
        spec_idem = PackageSpec(
            "IdempotentPkg",
            "First generation",
            ["Dev"],
            :standard,
            false
        )

        # Generate once
        generate_package(spec_idem, tmp_idem)
        @test isfile(joinpath(tmp_idem, "Project.toml"))

        # Generate again in the same directory (should not error)
        result = generate_package(spec_idem, tmp_idem)
        @test contains(result, "scaffolded successfully")
        @test isfile(joinpath(tmp_idem, "Project.toml"))
    end
end
