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
    @test contains(project_content, "name = "TestSpit"")
    @test contains(project_content, "authors = ["Author One"]")
end
