# SPDX-License-Identifier: PMPL-1.0-or-later
module Generator

using UUIDs
using Mustache
using JSON3

export PackageSpec, generate_package

struct PackageSpec
    name::String
    domain_summary::String
    authors::Vector{String}
    ci_profile::Symbol # :minimal, :standard, :strict
    ffi_enabled::Bool
end

"""
    generate_package(spec, target_dir)
Scaffolds a new Julia package repo using the RSR standard.
"""
function generate_package(spec::PackageSpec, target_dir::String)
    println("🛠️ Spitting out new package: $(spec.name)...")
    
    # Base template path (internal to the spitter package)
    tpl_dir = joinpath(@__DIR__, "..", "templates", "base")
    
    # 1. Create directory structure
    dirs = [
        "src",
        "test",
        "docs",
        "scripts",
        "contractiles",
        ".github/workflows",
        ".machine_readable"
    ]
    
    if spec.ffi_enabled
        push!(dirs, "ffi/zig")
        push!(dirs, "src/abi")
    end
    
    for d in dirs
        mkpath(joinpath(target_dir, d))
    end
    
    # 2. Generate Project.toml
    uuid = string(uuid4())
    project_tpl = """
    name = "{{name}}"
    uuid = "{{uuid}}"
    authors = [{{#authors}}"{{.}}"{{/authors}}]
    version = "0.1.0"

    [deps]
    Dates = "ade2ca70-3891-5945-931b-dc5ea821e773"

    [compat]
    julia = "1.10"
    """
    
    open(joinpath(target_dir, "Project.toml"), "w") do io
        print(io, render(project_tpl, name=spec.name, uuid=uuid, authors=spec.authors))
    end
    
    # 3. Generate main entry point
    src_tpl = """
    # SPDX-License-Identifier: PMPL-1.0-or-later
    module {{name}}

    # Domain: {{summary}}

    end # module
    """
    
    open(joinpath(target_dir, "src", "$(spec.name).jl"), "w") do io
        print(io, render(src_tpl, name=spec.name, summary=spec.domain_summary))
    end
    
    # 4. Use external templates for README and TASKS if they exist
    readme_tpl_path = joinpath(tpl_dir, "README.adoc.tpl")
    if isfile(readme_tpl_path)
        tpl_content = read(readme_tpl_path, String)
        open(joinpath(target_dir, "README.adoc"), "w") do io
            print(io, render(tpl_content, name=spec.name, summary=spec.domain_summary))
        end
    end

    tasks_tpl_path = joinpath(tpl_dir, "SONNET-TASKS.md.tpl")
    if isfile(tasks_tpl_path)
        tpl_content = read(tasks_tpl_path, String)
        open(joinpath(target_dir, "SONNET-TASKS.md"), "w") do io
            print(io, render(tpl_content, name=spec.name, summary=spec.domain_summary, uuid=uuid))
        end
    end
    
    return "Package $(spec.name) scaffolded successfully! 🚀"
end

end # module
