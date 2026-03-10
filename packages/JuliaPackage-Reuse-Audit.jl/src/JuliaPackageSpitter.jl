# SPDX-License-Identifier: PMPL-1.0-or-later

"""
    JuliaPackageSpitter

Automated Julia package scaffolding and reuse auditing tool. Generates
compliant package structures from templates and audits existing packages
for code reuse opportunities across the ecosystem.

# Key Features
- Package generation from configurable `PackageSpec` templates
- Automated Project.toml, test, and CI scaffolding
- Cross-package reuse pattern detection

# Example
```julia
using JuliaPackageSpitter
spec = PackageSpec(name="MyPackage", uuid=uuid4())
generate_package(spec, "/path/to/output")
```
"""
module JuliaPackageSpitter

include("generator.jl")

using .Generator
export PackageSpec, generate_package

end # module
