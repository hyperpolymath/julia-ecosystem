# SPDX-License-Identifier: PMPL-1.0-or-later
module JuliaPackageSpitter

include("generator.jl")

using .Generator
export PackageSpec, generate_package

end # module
