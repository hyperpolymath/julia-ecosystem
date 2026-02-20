# SPDX-License-Identifier: PMPL-1.0-or-later
# Copyright (c) 2026 Jonathan D.A. Jewell <jonathan.jewell@open.ac.uk>

using Test
using ZeroProb
using Distributions
using LinearAlgebra

@testset "ZeroProb.jl Tests" begin
    # Original test suites
    include("test_types.jl")
    include("test_measures.jl")
    include("test_paradoxes.jl")
    include("test_applications.jl")

    # Extended test suites (Phase 2, 3, 4, 5, 7)
    include("test_new_types.jl")
    include("test_new_measures.jl")
    include("test_new_paradoxes.jl")
end
