# SPDX-License-Identifier: PMPL-1.0-or-later

"""
Basic SAT checking example for SMTLib.jl

This example demonstrates the core workflow:
1. Create an SMT context
2. Declare variables
3. Add assertions
4. Check satisfiability
"""

using SMTLib

# Create a context with quantifier-free linear integer arithmetic
ctx = SMTContext(logic=:QF_LIA)

# Declare variables
declare(ctx, :x, Int)
declare(ctx, :y, Int)

# Add constraints
assert!(ctx, :(x > 0))
assert!(ctx, :(y > 0))
assert!(ctx, :(x + y == 10))
assert!(ctx, :(x < y))

# Show the generated SMT-LIB2 script
println("Generated SMT-LIB2 script:")
println("=" ^ 50)
script = SMTLib.build_script(ctx, true)
println(script)
println("=" ^ 50)

# Check satisfiability (requires an installed solver)
solvers = available_solvers()
if !isempty(solvers)
    println("\nAvailable solvers: ", solvers)
    result = check_sat(ctx)
    println("Result: ", result.status)
    if result.status == :sat
        println("Model: ", result.model)
    end
else
    println("\nNo SMT solver found. Install z3, cvc5, or yices to run the solver.")
end
