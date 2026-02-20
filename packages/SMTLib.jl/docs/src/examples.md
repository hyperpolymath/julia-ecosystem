# Examples

## Basic Constraint Solving

```julia
using SMTLib

# Linear integer arithmetic
ctx = SMTContext(logic=:QF_LIA)
declare(ctx, :x, Int)
declare(ctx, :y, Int)
assert!(ctx, :(x + 2*y == 10))
assert!(ctx, :(x >= 0))
assert!(ctx, :(y >= 0))

result = check_sat(ctx)
@show result.model  # Dict(:x => 4, :y => 3) or similar
```

## Incremental Solving

```julia
ctx = SMTContext(logic=:QF_LIA)
declare(ctx, :x, Int)
assert!(ctx, :(x > 5))

# First check
result1 = check_sat(ctx)  # sat

# Add more constraints
push!(ctx)
assert!(ctx, :(x < 3))
result2 = check_sat(ctx)  # unsat

# Backtrack
pop!(ctx)
result3 = check_sat(ctx)  # sat again
```

## Bitvector Constraints

```julia
ctx = SMTContext(logic=:QF_BV)
declare(ctx, :a, BitVec{8})
declare(ctx, :b, BitVec{8})

# Bitvector operations
assert!(ctx, :(bvadd(a, b) == 0xFF))
assert!(ctx, :(bvand(a, b) == 0x00))

result = check_sat(ctx)
```

## Real Arithmetic

```julia
ctx = SMTContext(logic=:QF_LRA)
declare(ctx, :x, Real)
declare(ctx, :y, Real)

assert!(ctx, :(x + y > 10.5))
assert!(ctx, :(x - y < 2.0))
assert!(ctx, :(x > 0))

result = check_sat(ctx)
```

## Array Theory

```julia
ctx = SMTContext(logic=:QF_AUFLIA)
declare(ctx, :arr, Array{Int,Int})
declare(ctx, :i, Int)
declare(ctx, :j, Int)

# Array constraints
assert!(ctx, :(select(arr, i) == 42))
assert!(ctx, :(select(store(arr, j, 100), j) == 100))
assert!(ctx, :(i != j))

result = check_sat(ctx)
```

## Unsat Core

```julia
ctx = SMTContext(logic=:QF_LIA)
declare(ctx, :x, Int)

# Named assertions
assert!(ctx, :(x > 10), name=:c1)
assert!(ctx, :(x < 5), name=:c2)
assert!(ctx, :(x >= 0), name=:c3)

result = check_sat(ctx, unsat_core=true)
@show result.unsat_core  # [:c1, :c2] - conflicting constraints
```

## Timeout Handling

```julia
ctx = SMTContext(logic=:QF_NRA, timeout=5000)  # 5 second timeout
declare(ctx, :x, Real)
assert!(ctx, :(x^5 + x^3 + x == 42))  # Hard nonlinear constraint

result = check_sat(ctx)
if result.status == :timeout
    println("Solver timed out")
end
```

## Using the @smt Macro

```julia
# Convenient syntax for simple queries
result = @smt begin
    x::Int
    y::Int
    x + y == 10
    x > y
    x > 0
end

if result.status == :sat
    println("x = ", result.model[:x])
    println("y = ", result.model[:y])
end
```

## Multiple Solvers

```julia
# Find all available solvers
solvers = available_solvers()
for solver in solvers
    println("Found: ", solver.kind, " at ", solver.path)
end

# Use a specific solver
z3 = find_solver(:z3)
ctx = SMTContext(solver=z3, logic=:QF_LIA)
```

## Complex Constraints

```julia
ctx = SMTContext(logic=:QF_NIA)
declare(ctx, :x, Int)
declare(ctx, :y, Int)
declare(ctx, :z, Int)

# Nonlinear constraints
assert!(ctx, :(x * y + z == 100))
assert!(ctx, :(x^2 - y^2 == 16))
assert!(ctx, :(z > 0))

result = check_sat(ctx)
if result.status == :sat
    @show result.model
end
```
