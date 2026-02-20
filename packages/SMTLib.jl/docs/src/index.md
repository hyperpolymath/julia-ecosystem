# SMTLib.jl

A lightweight Julia interface to SMT solvers via SMT-LIB2 format.

## Features

- **Auto-detection** of installed SMT solvers (Z3, CVC5, Yices, MathSAT)
- **Julia expression to SMT-LIB2** conversion
- **Multiple logics**: QF_LIA, QF_LRA, QF_NRA, QF_BV, arrays, and more
- **Model parsing** and counterexample extraction
- **Timeout support**
- **Incremental solving** with push/pop semantics
- **Zero dependencies** - pure Julia

## Quick Start

```julia
using SMTLib

# Create an SMT context
ctx = SMTContext(logic=:QF_LIA)

# Declare variables
declare(ctx, :x, Int)
declare(ctx, :y, Int)

# Add constraints
assert!(ctx, :(x + y == 10))
assert!(ctx, :(x > 0))
assert!(ctx, :(y > 0))

# Check satisfiability
result = check_sat(ctx)

if result.status == :sat
    println("Solution found:")
    println("x = ", result.model[:x])
    println("y = ", result.model[:y])
end
```

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/hyperpolymath/SMTLib.jl")
```

### Prerequisites

Install at least one SMT solver:

```bash
# Z3 (recommended)
brew install z3          # macOS
apt install z3           # Ubuntu/Debian
pacman -S z3             # Arch

# CVC5
brew install cvc5        # macOS
apt install cvc5         # Ubuntu/Debian
```

## What is SMT?

**Satisfiability Modulo Theories (SMT)** extends boolean satisfiability (SAT) with theories like arithmetic, arrays, and bitvectors. SMT solvers are used for:

- **Formal verification** - proving program correctness
- **Symbolic execution** - exploring execution paths
- **Constraint solving** - finding solutions to complex constraints
- **Test generation** - generating inputs that trigger bugs
- **Program synthesis** - generating programs from specifications

## Supported Solvers

- **Z3** (Microsoft Research) - Most feature-complete
- **CVC5** - Strong theory support
- **Yices** - Fast for linear arithmetic
- **MathSAT** - Good for optimization

## Supported Logics

| Logic | Description |
|-------|-------------|
| QF_LIA | Quantifier-free linear integer arithmetic |
| QF_LRA | Quantifier-free linear real arithmetic |
| QF_NIA | Quantifier-free nonlinear integer arithmetic |
| QF_NRA | Quantifier-free nonlinear real arithmetic |
| QF_BV | Quantifier-free bitvectors |
| QF_AUFLIA | Arrays, uninterpreted functions, linear integer arithmetic |
| LIA | Linear integer arithmetic with quantifiers |
| LRA | Linear real arithmetic with quantifiers |
| AUFLIRA | Arrays, uninterpreted functions, linear arithmetic |
| ALL | All supported theories |

## License

SMTLib.jl is licensed under the [PMPL-1.0-or-later](https://github.com/hyperpolymath/palimpsest-license) license.
