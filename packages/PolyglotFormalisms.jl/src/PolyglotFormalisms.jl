# SPDX-License-Identifier: PMPL-1.0-or-later
"""
    PolyglotFormalisms

Julia reference implementation of the aggregate-library (PolyglotFormalisms) Common Library.

This package provides formally verified implementations of the minimal overlap
functions specified in the PolyglotFormalisms project. Each function includes:

1. Implementation following the PolyglotFormalisms specification
2. Formal proofs of mathematical properties using Axiom.jl's @prove macro
3. Test cases matching the PolyglotFormalisms conformance suite

The goal is to demonstrate cross-language verification and serve as a reference
for semantic equivalence checking across ReScript, Julia, Gleam, Elixir, and
other hyperpolymath ecosystem languages.

# Modules

- `Arithmetic`: Basic arithmetic operations (add, subtract, multiply, divide, modulo)
- `Comparison`: Comparison operations (less_than, greater_than, equal, etc.)
- `Logical`: Boolean logic (and, or, not)
- `StringOps`: String operations (concat, length, substring, split, join, etc.)
- `Collection`: Collection operations (map, filter, fold, contains)
- `Conditional`: Conditional operations (if_then_else)

# Example

```julia
using PolyglotFormalisms

# All operations include formal proofs
result = Arithmetic.add(2, 3)  # Returns 5

# Properties are proven at compile time:
# @prove ∀a b. add(a, b) == add(b, a)  # Commutativity
# @prove ∀a b c. add(add(a, b), c) == add(a, add(b, c))  # Associativity
# @prove ∀a. add(a, 0) == a  # Identity
```

# Design Philosophy

This implementation follows the PolyglotFormalisms specification format:
- Minimal intersection across 7+ radically different languages
- Clear behavioral semantics
- Executable test cases
- Mathematical properties proven with Axiom.jl

For full specification details, see:
https://github.com/hyperpolymath/aggregate-library
"""
module PolyglotFormalisms

export Arithmetic, Comparison, Logical, StringOps, Collection, Conditional

include("arithmetic.jl")
include("comparison.jl")
include("logical.jl")
include("string.jl")
include("collection.jl")
include("conditional.jl")

end # module PolyglotFormalisms
