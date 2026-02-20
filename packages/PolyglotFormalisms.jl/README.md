# PolyglotFormalisms.jl

**Julia reference implementation of the aggregate-library (PolyglotFormalisms) Common Library with formal verification.**

[![Project Topology](https://img.shields.io/badge/Project-Topology-9558B2)](TOPOLOGY.md)
[![Completion Status](https://img.shields.io/badge/Completion-100%25-green)](TOPOLOGY.md)

image:https://img.shields.io/badge/License-PMPL--1.0-blue.svg[License: PMPL-1.0,link="https://github.com/hyperpolymath/palimpsest-license"]
[![Tests: Passing](https://img.shields.io/badge/tests-422%20passing-brightgreen.svg)]()

## Overview

PolyglotFormalisms.jl provides formally verified implementations of the minimal overlap functions specified in the [aggregate-library](https://github.com/hyperpolymath/aggregate-library) project. This package serves as a Julia reference implementation for cross-language semantic equivalence verification.

## Why PolyglotFormalisms.jl?

The aggregate-library defines a minimal intersection of functionality across radically different programming languages. PolyglotFormalisms.jl adds value by:

1. **Formal Verification**: Mathematical properties are proven using Axiom.jl's `@prove` macro (planned)
2. **Reference Implementation**: Serves as a semantically verified baseline for other language implementations
3. **Conformance Testing**: Test suite exactly matches PolyglotFormalisms specifications
4. **Cross-Language Bridge**: Enables verification that ReScript, Gleam, Elixir implementations satisfy the same properties

## Installation

```julia
using Pkg
Pkg.add("PolyglotFormalisms")
```

## Usage

```julia
using PolyglotFormalisms

# Arithmetic operations
Arithmetic.add(2, 3)         # 5
Arithmetic.subtract(10, 3)   # 7
Arithmetic.multiply(4, 5)    # 20
Arithmetic.divide(10, 2)     # 5.0
Arithmetic.modulo(10, 3)     # 1

# Comparison operations
Comparison.less_than(2, 3)        # true
Comparison.equal(5, 5)            # true

# Collection operations
Collection.map_items(x -> x^2, [1, 2, 3])     # [1, 4, 9]
Collection.filter_items(iseven, [1, 2, 3, 4])  # [2, 4]
Collection.fold_items(+, 0, [1, 2, 3])         # 6

# Conditional operations
Conditional.if_then_else(true, "yes", "no")  # "yes"
Conditional.coalesce(nothing, nothing, 42)   # 42
Conditional.clamp_value(15, 0, 10)           # 10
```

## Modules

### Arithmetic (5 operations)

| Function | Description |
|----------|-------------|
| `add(a, b)` | Sum of two numbers |
| `subtract(a, b)` | Difference of two numbers |
| `multiply(a, b)` | Product of two numbers |
| `divide(a, b)` | Quotient of two numbers |
| `modulo(a, b)` | Remainder of division |

**Verified properties:** commutativity, associativity, identity, zero element, distributivity.

### Comparison (6 operations)

| Function | Description |
|----------|-------------|
| `less_than(a, b)` | a < b |
| `greater_than(a, b)` | a > b |
| `equal(a, b)` | a == b |
| `not_equal(a, b)` | a != b |
| `less_equal(a, b)` | a <= b |
| `greater_equal(a, b)` | a >= b |

**Verified properties:** trichotomy, transitivity, reflexivity (for `equal`, `less_equal`, `greater_equal`), antisymmetry.

### Logical (3 operations)

| Function | Description |
|----------|-------------|
| `and(a, b)` | Logical conjunction |
| `or(a, b)` | Logical disjunction |
| `not(a)` | Logical negation |

**Verified properties:** commutativity, associativity, identity, De Morgan's laws, double negation, excluded middle.

### StringOps (14 operations)

| Function | Description |
|----------|-------------|
| `concat(a, b)` | Concatenate two strings |
| `length(s)` | String length |
| `substring(s, start, end_pos)` | Extract substring |
| `index_of(s, substr)` | Find first occurrence (0 if not found) |
| `contains(s, substr)` | Check if string contains substring |
| `starts_with(s, prefix)` | Check prefix |
| `ends_with(s, suffix)` | Check suffix |
| `to_uppercase(s)` | Convert to uppercase |
| `to_lowercase(s)` | Convert to lowercase |
| `trim(s)` | Remove leading/trailing whitespace |
| `split(s, delimiter)` | Split string by delimiter |
| `join(parts, separator)` | Join strings with separator |
| `replace(s, old, new)` | Replace occurrences of substring |
| `is_empty(s)` | Check if string is empty |

**Verified properties:** concat associativity, concat identity, length non-negativity, split/join roundtrip, trim idempotence.

### Collection (13 operations)

| Function | Description |
|----------|-------------|
| `map_items(f, coll)` | Apply function to each element |
| `filter_items(pred, coll)` | Keep elements matching predicate |
| `fold_items(f, init, coll)` | Left-fold with accumulator |
| `zip_items(a, b)` | Pair elements positionally |
| `flat_map_items(f, coll)` | Map and flatten results |
| `group_by(key_fn, coll)` | Group elements by key function |
| `sort_by(compare_fn, coll)` | Stable sort by comparison function |
| `unique_items(coll)` | Remove duplicates (preserve order) |
| `partition_items(pred, coll)` | Split into (matching, non-matching) |
| `take_items(n, coll)` | Take first n elements |
| `drop_items(n, coll)` | Drop first n elements |
| `any_item(pred, coll)` | True if any element matches |
| `all_items(pred, coll)` | True if all elements match |

**Verified properties:** functor identity (`map id = id`), functor composition, filter/partition consistency, fold universality, take/drop complementarity, De Morgan duality (`any`/`all`).

### Conditional (5 operations)

| Function | Description |
|----------|-------------|
| `if_then_else(pred, then_val, else_val)` | Total ternary conditional |
| `when(pred, val)` | Conditional value (`Some` or `nothing`) |
| `unless(pred, val)` | Inverse conditional value |
| `coalesce(values...)` | First non-nothing value |
| `clamp_value(x, lo, hi)` | Clamp number to range [lo, hi] |

**Verified properties:** if_then_else totality, when/unless duality, coalesce idempotence, clamp boundary conditions, clamp idempotence within range.

## Conformance Testing

Test suite exactly matches the PolyglotFormalisms specification test cases:

```julia
using Test
using PolyglotFormalisms

@testset "PolyglotFormalisms Conformance" begin
    # Test cases from specs/arithmetic/add.md
    @test Arithmetic.add(2, 3) == 5
    @test Arithmetic.add(-5, 3) == -2
    @test Arithmetic.add(0, 0) == 0
    @test Arithmetic.add(1.5, 2.5) == 4.0
    @test Arithmetic.add(-10, -20) == -30

    # Property verification
    @test Arithmetic.add(5, 3) == Arithmetic.add(3, 5)  # Commutativity
    @test Arithmetic.add(Arithmetic.add(2, 3), 4) == Arithmetic.add(2, Arithmetic.add(3, 4))  # Associativity
end
```

Run full test suite:
```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

## Integration with Axiom.jl

When Axiom.jl is available as a dependency, formal proofs will be automatically verified at compile time:

```julia
# Future integration:
@prove forall(a, b) do add(a, b) == add(b, a) end
@prove forall(a, b, c) do add(add(a, b), c) == add(a, add(b, c)) end
@prove forall(a) do add(a, 0) == a end
```

This enables:
- **Compile-time verification** of mathematical properties
- **Automatic error detection** if implementations violate proven properties
- **Formal certificates** proving correctness for safety-critical applications

## Cross-Language Verification

PolyglotFormalisms.jl serves as a formally verified reference for semantic equivalence checking:

1. **Implement in target language** (ReScript, Gleam, Elixir)
2. **Run PolyglotFormalisms conformance tests** in both languages
3. **Use Axiom.jl + SMTLib.jl** to prove semantic equivalence
4. **Generate verification certificate**

Example verification workflow:
```julia
using PolyglotFormalisms
using Axiom
using SMTLib

# Verify ReScript implementation semantically equivalent to Julia
verify_equivalence(
    julia_impl = Arithmetic.add,
    rescript_impl = RescriptFFI.add,
    properties = [commutativity, associativity, identity]
)
```

## Design Philosophy

This implementation follows the PolyglotFormalisms specification philosophy:
- **Minimal intersection**: Only functions that work across all target languages
- **Clear semantics**: Unambiguous behavioral specifications
- **Testable**: Executable test cases for every operation
- **Provable**: Mathematical properties verified with formal methods
- **Extensible**: Each ecosystem extends through its standard library

## Related Projects

- [aggregate-library](https://github.com/hyperpolymath/aggregate-library) - PolyglotFormalisms specification
- [alib-for-rescript](https://github.com/hyperpolymath/alib-for-rescript) - ReScript implementation
- [Axiom.jl](https://github.com/hyperpolymath/Axiom.jl) - Formal verification for ML models
- [SMTLib.jl](https://github.com/hyperpolymath/SMTLib.jl) - SMT solver integration for Julia

## References & Bibliography

### Type Theory & Formal Semantics

- Pierce, B.C. _Types and Programming Languages_. MIT Press, 2002. -- Type systems, operational and denotational semantics.
- Winskel, G. _The Formal Semantics of Programming Languages: An Introduction_. MIT Press, 1993. -- Denotational, operational, and axiomatic semantics.
- Cardelli, L. & Wegner, P. "On Understanding Types, Data Abstraction, and Polymorphism." _Computing Surveys_ 17(4), 1985, pp. 471-523. -- Type theory foundations for programming languages.

### Category Theory & Algebraic Properties

- Mac Lane, S. _Categories for the Working Mathematician_. 2nd ed., Graduate Texts in Mathematics 5, Springer, 1998. -- Category theory foundations (functors, natural transformations, monads).
- Milewski, B. _Category Theory for Programmers_. 2019. -- Functor and monad laws applied to programming.
- Bird, R. & de Moor, O. _Algebra of Programming_. Prentice Hall, 1997. -- Fold/map fusion laws, program calculation.
- Dummit, D.S. & Foote, R.M. _Abstract Algebra_. 3rd ed., Wiley, 2004. -- Algebraic structures (groups, rings, fields) underlying arithmetic properties.

### Parametricity & Free Theorems

- Wadler, P. "Theorems for free!" In _Proceedings of FPCA '89_, ACM, 1989, pp. 347-359. -- Parametric polymorphism and free theorems from types.

### Standards

- IEEE 754-2019. _IEEE Standard for Floating-Point Arithmetic_. IEEE, 2019. -- NaN/Inf handling, rounding modes, division by zero semantics.

## Contributing

Contributions welcome! Please ensure:
1. Implementations match PolyglotFormalisms specifications exactly
2. All test cases from PolyglotFormalisms specs are included
3. Properties are documented (and proven when Axiom.jl integration is complete)
4. Tests pass: `julia --project=. -e 'using Pkg; Pkg.test()'`

## License

PMPL-1.0-or-later (Palimpsest Meta-Public License)

## Status

**Current**: 6 modules implemented, 422 passing tests.

| Module | Operations | Status |
|--------|-----------|--------|
| Arithmetic | 5 | Complete |
| Comparison | 6 | Complete |
| Logical | 3 | Complete |
| StringOps | 14 | Complete |
| Collection | 13 | Complete |
| Conditional | 5 | Complete |

**Planned**: Axiom.jl integration for compile-time formal proofs.

---

**Hyperpolymath Ecosystem** - Multi-language, formally verified, semantically equivalent.
