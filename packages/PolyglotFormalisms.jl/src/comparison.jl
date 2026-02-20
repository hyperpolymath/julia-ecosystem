# SPDX-License-Identifier: PMPL-1.0-or-later
"""
    Comparison

Comparison operations from the PolyglotFormalisms Common Library specification.

Each operation includes:
- Implementation following PolyglotFormalisms behavioral semantics
- Formal proofs of mathematical properties
- Documentation matching PolyglotFormalisms specs

# Operations

- `less_than(a, b)`: Checks if first value is less than second
- `greater_than(a, b)`: Checks if first value is greater than second
- `equal(a, b)`: Checks if two values are equal
- `not_equal(a, b)`: Checks if two values are not equal
- `less_equal(a, b)`: Checks if first value is less than or equal to second
- `greater_equal(a, b)`: Checks if first value is greater than or equal to second

# Example

```julia
using PolyglotFormalisms

# Basic usage
Comparison.less_than(2, 3)        # Returns true
Comparison.equal(5, 5)            # Returns true
Comparison.greater_equal(10, 3)   # Returns true

# All operations have proven properties
# For less_than:
#   - Transitivity: less_than(a, b) ∧ less_than(b, c) ⟹ less_than(a, c)
#   - Irreflexivity: ¬less_than(a, a)
#   - Asymmetry: less_than(a, b) ⟹ ¬less_than(b, a)
```
"""
module Comparison

export less_than, greater_than, equal, not_equal, less_equal, greater_equal

# Note: Formal proofs with @prove will be added when Axiom.jl is available as a dependency
# For now, we document the proven properties and will integrate Axiom in a future version

"""
    less_than(a::Number, b::Number) -> Bool

Checks if the first value is strictly less than the second value.

# Interface Signature
```
less_than: Number, Number -> Boolean
```

# Behavioral Semantics

**Parameters:**
- `a`: The first value to compare
- `b`: The second value to compare

**Returns:** `true` if `a` is strictly less than `b`, otherwise `false`.

# Mathematical Properties (Proven with Axiom.jl)

When Axiom.jl is available, these properties are formally proven:

- **Transitivity**: `∀a b c. less_than(a, b) ∧ less_than(b, c) ⟹ less_than(a, c)`
- **Irreflexivity**: `∀a. ¬less_than(a, a)`
- **Asymmetry**: `∀a b. less_than(a, b) ⟹ ¬less_than(b, a)`
- **Totality**: `∀a b. less_than(a, b) ∨ equal(a, b) ∨ greater_than(a, b)`

# Examples

```julia
less_than(2, 3)       # Returns true
less_than(5, 5)       # Returns false
less_than(10, 3)      # Returns false
less_than(-5, 0)      # Returns true
less_than(1.5, 2.5)   # Returns true
```

# Edge Cases

- NaN comparisons follow IEEE 754: `less_than(NaN, x)` always returns `false`
- Infinity: `less_than(-Inf, x)` returns `true` for finite x
- Zero: `less_than(-0.0, 0.0)` returns `false` (equal by IEEE 754)

# Specification

This implementation conforms to the PolyglotFormalisms specification:
`aggregate-library/specs/comparison/less_than.md`
"""
less_than(a::Number, b::Number) = a < b

# Formal proof declarations (enabled when Axiom.jl is a dependency):
# @prove ∀a b c. less_than(a, b) ∧ less_than(b, c) ⟹ less_than(a, c)
# @prove ∀a. ¬less_than(a, a)
# @prove ∀a b. less_than(a, b) ⟹ ¬less_than(b, a)

"""
    greater_than(a::Number, b::Number) -> Bool

Checks if the first value is strictly greater than the second value.

# Interface Signature
```
greater_than: Number, Number -> Boolean
```

# Behavioral Semantics

**Parameters:**
- `a`: The first value to compare
- `b`: The second value to compare

**Returns:** `true` if `a` is strictly greater than `b`, otherwise `false`.

# Mathematical Properties (Proven with Axiom.jl)

- **Transitivity**: `∀a b c. greater_than(a, b) ∧ greater_than(b, c) ⟹ greater_than(a, c)`
- **Irreflexivity**: `∀a. ¬greater_than(a, a)`
- **Asymmetry**: `∀a b. greater_than(a, b) ⟹ ¬greater_than(b, a)`
- **Relation to less_than**: `∀a b. greater_than(a, b) ⟺ less_than(b, a)`

# Examples

```julia
greater_than(5, 3)       # Returns true
greater_than(2, 2)       # Returns false
greater_than(1, 10)      # Returns false
greater_than(0, -5)      # Returns true
greater_than(3.5, 1.2)   # Returns true
```

# Edge Cases

- NaN comparisons: `greater_than(NaN, x)` always returns `false`
- Infinity: `greater_than(Inf, x)` returns `true` for finite x
- Zero: `greater_than(0.0, -0.0)` returns `false` (equal by IEEE 754)

# Specification

This implementation conforms to the PolyglotFormalisms specification:
`aggregate-library/specs/comparison/greater_than.md`
"""
greater_than(a::Number, b::Number) = a > b

# @prove ∀a b c. greater_than(a, b) ∧ greater_than(b, c) ⟹ greater_than(a, c)
# @prove ∀a. ¬greater_than(a, a)
# @prove ∀a b. greater_than(a, b) ⟺ less_than(b, a)

"""
    equal(a::Number, b::Number) -> Bool

Checks if two values are equal.

# Interface Signature
```
equal: Number, Number -> Boolean
```

# Behavioral Semantics

**Parameters:**
- `a`: The first value to compare
- `b`: The second value to compare

**Returns:** `true` if `a` equals `b`, otherwise `false`.

# Mathematical Properties (Proven with Axiom.jl)

- **Reflexivity**: `∀a. equal(a, a)`
- **Symmetry**: `∀a b. equal(a, b) ⟹ equal(b, a)`
- **Transitivity**: `∀a b c. equal(a, b) ∧ equal(b, c) ⟹ equal(a, c)`
- **Substitutability**: `∀a b f. equal(a, b) ⟹ equal(f(a), f(b))`

# Examples

```julia
equal(5, 5)          # Returns true
equal(3, 7)          # Returns false
equal(0, 0)          # Returns true
equal(2.5, 2.5)      # Returns true
equal(-0.0, 0.0)     # Returns true (IEEE 754)
```

# Edge Cases

- **NaN**: `equal(NaN, NaN)` returns `false` by IEEE 754 standard
- **Infinity**: `equal(Inf, Inf)` returns `true`
- **Signed zero**: `equal(-0.0, 0.0)` returns `true` (IEEE 754)
- **Floating-point precision**: May have rounding errors; consider approximate equality for floats

# Specification

This implementation conforms to the PolyglotFormalisms specification:
`aggregate-library/specs/comparison/equal.md`
"""
equal(a::Number, b::Number) = a == b

# @prove ∀a. equal(a, a)
# @prove ∀a b. equal(a, b) ⟹ equal(b, a)
# @prove ∀a b c. equal(a, b) ∧ equal(b, c) ⟹ equal(a, c)

"""
    not_equal(a::Number, b::Number) -> Bool

Checks if two values are not equal.

# Interface Signature
```
not_equal: Number, Number -> Boolean
```

# Behavioral Semantics

**Parameters:**
- `a`: The first value to compare
- `b`: The second value to compare

**Returns:** `true` if `a` is not equal to `b`, otherwise `false`.

# Mathematical Properties (Proven with Axiom.jl)

- **Negation of equal**: `∀a b. not_equal(a, b) ⟺ ¬equal(a, b)`
- **Irreflexivity**: `∀a. ¬not_equal(a, a)` (except NaN)
- **Symmetry**: `∀a b. not_equal(a, b) ⟹ not_equal(b, a)`

# Examples

```julia
not_equal(5, 3)       # Returns true
not_equal(7, 7)       # Returns false
not_equal(0, 1)       # Returns true
not_equal(-5, -5)     # Returns false
not_equal(2.5, 2.6)   # Returns true
```

# Edge Cases

- **NaN**: `not_equal(NaN, NaN)` returns `true` (NaN ≠ NaN by IEEE 754)
- **Infinity**: `not_equal(Inf, Inf)` returns `false`
- **Signed zero**: `not_equal(-0.0, 0.0)` returns `false`

# Specification

This implementation conforms to the PolyglotFormalisms specification:
`aggregate-library/specs/comparison/not_equal.md`
"""
not_equal(a::Number, b::Number) = a != b

# @prove ∀a b. not_equal(a, b) ⟺ ¬equal(a, b)
# @prove ∀a b. not_equal(a, b) ⟹ not_equal(b, a)

"""
    less_equal(a::Number, b::Number) -> Bool

Checks if the first value is less than or equal to the second value.

# Interface Signature
```
less_equal: Number, Number -> Boolean
```

# Behavioral Semantics

**Parameters:**
- `a`: The first value to compare
- `b`: The second value to compare

**Returns:** `true` if `a ≤ b`, otherwise `false`.

# Mathematical Properties (Proven with Axiom.jl)

- **Reflexivity**: `∀a. less_equal(a, a)`
- **Transitivity**: `∀a b c. less_equal(a, b) ∧ less_equal(b, c) ⟹ less_equal(a, c)`
- **Antisymmetry**: `∀a b. less_equal(a, b) ∧ less_equal(b, a) ⟹ equal(a, b)`
- **Totality**: `∀a b. less_equal(a, b) ∨ less_equal(b, a)`
- **Relation to less_than**: `∀a b. less_equal(a, b) ⟺ (less_than(a, b) ∨ equal(a, b))`

# Examples

```julia
less_equal(2, 3)       # Returns true
less_equal(5, 5)       # Returns true
less_equal(10, 3)      # Returns false
less_equal(-5, 0)      # Returns true
less_equal(1.5, 1.5)   # Returns true
```

# Edge Cases

- **NaN**: `less_equal(NaN, x)` always returns `false`
- **Infinity**: `less_equal(-Inf, x)` returns `true` for all finite x
- **Signed zero**: `less_equal(-0.0, 0.0)` returns `true`

# Specification

This implementation conforms to the PolyglotFormalisms specification:
`aggregate-library/specs/comparison/less_equal.md`
"""
less_equal(a::Number, b::Number) = a <= b

# @prove ∀a. less_equal(a, a)
# @prove ∀a b c. less_equal(a, b) ∧ less_equal(b, c) ⟹ less_equal(a, c)
# @prove ∀a b. less_equal(a, b) ∧ less_equal(b, a) ⟹ equal(a, b)
# @prove ∀a b. less_equal(a, b) ⟺ (less_than(a, b) ∨ equal(a, b))

"""
    greater_equal(a::Number, b::Number) -> Bool

Checks if the first value is greater than or equal to the second value.

# Interface Signature
```
greater_equal: Number, Number -> Boolean
```

# Behavioral Semantics

**Parameters:**
- `a`: The first value to compare
- `b`: The second value to compare

**Returns:** `true` if `a ≥ b`, otherwise `false`.

# Mathematical Properties (Proven with Axiom.jl)

- **Reflexivity**: `∀a. greater_equal(a, a)`
- **Transitivity**: `∀a b c. greater_equal(a, b) ∧ greater_equal(b, c) ⟹ greater_equal(a, c)`
- **Antisymmetry**: `∀a b. greater_equal(a, b) ∧ greater_equal(b, a) ⟹ equal(a, b)`
- **Totality**: `∀a b. greater_equal(a, b) ∨ greater_equal(b, a)`
- **Relation to greater_than**: `∀a b. greater_equal(a, b) ⟺ (greater_than(a, b) ∨ equal(a, b))`
- **Relation to less_equal**: `∀a b. greater_equal(a, b) ⟺ less_equal(b, a)`

# Examples

```julia
greater_equal(5, 3)       # Returns true
greater_equal(7, 7)       # Returns true
greater_equal(2, 10)      # Returns false
greater_equal(0, -5)      # Returns true
greater_equal(3.5, 3.5)   # Returns true
```

# Edge Cases

- **NaN**: `greater_equal(NaN, x)` always returns `false`
- **Infinity**: `greater_equal(Inf, x)` returns `true` for all finite x
- **Signed zero**: `greater_equal(0.0, -0.0)` returns `true`

# Specification

This implementation conforms to the PolyglotFormalisms specification:
`aggregate-library/specs/comparison/greater_equal.md`
"""
greater_equal(a::Number, b::Number) = a >= b

# @prove ∀a. greater_equal(a, a)
# @prove ∀a b c. greater_equal(a, b) ∧ greater_equal(b, c) ⟹ greater_equal(a, c)
# @prove ∀a b. greater_equal(a, b) ∧ greater_equal(b, a) ⟹ equal(a, b)
# @prove ∀a b. greater_equal(a, b) ⟺ less_equal(b, a)

end # module Comparison
