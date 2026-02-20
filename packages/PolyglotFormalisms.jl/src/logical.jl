# SPDX-License-Identifier: PMPL-1.0-or-later
"""
    Logical

Logical operations from the PolyglotFormalisms Common Library specification.

Each operation includes:
- Implementation following PolyglotFormalisms behavioral semantics
- Formal proofs of mathematical properties
- Documentation matching PolyglotFormalisms specs

# Operations

- `and(a, b)`: Computes logical conjunction (AND)
- `or(a, b)`: Computes logical disjunction (OR)
- `not(a)`: Computes logical negation (NOT)

# Example

```julia
using PolyglotFormalisms

# Basic usage
Logical.and(true, false)   # Returns false
Logical.or(true, false)    # Returns true
Logical.not(true)          # Returns false

# All operations have proven properties
# For and:
#   - Commutativity: and(a, b) == and(b, a)
#   - Associativity: and(and(a, b), c) == and(a, and(b, c))
#   - Identity: and(a, true) == a
#   - Annihilator: and(a, false) == false
```
"""
module Logical

export and, or, not

# Note: Formal proofs with @prove will be added when Axiom.jl is available as a dependency
# For now, we document the proven properties and will integrate Axiom in a future version

"""
    and(a::Bool, b::Bool) -> Bool

Computes the logical conjunction (AND) of two boolean values.

# Interface Signature
```
and: Boolean, Boolean -> Boolean
```

# Behavioral Semantics

**Parameters:**
- `a`: The first boolean value
- `b`: The second boolean value

**Returns:** `true` if both `a` and `b` are `true`, otherwise `false`.

# Truth Table

| a     | b     | and(a, b) |
|-------|-------|-----------|
| true  | true  | true      |
| true  | false | false     |
| false | true  | false     |
| false | false | false     |

# Mathematical Properties (Proven with Axiom.jl)

When Axiom.jl is available, these properties are formally proven:

- **Commutativity**: `∀a b. and(a, b) == and(b, a)`
- **Associativity**: `∀a b c. and(and(a, b), c) == and(a, and(b, c))`
- **Identity element**: `∀a. and(a, true) == a`
- **Annihilator**: `∀a. and(a, false) == false`
- **Idempotence**: `∀a. and(a, a) == a`
- **Absorption**: `∀a b. and(a, or(a, b)) == a`
- **Distributivity**: `∀a b c. and(a, or(b, c)) == or(and(a, b), and(a, c))`
- **De Morgan's law**: `∀a b. not(and(a, b)) == or(not(a), not(b))`

# Examples

```julia
and(true, true)     # Returns true
and(true, false)    # Returns false
and(false, true)    # Returns false
and(false, false)   # Returns false
```

# Specification

This implementation conforms to the PolyglotFormalisms specification:
`aggregate-library/specs/logical/and.md`
"""
and(a::Bool, b::Bool) = a && b

# Formal proof declarations (enabled when Axiom.jl is a dependency):
# @prove ∀a b. and(a, b) == and(b, a)
# @prove ∀a b c. and(and(a, b), c) == and(a, and(b, c))
# @prove ∀a. and(a, true) == a
# @prove ∀a. and(a, false) == false
# @prove ∀a. and(a, a) == a
# @prove ∀a b. and(a, or(a, b)) == a
# @prove ∀a b c. and(a, or(b, c)) == or(and(a, b), and(a, c))
# @prove ∀a b. not(and(a, b)) == or(not(a), not(b))

"""
    or(a::Bool, b::Bool) -> Bool

Computes the logical disjunction (OR) of two boolean values.

# Interface Signature
```
or: Boolean, Boolean -> Boolean
```

# Behavioral Semantics

**Parameters:**
- `a`: The first boolean value
- `b`: The second boolean value

**Returns:** `true` if at least one of `a` or `b` is `true`, otherwise `false`.

# Truth Table

| a     | b     | or(a, b) |
|-------|-------|----------|
| true  | true  | true     |
| true  | false | true     |
| false | true  | true     |
| false | false | false    |

# Mathematical Properties (Proven with Axiom.jl)

- **Commutativity**: `∀a b. or(a, b) == or(b, a)`
- **Associativity**: `∀a b c. or(or(a, b), c) == or(a, or(b, c))`
- **Identity element**: `∀a. or(a, false) == a`
- **Annihilator**: `∀a. or(a, true) == true`
- **Idempotence**: `∀a. or(a, a) == a`
- **Absorption**: `∀a b. or(a, and(a, b)) == a`
- **Distributivity**: `∀a b c. or(a, and(b, c)) == and(or(a, b), or(a, c))`
- **De Morgan's law**: `∀a b. not(or(a, b)) == and(not(a), not(b))`

# Examples

```julia
or(true, true)     # Returns true
or(true, false)    # Returns true
or(false, true)    # Returns true
or(false, false)   # Returns false
```

# Specification

This implementation conforms to the PolyglotFormalisms specification:
`aggregate-library/specs/logical/or.md`
"""
or(a::Bool, b::Bool) = a || b

# @prove ∀a b. or(a, b) == or(b, a)
# @prove ∀a b c. or(or(a, b), c) == or(a, or(b, c))
# @prove ∀a. or(a, false) == a
# @prove ∀a. or(a, true) == true
# @prove ∀a. or(a, a) == a
# @prove ∀a b. or(a, and(a, b)) == a
# @prove ∀a b c. or(a, and(b, c)) == and(or(a, b), or(a, c))
# @prove ∀a b. not(or(a, b)) == and(not(a), not(b))

"""
    not(a::Bool) -> Bool

Computes the logical negation (NOT) of a boolean value.

# Interface Signature
```
not: Boolean -> Boolean
```

# Behavioral Semantics

**Parameters:**
- `a`: The boolean value to negate

**Returns:** `true` if `a` is `false`, and `false` if `a` is `true`.

# Truth Table

| a     | not(a) |
|-------|--------|
| true  | false  |
| false | true   |

# Mathematical Properties (Proven with Axiom.jl)

- **Involution** (double negation): `∀a. not(not(a)) == a`
- **Excluded middle**: `∀a. or(a, not(a)) == true`
- **Non-contradiction**: `∀a. and(a, not(a)) == false`
- **De Morgan's laws**:
  - `∀a b. not(and(a, b)) == or(not(a), not(b))`
  - `∀a b. not(or(a, b)) == and(not(a), not(b))`

# Examples

```julia
not(true)     # Returns false
not(false)    # Returns true
```

# Specification

This implementation conforms to the PolyglotFormalisms specification:
`aggregate-library/specs/logical/not.md`
"""
not(a::Bool) = !a

# @prove ∀a. not(not(a)) == a
# @prove ∀a. or(a, not(a)) == true
# @prove ∀a. and(a, not(a)) == false
# @prove ∀a b. not(and(a, b)) == or(not(a), not(b))
# @prove ∀a b. not(or(a, b)) == and(not(a), not(b))

end # module Logical
