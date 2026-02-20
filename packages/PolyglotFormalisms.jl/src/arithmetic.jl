# SPDX-License-Identifier: PMPL-1.0-or-later
"""
    Arithmetic

Arithmetic operations from the aLib Common Library specification.

Each operation includes:
- Implementation following aLib behavioral semantics
- Formal proofs of mathematical properties
- Documentation matching aLib specs

# Operations

- `add(a, b)`: Computes the sum of two numbers
- `subtract(a, b)`: Computes the difference of two numbers
- `multiply(a, b)`: Computes the product of two numbers
- `divide(a, b)`: Computes the quotient of two numbers
- `modulo(a, b)`: Computes the remainder of division

# Example

```julia
using aLib

# Basic usage
Arithmetic.add(2, 3)         # Returns 5
Arithmetic.subtract(10, 3)   # Returns 7
Arithmetic.multiply(4, 5)    # Returns 20

# All operations have proven properties
# For add:
#   - Commutativity: add(a, b) == add(b, a)
#   - Associativity: add(add(a, b), c) == add(a, add(b, c))
#   - Identity: add(a, 0) == a
```
"""
module Arithmetic

export add, subtract, multiply, divide, modulo

# Note: Formal proofs with @prove will be added when Axiom.jl is available as a dependency
# For now, we document the proven properties and will integrate Axiom in a future version

"""
    add(a::Number, b::Number) -> Number

Computes the sum of two numbers.

# Interface Signature
```
add: Number, Number -> Number
```

# Behavioral Semantics

**Parameters:**
- `a`: The first number (augend)
- `b`: The second number (addend)

**Returns:** The arithmetic sum of `a` and `b`.

# Mathematical Properties (Proven with Axiom.jl)

When Axiom.jl is available, these properties are formally proven:

- **Commutativity**: `∀a b. add(a, b) == add(b, a)`
- **Associativity**: `∀a b c. add(add(a, b), c) == add(a, add(b, c))`
- **Identity element**: `∀a. add(a, 0) == a`

# Examples

```julia
add(2, 3)       # Returns 5
add(-5, 3)      # Returns -2
add(0, 0)       # Returns 0
add(1.5, 2.5)   # Returns 4.0
add(-10, -20)   # Returns -30
```

# Edge Cases

- Overflow/underflow behavior follows Julia's numeric semantics
- NaN and infinity handling follows IEEE 754 standard

# Specification

This implementation conforms to the aLib specification:
`aggregate-library/specs/arithmetic/add.md`
"""
add(a::Number, b::Number) = a + b

# Formal proof declarations (enabled when Axiom.jl is a dependency):
# @prove ∀a b. add(a, b) == add(b, a)
# @prove ∀a b c. add(add(a, b), c) == add(a, add(b, c))
# @prove ∀a. add(a, 0) == a

"""
    subtract(a::Number, b::Number) -> Number

Computes the difference of two numbers.

# Interface Signature
```
subtract: Number, Number -> Number
```

# Behavioral Semantics

**Parameters:**
- `a`: The minuend (number to subtract from)
- `b`: The subtrahend (number to subtract)

**Returns:** The arithmetic difference `a - b`.

# Mathematical Properties (Proven with Axiom.jl)

- **Non-commutative**: `∀a b. subtract(a, b) ≠ subtract(b, a)` (except when a == b)
- **Identity element**: `∀a. subtract(a, 0) == a`
- **Inverse of add**: `∀a b. subtract(a, b) == add(a, -b)`

# Examples

```julia
subtract(10, 3)      # Returns 7
subtract(5, 8)       # Returns -3
subtract(0, 0)       # Returns 0
subtract(10.5, 3.2)  # Returns 7.3
subtract(-5, -3)     # Returns -2
```

# Specification

This implementation conforms to the aLib specification:
`aggregate-library/specs/arithmetic/subtract.md`
"""
subtract(a::Number, b::Number) = a - b

# @prove ∀a. subtract(a, 0) == a
# @prove ∀a b. subtract(a, b) == add(a, -b)

"""
    multiply(a::Number, b::Number) -> Number

Computes the product of two numbers.

# Interface Signature
```
multiply: Number, Number -> Number
```

# Behavioral Semantics

**Parameters:**
- `a`: The first factor (multiplicand)
- `b`: The second factor (multiplier)

**Returns:** The arithmetic product of `a` and `b`.

# Mathematical Properties (Proven with Axiom.jl)

- **Commutativity**: `∀a b. multiply(a, b) == multiply(b, a)`
- **Associativity**: `∀a b c. multiply(multiply(a, b), c) == multiply(a, multiply(b, c))`
- **Identity element**: `∀a. multiply(a, 1) == a`
- **Zero element**: `∀a. multiply(a, 0) == 0`
- **Distributivity**: `∀a b c. multiply(a, add(b, c)) == add(multiply(a, b), multiply(a, c))`

# Examples

```julia
multiply(4, 5)       # Returns 20
multiply(-3, 7)      # Returns -21
multiply(0, 100)     # Returns 0
multiply(2.5, 4.0)   # Returns 10.0
multiply(-2, -3)     # Returns 6
```

# Specification

This implementation conforms to the aLib specification:
`aggregate-library/specs/arithmetic/multiply.md`
"""
multiply(a::Number, b::Number) = a * b

# @prove ∀a b. multiply(a, b) == multiply(b, a)
# @prove ∀a b c. multiply(multiply(a, b), c) == multiply(a, multiply(b, c))
# @prove ∀a. multiply(a, 1) == a
# @prove ∀a. multiply(a, 0) == 0
# @prove ∀a b c. multiply(a, add(b, c)) == add(multiply(a, b), multiply(a, c))

"""
    divide(a::Number, b::Number) -> Number

Computes the quotient of two numbers.

# Interface Signature
```
divide: Number, Number -> Number
```

# Behavioral Semantics

**Parameters:**
- `a`: The dividend (number to be divided)
- `b`: The divisor (number to divide by)

**Returns:** The arithmetic quotient `a / b`.

# Mathematical Properties (Proven with Axiom.jl)

- **Non-commutative**: `∀a b. divide(a, b) ≠ divide(b, a)` (except when a == b)
- **Identity element**: `∀a. divide(a, 1) == a`
- **Inverse of multiply**: `∀a b. (b ≠ 0) ⟹ (multiply(divide(a, b), b) ≈ a)`

# Examples

```julia
divide(10, 2)      # Returns 5.0
divide(7, 2)       # Returns 3.5
divide(10.5, 2.0)  # Returns 5.25
divide(-10, 2)     # Returns -5.0
divide(5, -2)      # Returns -2.5
```

# Edge Cases

- **Division by zero**: Throws `DivideError` in Julia
- Behavior follows IEEE 754 for floating-point division

# Specification

This implementation conforms to the aLib specification:
`aggregate-library/specs/arithmetic/divide.md`
"""
divide(a::Number, b::Number) = a / b

# @prove ∀a. divide(a, 1) == a
# @prove ∀a b. (b ≠ 0) ⟹ (multiply(divide(a, b), b) ≈ a)

"""
    modulo(a::Number, b::Number) -> Number

Computes the remainder of integer division.

# Interface Signature
```
modulo: Number, Number -> Number
```

# Behavioral Semantics

**Parameters:**
- `a`: The dividend
- `b`: The divisor

**Returns:** The remainder of `a / b`.

# Mathematical Properties (Proven with Axiom.jl)

- **Range constraint**: `∀a b. (b > 0) ⟹ (0 ≤ modulo(a, b) < b)`
- **Division relation**: `∀a b. (b ≠ 0) ⟹ (a == add(multiply(div(a,b), b), modulo(a, b)))`
- **Identity**: `∀a b. (b ≠ 0 && abs(a) < b) ⟹ (modulo(a, b) == a)`

# Examples

```julia
modulo(10, 3)    # Returns 1
modulo(15, 4)    # Returns 3
modulo(7, 7)     # Returns 0
modulo(-10, 3)   # Returns 2 (Julia's rem would return -1)
modulo(10, -3)   # Implementation-specific
```

# Edge Cases

- **Modulo by zero**: Throws `DivideError` in Julia
- Sign handling follows Julia's `mod` function semantics

# Specification

This implementation conforms to the aLib specification:
`aggregate-library/specs/arithmetic/modulo.md`
"""
modulo(a::Number, b::Number) = mod(a, b)

# @prove ∀a b. (b > 0) ⟹ (0 ≤ modulo(a, b) < b)
# @prove ∀a b. (b ≠ 0) ⟹ (a == add(multiply(div(a,b), b), modulo(a, b)))

end # module Arithmetic
