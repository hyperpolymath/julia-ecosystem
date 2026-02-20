# SPDX-License-Identifier: PMPL-1.0-or-later
"""
    String

String operations from the PolyglotFormalisms Common Library specification.

Julia implementation matching aggregate-library behavioral semantics.

Each operation includes:
- Implementation following PolyglotFormalisms specification
- Type signatures for string operations
- Documentation matching specification format
"""
module StringOps

export concat, length, substring, index_of, contains, starts_with, ends_with,
       to_uppercase, to_lowercase, trim, split, join, replace, is_empty

"""
    concat(a::AbstractString, b::AbstractString) -> String

Concatenates two strings.

# Behavioral Semantics
- Parameters: a (first string), b (second string)
- Returns: The concatenation of a and b

# Mathematical Properties
- Associativity: concat(concat(a, b), c) == concat(a, concat(b, c))
- Identity element: concat(a, "") == concat("", a) == a
- Non-commutativity: concat(a, b) != concat(b, a) (in general)

# Examples
```julia
concat("Hello", " World")  # Returns "Hello World"
concat("", "test")          # Returns "test"
concat("test", "")          # Returns "test"
concat("a", "b")            # Returns "ab"
```
"""
function concat(a::AbstractString, b::AbstractString)::String
    return string(a, b)
end

"""
    length(s::AbstractString) -> Int

Returns the length of a string (number of characters).

# Behavioral Semantics
- Parameters: s (string to measure)
- Returns: Number of Unicode characters (code points) in the string

# Mathematical Properties
- Non-negativity: length(s) >= 0
- Empty string: length("") == 0
- Concatenation: length(concat(a, b)) == length(a) + length(b)

# Examples
```julia
length("Hello")      # Returns 5
length("")           # Returns 0
length("🎉")         # Returns 1 (Unicode character)
length("Test 123")   # Returns 8
```
"""
function length(s::AbstractString)::Int
    return Base.length(s)
end

"""
    substring(s::AbstractString, start::Int, end_pos::Int) -> String

Extracts a substring from start index to end index (inclusive, 1-indexed).

# Behavioral Semantics
- Parameters: s (source string), start (start index), end_pos (end index)
- Returns: Substring from start to end_pos (inclusive)
- Indexing: 1-based (Julia convention)

# Edge Cases
- If start > end_pos: returns empty string
- If indices out of bounds: throws BoundsError

# Examples
```julia
substring("Hello World", 1, 5)   # Returns "Hello"
substring("Hello World", 7, 11)  # Returns "World"
substring("Test", 1, 1)          # Returns "T"
substring("Test", 3, 2)          # Returns ""
```
"""
function substring(s::AbstractString, start::Int, end_pos::Int)::String
    if start > end_pos
        return ""
    end
    return String(s[start:end_pos])
end

"""
    index_of(s::AbstractString, substr::AbstractString) -> Int

Finds the first occurrence of a substring.

# Behavioral Semantics
- Parameters: s (string to search), substr (substring to find)
- Returns: 1-based index of first occurrence, or 0 if not found

# Mathematical Properties
- Not found convention: returns 0 (Julia-specific, some languages use -1)
- Empty substring: index_of(s, "") == 1 (found at start)

# Examples
```julia
index_of("Hello World", "World")  # Returns 7
index_of("Hello World", "o")      # Returns 5 (first 'o')
index_of("Test", "xyz")           # Returns 0 (not found)
index_of("Test", "")              # Returns 1
```
"""
function index_of(s::AbstractString, substr::AbstractString)::Int
    if isempty(substr)
        return 1
    end
    result = findfirst(substr, s)
    return result === nothing ? 0 : first(result)
end

"""
    contains(s::AbstractString, substr::AbstractString) -> Bool

Checks if a string contains a substring.

# Behavioral Semantics
- Parameters: s (string to search), substr (substring to find)
- Returns: true if substr is found in s, false otherwise

# Mathematical Properties
- Empty substring: contains(s, "") == true (always)
- Reflexivity: contains(s, s) == true
- Transitivity: If contains(a, b) && contains(b, c), then contains(a, c)

# Examples
```julia
contains("Hello World", "World")  # Returns true
contains("Hello World", "xyz")    # Returns false
contains("Test", "")              # Returns true
contains("", "Test")              # Returns false
```
"""
function contains(s::AbstractString, substr::AbstractString)::Bool
    return occursin(substr, s)
end

"""
    starts_with(s::AbstractString, prefix::AbstractString) -> Bool

Checks if a string starts with a given prefix.

# Behavioral Semantics
- Parameters: s (string to check), prefix (prefix to match)
- Returns: true if s starts with prefix, false otherwise

# Mathematical Properties
- Empty prefix: starts_with(s, "") == true (always)
- Reflexivity: starts_with(s, s) == true

# Examples
```julia
starts_with("Hello World", "Hello")  # Returns true
starts_with("Hello World", "World")  # Returns false
starts_with("Test", "")              # Returns true
starts_with("", "Test")              # Returns false
```
"""
function starts_with(s::AbstractString, prefix::AbstractString)::Bool
    return startswith(s, prefix)
end

"""
    ends_with(s::AbstractString, suffix::AbstractString) -> Bool

Checks if a string ends with a given suffix.

# Behavioral Semantics
- Parameters: s (string to check), suffix (suffix to match)
- Returns: true if s ends with suffix, false otherwise

# Mathematical Properties
- Empty suffix: ends_with(s, "") == true (always)
- Reflexivity: ends_with(s, s) == true

# Examples
```julia
ends_with("Hello World", "World")  # Returns true
ends_with("Hello World", "Hello")  # Returns false
ends_with("Test", "")              # Returns true
ends_with("", "Test")              # Returns false
```
"""
function ends_with(s::AbstractString, suffix::AbstractString)::Bool
    return endswith(s, suffix)
end

"""
    to_uppercase(s::AbstractString) -> String

Converts all characters in a string to uppercase.

# Behavioral Semantics
- Parameters: s (string to convert)
- Returns: New string with all characters in uppercase

# Mathematical Properties
- Idempotence: to_uppercase(to_uppercase(s)) == to_uppercase(s)
- Unicode aware: properly handles non-ASCII characters

# Examples
```julia
to_uppercase("Hello World")  # Returns "HELLO WORLD"
to_uppercase("test")         # Returns "TEST"
to_uppercase("TEST")         # Returns "TEST"
to_uppercase("café")         # Returns "CAFÉ"
```
"""
function to_uppercase(s::AbstractString)::String
    return uppercase(s)
end

"""
    to_lowercase(s::AbstractString) -> String

Converts all characters in a string to lowercase.

# Behavioral Semantics
- Parameters: s (string to convert)
- Returns: New string with all characters in lowercase

# Mathematical Properties
- Idempotence: to_lowercase(to_lowercase(s)) == to_lowercase(s)
- Unicode aware: properly handles non-ASCII characters

# Examples
```julia
to_lowercase("Hello World")  # Returns "hello world"
to_lowercase("TEST")         # Returns "test"
to_lowercase("test")         # Returns "test"
to_lowercase("CAFÉ")         # Returns "café"
```
"""
function to_lowercase(s::AbstractString)::String
    return lowercase(s)
end

"""
    trim(s::AbstractString) -> String

Removes leading and trailing whitespace from a string.

# Behavioral Semantics
- Parameters: s (string to trim)
- Returns: New string with whitespace removed from both ends
- Whitespace: spaces, tabs, newlines, and other Unicode whitespace

# Mathematical Properties
- Idempotence: trim(trim(s)) == trim(s)
- No internal whitespace affected: only leading/trailing removed

# Examples
```julia
trim("  Hello World  ")  # Returns "Hello World"
trim("\\n\\tTest\\n")      # Returns "Test"
trim("NoSpaces")         # Returns "NoSpaces"
trim("   ")              # Returns ""
```
"""
function trim(s::AbstractString)::String
    return strip(s)
end

"""
    split(s::AbstractString, delimiter::AbstractString) -> Vector{String}

Splits a string into parts using a delimiter.

# Behavioral Semantics
- Parameters: s (string to split), delimiter (separator)
- Returns: Array of substrings
- Empty delimiter: returns array of individual characters

# Edge Cases
- Empty string: split("", delim) returns [""]
- Delimiter not found: returns [s]
- Consecutive delimiters: create empty strings in result

# Examples
```julia
split("a,b,c", ",")           # Returns ["a", "b", "c"]
split("Hello World", " ")      # Returns ["Hello", "World"]
split("test", ",")             # Returns ["test"]
split("a,,b", ",")             # Returns ["a", "", "b"]
```
"""
function split(s::AbstractString, delimiter::AbstractString)::Vector{String}
    if isempty(delimiter)
        return [string(c) for c in s]
    end
    return Base.split(s, delimiter) |> collect .|> String
end

"""
    join(parts::Vector{<:AbstractString}, separator::AbstractString) -> String

Joins an array of strings with a separator.

# Behavioral Semantics
- Parameters: parts (array of strings), separator (string to insert between parts)
- Returns: Single string with parts joined by separator

# Mathematical Properties
- Empty array: join([], sep) == ""
- Single element: join([s], sep) == s
- Inverse of split: join(split(s, d), d) ≈ s (for simple cases)

# Examples
```julia
join(["a", "b", "c"], ",")       # Returns "a,b,c"
join(["Hello", "World"], " ")     # Returns "Hello World"
join(["test"], ",")               # Returns "test"
join(String[], ",")               # Returns ""
```
"""
function join(parts::Vector{<:AbstractString}, separator::AbstractString)::String
    return Base.join(parts, separator)
end

"""
    replace(s::AbstractString, old::AbstractString, new::AbstractString) -> String

Replaces all occurrences of a substring with another string.

# Behavioral Semantics
- Parameters: s (source string), old (substring to replace), new (replacement)
- Returns: New string with all occurrences replaced

# Edge Cases
- old not found: returns original string unchanged
- Empty old: returns original string unchanged
- new can be empty (deletion)

# Examples
```julia
replace("Hello World", "World", "Universe")  # Returns "Hello Universe"
replace("test test", "test", "demo")         # Returns "demo demo"
replace("Hello", "xyz", "abc")               # Returns "Hello"
replace("Hello", "l", "")                    # Returns "Heo"
```
"""
function replace(s::AbstractString, old::AbstractString, new::AbstractString)::String
    if isempty(old)
        return String(s)
    end
    return Base.replace(s, old => new)
end

"""
    is_empty(s::AbstractString) -> Bool

Checks if a string is empty.

# Behavioral Semantics
- Parameters: s (string to check)
- Returns: true if string has zero length, false otherwise

# Mathematical Properties
- Equivalent to: length(s) == 0
- Empty string: is_empty("") == true

# Examples
```julia
is_empty("")        # Returns true
is_empty("test")    # Returns false
is_empty(" ")       # Returns false (space is a character)
```
"""
function is_empty(s::AbstractString)::Bool
    return isempty(s)
end

end # module StringOps
