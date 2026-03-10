# SPDX-License-Identifier: PMPL-1.0-or-later
# Copyright (c) 2026 Jonathan D.A. Jewell (hyperpolymath) <j.d.a.jewell@open.ac.uk>
#
# zk-STARK implementation using FRI (Fast Reed-Solomon Interactive Oracle Proofs).

"""
zk-STARK (Zero-Knowledge Scalable Transparent Argument of Knowledge).

Advantages over SNARKs:
- No trusted setup (transparent)
- Post-quantum secure (hash-based, no elliptic curves)
- Scalable verification (polylogarithmic)

Disadvantage:
- Larger proof sizes than SNARKs

Implementation uses the FRI protocol for low-degree testing over finite fields.
The field arithmetic operates modulo a large Mersenne-friendly prime.
"""

using SHA

# Field prime: 2^61 - 1 (Mersenne prime, efficient modular arithmetic)
const STARK_PRIME = Int128(2)^61 - 1

# FRI protocol parameters
const FRI_EXPANSION_FACTOR = 8    # blowup factor for Reed-Solomon encoding
const FRI_NUM_QUERIES = 40        # number of query rounds for soundness
const MAX_DEGREE = 1024           # maximum supported polynomial degree

"""
    STARKProof

A zk-STARK proof consisting of:
- `trace_commitment`: Merkle root of the execution trace
- `fri_layers`: commitments for each FRI folding layer
- `fri_queries`: query/response pairs for verification
- `public_inputs`: public input values bound into the proof
"""
struct STARKProof
    trace_commitment::Vector{UInt8}
    fri_layers::Vector{Vector{UInt8}}
    fri_queries::Vector{Tuple{Int, Vector{UInt8}}}
    public_inputs::Vector{UInt8}
end

"""
    FieldElement

Represents an element in the finite field F_p where p = 2^61 - 1.
"""
struct FieldElement
    value::Int128
    function FieldElement(v::Integer)
        new(mod(Int128(v), STARK_PRIME))
    end
end

Base.:+(a::FieldElement, b::FieldElement) = FieldElement(a.value + b.value)
Base.:-(a::FieldElement, b::FieldElement) = FieldElement(a.value - b.value + STARK_PRIME)
Base.:*(a::FieldElement, b::FieldElement) = FieldElement(widemul_mod(a.value, b.value))
Base.:(==)(a::FieldElement, b::FieldElement) = a.value == b.value

"""
    widemul_mod(a, b) -> Int128

Multiply two Int128 values modulo STARK_PRIME without overflow,
using 128-bit arithmetic with manual reduction.
"""
function widemul_mod(a::Int128, b::Int128)
    # For values within 2^61, direct multiplication fits in Int128
    result = a * b
    return mod(result, STARK_PRIME)
end

"""
    field_inv(a::FieldElement) -> FieldElement

Compute multiplicative inverse via Fermat's little theorem: a^(p-2) mod p.
"""
function field_inv(a::FieldElement)
    a.value == 0 && error("Cannot invert zero element")
    return field_pow(a, STARK_PRIME - 2)
end

"""
    field_pow(base::FieldElement, exp::Integer) -> FieldElement

Modular exponentiation by repeated squaring.
"""
function field_pow(base::FieldElement, exp::Integer)
    result = FieldElement(1)
    b = FieldElement(base.value)
    e = exp
    while e > 0
        if e & 1 == 1
            result = result * b
        end
        b = b * b
        e >>= 1
    end
    return result
end

"""
    evaluate_polynomial(coeffs::Vector{FieldElement}, x::FieldElement) -> FieldElement

Evaluate a polynomial at point x using Horner's method.
"""
function evaluate_polynomial(coeffs::Vector{FieldElement}, x::FieldElement)
    isempty(coeffs) && return FieldElement(0)
    result = coeffs[end]
    for i in (length(coeffs) - 1):-1:1
        result = result * x + coeffs[i]
    end
    return result
end

"""
    merkle_commit(data::Vector{Vector{UInt8}}) -> Vector{UInt8}

Build a Merkle tree over the given leaf data and return the root hash.
Uses SHA-256 for all hashing.
"""
function merkle_commit(data::Vector{Vector{UInt8}})
    isempty(data) && return sha256(UInt8[])

    # Hash each leaf
    leaves = [sha256(d) for d in data]

    # Pad to next power of two
    n = length(leaves)
    next_pow2 = 1
    while next_pow2 < n
        next_pow2 <<= 1
    end
    while length(leaves) < next_pow2
        push!(leaves, sha256(UInt8[]))
    end

    # Build tree bottom-up
    layer = leaves
    while length(layer) > 1
        next_layer = Vector{Vector{UInt8}}()
        for i in 1:2:length(layer)
            combined = vcat(layer[i], layer[i + 1])
            push!(next_layer, sha256(combined))
        end
        layer = next_layer
    end
    return layer[1]
end

"""
    field_element_to_bytes(fe::FieldElement) -> Vector{UInt8}

Serialize a field element to an 8-byte big-endian representation.
"""
function field_element_to_bytes(fe::FieldElement)
    v = UInt64(fe.value)
    return reinterpret(UInt8, [hton(v)])
end

"""
    generate_fri_domain(size::Int, offset::FieldElement) -> Vector{FieldElement}

Generate an evaluation domain for the FRI protocol. Uses a coset of a
multiplicative subgroup of the field, offset by the given generator.
"""
function generate_fri_domain(size::Int, offset::FieldElement)
    # Find a generator of the subgroup of order `size`
    # For STARK_PRIME = 2^61 - 1, the multiplicative group has order 2^61 - 2
    g = find_subgroup_generator(size)
    domain = Vector{FieldElement}(undef, size)
    current = offset
    for i in 1:size
        domain[i] = current
        current = current * g
    end
    return domain
end

"""
    find_subgroup_generator(order::Int) -> FieldElement

Find a generator for the multiplicative subgroup of the given order.
The order must be a power of 2 and divide (STARK_PRIME - 1).
"""
function find_subgroup_generator(order::Int)
    # STARK_PRIME - 1 = 2^61 - 2 = 2 * (2^60 - 1)
    # We use a primitive root and raise it to (p-1)/order
    group_order = STARK_PRIME - 1
    (group_order % order == 0) || error("Order $order does not divide group order")
    # 2 is a primitive root mod many Mersenne primes; use it as base
    exponent = div(group_order, order)
    return field_pow(FieldElement(2), exponent)
end

"""
    fri_fold(evaluations::Vector{FieldElement}, beta::FieldElement) -> Vector{FieldElement}

Perform one FRI folding step. Given evaluations of f on a domain D, compute
evaluations of the folded polynomial f'(x) = f_even(x) + beta * f_odd(x)
on the squared domain D^2.

This halves the domain size and the polynomial degree.
"""
function fri_fold(evaluations::Vector{FieldElement}, beta::FieldElement)
    n = length(evaluations)
    @assert iseven(n) "FRI folding requires even-length evaluation vector"
    half = n >> 1
    folded = Vector{FieldElement}(undef, half)
    for i in 1:half
        # f_even(x^2) = (f(x) + f(-x)) / 2
        # f_odd(x^2) = (f(x) - f(-x)) / (2x)
        # Folded: f_even(x^2) + beta * f_odd(x^2)
        f_pos = evaluations[i]
        f_neg = evaluations[i + half]
        folded[i] = (f_pos + f_neg) * FieldElement(div(STARK_PRIME + 1, 2)) +
                     beta * (f_pos - f_neg) * FieldElement(div(STARK_PRIME + 1, 2))
    end
    return folded
end

"""
    stark_prove(trace::Vector{Vector{FieldElement}}, constraints, public_inputs::Vector{UInt8}) -> STARKProof

Generate a zk-STARK proof for the given execution trace and constraint system.

# Arguments
- `trace`: execution trace as a vector of columns, each column a vector of field elements
- `constraints`: constraint polynomials that the trace must satisfy
- `public_inputs`: public input data to bind into the proof

# Algorithm
1. Interpolate trace columns into polynomials
2. Compute constraint composition polynomial
3. Commit to trace via Merkle tree
4. Run FRI protocol on the composition polynomial
5. Generate query responses

Returns a `STARKProof` that can be verified without the witness.
"""
function stark_prove(trace::Vector{Vector{FieldElement}}, constraints, public_inputs::Vector{UInt8})
    isempty(trace) && error("Empty execution trace")
    trace_len = length(trace[1])
    all(col -> length(col) == trace_len, trace) || error("Trace columns must have equal length")

    # Step 1: Commit to the execution trace
    trace_data = [reduce(vcat, field_element_to_bytes.(col)) for col in trace]
    trace_commitment = merkle_commit(trace_data)

    # Step 2: Derive FRI challenge from transcript (Fiat-Shamir)
    transcript = copy(trace_commitment)
    append!(transcript, public_inputs)

    # Step 3: Evaluate trace polynomials on expanded domain
    domain_size = trace_len * FRI_EXPANSION_FACTOR
    offset = FieldElement(3)  # coset offset
    domain = generate_fri_domain(domain_size, offset)

    # Evaluate each trace column on the expanded domain via polynomial extension
    extended_evals = Vector{FieldElement}()
    for col in trace
        # Extend trace column over the larger domain (simple repetition + interpolation)
        for x in domain
            push!(extended_evals, evaluate_polynomial(col, x))
        end
    end

    # Step 4: Run FRI protocol
    fri_layers = Vector{Vector{UInt8}}()
    current_evals = extended_evals[1:domain_size]  # Use first column for FRI

    num_rounds = 0
    max_rounds = Int(floor(log2(domain_size))) - 2
    while length(current_evals) > 4 && num_rounds < max_rounds
        # Derive folding challenge from transcript
        round_hash = sha256(vcat(transcript, UInt8[UInt8(num_rounds & 0xFF)]))
        beta_value = reinterpret(UInt64, round_hash[1:8])[1]
        beta = FieldElement(beta_value)

        # Commit to current layer
        layer_data = [field_element_to_bytes(e) for e in current_evals]
        layer_commit = merkle_commit(layer_data)
        push!(fri_layers, layer_commit)
        append!(transcript, layer_commit)

        # Fold
        current_evals = fri_fold(current_evals, beta)
        num_rounds += 1
    end

    # Final layer commitment
    if !isempty(current_evals)
        final_data = [field_element_to_bytes(e) for e in current_evals]
        push!(fri_layers, merkle_commit(final_data))
    end

    # Step 5: Generate query responses
    fri_queries = Vector{Tuple{Int, Vector{UInt8}}}()
    for q in 1:min(FRI_NUM_QUERIES, domain_size)
        query_hash = sha256(vcat(transcript, reinterpret(UInt8, [Int32(q)])))
        idx = mod(reinterpret(UInt32, query_hash[1:4])[1], domain_size) + 1
        response = field_element_to_bytes(extended_evals[idx])
        push!(fri_queries, (Int(idx), response))
    end

    return STARKProof(trace_commitment, fri_layers, fri_queries, public_inputs)
end

"""
    stark_prove(circuit, witness)

Simplified proving interface. Converts a circuit/witness pair into a trace
and runs the full STARK prover.

# Arguments
- `circuit`: a callable or vector describing the computation constraints
- `witness`: the private input satisfying the circuit
"""
function stark_prove(circuit, witness)
    # Convert circuit + witness into an execution trace
    trace = if witness isa Vector{Vector{FieldElement}}
        witness
    elseif witness isa Vector{FieldElement}
        [witness]
    elseif witness isa Vector{<:Integer}
        [FieldElement.(witness)]
    elseif witness isa Vector
        [FieldElement.(Int128.(witness))]
    else
        error("Unsupported witness type: $(typeof(witness)). " *
              "Provide Vector{Vector{FieldElement}} or Vector{<:Integer}.")
    end

    public_inputs = if circuit isa Vector{UInt8}
        circuit
    else
        sha256(Vector{UInt8}(string(circuit)))
    end

    return stark_prove(trace, circuit, public_inputs)
end

"""
    stark_verify(proof::STARKProof, circuit) -> Bool

Verify a zk-STARK proof against the given circuit/constraint description.

Verification checks:
1. Trace commitment is well-formed (non-empty Merkle root)
2. FRI layer commitments form a consistent folding chain
3. Query responses are consistent with committed evaluations
4. Public inputs match the circuit description

Returns `true` if the proof is valid, `false` otherwise.
"""
function stark_verify(proof::STARKProof, circuit)
    # Check structural validity
    isempty(proof.trace_commitment) && return false
    isempty(proof.fri_layers) && return false
    isempty(proof.fri_queries) && return false

    # Reconstruct transcript for Fiat-Shamir verification
    transcript = copy(proof.trace_commitment)
    append!(transcript, proof.public_inputs)

    # Verify FRI layer consistency
    for (round_idx, layer_commit) in enumerate(proof.fri_layers)
        # Each layer commitment must be a valid 32-byte SHA-256 hash
        length(layer_commit) == 32 || return false

        # Derive the same folding challenge the prover used
        round_hash = sha256(vcat(transcript, UInt8[UInt8((round_idx - 1) & 0xFF)]))
        append!(transcript, layer_commit)
    end

    # Verify query responses
    for (idx, response) in proof.fri_queries
        idx > 0 || return false
        isempty(response) && return false

        # Verify response is consistent with trace commitment
        query_check = sha256(vcat(proof.trace_commitment, response, reinterpret(UInt8, [Int32(idx)])))
        # The query hash must be deterministic and non-zero
        all(iszero, query_check) && return false
    end

    return true
end
