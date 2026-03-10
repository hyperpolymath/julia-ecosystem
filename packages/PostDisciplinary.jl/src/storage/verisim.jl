# SPDX-License-Identifier: PMPL-1.0-or-later
# Copyright (c) 2026 Jonathan D.A. Jewell (hyperpolymath) <j.d.a.jewell@open.ac.uk>
#
# VeriSimDB bridge for PostDisciplinary.jl.
# Provides a client interface for storing and querying multidisciplinary
# knowledge units (Hexads) in VeriSimDB using its six-modality storage model.

module VeriSimBridge

using HTTP
using JSON3
using Dates

export VeriSimClient, VeriSimError, Hexad, QueryResult
export store_hexad, vql_query, batch_store, health_check

# ============================================================================
# Types
# ============================================================================

"""
    VeriSimClient(endpoint, api_key; timeout_s, max_retries)

Client for communicating with a VeriSimDB instance. Provides authenticated
access to the hexad storage and VQL query APIs.

# Fields
- `endpoint`: base URL of the VeriSimDB server (e.g., "http://localhost:8420")
- `api_key`: API authentication key
- `timeout_s`: request timeout in seconds (default: 30)
- `max_retries`: maximum retry attempts for transient failures (default: 3)
"""
struct VeriSimClient
    endpoint::String
    api_key::String
    timeout_s::Int
    max_retries::Int

    function VeriSimClient(endpoint::String, api_key::String;
                           timeout_s::Int=30, max_retries::Int=3)
        # Normalise endpoint: strip trailing slash
        clean_endpoint = rstrip(endpoint, '/')
        new(clean_endpoint, api_key, timeout_s, max_retries)
    end
end

"""
    VeriSimError(message, status_code, response_body)

Error returned from VeriSimDB operations.
"""
struct VeriSimError <: Exception
    message::String
    status_code::Int
    response_body::String
end

Base.showerror(io::IO, e::VeriSimError) =
    print(io, "VeriSimError($(e.status_code)): $(e.message)")

"""
    Hexad

A multidisciplinary knowledge unit stored in VeriSimDB's six-modality model.

The six modalities represent orthogonal dimensions of knowledge:
1. `textual` - natural language descriptions and narratives
2. `numeric` - quantitative measurements and statistical data
3. `relational` - graph edges and entity relationships
4. `temporal` - timestamps, durations, and temporal orderings
5. `spatial` - geographic coordinates and spatial relations
6. `symbolic` - formal logic, equations, and structured representations
"""
struct Hexad
    id::String
    textual::Union{String, Nothing}
    numeric::Union{Dict{String, Float64}, Nothing}
    relational::Union{Vector{Tuple{String, String, String}}, Nothing}  # (subject, predicate, object)
    temporal::Union{Dict{String, DateTime}, Nothing}
    spatial::Union{Dict{String, Tuple{Float64, Float64}}, Nothing}     # name => (lat, lon)
    symbolic::Union{Dict{String, Any}, Nothing}
    metadata::Dict{String, String}
end

"""Convenience constructor for a Hexad with minimal required fields."""
function Hexad(id::String; textual=nothing, numeric=nothing, relational=nothing,
               temporal=nothing, spatial=nothing, symbolic=nothing,
               metadata=Dict{String,String}())
    return Hexad(id, textual, numeric, relational, temporal, spatial, symbolic, metadata)
end

"""
    QueryResult

Result of a VQL (VeriSim Query Language) query.
"""
struct QueryResult
    hexads::Vector{Hexad}
    total_count::Int
    query_time_ms::Float64
    continuation_token::Union{String, Nothing}
end

# ============================================================================
# API operations
# ============================================================================

"""
    store_hexad(client::VeriSimClient, hexad::Hexad) -> Symbol

Persist a multidisciplinary knowledge unit (Hexad) into VeriSimDB.
The hexad is stored across all six modalities simultaneously, enabling
cross-modal queries.

# Arguments
- `client`: authenticated VeriSimDB client
- `hexad`: the knowledge unit to store

# Returns
`:stored` on success.

# Throws
`VeriSimError` on server or authentication errors.
"""
function store_hexad(c::VeriSimClient, hexad::Hexad)
    payload = _hexad_to_json(hexad)
    response = _request(c, "POST", "/api/v1/hexads", payload)
    return :stored
end

"""Backward-compatible store_hexad that accepts raw data."""
function store_hexad(c::VeriSimClient, data)
    hexad = if data isa Hexad
        data
    else
        # Wrap raw data as a textual hexad
        Hexad(
            string(hash(data)),
            textual = string(data),
            metadata = Dict("source" => "PostDisciplinary.jl", "created_at" => string(now()))
        )
    end
    return store_hexad(c, hexad)
end

"""
    vql_query(client::VeriSimClient, query::String; limit=100, offset=0) -> QueryResult

Execute a VQL (VeriSim Query Language) query against the database.

VQL supports cross-modal queries that span multiple knowledge dimensions:
- `SELECT FROM textual WHERE ...` - text search
- `SELECT FROM numeric WHERE field > value` - numeric range queries
- `SELECT FROM relational WHERE subject = ...` - graph traversal
- `SELECT FROM temporal WHERE after '2026-01-01'` - temporal queries
- `SELECT CROSS textual, numeric WHERE ...` - cross-modal joins

# Arguments
- `client`: authenticated VeriSimDB client
- `query`: VQL query string
- `limit`: maximum results to return (default: 100)
- `offset`: pagination offset (default: 0)

# Returns
A `QueryResult` containing matching hexads and pagination metadata.
"""
function vql_query(c::VeriSimClient, query::String; limit::Int=100, offset::Int=0)
    payload = Dict(
        "query" => query,
        "limit" => limit,
        "offset" => offset
    )
    response = _request(c, "POST", "/api/v1/query", payload)
    return _parse_query_result(response)
end

"""
    batch_store(client::VeriSimClient, hexads::Vector{Hexad}) -> Int

Store multiple hexads in a single batch operation. More efficient than
individual store_hexad calls for bulk ingestion.

# Returns
The number of hexads successfully stored.
"""
function batch_store(c::VeriSimClient, hexads::Vector{Hexad})
    payload = Dict("hexads" => [_hexad_to_json_dict(h) for h in hexads])
    response = _request(c, "POST", "/api/v1/hexads/batch", payload)

    result = JSON3.read(String(response.body))
    return get(result, :stored_count, length(hexads))
end

"""
    health_check(client::VeriSimClient) -> NamedTuple

Check the health and status of the VeriSimDB server.

# Returns
A named tuple with `:healthy` (Bool), `:version` (String), and `:uptime_s` (Int).
"""
function health_check(c::VeriSimClient)
    try
        response = _request(c, "GET", "/api/v1/health", nothing)
        result = JSON3.read(String(response.body))
        return (
            healthy = get(result, :status, "unknown") == "ok",
            version = string(get(result, :version, "unknown")),
            uptime_s = get(result, :uptime_seconds, 0)
        )
    catch e
        if e isa VeriSimError
            return (healthy=false, version="unknown", uptime_s=0)
        end
        rethrow(e)
    end
end

# ============================================================================
# Internal helpers
# ============================================================================

"""Make an authenticated HTTP request to VeriSimDB with retry logic."""
function _request(c::VeriSimClient, method::String, path::String, payload)
    url = c.endpoint * path
    headers = [
        "Authorization" => "Bearer $(c.api_key)",
        "Content-Type" => "application/json",
        "User-Agent" => "PostDisciplinary.jl/1.0",
    ]

    last_error = nothing
    for attempt in 1:c.max_retries
        try
            response = if method == "GET"
                HTTP.get(url; headers=headers, readtimeout=c.timeout_s, retry=false)
            elseif method == "POST"
                body = payload !== nothing ? JSON3.write(payload) : ""
                HTTP.post(url; headers=headers, body=body, readtimeout=c.timeout_s, retry=false)
            else
                error("Unsupported HTTP method: $method")
            end

            if response.status >= 400
                throw(VeriSimError(
                    "HTTP $(response.status) from VeriSimDB",
                    response.status,
                    String(response.body)
                ))
            end

            return response
        catch e
            last_error = e
            if e isa VeriSimError && e.status_code < 500
                # Client errors (4xx) should not be retried
                rethrow(e)
            end
            # Retry on server errors (5xx) and connection errors
            if attempt < c.max_retries
                sleep(0.5 * (2 ^ (attempt - 1)))  # exponential backoff
            end
        end
    end

    if last_error isa VeriSimError
        throw(last_error)
    end
    throw(VeriSimError("Failed after $(c.max_retries) attempts", 0, string(last_error)))
end

"""Convert a Hexad to a JSON-compatible Dict for API payloads."""
function _hexad_to_json_dict(h::Hexad)
    d = Dict{String, Any}("id" => h.id)

    h.textual !== nothing && (d["textual"] = h.textual)

    if h.numeric !== nothing
        d["numeric"] = h.numeric
    end

    if h.relational !== nothing
        d["relational"] = [Dict("subject" => s, "predicate" => p, "object" => o)
                           for (s, p, o) in h.relational]
    end

    if h.temporal !== nothing
        d["temporal"] = Dict(k => string(v) for (k, v) in h.temporal)
    end

    if h.spatial !== nothing
        d["spatial"] = Dict(k => Dict("lat" => lat, "lon" => lon)
                           for (k, (lat, lon)) in h.spatial)
    end

    if h.symbolic !== nothing
        d["symbolic"] = h.symbolic
    end

    if !isempty(h.metadata)
        d["metadata"] = h.metadata
    end

    return d
end

"""Convert a Hexad to a JSON payload Dict (wrapping for single-hexad endpoints)."""
function _hexad_to_json(h::Hexad)
    return _hexad_to_json_dict(h)
end

"""Parse a VQL query response into a QueryResult."""
function _parse_query_result(response)
    result = JSON3.read(String(response.body))

    hexads = Hexad[]
    raw_hexads = get(result, :hexads, [])
    for raw in raw_hexads
        h = Hexad(
            string(get(raw, :id, "")),
            textual = _get_or_nothing(raw, :textual, String),
            numeric = _parse_numeric(raw),
            metadata = Dict{String,String}(
                string(k) => string(v) for (k, v) in get(raw, :metadata, Dict())
            )
        )
        push!(hexads, h)
    end

    return QueryResult(
        hexads,
        get(result, :total_count, length(hexads)),
        get(result, :query_time_ms, 0.0),
        _get_or_nothing(result, :continuation_token, String)
    )
end

"""Get a typed value from a JSON object or return nothing."""
function _get_or_nothing(obj, key::Symbol, ::Type{T}) where T
    val = get(obj, key, nothing)
    val === nothing && return nothing
    return T(val)
end

"""Parse the numeric modality from a JSON response."""
function _parse_numeric(raw)
    nums = get(raw, :numeric, nothing)
    nums === nothing && return nothing
    return Dict{String, Float64}(string(k) => Float64(v) for (k, v) in nums)
end

end # module
