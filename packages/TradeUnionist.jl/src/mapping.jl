# SPDX-License-Identifier: PMPL-1.0-or-later
# Copyright (c) 2026 Jonathan D.A. Jewell (hyperpolymath) <j.d.a.jewell@open.ac.uk>
#
# Geospatial mapping module for TradeUnionist.jl.
# Provides spatial analysis for union organising: finding members near worksites,
# computing commute distances, and performing spatial joins with external data.

module Mapping

using ..Types
using LibGEOS
using GeoInterface
using DataFrames
using DuckDB

export find_members_near, spatial_mashup, calc_commute_distance

# Earth's mean radius in kilometres (WGS-84)
const EARTH_RADIUS_KM = 6371.0088

"""
    find_members_near(site::Worksite, members::Vector{MemberRecord}, radius::Float64) -> Vector{MemberRecord}

Find all union members whose home location is within `radius` kilometres
of a worksite. Uses the Haversine formula for accurate great-circle
distance on the Earth's surface.

# Arguments
- `site`: the worksite to search around
- `members`: pool of members to filter
- `radius`: maximum distance in kilometres

# Returns
A vector of `MemberRecord`s within the specified radius, sorted by distance
(nearest first).
"""
function find_members_near(site::Worksite, members::Vector{MemberRecord}, radius::Float64)
    radius > 0 || throw(ArgumentError("Radius must be positive (got: $radius)"))

    if site.geo === nothing
        return MemberRecord[]
    end

    # Compute distances and filter
    nearby = Tuple{MemberRecord, Float64}[]
    for m in members
        if m.home_geo !== nothing
            d = haversine_distance(site.geo, m.home_geo)
            if d <= radius
                push!(nearby, (m, d))
            end
        end
    end

    # Sort by distance (nearest first)
    sort!(nearby; by=x -> x[2])
    return [m for (m, _) in nearby]
end

"""
    haversine_distance(loc1::GeoLocation, loc2::GeoLocation) -> Float64

Compute the great-circle distance between two geographic coordinates using
the Haversine formula. Returns distance in kilometres.

The Haversine formula is numerically stable for small distances and
accurate to within 0.5% for any two points on Earth.

# Arguments
- `loc1`, `loc2`: geographic coordinates with `lat` and `lon` fields (in degrees)
"""
function haversine_distance(l1::GeoLocation, l2::GeoLocation)
    # Convert degrees to radians
    lat1 = deg2rad(l1.lat)
    lat2 = deg2rad(l2.lat)
    dlat = deg2rad(l2.lat - l1.lat)
    dlon = deg2rad(l2.lon - l1.lon)

    # Haversine formula
    a = sin(dlat / 2)^2 + cos(lat1) * cos(lat2) * sin(dlon / 2)^2
    c = 2 * atan(sqrt(a), sqrt(1 - a))

    return EARTH_RADIUS_KM * c
end

"""
    calc_commute_distance(member::MemberRecord, site::Worksite) -> Union{Float64, Nothing}

Calculate the straight-line (great-circle) commute distance in kilometres
between a member's home and their worksite. Returns `nothing` if either
location is unavailable.

Note: This is the geodesic distance, not the actual road/transit distance.
For routing-based distances, integrate with an external routing API.

# Arguments
- `member`: the union member
- `site`: the worksite

# Returns
Distance in kilometres, or `nothing` if coordinates are missing.
"""
function calc_commute_distance(member::MemberRecord, site::Worksite)
    if member.home_geo === nothing || site.geo === nothing
        return nothing
    end
    return haversine_distance(member.home_geo, site.geo)
end

"""
    spatial_mashup(df::DataFrame, geojson_path::String) -> DataFrame

Combine union membership data with external geographic boundary data
(e.g., Census tracts, transit lines, voting districts) using DuckDB's
spatial extension for high-performance point-in-polygon joins.

The input DataFrame must contain `lat` and `lon` columns with member
coordinates. The GeoJSON file provides polygon geometries to join against.

# Arguments
- `df`: DataFrame with at least `lat::Float64` and `lon::Float64` columns
- `geojson_path`: path to a GeoJSON file with polygon/multipolygon features

# Returns
A DataFrame with member rows enriched by the GeoJSON feature properties
for the polygon each member falls within.

# Throws
`ArgumentError` if the DataFrame lacks `lat`/`lon` columns or the GeoJSON
file does not exist.
"""
function spatial_mashup(df::DataFrame, geojson_path::String)
    # Validate inputs
    hasproperty(df, :lat) || throw(ArgumentError("DataFrame must have a 'lat' column"))
    hasproperty(df, :lon) || throw(ArgumentError("DataFrame must have a 'lon' column"))
    isfile(geojson_path) || throw(ArgumentError("GeoJSON file not found: $geojson_path"))

    db = DuckDB.DB()

    # Install and load the spatial extension
    DuckDB.execute(db, "INSTALL spatial; LOAD spatial;")

    # Register the DataFrame as a table
    DuckDB.register_data_frame(db, df, "members")

    # Perform a spatial join: find which GeoJSON polygon each member point falls within
    query = """
    SELECT m.*, geo.properties
    FROM members m
    LEFT JOIN ST_Read('$(escape_sql_string(geojson_path))') geo
    ON ST_Within(ST_Point(m.lon, m.lat), geo.geom)
    """

    result = DuckDB.execute(db, query)
    return DuckDB.to_data_frame(result)
end

"""Escape single quotes in a string for safe inclusion in SQL."""
function escape_sql_string(s::String)
    return replace(s, "'" => "''")
end

end # module
