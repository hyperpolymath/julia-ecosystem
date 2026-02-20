# SPDX-License-Identifier: PMPL-1.0-or-later
module Mapping

using ..Types
using LibGEOS
using GeoInterface
using DataFrames
using DuckDB

export find_members_near, spatial_mashup, calc_commute_distance

"""
    find_members_near(worksite::Worksite, members::Vector{MemberRecord}, radius_km::Float64)
Finds members living within a certain distance of a worksite.
"""
function find_members_near(site::Worksite, members::Vector{MemberRecord}, radius::Float64)
    if site.geo === nothing return MemberRecord[] end
    
    nearby = MemberRecord[]
    for m in members
        if m.home_geo !== nothing
            d = calc_distance(site.geo, m.home_geo)
            if d <= radius
                push!(nearby, m)
            end
        end
    end
    return nearby
end

"""
    calc_distance(loc1, loc2)
Haversine distance approximation in km.
"""
function calc_distance(l1::GeoLocation, l2::GeoLocation)
    # Simple Euclidean for demo, would be Haversine in production
    return sqrt((l1.lat - l2.lat)^2 + (l1.lon - l2.lon)^2) * 111.0 
end

"""
    spatial_mashup(union_data::DataFrame, external_geojson_path::String)
Combines union membership data with external geographic data (e.g. Census tracts, Transit lines).
Uses DuckDB's spatial extension for high-performance joins.
"""
function spatial_mashup(df::DataFrame, geojson_path::String)
    db = DuckDB.DB()
    DuckDB.execute(db, "INSTALL spatial; LOAD spatial;")
    
    # Register the dataframe as a table
    DuckDB.register_data_frame(db, df, "members")
    
    # Query linking member points to polygons in the geojson
    query = """
    SELECT m.*, geo.properties
    FROM members m, ST_Read('$geojson_path') geo
    WHERE ST_Within(ST_Point(m.lon, m.lat), geo.geom)
    """
    
    return DuckDB.to_data_frame(DuckDB.execute(db, query))
end

end # module
