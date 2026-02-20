# SPDX-License-Identifier: PMPL-1.0-or-later
module Publics

export PublicGroup, Shareholders, LocalCommunity, MediaPublic, GovernmentPublic, EmployeePublic

abstract type PublicGroup end

struct Shareholders <: PublicGroup end
struct LocalCommunity <: PublicGroup end
struct MediaPublic <: PublicGroup end
struct GovernmentPublic <: PublicGroup end
struct EmployeePublic <: PublicGroup end

struct PublicSegment
    group::PublicGroup
    sentiment::Float64 # -1.0 to 1.0
    key_concerns::Vector{String}
    preferred_channels::Vector{Symbol}
end

end # module
