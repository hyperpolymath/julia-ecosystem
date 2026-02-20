# SPDX-License-Identifier: PMPL-1.0-or-later
module InternalComms

using ..Types

export EmployeeEngagement, InternalNewsletter, log_engagement

struct EmployeeEngagement
    department::Symbol
    participation_rate::Float64
    sentiment_score::Float64
end

struct InternalNewsletter
    id::Symbol
    volume::Int
    headline::String
    read_rate::Float64
end

function log_engagement(dept, rate, sentiment)
    return EmployeeEngagement(dept, rate, sentiment)
end

end # module
