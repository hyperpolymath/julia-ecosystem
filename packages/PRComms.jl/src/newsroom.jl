# SPDX-License-Identifier: PMPL-1.0-or-later
module Newsroom

using ..Types
using Dates

export draft_release, review_release, publish_release

function draft_release(id::Symbol, title::String, body::String)
    return PressRelease(id, title, body, :draft, nothing, nothing)
end

function review_release(release::PressRelease)
    release.status = :review
    return "Press release '$(release.title)' is now in review. 👀"
end

function publish_release(release::PressRelease; embargo::Union{DateTime, Nothing}=nothing)
    if embargo !== nothing
        release.status = :embargoed
        release.embargo_at = embargo
        return "Release scheduled for $(embargo) ⏳"
    else
        release.status = :published
        return "Release LIVE! 🚀"
    end
end

end # module
