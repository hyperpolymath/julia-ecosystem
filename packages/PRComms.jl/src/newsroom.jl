# SPDX-License-Identifier: PMPL-1.0-or-later
# Copyright (c) 2026 Jonathan D.A. Jewell (hyperpolymath) <j.d.a.jewell@open.ac.uk>
#
# Newsroom module for PRComms.jl.
# Manages the press release lifecycle: drafting, review, embargo scheduling,
# approval workflows, and publication.

module Newsroom

using ..Types
using Dates

export draft_release, review_release, publish_release
export approve_release, reject_release, schedule_embargo
export release_status_report

# Maximum body length for press releases (industry standard: ~500 words)
const MAX_BODY_LENGTH = 5000
const MIN_BODY_LENGTH = 50

"""
    draft_release(id::Symbol, title::String, body::String) -> PressRelease

Create a new press release in `:draft` status. Validates that the title
and body meet minimum content requirements.

# Arguments
- `id`: unique identifier for the release
- `title`: headline (should be concise and descriptive)
- `body`: full press release text

# Returns
A new `PressRelease` in draft status.

# Throws
`ArgumentError` if title is empty or body is too short/long.
"""
function draft_release(id::Symbol, title::String, body::String)
    isempty(strip(title)) && throw(ArgumentError("Press release title cannot be empty"))

    stripped_body = strip(body)
    if length(stripped_body) < MIN_BODY_LENGTH
        throw(ArgumentError(
            "Press release body too short ($(length(stripped_body)) chars, " *
            "minimum $(MIN_BODY_LENGTH)). Provide substantive content."
        ))
    end

    if length(stripped_body) > MAX_BODY_LENGTH
        throw(ArgumentError(
            "Press release body too long ($(length(stripped_body)) chars, " *
            "maximum $(MAX_BODY_LENGTH)). Edit for conciseness."
        ))
    end

    return PressRelease(id, strip(title), stripped_body, :draft, nothing, nothing)
end

"""
    review_release(release::PressRelease) -> String

Transition a press release from `:draft` to `:review` status. Only drafts
can be submitted for review.

# Returns
A status message confirming the transition.

# Throws
`ArgumentError` if the release is not in `:draft` status.
"""
function review_release(release::PressRelease)
    if release.status != :draft
        throw(ArgumentError(
            "Cannot submit for review: release '$(release.title)' is in " *
            "'$(release.status)' status (expected :draft)"
        ))
    end

    release.status = :review
    return "Press release '$(release.title)' is now in review."
end

"""
    approve_release(release::PressRelease) -> String

Approve a press release that is currently in review. Sets the approval
timestamp and transitions to `:approved` status, ready for publication
or embargo scheduling.

# Returns
A status message with the approval timestamp.

# Throws
`ArgumentError` if the release is not in `:review` status.
"""
function approve_release(release::PressRelease)
    if release.status != :review
        throw(ArgumentError(
            "Cannot approve: release '$(release.title)' is in " *
            "'$(release.status)' status (expected :review)"
        ))
    end

    release.status = :approved
    release.approved_at = now()
    return "Press release '$(release.title)' approved at $(release.approved_at)."
end

"""
    reject_release(release::PressRelease; reason::String="") -> String

Reject a press release and return it to `:draft` status for revisions.
Optionally provide a reason for rejection.

# Returns
A status message indicating the release was returned for revisions.
"""
function reject_release(release::PressRelease; reason::String="")
    if release.status == :published
        throw(ArgumentError("Cannot reject an already-published release"))
    end

    release.status = :draft
    release.approved_at = nothing
    release.embargo_at = nothing

    reason_text = isempty(reason) ? "" : " Reason: $reason"
    return "Press release '$(release.title)' returned to draft.$reason_text"
end

"""
    schedule_embargo(release::PressRelease, embargo_time::DateTime) -> String

Set an embargo time on an approved press release. The release will
automatically become available for publication at the embargo time.

# Arguments
- `release`: an approved press release
- `embargo_time`: the DateTime when the embargo lifts (must be in the future)

# Returns
A status message with the scheduled embargo time.

# Throws
`ArgumentError` if the release is not approved or the time is in the past.
"""
function schedule_embargo(release::PressRelease, embargo_time::DateTime)
    if release.status != :approved && release.status != :review
        throw(ArgumentError(
            "Cannot embargo: release must be in :approved or :review status " *
            "(currently :$(release.status))"
        ))
    end

    if embargo_time <= now()
        throw(ArgumentError("Embargo time must be in the future (got: $embargo_time)"))
    end

    release.status = :embargoed
    release.embargo_at = embargo_time
    return "Press release '$(release.title)' embargoed until $(embargo_time)."
end

"""
    publish_release(release::PressRelease; embargo::Union{DateTime, Nothing}=nothing) -> String

Publish a press release immediately, or schedule it for embargoed release.

If `embargo` is provided, the release is set to `:embargoed` status and will
become available at the specified time. If `embargo` is `nothing`, the release
is published immediately.

Only releases in `:draft`, `:review`, `:approved`, or `:embargoed` status
(with passed embargo time) can be published.

# Arguments
- `release`: the press release to publish
- `embargo`: optional future DateTime for embargoed release

# Returns
A status message confirming publication or scheduled embargo.
"""
function publish_release(release::PressRelease; embargo::Union{DateTime, Nothing}=nothing)
    if release.status == :published
        return "Press release '$(release.title)' is already published."
    end

    if embargo !== nothing
        if embargo <= now()
            throw(ArgumentError("Embargo time must be in the future (got: $embargo)"))
        end
        release.status = :embargoed
        release.embargo_at = embargo
        return "Press release '$(release.title)' scheduled for $(embargo)."
    end

    # Check if embargoed release has reached its time
    if release.status == :embargoed && release.embargo_at !== nothing
        if release.embargo_at > now()
            remaining = release.embargo_at - now()
            return "Cannot publish yet: embargo active until $(release.embargo_at) " *
                   "($(round(remaining, Dates.Minute)) remaining)."
        end
    end

    release.status = :published
    if release.approved_at === nothing
        release.approved_at = now()
    end
    return "Press release '$(release.title)' is now LIVE."
end

"""
    release_status_report(releases::Vector{PressRelease}) -> String

Generate a formatted status report for a collection of press releases,
grouped by status.

# Returns
A multi-line string summarising all releases and their current status.
"""
function release_status_report(releases::Vector{PressRelease})
    isempty(releases) && return "No press releases in the system."

    lines = String["Press Release Status Report", "=" ^ 40, ""]

    # Group by status
    status_order = [:draft, :review, :approved, :embargoed, :published]
    for status in status_order
        matching = filter(r -> r.status == status, releases)
        isempty(matching) && continue

        push!(lines, "$(uppercase(string(status))) ($(length(matching))):")
        for r in matching
            embargo_info = if r.embargo_at !== nothing
                " [embargo: $(r.embargo_at)]"
            else
                ""
            end
            approval_info = if r.approved_at !== nothing
                " [approved: $(r.approved_at)]"
            else
                ""
            end
            push!(lines, "  - :$(r.id) \"$(r.title)\"$(embargo_info)$(approval_info)")
        end
        push!(lines, "")
    end

    push!(lines, "Total: $(length(releases)) release(s)")
    return join(lines, '\n')
end

end # module
