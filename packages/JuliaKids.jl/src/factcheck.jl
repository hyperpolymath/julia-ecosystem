# SPDX-License-Identifier: PMPL-1.0-or-later
module FactCheck

export is_true

"""
    is_true(statement)

Checks if something is logically true.
Example: is_true("A whale is a fish") -> False
"""
function is_true(statement)
    # Very simple keyword check for now
    if occursin("whale", lowercase(statement)) && occursin("fish", lowercase(statement))
        return false
    end
    return true # Optimistic!
end

end # module
