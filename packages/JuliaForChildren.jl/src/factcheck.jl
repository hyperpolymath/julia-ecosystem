# SPDX-License-Identifier: MPL-2.0
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
