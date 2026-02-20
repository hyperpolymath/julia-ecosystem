# SPDX-License-Identifier: PMPL-1.0-or-later
"""
Dempster-Shafer theory of evidence for combining uncertain information.

Provides belief functions (basic probability assignments) for modeling
epistemic uncertainty. Unlike Bayesian probability, D-S explicitly represents
ignorance and handles conflicting evidence gracefully.
"""
module DempsterShafer

export MassAssignment, Belief, Plausibility
export mass, belief, plausibility, uncertainty
export combine_dempster, discount, pignistic_transform
export conflict_measure

"""
    MassAssignment{T}

Basic probability assignment (BPA) over a frame of discernment.
Maps sets of hypotheses to masses in [0,1] that sum to 1.

# Example
```julia
frame = [:A, :B, :C]
masses = Dict(
    Set([:A]) => 0.4,
    Set([:B]) => 0.3,
    Set([:A, :B]) => 0.2,
    Set([:A, :B, :C]) => 0.1  # ignorance
)
m = MassAssignment(frame, masses)
```
"""
struct MassAssignment{T}
    frame::Vector{T}
    masses::Dict{Set{T}, Float64}

    function MassAssignment{T}(frame::Vector{T}, masses::Dict{Set{T}, Float64}) where T
        total = sum(values(masses))
        # Allow empty mass assignments as edge case
        if !isempty(masses) && !isapprox(total, 1.0, atol=1e-10)
            error("Masses must sum to 1.0, got $(total)")
        end

        frame_set = Set(frame)
        for s in keys(masses)
            if !issubset(s, frame_set)
                error("Mass assigned to set outside frame: $(s)")
            end
        end

        new{T}(frame, masses)
    end
end

MassAssignment(frame::Vector{T}, masses::Dict{Set{T}, Float64}) where T =
    MassAssignment{T}(frame, masses)

mass(m::MassAssignment{T}, hypothesis::Set{T}) where T = get(m.masses, hypothesis, 0.0)

"""
    belief(m::MassAssignment, hypothesis)

Lower probability: sum of masses of all subsets.
"""
function belief(m::MassAssignment{T}, hypothesis::Set{T}) where T
    sum(mass_val for (s, mass_val) in m.masses if issubset(s, hypothesis); init=0.0)
end

"""
    plausibility(m::MassAssignment, hypothesis)

Upper probability: sum of masses of all intersecting sets.
"""
function plausibility(m::MassAssignment{T}, hypothesis::Set{T}) where T
    sum(mass_val for (s, mass_val) in m.masses if !isempty(intersect(s, hypothesis)); init=0.0)
end

uncertainty(m::MassAssignment{T}, hypothesis::Set{T}) where T =
    (belief(m, hypothesis), plausibility(m, hypothesis))

"""
    combine_dempster(m1, m2)

Dempster's rule of combination: merge two bodies of evidence.
"""
function combine_dempster(m1::MassAssignment{T}, m2::MassAssignment{T}) where T
    m1.frame == m2.frame || error("Frames must match")

    combined = Dict{Set{T}, Float64}()
    conflict = 0.0

    for (A, m1_A) in m1.masses, (B, m2_B) in m2.masses
        inter = intersect(A, B)
        if isempty(inter)
            conflict += m1_A * m2_B
        else
            combined[inter] = get(combined, inter, 0.0) + m1_A * m2_B
        end
    end

    conflict >= 1.0 && error("Total conflict: evidence contradictory")

    if conflict >= 1.0 - 1e-10  # Use epsilon tolerance
        throw(ArgumentError("Total conflict ($(conflict)): evidence is contradictory"))
    end
    norm = 1.0 / (1.0 - conflict)
    MassAssignment(m1.frame, Dict(s => m * norm for (s, m) in combined))
end

"""
    discount(m, reliability)

Discount unreliable evidence by transferring mass to ignorance.
"""
function discount(m::MassAssignment{T}, α::Float64) where T
    0.0 <= α <= 1.0 || error("Reliability must be in [0,1]")

    discounted = Dict{Set{T}, Float64}()
    frame_set = Set(m.frame)
    ignorance = 1.0 - α

    for (s, mass_val) in m.masses
        if s == frame_set
            discounted[s] = α * mass_val + ignorance
        else
            discounted[s] = α * mass_val
        end
    end

    MassAssignment(m.frame, discounted)
end

"""
    pignistic_transform(m)

Convert belief function to probability via pignistic transformation.
"""
function pignistic_transform(m::MassAssignment{T}) where T
    probs = Dict{T, Float64}(h => 0.0 for h in m.frame)

    for (s, mass_val) in m.masses
        if !isempty(s)
            share = mass_val / length(s)
            for element in s
                probs[element] += share
            end
        end
    end

    probs
end

conflict_measure(m1::MassAssignment{T}, m2::MassAssignment{T}) where T =
    sum(m1_A * m2_B for (A, m1_A) in m1.masses, (B, m2_B) in m2.masses if isempty(intersect(A, B)))

end # module DempsterShafer
