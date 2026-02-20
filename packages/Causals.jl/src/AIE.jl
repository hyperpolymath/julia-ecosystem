# SPDX-License-Identifier: PMPL-1.0-or-later
"""
Applied Information Economics (AIE) implementation based on Hubbard's framework.
Focuses on the Value of Information (VoI) and reducing uncertainty in causal models.
"""
module AIE

export evoi, reduce_uncertainty

"""
    evoi(opportunity_loss, prob_of_wrong_decision)
Calculates the Expected Value of Information. 
Helps determine if further investigation/data collection is economically justified.
"""
function evoi(loss::Float64, p_wrong::Float64)
    println("Calculating EVOI for decision node... 💰")
    return loss * p_wrong
end

"""
    reduce_uncertainty(prior_range, new_data)
AIE-style uncertainty reduction.
"""
function reduce_uncertainty(prior, data)
    println("Applying AIE calibration to reduce uncertainty... 📉")
    return (lower=prior.lower * 0.9, upper=prior.upper * 0.9) # Simplified shrinkage
end

end # module
