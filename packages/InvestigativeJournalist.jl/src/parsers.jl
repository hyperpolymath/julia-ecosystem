# SPDX-License-Identifier: PMPL-1.0-or-later
module Parsers

using ..Types
using TextAnalysis
using StringDistances

export parse_contract, parse_job_description, compare_policies

"""
    parse_contract(text)
Extracts key legal entities and suspicious clauses from contract text.
"""
function parse_contract(text::String)
    # Simple entity extraction using TextAnalysis
    doc = StringDocument(text)
    # find_entities!(doc) # placeholder for real NER
    
    suspicious_terms = ["indemnify", "non-disclosure", "irrevocable", "arbitration"]
    found = filter(t -> occursin(t, lowercase(text)), suspicious_terms)
    
    return (
        entities = ["Entity A", "Entity B"], # Placeholder
        red_flags = found,
        word_count = length(tokens(doc))
    )
end

"""
    parse_job_description(text)
Analyzes a job description for bias or 'Ghost Job' indicators.
"""
function parse_job_description(text::String)
    indicators = ["rockstar", "ninja", "guru", "fast-paced environment"]
    score = count(i -> occursin(i, lowercase(text)), indicators)
    
    return (
        jargon_score = score,
        is_suspicious = score > 2
    )
end

"""
    compare_policies(old_text, new_text)
Calculates the distance between two policy versions to find hidden edits.
"""
function compare_policies(old::String, new::String)
    d = compare(old, new, Levenshtein())
    return (
        similarity = d,
        has_major_changes = d < 0.8
    )
end

end # module
