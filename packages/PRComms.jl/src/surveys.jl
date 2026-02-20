# SPDX-License-Identifier: PMPL-1.0-or-later
module Surveys

using JSON3

export Question, SurveyTemplate, add_question!, build_survey_json

struct Question
    text::String
    type::Symbol # :likert, :text, :multiple_choice, :nps
    options::Vector{String}
end

Question(text, type) = Question(text, type, String[])

mutable struct SurveyTemplate
    id::Symbol
    title::String
    description::String
    questions::Vector{Question}
end

function SurveyTemplate(id::Symbol, title::String)
    return SurveyTemplate(id, title, "", Question[])
end

"""
    add_question!(survey, text, type; options=[])
Adds a new question to the survey developer.
"""
function add_question!(s::SurveyTemplate, text::String, type::Symbol; options=String[])
    push!(s.questions, Question(text, type, options))
    return "Question added to '$(s.title)'"
end

"""
    build_survey_json(survey)
Exports the survey design to JSON for web/app integration.
"""
function build_survey_json(s::SurveyTemplate)
    return JSON3.write(s)
end

end # module
