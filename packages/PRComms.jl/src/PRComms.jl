# SPDX-License-Identifier: PMPL-1.0-or-later
"""
    PRComms

Public relations and communications management toolkit. Covers message pillar
creation, press release drafting, crisis communications, stakeholder surveys,
brand equity modelling, and multi-level communications planning (strategic,
tactical, operational).

# Key Features
- Message pillars with audience-specific variants
- Press release lifecycle (draft, review, publish)
- Crisis playbook activation and holding statements
- NPS calculation, share-of-voice analysis, and survey builder
- Specialist modules for internal comms, personal PR, and public groups

# Example
```julia
using PRComms
pillar = create_pillar("Sustainability", "Our green commitments")
release = draft_release("Launch Event", pillar)
```
"""
module PRComms

include("types.jl")
include("messaging.jl")
include("newsroom.jl")
include("crisis.jl")
include("analytics.jl")
include("strategy.jl")
include("surveys.jl")
include("planning/levels.jl")
include("boundary_objects.jl")
include("specialists/internal.jl")
include("specialists/personal.jl")
include("specialists/publics.jl")
include("assets/templates.jl")

using .Types
using .Messaging
using .Newsroom
using .Crisis
using .Analytics
using .Strategy
using .Surveys
using .PlanningLevels
using .BoundaryObjects
using .InternalComms
using .PersonalPR
using .Publics
using .AssetTemplates

# Re-export core operations
export MessagePillar, AudienceVariant, PressRelease, MediaContact, PitchRecord, Campaign, CrisisPlaybook
export create_pillar, generate_variant
export draft_release, review_release, publish_release
export activate_crisis_mode, issue_holding_statement
export SurveyResult, calc_nps, share_of_voice
export first_order_ratio, second_order_ratio, third_order_ratio
export CommsPlan, add_milestone, brand_equity_valuation
export Question, SurveyTemplate, add_question!, build_survey_json

# Re-export specialized structures
export PRActivity, BusinessPR, StrategicPR, TacticalPR, OperationalPR
export MessageHouse, SharedGlossary, create_message_house
export EmployeeEngagement, InternalNewsletter, log_engagement
export ThoughtLeaderProfile, PresentationRecord, track_appearance
export PublicGroup, Shareholders, LocalCommunity, MediaPublic, GovernmentPublic, EmployeePublic
export make_email_signature, make_business_card, make_letterhead

end # module PRComms
