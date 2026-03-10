# SPDX-License-Identifier: PMPL-1.0-or-later
using Test
using PRComms
using Dates
using DataFrames

@testset "PRComms.jl" begin

    # -----------------------------------------------------------------------
    # Types
    # -----------------------------------------------------------------------
    @testset "Types" begin
        @testset "MessagePillar" begin
            pillar = MessagePillar(:p1, "Sustainability", ["Net zero by 2030", "Green supply chain"],
                                   "VP Comms", DateTime(2026, 3, 1))
            @test pillar.id == :p1
            @test pillar.theme == "Sustainability"
            @test length(pillar.key_points) == 2
            @test pillar.approved_by == "VP Comms"
            @test pillar.approved_at == DateTime(2026, 3, 1)
        end

        @testset "AudienceVariant" begin
            variant = AudienceVariant(:v1, :p1, :investors, :linkedin, :formal,
                                      "We are committed to sustainability.")
            @test variant.id == :v1
            @test variant.pillar_id == :p1
            @test variant.audience == :investors
            @test variant.channel == :linkedin
            @test variant.tone == :formal
            @test variant.body == "We are committed to sustainability."
        end

        @testset "AudienceVariant audience/channel/tone variants" begin
            for aud in [:investors, :customers, :employees, :media]
                v = AudienceVariant(:v, :p, aud, :twitter, :bold, "")
                @test v.audience == aud
            end
            for ch in [:twitter, :linkedin, :press_release, :internal]
                v = AudienceVariant(:v, :p, :media, ch, :formal, "")
                @test v.channel == ch
            end
            for tone in [:formal, :empathetic, :bold]
                v = AudienceVariant(:v, :p, :media, :twitter, tone, "")
                @test v.tone == tone
            end
        end

        @testset "PressRelease" begin
            pr = PressRelease(:pr1, "Big Announcement", "Body text here",
                              :draft, nothing, nothing)
            @test pr.id == :pr1
            @test pr.title == "Big Announcement"
            @test pr.status == :draft
            @test pr.embargo_at === nothing
            @test pr.approved_at === nothing

            # Mutable - test status change
            pr.status = :published
            @test pr.status == :published
        end

        @testset "MediaContact" begin
            contact = MediaContact(:mc1, "Sarah Reporter", "The Times",
                                   [:tech, :finance], "sarah@times.com", nothing)
            @test contact.id == :mc1
            @test contact.name == "Sarah Reporter"
            @test contact.outlet == "The Times"
            @test :tech in contact.beat
            @test :finance in contact.beat
            @test contact.email == "sarah@times.com"
            @test contact.last_contacted_at === nothing
        end

        @testset "PitchRecord" begin
            pitch = PitchRecord(:pi1, :mc1, :pr1, DateTime(2026, 2, 15), :sent)
            @test pitch.id == :pi1
            @test pitch.contact_id == :mc1
            @test pitch.release_id == :pr1
            @test pitch.response_status == :sent

            for status in [:sent, :opened, :responded, :declined]
                p = PitchRecord(:pi, :mc, :pr, now(), status)
                @test p.response_status == status
            end
        end

        @testset "Campaign" begin
            camp = Campaign(:camp1, "Q1 Product Launch",
                            ["Awareness", "Lead gen"],
                            DateTime(2026, 1, 1), DateTime(2026, 3, 31))
            @test camp.id == :camp1
            @test camp.name == "Q1 Product Launch"
            @test length(camp.goals) == 2
            @test camp.start_at < camp.end_at
        end

        @testset "CrisisPlaybook" begin
            playbook = CrisisPlaybook(:cpb1, :data_breach, 4,
                                      ["We are investigating", "More details to follow"],
                                      ["CISO", "CEO", "Legal", "Comms"])
            @test playbook.id == :cpb1
            @test playbook.incident_type == :data_breach
            @test playbook.severity == 4
            @test length(playbook.holding_statements) == 2
            @test length(playbook.escalation_tree) == 4
        end
    end

    # -----------------------------------------------------------------------
    # Messaging
    # -----------------------------------------------------------------------
    @testset "Messaging" begin
        @testset "create_pillar" begin
            pillar = create_pillar(:test_p, "Innovation",
                                   ["AI-first", "Open source commitment"])
            @test pillar isa MessagePillar
            @test pillar.id == :test_p
            @test pillar.theme == "Innovation"
            @test length(pillar.key_points) == 2
            @test pillar.approved_by == "Chief Comms"  # default
            @test pillar.approved_at <= now()
        end

        @testset "create_pillar custom approver" begin
            pillar = create_pillar(:test_p2, "Safety", ["Zero incidents"];
                                   approver="Head of Safety")
            @test pillar.approved_by == "Head of Safety"
        end

        @testset "generate_variant" begin
            pillar = create_pillar(:gv_p, "Growth", ["Revenue up 20%", "New markets"])
            variant = generate_variant(pillar, :investors, :linkedin)
            @test variant isa AudienceVariant
            @test variant.pillar_id == :gv_p
            @test variant.audience == :investors
            @test variant.channel == :linkedin
            @test variant.tone == :formal  # default
            @test occursin("investors", variant.body)
            @test occursin("Revenue up 20%", variant.body)
        end

        @testset "generate_variant custom tone" begin
            pillar = create_pillar(:tone_p, "Culture", ["We care"])
            variant = generate_variant(pillar, :employees, :internal; tone=:empathetic)
            @test variant.tone == :empathetic
        end
    end

    # -----------------------------------------------------------------------
    # Newsroom
    # -----------------------------------------------------------------------
    @testset "Newsroom" begin
        @testset "draft_release" begin
            pr = draft_release(:nr1, "Launch Day", "We are excited to announce...")
            @test pr isa PressRelease
            @test pr.id == :nr1
            @test pr.title == "Launch Day"
            @test pr.body == "We are excited to announce..."
            @test pr.status == :draft
            @test pr.embargo_at === nothing
            @test pr.approved_at === nothing
        end

        @testset "review_release" begin
            pr = draft_release(:nr2, "Review Test", "Body")
            result = review_release(pr)
            @test result isa String
            @test pr.status == :review
            @test occursin("Review Test", result)
        end

        @testset "publish_release without embargo" begin
            pr = draft_release(:nr3, "Publish Test", "Body")
            result = publish_release(pr)
            @test result isa String
            @test pr.status == :published
            @test occursin("LIVE", result)
        end

        @testset "publish_release with embargo" begin
            pr = draft_release(:nr4, "Embargo Test", "Body")
            embargo_time = DateTime(2026, 6, 1, 9, 0)
            result = publish_release(pr; embargo=embargo_time)
            @test result isa String
            @test pr.status == :embargoed
            @test pr.embargo_at == embargo_time
        end
    end

    # -----------------------------------------------------------------------
    # Crisis
    # -----------------------------------------------------------------------
    @testset "Crisis" begin
        @testset "activate_crisis_mode" begin
            playbook = CrisisPlaybook(:crisis1, :product_recall, 3,
                                      ["Statement 1"], ["Operations", "Legal", "CEO"])
            result = activate_crisis_mode(playbook)
            @test result isa String
            @test occursin("Incident Management", result)
        end

        @testset "issue_holding_statement" begin
            playbook = CrisisPlaybook(:crisis2, :cyber_attack, 5,
                                      ["We are aware", "Investigating now", "Update soon"],
                                      ["CISO"])
            stmt = issue_holding_statement(playbook, 1)
            @test stmt isa String
            @test occursin("We are aware", stmt)

            stmt2 = issue_holding_statement(playbook, 3)
            @test occursin("Update soon", stmt2)
        end

        @testset "issue_holding_statement bounds" begin
            playbook = CrisisPlaybook(:crisis3, :flood, 2,
                                      ["Only one statement"], ["Ops"])
            @test_throws BoundsError issue_holding_statement(playbook, 5)
        end
    end

    # -----------------------------------------------------------------------
    # Analytics
    # -----------------------------------------------------------------------
    @testset "Analytics" begin
        @testset "SurveyResult" begin
            sr = SurveyResult(:sr1, "Customer Satisfaction", 100,
                              [9, 10, 8, 7, 3, 10, 9, 6, 5, 10],
                              ["Great product", "Needs work"])
            @test sr.id == :sr1
            @test sr.topic == "Customer Satisfaction"
            @test sr.respondents == 100
            @test length(sr.scores) == 10
            @test length(sr.verbatim) == 2
        end

        @testset "calc_nps" begin
            # Promoters (9-10): 5 scores, Detractors (1-6): 3 scores, Total: 10
            sr = SurveyResult(:nps1, "NPS Test", 10,
                              [9, 10, 10, 9, 10, 3, 5, 6, 7, 8], String[])
            nps = calc_nps(sr)
            @test nps isa Float64
            # 5 promoters, 3 detractors, 10 total => (5-3)/10 * 100 = 20.0
            @test nps == 20.0
        end

        @testset "calc_nps all promoters" begin
            sr = SurveyResult(:nps2, "Great", 5, [10, 10, 10, 9, 9], String[])
            @test calc_nps(sr) == 100.0
        end

        @testset "calc_nps all detractors" begin
            sr = SurveyResult(:nps3, "Bad", 5, [1, 2, 3, 4, 5], String[])
            @test calc_nps(sr) == -100.0
        end

        @testset "calc_nps empty scores" begin
            sr = SurveyResult(:nps4, "Empty", 0, Int[], String[])
            @test calc_nps(sr) == 0.0
        end

        @testset "share_of_voice" begin
            @test share_of_voice(250, 1000) == 25.0
            @test share_of_voice(0, 1000) == 0.0
            @test share_of_voice(0, 0) == 0.0
            @test share_of_voice(500, 500) == 100.0
        end

        @testset "first_order_ratio" begin
            @test first_order_ratio(50, 200) == 0.25
            @test first_order_ratio(0, 100) == 0.0
            @test first_order_ratio(10, 0) == 0.0  # divide by zero guard
        end

        @testset "second_order_ratio" begin
            @test second_order_ratio(0.5, 1000) == 0.0005
            @test second_order_ratio(1.0, 0) == 0.0  # divide by zero guard
        end

        @testset "third_order_ratio" begin
            @test third_order_ratio(0.8, 0.6, 2.0) == 0.1
            @test third_order_ratio(0.5, 0.5, 0) == 0.0  # divide by zero guard
        end
    end

    # -----------------------------------------------------------------------
    # Strategy
    # -----------------------------------------------------------------------
    @testset "Strategy" begin
        @testset "CommsPlan construction" begin
            plan = CommsPlan(:cp1, "Q2 Campaign")
            @test plan.id == :cp1
            @test plan.name == "Q2 Campaign"
            @test plan.timeline isa DataFrame
            @test nrow(plan.timeline) == 0
        end

        @testset "add_milestone" begin
            plan = CommsPlan(:cp2, "Launch Plan")
            result = add_milestone(plan, Date(2026, 4, 1), :twitter,
                                   "Announce product", "Alice")
            @test result isa String
            @test occursin("Announce product", result)
            @test nrow(plan.timeline) == 1

            add_milestone(plan, Date(2026, 3, 15), :linkedin,
                          "Teaser post", "Bob")
            @test nrow(plan.timeline) == 2
            # Should be sorted by date
            @test plan.timeline.Date[1] == Date(2026, 3, 15)
            @test plan.timeline.Date[2] == Date(2026, 4, 1)
        end

        @testset "brand_equity_valuation" begin
            result = brand_equity_valuation(1_000_000.0, 0.5)
            @test hasproperty(result, :value)
            @test hasproperty(result, :royalty_rate_percent)
            @test hasproperty(result, :explanation)
            @test result.value isa Float64
            @test result.value > 0

            # brand_strength_index = 0.0 => royalty_rate = 0.01, value = 1M * 0.01 * 5 = 50000
            result_low = brand_equity_valuation(1_000_000.0, 0.0)
            @test result_low.value == 50_000.0
            @test result_low.royalty_rate_percent == 1.0

            # brand_strength_index = 1.0 => royalty_rate = 0.05, value = 1M * 0.05 * 5 = 250000
            result_high = brand_equity_valuation(1_000_000.0, 1.0)
            @test result_high.value == 250_000.0
            @test result_high.royalty_rate_percent == 5.0
        end
    end

    # -----------------------------------------------------------------------
    # Surveys
    # -----------------------------------------------------------------------
    @testset "Surveys" begin
        @testset "Question" begin
            q = Question("How satisfied are you?", :likert,
                         ["Very", "Somewhat", "Not at all"])
            @test q.text == "How satisfied are you?"
            @test q.type == :likert
            @test length(q.options) == 3
        end

        @testset "Question short constructor" begin
            q = Question("What do you think?", :text)
            @test q.text == "What do you think?"
            @test q.type == :text
            @test isempty(q.options)
        end

        @testset "SurveyTemplate" begin
            st = SurveyTemplate(:st1, "Customer Feedback")
            @test st.id == :st1
            @test st.title == "Customer Feedback"
            @test st.description == ""
            @test isempty(st.questions)
        end

        @testset "add_question!" begin
            st = SurveyTemplate(:st2, "Employee Survey")
            result = add_question!(st, "Rate your manager", :likert;
                                   options=["1", "2", "3", "4", "5"])
            @test result isa String
            @test occursin("Employee Survey", result)
            @test length(st.questions) == 1
            @test st.questions[1].text == "Rate your manager"

            add_question!(st, "Comments?", :text)
            @test length(st.questions) == 2
        end

        @testset "build_survey_json" begin
            st = SurveyTemplate(:st3, "JSON Test")
            add_question!(st, "Q1", :nps)
            json_str = build_survey_json(st)
            @test json_str isa String
            @test occursin("JSON Test", json_str)
            @test occursin("Q1", json_str)
            @test occursin("nps", json_str)
        end
    end

    # -----------------------------------------------------------------------
    # PlanningLevels
    # -----------------------------------------------------------------------
    @testset "PlanningLevels" begin
        @testset "PRLevel subtypes" begin
            @test BusinessPR() isa PRComms.PlanningLevels.PRLevel
            @test StrategicPR() isa PRComms.PlanningLevels.PRLevel
            @test TacticalPR() isa PRComms.PlanningLevels.PRLevel
            @test OperationalPR() isa PRComms.PlanningLevels.PRLevel
        end

        @testset "PRActivity" begin
            activity = PRActivity(StrategicPR(), :HR,
                                  "Improve employer brand", :in_progress)
            @test activity.level isa StrategicPR
            @test activity.function_area == :HR
            @test activity.objective == "Improve employer brand"
            @test activity.status == :in_progress
        end
    end

    # -----------------------------------------------------------------------
    # BoundaryObjects
    # -----------------------------------------------------------------------
    @testset "BoundaryObjects" begin
        @testset "MessageHouse" begin
            house = MessageHouse("We build the future",
                                 ["Innovation", "Trust", "Scale"],
                                 ["100M users", "99.9% uptime", "ISO 27001"])
            @test house.roof == "We build the future"
            @test length(house.pillars) == 3
            @test length(house.foundation) == 3
        end

        @testset "create_message_house" begin
            house = create_message_house(
                "Sustainability Leader",
                ["Carbon neutral ops", "Green supply chain"],
                ["Reduced emissions 40%", "B Corp certified"]
            )
            @test house isa MessageHouse
            @test house.roof == "Sustainability Leader"
        end

        @testset "SharedGlossary" begin
            glossary = SharedGlossary(Dict(
                "ROI" => "Return on Investment",
                "SOV" => "Share of Voice"
            ))
            @test glossary.terms["ROI"] == "Return on Investment"
            @test length(glossary.terms) == 2
        end
    end

    # -----------------------------------------------------------------------
    # InternalComms
    # -----------------------------------------------------------------------
    @testset "InternalComms" begin
        @testset "EmployeeEngagement" begin
            ee = EmployeeEngagement(:engineering, 0.85, 4.2)
            @test ee.department == :engineering
            @test ee.participation_rate == 0.85
            @test ee.sentiment_score == 4.2
        end

        @testset "InternalNewsletter" begin
            nl = InternalNewsletter(:nl1, 42, "Q1 Highlights", 0.67)
            @test nl.id == :nl1
            @test nl.volume == 42
            @test nl.headline == "Q1 Highlights"
            @test nl.read_rate == 0.67
        end

        @testset "log_engagement" begin
            result = log_engagement(:sales, 0.72, 3.8)
            @test result isa EmployeeEngagement
            @test result.department == :sales
            @test result.participation_rate == 0.72
            @test result.sentiment_score == 3.8
        end
    end

    # -----------------------------------------------------------------------
    # PersonalPR
    # -----------------------------------------------------------------------
    @testset "PersonalPR" begin
        @testset "ThoughtLeaderProfile" begin
            profile = ThoughtLeaderProfile("Jane CEO",
                                           ["AI", "Leadership"],
                                           ["AI transforms business", "Trust is key"])
            @test profile.executive_name == "Jane CEO"
            @test length(profile.expertise_areas) == 2
            @test length(profile.key_narratives) == 2
        end

        @testset "PresentationRecord" begin
            rec = PresentationRecord("TechConf 2026", Date(2026, 5, 1),
                                     "Future of AI", 5000)
            @test rec.event_name == "TechConf 2026"
            @test rec.topic == "Future of AI"
            @test rec.audience_reach == 5000
        end

        @testset "track_appearance" begin
            profile = ThoughtLeaderProfile("Bob CTO", ["Security"], ["Zero trust"])
            result = track_appearance(profile, "SecCon", Date(2026, 6, 1),
                                      "Zero Trust Architecture", 2000)
            @test result isa PresentationRecord
            @test result.event_name == "SecCon"
            @test result.topic == "Zero Trust Architecture"
            @test result.audience_reach == 2000
        end
    end

    # -----------------------------------------------------------------------
    # Publics
    # -----------------------------------------------------------------------
    @testset "Publics" begin
        @testset "PublicGroup subtypes" begin
            @test Shareholders() isa PublicGroup
            @test LocalCommunity() isa PublicGroup
            @test MediaPublic() isa PublicGroup
            @test GovernmentPublic() isa PublicGroup
            @test EmployeePublic() isa PublicGroup
        end
    end

    # -----------------------------------------------------------------------
    # AssetTemplates
    # -----------------------------------------------------------------------
    @testset "AssetTemplates" begin
        @testset "make_email_signature" begin
            sig = make_email_signature("Alice Smith", "Director of Comms",
                                       "alice@example.com", "+44 123 456")
            @test sig isa String
            @test occursin("Alice Smith", sig)
            @test occursin("Director of Comms", sig)
            @test occursin("alice@example.com", sig)
            @test occursin("+44 123 456", sig)
        end
    end

end # top-level testset
