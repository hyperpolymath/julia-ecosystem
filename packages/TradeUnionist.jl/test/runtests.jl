# SPDX-License-Identifier: PMPL-1.0-or-later
using Test
using TradeUnionist
using Dates
using DataFrames

@testset "TradeUnionist.jl" begin

    # ─────────────────────────────────
    # Types
    # ─────────────────────────────────
    @testset "Types" begin

        @testset "GeoLocation" begin
            geo = GeoLocation(51.5074, -0.1278)
            @test geo.lat == 51.5074
            @test geo.lon == -0.1278
        end

        @testset "Worksite construction" begin
            site = Worksite(:site1, "Acme Corp", "London HQ", "Engineering", 200, nothing)
            @test site.id == :site1
            @test site.employer == "Acme Corp"
            @test site.location_name == "London HQ"
            @test site.unit == "Engineering"
            @test site.headcount_estimate == 200
            @test site.geo === nothing

            # With geo
            geo = GeoLocation(51.5, -0.1)
            site2 = Worksite(:site2, "Beta Ltd", "Manchester", "Ops", 50, geo)
            @test site2.geo.lat == 51.5
        end

        @testset "MemberRecord construction" begin
            t = now()
            mr = MemberRecord(:m1, :site1, :member, :steward, ["pay", "safety"], t, nothing)
            @test mr.id == :m1
            @test mr.worksite_id == :site1
            @test mr.status == :member
            @test mr.role == :steward
            @test length(mr.issues) == 2
            @test mr.last_contact_at == t
            @test mr.home_geo === nothing
        end

        @testset "MemberRecord is mutable" begin
            mr = MemberRecord(:m2, :site1, :non_member, :worker, String[], nothing, nothing)
            mr.status = :supporter
            @test mr.status == :supporter
            mr.role = :organizer
            @test mr.role == :organizer
            push!(mr.issues, "workload")
            @test length(mr.issues) == 1
        end

        @testset "OrganizerConversation" begin
            t = now()
            conv = OrganizerConversation(:c1, :m1, ["wages", "hours"], 0.6, "Schedule follow-up", t)
            @test conv.id == :c1
            @test conv.member_id == :m1
            @test conv.sentiment == 0.6
            @test "wages" in conv.topic_tags
            @test conv.next_step == "Schedule follow-up"
            @test conv.timestamp == t
        end

        @testset "GrievanceCase construction and mutability" begin
            filed = now()
            due = filed + Day(14)
            gc = GrievanceCase(:g1, :m1, filed, :intake, ["memo.pdf"], due)
            @test gc.id == :g1
            @test gc.status == :intake
            @test length(gc.evidence_refs) == 1

            gc.status = :step1
            @test gc.status == :step1
        end

        @testset "ContractClause" begin
            cc = ContractClause(:cl1, "Article 5", "Old text", "New text", :high)
            @test cc.id == :cl1
            @test cc.section == "Article 5"
            @test cc.current_text == "Old text"
            @test cc.proposed_text == "New text"
            @test cc.priority == :high
        end

        @testset "BargainingProposal" begin
            bp = BargainingProposal(:bp1, :cl1, "Cost of living increase", 500.0, :draft)
            @test bp.id == :bp1
            @test bp.clause_id == :cl1
            @test bp.rationale == "Cost of living increase"
            @test bp.cost_estimate == 500.0
            @test bp.status == :draft
        end

        @testset "MobilizationPlan" begin
            t = now()
            mp = MobilizationPlan(:mp1, :rally, t, 100, 85)
            @test mp.id == :mp1
            @test mp.action_type == :rally
            @test mp.target_turnout == 100
            @test mp.actual_turnout == 85
        end
    end

    # ─────────────────────────────────
    # Organizing
    # ─────────────────────────────────
    @testset "Organizing" begin

        @testset "register_worksite" begin
            site = register_worksite("Acme Corp", "London", "Engineering", 150)
            @test site isa Worksite
            @test site.employer == "Acme Corp"
            @test site.location_name == "London"
            @test site.unit == "Engineering"
            @test site.headcount_estimate == 150
            @test site.id isa Symbol
        end

        @testset "upsert_member" begin
            member = upsert_member(:site1, :alice, :member, :steward)
            @test member isa MemberRecord
            @test member.id == :alice
            @test member.worksite_id == :site1
            @test member.status == :member
            @test member.role == :steward
            @test isempty(member.issues)
            @test member.last_contact_at isa DateTime
        end

        @testset "log_conversation" begin
            conv = log_conversation(:alice, ["pay", "safety"], 0.8, "Follow up next week")
            @test conv isa OrganizerConversation
            @test conv.member_id == :alice
            @test conv.sentiment == 0.8
            @test "pay" in conv.topic_tags
            @test conv.next_step == "Follow up next week"
            @test conv.timestamp isa DateTime
        end
    end

    # ─────────────────────────────────
    # Grievances
    # ─────────────────────────────────
    @testset "Grievances" begin

        @testset "open_grievance" begin
            gc = open_grievance(:alice, "Unfair scheduling")
            @test gc isa GrievanceCase
            @test gc.member_id == :alice
            @test gc.status == :intake
            @test "Unfair scheduling" in gc.evidence_refs
            @test gc.due_date > gc.filed_at

            # Custom due_in_days
            gc2 = open_grievance(:bob, "Safety violation", 30)
            @test gc2.due_date > gc2.filed_at
        end

        @testset "update_grievance_status" begin
            gc = open_grievance(:alice, "Contract breach")
            result = update_grievance_status(gc, :step1)
            @test gc.status == :step1
            @test occursin("step1", result)

            update_grievance_status(gc, :arbitration)
            @test gc.status == :arbitration

            update_grievance_status(gc, :resolved)
            @test gc.status == :resolved
        end
    end

    # ─────────────────────────────────
    # Bargaining
    # ─────────────────────────────────
    @testset "Bargaining" begin

        @testset "compare_clauses" begin
            clause = ContractClause(:cl1, "Wages", "15/hr", "18/hr", :high)
            df = compare_clauses(clause)
            @test df isa DataFrame
            @test nrow(df) == 1
            @test df.Section[1] == "Wages"
            @test df.Current[1] == "15/hr"
            @test df.Proposed[1] == "18/hr"
            @test df.Priority[1] == :high
        end

        @testset "cost_proposal" begin
            bp = BargainingProposal(:bp1, :cl1, "Raise", 500.0, :draft)
            total = cost_proposal(bp, 200)
            @test total == 100_000.0

            # Zero headcount
            @test cost_proposal(bp, 0) == 0.0

            # Single worker
            @test cost_proposal(bp, 1) == 500.0
        end
    end

    # ─────────────────────────────────
    # Metrics
    # ─────────────────────────────────
    @testset "Metrics" begin

        @testset "UnionMetrics construction" begin
            um = UnionMetrics(:site1, 65.0, 80.0, 0.05, 12.5)
            @test um.site_id == :site1
            @test um.density == 65.0
            @test um.coverage == 80.0
            @test um.leadership_ratio == 0.05
            @test um.avg_wage_premium == 12.5
        end

        @testset "calc_density" begin
            @test calc_density(50, 100) == 50.0
            @test calc_density(100, 100) == 100.0
            @test calc_density(0, 100) == 0.0
            @test calc_density(0, 0) == 0.0  # edge case: zero eligible
        end

        @testset "calc_coverage" begin
            @test calc_coverage(80, 100) == 80.0
            @test calc_coverage(100, 100) == 100.0
            @test calc_coverage(0, 0) == 0.0
        end

        @testset "calc_leadership_ratio" begin
            @test calc_leadership_ratio(5, 100) == 0.05
            @test calc_leadership_ratio(0, 100) == 0.0
            @test calc_leadership_ratio(0, 0) == 0.0  # edge case
        end

        @testset "wage_gini_coefficient" begin
            # Perfect equality: all same wage
            equal_wages = [50_000.0, 50_000.0, 50_000.0, 50_000.0]
            @test wage_gini_coefficient(equal_wages) ≈ 0.0

            # Empty wages
            @test wage_gini_coefficient(Float64[]) == 0.0

            # Single wage
            @test wage_gini_coefficient([30_000.0]) ≈ 0.0

            # Known inequality: Gini should be positive
            unequal_wages = [10_000.0, 20_000.0, 30_000.0, 100_000.0]
            gini = wage_gini_coefficient(unequal_wages)
            @test gini > 0.0
            @test gini < 1.0
        end
    end

    # ─────────────────────────────────
    # Planning
    # ─────────────────────────────────
    @testset "Planning" begin

        @testset "PlanningLevel subtypes" begin
            @test StrategicGoal() isa TradeUnionist.Planning.PlanningLevel
            @test TacticalObjective() isa TradeUnionist.Planning.PlanningLevel
            @test OperationalTask() isa TradeUnionist.Planning.PlanningLevel
        end

        @testset "UnionActivity construction" begin
            act = UnionActivity(
                StrategicGoal(),
                :Organizing,
                "Achieve 50% density by Q4",
                "Jane Doe",
                :in_progress,
            )
            @test act.level isa TradeUnionist.Planning.PlanningLevel
            @test act.function_area == :Organizing
            @test act.description == "Achieve 50% density by Q4"
            @test act.owner == "Jane Doe"
            @test act.status == :in_progress
        end

        @testset "UnionActivity with different levels" begin
            tactical = UnionActivity(TacticalObjective(), :Bargaining, "Survey members", "Team Lead", :planned)
            @test tactical.level isa TacticalObjective

            operational = UnionActivity(OperationalTask(), :Admin, "Print flyers", "Volunteer", :done)
            @test operational.level isa OperationalTask
            @test operational.status == :done
        end
    end

    # ─────────────────────────────────
    # Events
    # ─────────────────────────────────
    @testset "Events" begin

        @testset "EventType subtypes" begin
            @test StrikeVote() isa TradeUnionist.Events.EventType
            @test Rally() isa TradeUnionist.Events.EventType
            @test TownHall() isa TradeUnionist.Events.EventType
            @test CommitteeMeeting() isa TradeUnionist.Events.EventType
        end

        @testset "UnionEvent construction" begin
            t = now()
            evt = UnionEvent(Rally(), "May Day March", t, "City Square", 500, [:m1, :m2, :m3])
            @test evt.type isa Rally
            @test evt.title == "May Day March"
            @test evt.date == t
            @test evt.location == "City Square"
            @test evt.target_attendance == 500
            @test length(evt.check_in_list) == 3
        end

        @testset "create_event_template" begin
            t = DateTime(2026, 5, 1, 10, 0, 0)

            evt = create_event_template(StrikeVote(), "Authorization Vote", t)
            @test evt isa UnionEvent
            @test evt.type isa StrikeVote
            @test evt.title == "Authorization Vote"
            @test evt.date == t
            @test evt.location == "TBD"
            @test evt.target_attendance == 0
            @test isempty(evt.check_in_list)

            # Different event types
            rally = create_event_template(Rally(), "Solidarity Rally", t)
            @test rally.type isa Rally

            th = create_event_template(TownHall(), "Open Forum", t)
            @test th.type isa TownHall

            cm = create_event_template(CommitteeMeeting(), "Executive Board", t)
            @test cm.type isa CommitteeMeeting
        end
    end

    # ─────────────────────────────────
    # Branding
    # ─────────────────────────────────
    @testset "Branding" begin

        @testset "UnionLeaderProfile construction" begin
            leader = UnionLeaderProfile(
                "Rosa Parks",
                "Chief Steward",
                "1199",
                ["Workers deserve dignity", "Together we bargain"],
            )
            @test leader.name == "Rosa Parks"
            @test leader.title == "Chief Steward"
            @test leader.local_number == "1199"
            @test length(leader.narratives) == 2
            @test "Workers deserve dignity" in leader.narratives
        end
    end

end # top-level testset
