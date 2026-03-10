# SPDX-License-Identifier: PMPL-1.0-or-later
using Test
using PostDisciplinary
using UUIDs
using Dates
using DataFrames

@testset "PostDisciplinary.jl" begin

    # ─────────────────────────────────────────────
    # Core types: LinkedEntity and ResearchProject
    # ─────────────────────────────────────────────
    @testset "LinkedEntity" begin
        id = uuid4()
        e = LinkedEntity(id, :Cliodynamics, :polity_42, :Claim, Dict(:confidence => 0.9))
        @test e.id == id
        @test e.source_library == :Cliodynamics
        @test e.original_id == :polity_42
        @test e.kind == :Claim
        @test e.metadata[:confidence] == 0.9
    end

    @testset "ResearchProject construction" begin
        p = ResearchProject("Climate Justice")
        @test p.name == "Climate Justice"
        @test p.id isa Symbol
        @test p.graph isa MetaGraphsNext.MetaGraph
    end

    @testset "add_link! and generate_synthesis" begin
        p = ResearchProject("Test Synthesis")

        e1 = LinkedEntity(uuid4(), :Cliodynamics, :secular_cycle, :Model,
                          Dict(:period => "200yr"))
        e2 = LinkedEntity(uuid4(), :Axiology, :justice_theory, :Claim,
                          Dict(:school => "Rawlsian"))

        result = add_link!(p, e1, e2, :supports)
        @test occursin("Link established", result)
        @test occursin("Model", result)
        @test occursin("Claim", result)

        synth = generate_synthesis(p)
        @test synth.report_title == "Synthesis: Test Synthesis"
        @test synth.entities isa DataFrame
        @test nrow(synth.entities) == 2
        @test synth.graph_density isa Float64
    end

    @testset "@link macro" begin
        p = ResearchProject("Macro Test")
        # The @link macro prints and executes the block; just verify no error
        @test begin
            @link p begin
                x = 1 + 1
            end
            true
        end
    end

    # ─────────────────────────────────
    # RaftConsensus
    # ─────────────────────────────────
    @testset "RaftConsensus" begin
        node = RaftNode()
        @test node.state == PostDisciplinary.RaftConsensus.Follower
        @test node.current_term == 0
        @test node.voted_for === nothing
        @test isempty(node.log)

        candidate_id = uuid4()

        # Vote granted when term is higher
        @test request_vote(node, 1, candidate_id) == true
        @test node.current_term == 1
        @test node.voted_for == candidate_id

        # Vote denied when term is not higher
        @test request_vote(node, 1, uuid4()) == false
        @test request_vote(node, 0, uuid4()) == false

        # AppendEntries with equal or higher term succeeds
        leader_id = uuid4()
        entries = ["fact1", "fact2"]
        @test append_entries(node, 1, leader_id, entries) == true
        @test length(node.log) == 2
        @test node.log == ["fact1", "fact2"]

        # AppendEntries with lower term fails
        @test append_entries(node, 0, leader_id, ["nope"]) == false
        @test length(node.log) == 2  # unchanged

        # AppendEntries with higher term updates term
        append_entries(node, 5, leader_id, ["fact3"])
        @test node.current_term == 5
        @test length(node.log) == 3
    end

    # ─────────────────────────────────
    # MetaAnalysis
    # ─────────────────────────────────
    @testset "MetaAnalysis" begin
        @testset "EffectSize construction" begin
            es = EffectSize(0.5, 0.04, 25.0)
            @test es.value == 0.5
            @test es.variance == 0.04
            @test es.weight == 25.0
        end

        @testset "aggregate_effects" begin
            studies = [
                EffectSize(0.3, 0.01, 100.0),
                EffectSize(0.5, 0.02, 50.0),
                EffectSize(0.7, 0.03, 50.0),
            ]
            pooled = aggregate_effects(studies)
            # Manual: (0.3*100 + 0.5*50 + 0.7*50) / (100+50+50) = (30+25+35)/200 = 0.45
            @test pooled ≈ 0.45

            # Single study returns its own value
            @test aggregate_effects([EffectSize(0.8, 0.01, 1.0)]) ≈ 0.8
        end

        @testset "heterogeneity_q" begin
            studies = [
                EffectSize(0.3, 0.01, 100.0),
                EffectSize(0.5, 0.02, 50.0),
                EffectSize(0.7, 0.03, 50.0),
            ]
            pooled = 0.45
            q = heterogeneity_q(studies, pooled)
            # Q = 100*(0.3-0.45)^2 + 50*(0.5-0.45)^2 + 50*(0.7-0.45)^2
            #   = 100*0.0225 + 50*0.0025 + 50*0.0625
            #   = 2.25 + 0.125 + 3.125 = 5.5
            @test q ≈ 5.5

            # All identical effects => Q = 0
            uniform = [EffectSize(0.4, 0.01, 10.0) for _ in 1:5]
            @test heterogeneity_q(uniform, 0.4) ≈ 0.0
        end
    end

    # ─────────────────────────────────
    # KnowledgeDiffing
    # ─────────────────────────────────
    @testset "KnowledgeDiffing" begin
        p1 = ResearchProject("Era 1")
        p2 = ResearchProject("Era 2")
        diff = diff_knowledge_graphs(p1, p2)
        @test diff isa KnowledgeDiff
        @test :NewTheory in diff.added_entities
        @test :DebunkedClaim in diff.removed_entities
        @test (:TheoryA, :TheoryB) in diff.changed_relationships
    end

    # ─────────────────────────────────
    # Memetics
    # ─────────────────────────────────
    @testset "Memetics" begin
        meme_id = uuid4()
        m = Meme(meme_id, "Dawkins original meme", :Cliodynamics, UUID[])
        @test m.id == meme_id
        @test m.content == "Dawkins original meme"
        @test isempty(m.variants)

        r = Replicator(m, 0.8, 0.5, now())
        @test r.fitness == 0.8
        @test r.virality == 0.5

        new_id = mutate(r, 0.1)
        @test new_id isa UUID
        @test length(r.meme.variants) == 1
        @test r.meme.variants[1] == new_id

        fitness = calculate_fitness(r, nothing)
        @test fitness ≈ 0.8 * 1.1
    end

    # ─────────────────────────────────
    # Methodology (ResearchStrategy)
    # ─────────────────────────────────
    @testset "Methodology" begin
        @test MultiDisciplinary() isa ResearchStrategy
        @test InterDisciplinary() isa ResearchStrategy
        @test TransDisciplinary() isa ResearchStrategy
        @test AntiDisciplinary() isa ResearchStrategy

        p = ResearchProject("Strategy Test")
        # execute_strategy returns nothing (prints only); just verify no error
        @test execute_strategy(MultiDisciplinary(), p) === nothing
        @test execute_strategy(TransDisciplinary(), p) === nothing
        @test execute_strategy(AntiDisciplinary(), p) === nothing
    end

    # ─────────────────────────────────
    # VeriSimBridge
    # ─────────────────────────────────
    @testset "VeriSimBridge" begin
        client = VeriSimClient("http://localhost:9090", "test-key")
        @test client.endpoint == "http://localhost:9090"
        @test client.api_key == "test-key"

        @test store_hexad(client, Dict(:test => true)) == :stored
        @test vql_query(client, "SELECT * FROM hexads") == []
    end

    # ─────────────────────────────────
    # LithoglyphBridge
    # ─────────────────────────────────
    @testset "LithoglyphBridge" begin
        client = LithoglyphClient("http://localhost:8080")
        @test client.endpoint == "http://localhost:8080"

        @test store_glyph(client, "glyph-data") == :registered
        @test find_symbol(client, "ankh") == []
    end

    # ─────────────────────────────────
    # Triangulation
    # ─────────────────────────────────
    @testset "Triangulation" begin
        p = ResearchProject("Triangulation Test")
        report = triangulate_findings(p, [:History, :Economics, :Ethics])
        @test report isa CorrelationReport
        @test report.agreement_score == 0.85
        @test "None" in report.contradictions
        @test :HumanProgress in report.shared_entities
    end

    # ─────────────────────────────────
    # ExpertSynthesis
    # ─────────────────────────────────
    @testset "ExpertSynthesis" begin
        @testset "ExpertRouter" begin
            expert = PostDisciplinary.ExpertSynthesis.DisciplinaryExpert(:e1, :History, 0.95)
            router = ExpertRouter([expert])
            @test length(router.experts) == 1

            result = route_to_discipline(router, "What caused the fall of Rome?")
            @test result.id == :e1
            @test result.domain == :History
            @test result.reliability == 0.95
        end

        @testset "GraphOfThought" begin
            got = GraphOfThought(:proj1, String[], Symbol[])
            @test isempty(got.thought_nodes)
            @test isempty(got.transformations)

            evolve_thought!(got, "Initial hypothesis", :abduction)
            @test length(got.thought_nodes) == 1
            @test got.thought_nodes[1] == "Initial hypothesis"
            @test got.transformations[1] == :abduction

            evolve_thought!(got, "Refined theory", :deduction)
            @test length(got.thought_nodes) == 2
            @test length(got.transformations) == 2
        end
    end

    # ─────────────────────────────────
    # Hermeneutics
    # ─────────────────────────────────
    @testset "Hermeneutics" begin
        circle = HermeneuticCircle(:proj1, Dict{Symbol,Any}(:era => "Modern"))
        @test circle.project_id == :proj1
        @test circle.global_context[:era] == "Modern"

        entity = LinkedEntity(uuid4(), :History, :event_1, :Finding, Dict{Symbol,Any}())
        result = interpret_part(circle, entity)
        @test result == :interpreted_result

        entities = [
            LinkedEntity(uuid4(), :Economics, :gdp_trend, :Data, Dict{Symbol,Any}()),
            LinkedEntity(uuid4(), :Ethics, :justice_claim, :Claim, Dict{Symbol,Any}()),
        ]
        # synthesize_context returns nothing; verify no error
        @test synthesize_context(circle, entities) === nothing
    end

    # ─────────────────────────────────
    # MixedMethods (ResearchDesign)
    # ─────────────────────────────────
    @testset "MixedMethods" begin
        @test QuantDesign() isa ResearchDesign
        @test QualDesign() isa ResearchDesign
        @test MixedDesign() isa ResearchDesign

        p = ResearchProject("Design Test")
        @test run_design(QuantDesign(), p) === nothing
        @test run_design(QualDesign(), p) === nothing
        @test run_design(MixedDesign(), p) === nothing
    end

    # ─────────────────────────────────
    # KnowledgeTransfer (Impact)
    # ─────────────────────────────────
    @testset "KnowledgeTransfer" begin
        @testset "ImpactRecord construction" begin
            rec = ImpactRecord(:proj1, :policy_brief, now(), "Used in parliamentary briefing")
            @test rec.project_id == :proj1
            @test rec.utilisation_type == :policy_brief
            @test rec.description == "Used in parliamentary briefing"
        end

        @testset "log_utilisation" begin
            p = ResearchProject("Impact Test")
            record = log_utilisation(p, :public_campaign, "Campaign launched")
            @test record isa ImpactRecord
            @test record.utilisation_type == :public_campaign
            @test record.description == "Campaign launched"
            @test record.project_id == p.id
        end

        @testset "measure_influence" begin
            p = ResearchProject("Influence Test")
            result = measure_influence(p)
            @test result.adoption_rate == 0.65
            @test result.policy_hits == 12
        end
    end

    # ─────────────────────────────────
    # ResearchTemplates
    # ─────────────────────────────────
    @testset "ResearchTemplates" begin
        @test WickedProblem() isa ResearchTemplate
        @test StructuralForensics() isa ResearchTemplate
        @test ActionResearch() isa ResearchTemplate

        @testset "scaffold_project - WickedProblem" begin
            p = scaffold_project(WickedProblem(), "Housing Crisis")
            @test p isa ResearchProject
            @test p.name == "Housing Crisis"
        end

        @testset "scaffold_project - StructuralForensics" begin
            p = scaffold_project(StructuralForensics(), "Colonial Legacy")
            @test p isa ResearchProject
            @test p.name == "Colonial Legacy"
        end

        @testset "scaffold_project - ActionResearch" begin
            p = scaffold_project(ActionResearch(), "Community Organising")
            @test p isa ResearchProject
            @test p.name == "Community Organising"
        end
    end

end # top-level testset
