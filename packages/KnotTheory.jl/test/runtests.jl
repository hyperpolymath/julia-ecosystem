# SPDX-License-Identifier: PMPL-1.0-or-later
using Test
using Graphs
using Polynomials
using LinearAlgebra
using KnotTheory

@testset "KnotTheory" begin

    # -----------------------------------------------------------------------
    # Basic constructors and types
    # -----------------------------------------------------------------------
    @testset "Basics" begin
        k = unknot()
        @test k isa Knot
        @test k.name == :unknot
        @test crossing_number(k) == 0
        @test k.pd !== nothing
        @test isempty(k.pd.crossings)

        t = trefoil()
        @test t isa Knot
        @test t.name == :trefoil
        @test crossing_number(t) == 3
        @test t.pd !== nothing
        @test t.dt !== nothing
        @test t.dt.code == [4, 6, 2]

        fe = figure_eight()
        @test fe isa Knot
        @test fe.name == :figure_eight
        @test crossing_number(fe) == 4
        @test fe.pd !== nothing
        @test fe.dt !== nothing
        @test fe.dt.code == [4, 6, 8, 2]

        c5 = cinquefoil()
        @test c5 isa Knot
        @test c5.name == Symbol("5_1")
        @test crossing_number(c5) == 5
        @test c5.pd !== nothing
        @test c5.dt !== nothing
        @test c5.dt.code == [6, 8, 10, 2, 4]
    end

    # -----------------------------------------------------------------------
    # PD Code construction and extraction
    # -----------------------------------------------------------------------
    @testset "PD Code" begin
        pd = pdcode([(1, 2, 3, 4, 1)])
        k = Knot(:sample, pd, nothing)
        @test crossing_number(k) == 1
        @test writhe(k) == 1
        @test seifert_circles(pd) >= 0
        @test braid_index_estimate(pd) >= 1

        # Round-trip: build from pd, extract entries.
        entries = pdcode(k)
        @test length(entries) == 1
        @test entries[1] == (1, 2, 3, 4, 1)

        # Multi-crossing PD.
        pd2 = pdcode([(1, 2, 3, 4, 1), (5, 6, 7, 8, -1)])
        @test length(pd2.crossings) == 2
        @test pd2.crossings[1].sign == 1
        @test pd2.crossings[2].sign == -1
    end

    # -----------------------------------------------------------------------
    # DT Code
    # -----------------------------------------------------------------------
    @testset "DT Code" begin
        k = trefoil()
        dt = dtcode(k)
        @test dt.code == [4, 6, 2]
        @test crossing_number(k) == 3

        # DT-only knot still reports crossing number.
        k2 = Knot(:dt_only, nothing, DTCode([4, 6, 2]))
        @test crossing_number(k2) == 3
    end

    # -----------------------------------------------------------------------
    # Writhe
    # -----------------------------------------------------------------------
    @testset "Writhe" begin
        t = trefoil()
        @test writhe(t) == 3  # All positive crossings.

        fe = figure_eight()
        @test writhe(fe) == 0  # Alternating: 2 positive + 2 negative.

        c5 = cinquefoil()
        @test writhe(c5) == 5  # All positive crossings.
    end

    # -----------------------------------------------------------------------
    # Seifert circles
    # -----------------------------------------------------------------------
    @testset "Seifert circles" begin
        # Unknot: 0 crossings -> 0 circles from the diagram.
        uk = unknot()
        @test seifert_circles(uk.pd) == 0

        # Trefoil: 3 crossings, 2 Seifert circles.
        t = trefoil()
        n_sc, arc_map = seifert_circles_with_map(t.pd)
        @test n_sc == 2
        @test !isempty(arc_map)
        # All arc labels should be mapped.
        all_arcs = Set{Int}()
        for c in t.pd.crossings
            for a in c.arcs
                push!(all_arcs, a)
            end
        end
        for a in all_arcs
            @test haskey(arc_map, a)
            @test 1 <= arc_map[a] <= n_sc
        end

        # Figure-eight: 4 crossings, 3 Seifert circles.
        fe = figure_eight()
        n_fe = seifert_circles(fe.pd)
        @test n_fe == 3

        # Cinquefoil: 5 crossings, 2 Seifert circles.
        c5 = cinquefoil()
        n_c5 = seifert_circles(c5.pd)
        @test n_c5 == 2
    end

    # -----------------------------------------------------------------------
    # Seifert matrix
    # -----------------------------------------------------------------------
    @testset "Seifert matrix" begin
        # Unknot: empty matrix.
        uk = unknot()
        V_uk = seifert_matrix(uk.pd)
        @test size(V_uk) == (0, 0)

        # Trefoil: 2x2 matrix (2g = c - s + 1 = 3 - 2 + 1 = 2).
        t = trefoil()
        V_t = seifert_matrix(t.pd)
        @test size(V_t) == (2, 2)
        # The Seifert matrix of the trefoil should satisfy:
        # V + V' has determinant with |det| = 3 (the knot determinant).
        S_t = V_t + transpose(V_t)
        @test abs(round(Int, det(Float64.(S_t)))) == 3

        # Figure-eight: 2x2 matrix (2g = 4 - 3 + 1 = 2).
        fe = figure_eight()
        V_fe = seifert_matrix(fe.pd)
        @test size(V_fe) == (2, 2)
        # Determinant of V + V' should be 5.
        S_fe = V_fe + transpose(V_fe)
        @test abs(round(Int, det(Float64.(S_fe)))) == 5
    end

    # -----------------------------------------------------------------------
    # Signature
    # -----------------------------------------------------------------------
    @testset "Signature" begin
        # Unknot: signature 0.
        uk = unknot()
        @test signature(uk.pd) == 0

        # Trefoil: signature -2.
        t = trefoil()
        sig_t = signature(t.pd)
        @test sig_t == -2

        # Figure-eight: signature 0.
        fe = figure_eight()
        sig_fe = signature(fe.pd)
        @test sig_fe == 0
    end

    # -----------------------------------------------------------------------
    # Determinant
    # -----------------------------------------------------------------------
    @testset "Determinant" begin
        # Unknot: det = 1.
        uk = unknot()
        @test determinant(uk.pd) == 1

        # Trefoil: det = 3.
        t = trefoil()
        @test determinant(t.pd) == 3

        # Figure-eight: det = 5.
        fe = figure_eight()
        @test determinant(fe.pd) == 5
    end

    # -----------------------------------------------------------------------
    # Alexander polynomial
    # -----------------------------------------------------------------------
    @testset "Alexander polynomial" begin
        # Unknot: Delta(t) = 1.
        uk = unknot()
        alex_uk = alexander_polynomial(uk.pd)
        @test alex_uk == Dict(0 => 1)

        # Trefoil: Delta(t) = t^{-1} - 1 + t
        # => {-1 => 1, 0 => -1, 1 => 1}
        t = trefoil()
        alex_t = alexander_polynomial(t.pd)
        @test haskey(alex_t, 0)
        # Check symmetry: coefficient of k equals coefficient of -k.
        for (e, c) in alex_t
            @test get(alex_t, -e, 0) == c
        end
        # The Alexander polynomial of the trefoil evaluated at t=1 should be +-1.
        alex_at_1 = sum(values(alex_t))
        @test abs(alex_at_1) == 1

        # Figure-eight: Delta(t) = -t^{-1} + 3 - t
        fe = figure_eight()
        alex_fe = alexander_polynomial(fe.pd)
        @test haskey(alex_fe, 0)
        # Symmetry check.
        for (e, c) in alex_fe
            @test get(alex_fe, -e, 0) == c
        end
        alex_fe_at_1 = sum(values(alex_fe))
        @test abs(alex_fe_at_1) == 1
    end

    # -----------------------------------------------------------------------
    # Conway polynomial
    # -----------------------------------------------------------------------
    @testset "Conway polynomial" begin
        # Unknot: nabla(z) = 1.
        uk = unknot()
        conway_uk = conway_polynomial(uk.pd)
        @test get(conway_uk, 0, 0) == 1

        # Trefoil: nabla(z) = 1 + z^2 => {0 => 1, 2 => 1}
        t = trefoil()
        conway_t = conway_polynomial(t.pd)
        @test get(conway_t, 0, 0) != 0  # Constant term is nonzero.
        # Conway polynomial has only even powers for knots.
        for (e, _) in conway_t
            @test iseven(e)
        end
    end

    # -----------------------------------------------------------------------
    # Jones polynomial
    # -----------------------------------------------------------------------
    @testset "Jones polynomial" begin
        # Unknot.
        uk = unknot()
        jones_uk = jones_polynomial(uk.pd; wr=0)
        @test jones_uk == Dict(0 => 1)

        # Single crossing.
        pd = pdcode([(1, 2, 3, 4, 1)])
        jones = jones_polynomial(pd; wr=1)
        @test !isempty(jones)

        # Trefoil: should produce a nontrivial polynomial.
        t = trefoil()
        jones_t = jones_polynomial(t.pd; wr=writhe(t))
        @test !isempty(jones_t)
        @test length(jones_t) >= 3  # Jones poly of trefoil has multiple terms.

        # Polynomial conversion.
        poly, offset = to_polynomial(jones_t)
        @test poly isa Polynomials.Polynomial
        @test offset isa Int
    end

    # -----------------------------------------------------------------------
    # HOMFLY-PT polynomial
    # -----------------------------------------------------------------------
    @testset "HOMFLY-PT polynomial" begin
        # Unknot: P(a,z) = 1.
        uk = unknot()
        homfly_uk = homfly_polynomial(uk.pd)
        @test homfly_uk == Dict((0, 0) => 1)

        # Trefoil: should have multiple terms.
        t = trefoil()
        homfly_t = homfly_polynomial(t.pd)
        @test !isempty(homfly_t)

        # Figure-eight.
        fe = figure_eight()
        homfly_fe = homfly_polynomial(fe.pd)
        @test !isempty(homfly_fe)
    end

    # -----------------------------------------------------------------------
    # R1 Simplification
    # -----------------------------------------------------------------------
    @testset "R1 simplification" begin
        # Crossing with repeated arcs (kink) should be removed.
        pd = pdcode([(1, 1, 2, 2, 1)])
        reduced = r1_simplify(pd)
        @test length(reduced.crossings) == 0

        # Non-degenerate crossing should be kept.
        pd2 = pdcode([(1, 2, 3, 4, 1)])
        reduced2 = r1_simplify(pd2)
        @test length(reduced2.crossings) == 1
    end

    # -----------------------------------------------------------------------
    # R2 Simplification
    # -----------------------------------------------------------------------
    @testset "R2 simplification" begin
        # Construct two crossings sharing 2 arcs with opposite signs.
        # This is a bigon (R2 pair) that should be removable.
        pd = pdcode([
            (1, 3, 2, 4, 1),
            (2, 4, 5, 6, -1),
        ])
        reduced = r2_simplify(pd)
        @test length(reduced.crossings) < length(pd.crossings)

        # Two crossings with same sign should NOT be removed by R2.
        pd2 = pdcode([
            (1, 3, 2, 4, 1),
            (2, 4, 5, 6, 1),
        ])
        reduced2 = r2_simplify(pd2)
        @test length(reduced2.crossings) == 2
    end

    # -----------------------------------------------------------------------
    # R3 Simplification (topology-preserving, no crossing reduction)
    # -----------------------------------------------------------------------
    @testset "R3 simplification" begin
        pd = pdcode([(1, 2, 3, 4, 1), (3, 5, 6, 2, 1), (6, 4, 7, 5, 1)])
        reduced = r3_simplify(pd)
        # R3 does not change crossing count, just rearranges.
        @test length(reduced.crossings) == length(pd.crossings)
    end

    # -----------------------------------------------------------------------
    # Simplify PD (combined R1 + R2 + R3)
    # -----------------------------------------------------------------------
    @testset "simplify_pd" begin
        # A kink should be removed.
        pd = pdcode([(1, 1, 2, 2, 1)])
        reduced = simplify_pd(pd)
        @test length(reduced.crossings) == 0

        # Already simple diagram should be unchanged.
        t = trefoil()
        reduced_t = simplify_pd(t.pd)
        @test length(reduced_t.crossings) == 3
    end

    # -----------------------------------------------------------------------
    # Knot table
    # -----------------------------------------------------------------------
    @testset "Knot Table" begin
        table = knot_table()

        # Should have at least 15 entries.
        @test length(table) >= 15

        # Verify specific entries.
        @test haskey(table, :unknot)
        @test haskey(table, :trefoil)
        @test haskey(table, :figure_eight)
        @test haskey(table, Symbol("5_1"))
        @test haskey(table, Symbol("5_2"))
        @test haskey(table, Symbol("6_1"))
        @test haskey(table, Symbol("6_2"))
        @test haskey(table, Symbol("6_3"))
        @test haskey(table, Symbol("7_1"))
        @test haskey(table, Symbol("7_2"))
        @test haskey(table, Symbol("7_3"))
        @test haskey(table, Symbol("7_4"))
        @test haskey(table, Symbol("7_5"))
        @test haskey(table, Symbol("7_6"))
        @test haskey(table, Symbol("7_7"))

        # Verify crossing counts.
        @test table[:unknot].crossings == 0
        @test table[:trefoil].crossings == 3
        @test table[:figure_eight].crossings == 4
        @test table[Symbol("5_1")].crossings == 5
        @test table[Symbol("5_2")].crossings == 5
        @test table[Symbol("6_1")].crossings == 6
        @test table[Symbol("7_1")].crossings == 7
        @test table[Symbol("7_7")].crossings == 7

        # Verify DT codes have correct length (= crossing number).
        for (name, entry) in table
            @test length(entry.dt) == entry.crossings
        end

        # Verify all entries have descriptions.
        for (name, entry) in table
            @test haskey(entry, :description)
            @test !isempty(entry.description)
        end

        # Lookup function.
        @test lookup_knot(:trefoil) !== nothing
        @test lookup_knot(:trefoil).crossings == 3
        @test lookup_knot(:nonexistent) === nothing
    end

    # -----------------------------------------------------------------------
    # Graphs
    # -----------------------------------------------------------------------
    @testset "Graphs" begin
        pd = pdcode([(1, 2, 3, 4, 1)])
        g = to_graph(pd)
        @test nv(g) >= 4
        @test ne(g) >= 4

        # Trefoil graph.
        t = trefoil()
        g_t = to_graph(t.pd)
        @test nv(g_t) >= 6
    end

    # -----------------------------------------------------------------------
    # Polynomial conversion
    # -----------------------------------------------------------------------
    @testset "Polynomial conversion" begin
        alex = Dict(-1 => 1, 0 => -1, 1 => 1)
        poly, offset = to_polynomial(alex)
        @test poly isa Polynomials.Polynomial
        @test offset == -1

        # Empty polynomial.
        poly_empty, off_empty = to_polynomial(Dict{Int, Int}())
        @test off_empty == 0
    end

    # -----------------------------------------------------------------------
    # Braid word support
    # -----------------------------------------------------------------------
    @testset "Braid words" begin
        # Trefoil from braid word: sigma_1^3
        k = from_braid_word("s1.s1.s1")
        @test k isa Knot
        @test crossing_number(k) == 3
        @test k.pd !== nothing

        # Figure-eight from braid word: s1.S2.s1.S2
        k2 = from_braid_word("s1.S2.s1.S2")
        @test k2 isa Knot
        @test crossing_number(k2) == 4

        # Empty braid word -> unknot.
        k3 = from_braid_word("")
        @test crossing_number(k3) == 0

        # Round-trip: from_braid_word -> to_braid_word.
        word = "s1.s1.s1"
        k4 = from_braid_word(word)
        recovered = to_braid_word(k4)
        @test recovered == word

        # Braid word with inverse generators.
        k5 = from_braid_word("S1.S1.S1")
        @test crossing_number(k5) == 3
        # All crossings should be negative.
        for c in k5.pd.crossings
            @test c.sign == -1
        end

        # Higher generator index.
        k6 = from_braid_word("s2.s2.s1")
        @test crossing_number(k6) == 3
        @test k6.pd !== nothing
    end

    # -----------------------------------------------------------------------
    # JSON round-trip
    # -----------------------------------------------------------------------
    @testset "JSON" begin
        # Simple knot.
        k = Knot(:sample, pdcode([(1, 2, 3, 4, -1)]), DTCode([4, 6, 2]))
        path = joinpath(@__DIR__, "knot.json")
        write_knot_json(path, k)
        k2 = read_knot_json(path)
        @test k2.name == :sample
        @test crossing_number(k2) == 1
        @test k2.dt !== nothing
        @test k2.dt.code == [4, 6, 2]
        rm(path, force=true)

        # Trefoil with PD.
        t = trefoil()
        path_t = joinpath(@__DIR__, "trefoil.json")
        write_knot_json(path_t, t)
        t2 = read_knot_json(path_t)
        @test t2.name == :trefoil
        @test crossing_number(t2) == 3
        @test t2.pd !== nothing
        @test t2.dt !== nothing
        rm(path_t, force=true)

        # Figure-eight.
        fe = figure_eight()
        path_fe = joinpath(@__DIR__, "figure_eight.json")
        write_knot_json(path_fe, fe)
        fe2 = read_knot_json(path_fe)
        @test fe2.name == :figure_eight
        @test crossing_number(fe2) == 4
        rm(path_fe, force=true)

        # Cinquefoil.
        c5 = cinquefoil()
        path_c5 = joinpath(@__DIR__, "cinquefoil.json")
        write_knot_json(path_c5, c5)
        c52 = read_knot_json(path_c5)
        @test c52.name == Symbol("5_1")
        @test crossing_number(c52) == 5
        rm(path_c5, force=true)
    end

    # -----------------------------------------------------------------------
    # Cross-invariant consistency checks
    # -----------------------------------------------------------------------
    @testset "Invariant consistency" begin
        # For the trefoil, verify internal consistency of all invariants.
        t = trefoil()
        pd = t.pd

        # Seifert circles.
        n_sc = seifert_circles(pd)
        @test n_sc == 2

        # Seifert matrix dimensions: 2g = c - s + 1.
        V = seifert_matrix(pd)
        expected_dim = length(pd.crossings) - n_sc + 1
        @test size(V) == (expected_dim, expected_dim)

        # Signature is even (always true for knots).
        sig = signature(pd)
        @test iseven(sig)

        # Determinant equals |det(V + V')|.
        d = determinant(pd)
        @test d == abs(round(Int, det(Float64.(V + transpose(V)))))

        # Alexander polynomial at t=1 is +-1.
        alex = alexander_polynomial(pd)
        @test abs(sum(values(alex))) == 1

        # Conway polynomial constant term: evaluate at z=0.
        conway = conway_polynomial(pd)
        @test abs(get(conway, 0, 0)) == 1

        # For figure-eight, do the same.
        fe = figure_eight()
        pd_fe = fe.pd
        V_fe = seifert_matrix(pd_fe)
        @test determinant(pd_fe) == abs(round(Int, det(Float64.(V_fe + transpose(V_fe)))))
        alex_fe = alexander_polynomial(pd_fe)
        @test abs(sum(values(alex_fe))) == 1

        # Cinquefoil consistency.
        c5 = cinquefoil()
        pd_c5 = c5.pd
        V_c5 = seifert_matrix(pd_c5)
        n_sc_c5 = seifert_circles(pd_c5)
        @test n_sc_c5 == 2  # (2,5)-torus knot has 2 Seifert circles.
        @test size(V_c5) == (4, 4)  # g = 5 - 2 + 1 = 4
        @test determinant(pd_c5) == 5
        alex_c5 = alexander_polynomial(pd_c5)
        @test abs(sum(values(alex_c5))) == 1  # Delta(1) = +-1
        @test signature(pd_c5) == -4  # Signature of 5_1 is -4.
    end

    # -----------------------------------------------------------------------
    # Specific known values for standard knots
    # -----------------------------------------------------------------------
    @testset "Known invariant values" begin
        # Trefoil exact Alexander polynomial: t^{-1} - 1 + t.
        t = trefoil()
        alex_t = alexander_polynomial(t.pd)
        @test get(alex_t, -1, 0) == 1
        @test get(alex_t, 0, 0) == -1
        @test get(alex_t, 1, 0) == 1

        # Trefoil Conway polynomial: 1 + z^2.
        conway_t = conway_polynomial(t.pd)
        @test get(conway_t, 0, 0) == 1
        @test get(conway_t, 2, 0) == 1

        # Trefoil Seifert matrix: V = [[-1, 1], [0, -1]].
        V_t = seifert_matrix(t.pd)
        @test V_t[1, 1] == -1
        @test V_t[1, 2] == 1
        @test V_t[2, 1] == 0
        @test V_t[2, 2] == -1

        # Trefoil signature -2, determinant 3.
        @test signature(t.pd) == -2
        @test determinant(t.pd) == 3

        # Figure-eight exact Alexander polynomial: t^{-1} - 3 + t
        # (or equivalently -t^{-1} + 3 - t, depending on sign convention).
        fe = figure_eight()
        alex_fe = alexander_polynomial(fe.pd)
        @test abs(get(alex_fe, 0, 0)) == 3
        @test abs(get(alex_fe, 1, 0)) == 1
        @test abs(get(alex_fe, -1, 0)) == 1
        @test get(alex_fe, 1, 0) == get(alex_fe, -1, 0)  # symmetric

        # Figure-eight Conway polynomial: 1 - z^2.
        conway_fe = conway_polynomial(fe.pd)
        @test abs(get(conway_fe, 0, 0)) == 1
        @test abs(get(conway_fe, 2, 0)) == 1

        # Figure-eight signature 0, determinant 5.
        @test signature(fe.pd) == 0
        @test determinant(fe.pd) == 5

        # Figure-eight Seifert matrix: 2x2 with |det(V+V')| = 5.
        V_fe = seifert_matrix(fe.pd)
        @test size(V_fe) == (2, 2)
        S_fe = V_fe + transpose(V_fe)
        @test abs(round(Int, det(Float64.(S_fe)))) == 5

        # Figure-eight has 3 Seifert circles.
        @test seifert_circles(fe.pd) == 3

        # Unknot: all invariants trivial.
        uk = unknot()
        @test alexander_polynomial(uk.pd) == Dict(0 => 1)
        @test conway_polynomial(uk.pd) == Dict(0 => 1)
        @test signature(uk.pd) == 0
        @test determinant(uk.pd) == 1

        # Cinquefoil determinant is 5.
        c5 = cinquefoil()
        @test determinant(c5.pd) == 5
    end

    # -----------------------------------------------------------------------
    # Wirtinger generators
    # -----------------------------------------------------------------------
    @testset "Wirtinger generators" begin
        # Trefoil: 3 crossings -> 3 Wirtinger generators.
        t = trefoil()
        gen_map, n_gens = KnotTheory._wirtinger_generators(t.pd)
        @test n_gens == 3
        @test !isempty(gen_map)
        # All arc labels should be mapped.
        for c in t.pd.crossings
            for a in c.arcs
                @test haskey(gen_map, a)
                @test 1 <= gen_map[a] <= n_gens
            end
        end
        # Over-arcs at each crossing should map to the same generator.
        for c in t.pd.crossings
            @test gen_map[c.arcs[2]] == gen_map[c.arcs[4]]  # b and d
        end

        # Unknot: 0 crossings -> 0 generators.
        gen_map_uk, n_uk = KnotTheory._wirtinger_generators(unknot().pd)
        @test n_uk == 0

        # Figure-eight: 4 crossings -> 4 generators.
        fe = figure_eight()
        gen_map_fe, n_fe = KnotTheory._wirtinger_generators(fe.pd)
        @test n_fe == 4
    end

    # -----------------------------------------------------------------------
    # Alexander polynomial normalization properties
    # -----------------------------------------------------------------------
    @testset "Alexander normalization" begin
        # All Alexander polynomials should be symmetric: Delta(t) = Delta(1/t).
        for knot_fn in [trefoil, figure_eight, cinquefoil]
            k = knot_fn()
            alex = alexander_polynomial(k.pd)
            for (e, c) in alex
                @test get(alex, -e, 0) == c
            end
        end

        # Leading coefficient should be positive.
        for knot_fn in [trefoil, figure_eight, cinquefoil]
            k = knot_fn()
            alex = alexander_polynomial(k.pd)
            if !isempty(alex)
                max_e = maximum(keys(alex))
                @test alex[max_e] > 0
            end
        end
    end

end
