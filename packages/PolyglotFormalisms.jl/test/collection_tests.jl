# SPDX-License-Identifier: PMPL-1.0-or-later
"""
Collection conformance tests matching PolyglotFormalisms specification.

These tests correspond to the test cases defined in:
- aggregate-library/specs/collection/*.md

Each test case matches the PolyglotFormalisms format:
- input: function arguments
- output: expected_result
- description: "Test case description"

Tests verify both basic functionality and mathematical properties
documented in the Collection module.
"""

@testset "Collection Operations" begin
    @testset "map_items" begin
        # Basic functionality tests
        @test Collection.map_items(x -> x * 2, [1, 2, 3]) == [2, 4, 6]
        @test Collection.map_items(string, [1, 2, 3]) == ["1", "2", "3"]
        @test Collection.map_items(x -> x + 1, [10, 20, 30]) == [11, 21, 31]

        @testset "Length preservation" begin
            xs = [1, 2, 3, 4, 5]
            @test Base.length(Collection.map_items(x -> x * 2, xs)) == Base.length(xs)
            @test Base.length(Collection.map_items(string, xs)) == Base.length(xs)
        end

        @testset "Composition" begin
            f = x -> x * 2
            g = x -> x + 1
            xs = [1, 2, 3]
            @test Collection.map_items(f ∘ g, xs) == Collection.map_items(f, Collection.map_items(g, xs))
        end

        @testset "Identity" begin
            xs = [1, 2, 3, 4]
            @test Collection.map_items(identity, xs) == xs
        end

        @testset "Empty collection" begin
            @test Collection.map_items(x -> x * 2, Int[]) == Int[]
        end
    end

    @testset "filter_items" begin
        # Basic functionality tests
        @test Collection.filter_items(iseven, [1, 2, 3, 4, 5]) == [2, 4]
        @test Collection.filter_items(x -> x > 3, [1, 2, 3, 4, 5]) == [4, 5]

        @testset "Idempotence" begin
            xs = [1, 2, 3, 4, 5]
            @test Collection.filter_items(iseven, Collection.filter_items(iseven, xs)) ==
                  Collection.filter_items(iseven, xs)
        end

        @testset "True predicate" begin
            @test Collection.filter_items(_ -> true, [1, 2, 3]) == [1, 2, 3]
        end

        @testset "False predicate" begin
            @test Collection.filter_items(_ -> false, [1, 2, 3]) == []
        end

        @testset "Empty collection" begin
            @test Collection.filter_items(iseven, Int[]) == Int[]
        end

        @testset "Length bound" begin
            xs = [1, 2, 3, 4, 5]
            @test Base.length(Collection.filter_items(iseven, xs)) <= Base.length(xs)
        end
    end

    @testset "fold_items" begin
        # Basic functionality tests
        @test Collection.fold_items(+, 0, [1, 2, 3, 4]) == 10
        @test Collection.fold_items(*, 1, [1, 2, 3, 4]) == 24

        @testset "Identity (empty)" begin
            @test Collection.fold_items(+, 0, Int[]) == 0
            @test Collection.fold_items(*, 1, Int[]) == 1
            @test Collection.fold_items(+, 42, Int[]) == 42
        end

        @testset "Single element" begin
            @test Collection.fold_items(+, 0, [5]) == 5
            @test Collection.fold_items(*, 1, [7]) == 7
        end

        @testset "String concatenation" begin
            @test Collection.fold_items(*, "", ["a", "b", "c"]) == "abc"
        end
    end

    @testset "zip_items" begin
        # Basic functionality tests
        @test Collection.zip_items([1, 2, 3], ["a", "b", "c"]) == [(1, "a"), (2, "b"), (3, "c")]

        @testset "Length property" begin
            a = [1, 2, 3]
            b = ["a", "b", "c"]
            @test Base.length(Collection.zip_items(a, b)) == min(Base.length(a), Base.length(b))
        end

        @testset "Unequal lengths" begin
            @test Collection.zip_items([1, 2], ["a", "b", "c"]) == [(1, "a"), (2, "b")]
            @test Collection.zip_items([1, 2, 3], ["a", "b"]) == [(1, "a"), (2, "b")]
        end

        @testset "Empty collections" begin
            @test Collection.zip_items(Int[], String[]) == []
            @test Collection.zip_items([1, 2], String[]) == []
        end

        @testset "Unzip first element" begin
            a = [1, 2, 3]
            b = ["a", "b", "c"]
            zipped = Collection.zip_items(a, b)
            @test Collection.map_items(first, zipped) == a
        end
    end

    @testset "flat_map_items" begin
        # Basic functionality tests
        @test Collection.flat_map_items(x -> [x, x * 10], [1, 2, 3]) == [1, 10, 2, 20, 3, 30]

        @testset "Monad left identity" begin
            f = x -> [x, x * 2]
            @test Collection.flat_map_items(f, [5]) == f(5)
        end

        @testset "Monad right identity" begin
            xs = [1, 2, 3]
            @test Collection.flat_map_items(x -> [x], xs) == xs
        end

        @testset "Empty results" begin
            @test Collection.flat_map_items(x -> Int[], [1, 2, 3]) == []
        end

        @testset "Empty collection" begin
            @test Collection.flat_map_items(x -> [x, x], Int[]) == []
        end
    end

    @testset "group_by" begin
        # Basic functionality tests
        result = Collection.group_by(iseven, [1, 2, 3, 4, 5])
        @test result[true] == [2, 4]
        @test result[false] == [1, 3, 5]

        @testset "Partition property (size preservation)" begin
            xs = [1, 2, 3, 4, 5]
            groups = Collection.group_by(iseven, xs)
            total = sum(Base.length(v) for v in values(groups))
            @test total == Base.length(xs)
        end

        @testset "Modulo grouping" begin
            result = Collection.group_by(x -> x % 3, [1, 2, 3, 4, 5, 6])
            @test result[1] == [1, 4]
            @test result[2] == [2, 5]
            @test result[0] == [3, 6]
        end

        @testset "Empty collection" begin
            @test Collection.group_by(identity, Int[]) == Dict()
        end
    end

    @testset "sort_by" begin
        # Basic functionality tests
        @test Collection.sort_by(x -> -x, [3, 1, 4, 1, 5]) == [5, 4, 3, 1, 1]
        @test Collection.sort_by(abs, [-3, 1, -2]) == [1, -2, -3]

        @testset "Length preservation" begin
            xs = [3, 1, 4, 1, 5]
            @test Base.length(Collection.sort_by(identity, xs)) == Base.length(xs)
        end

        @testset "Idempotence" begin
            xs = [3, 1, 4, 1, 5]
            @test Collection.sort_by(identity, Collection.sort_by(identity, xs)) ==
                  Collection.sort_by(identity, xs)
        end

        @testset "Empty collection" begin
            @test Collection.sort_by(identity, Int[]) == Int[]
        end

        @testset "Already sorted" begin
            @test Collection.sort_by(identity, [1, 2, 3, 4]) == [1, 2, 3, 4]
        end
    end

    @testset "unique_items" begin
        # Basic functionality tests
        @test Collection.unique_items([1, 2, 3, 2, 1]) == [1, 2, 3]
        @test Collection.unique_items([1, 1, 1]) == [1]

        @testset "Idempotence" begin
            xs = [1, 2, 3, 2, 1]
            @test Collection.unique_items(Collection.unique_items(xs)) ==
                  Collection.unique_items(xs)
        end

        @testset "No duplicates in result" begin
            result = Collection.unique_items([1, 2, 3, 2, 1, 3])
            @test Base.length(result) == Base.length(unique(result))
        end

        @testset "Empty collection" begin
            @test Collection.unique_items(Int[]) == Int[]
        end

        @testset "Already unique" begin
            @test Collection.unique_items([1, 2, 3]) == [1, 2, 3]
        end

        @testset "Length bound" begin
            xs = [1, 2, 3, 2, 1]
            @test Base.length(Collection.unique_items(xs)) <= Base.length(xs)
        end
    end

    @testset "partition_items" begin
        # Basic functionality tests
        @test Collection.partition_items(iseven, [1, 2, 3, 4, 5]) == ([2, 4], [1, 3, 5])

        @testset "Completeness (size preservation)" begin
            xs = [1, 2, 3, 4, 5]
            yes, no = Collection.partition_items(iseven, xs)
            @test Base.length(yes) + Base.length(no) == Base.length(xs)
        end

        @testset "Relation to filter" begin
            xs = [1, 2, 3, 4, 5]
            yes, _ = Collection.partition_items(iseven, xs)
            @test yes == Collection.filter_items(iseven, xs)
        end

        @testset "True predicate" begin
            @test Collection.partition_items(_ -> true, [1, 2, 3]) == ([1, 2, 3], [])
        end

        @testset "False predicate" begin
            @test Collection.partition_items(_ -> false, [1, 2]) == ([], [1, 2])
        end

        @testset "Empty collection" begin
            @test Collection.partition_items(iseven, Int[]) == (Int[], Int[])
        end
    end

    @testset "take_items" begin
        # Basic functionality tests
        @test Collection.take_items(2, [1, 2, 3, 4, 5]) == [1, 2]
        @test Collection.take_items(10, [1, 2, 3]) == [1, 2, 3]

        @testset "Zero take" begin
            @test Collection.take_items(0, [1, 2, 3]) == []
        end

        @testset "Negative take" begin
            @test Collection.take_items(-1, [1, 2, 3]) == []
        end

        @testset "Identity" begin
            xs = [1, 2, 3]
            @test Collection.take_items(Base.length(xs), xs) == xs
        end

        @testset "Empty collection" begin
            @test Collection.take_items(5, Int[]) == Int[]
        end

        @testset "Complement with drop" begin
            xs = [1, 2, 3, 4, 5]
            n = 3
            @test vcat(Collection.take_items(n, xs), Collection.drop_items(n, xs)) == xs
        end
    end

    @testset "drop_items" begin
        # Basic functionality tests
        @test Collection.drop_items(2, [1, 2, 3, 4, 5]) == [3, 4, 5]
        @test Collection.drop_items(0, [1, 2, 3]) == [1, 2, 3]

        @testset "Drop all" begin
            @test Collection.drop_items(10, [1, 2]) == []
        end

        @testset "Negative drop" begin
            @test Collection.drop_items(-1, [1, 2, 3]) == [1, 2, 3]
        end

        @testset "Empty collection" begin
            @test Collection.drop_items(5, Int[]) == Int[]
        end

        @testset "Length property" begin
            xs = [1, 2, 3, 4, 5]
            @test Base.length(Collection.drop_items(2, xs)) == Base.length(xs) - 2
        end
    end

    @testset "any_item" begin
        # Basic functionality tests
        @test Collection.any_item(iseven, [1, 2, 3]) == true
        @test Collection.any_item(iseven, [1, 3, 5]) == false

        @testset "Empty collection" begin
            @test Collection.any_item(iseven, Int[]) == false
        end

        @testset "True predicate" begin
            @test Collection.any_item(_ -> true, [1, 2, 3]) == true
        end

        @testset "False predicate" begin
            @test Collection.any_item(_ -> false, [1, 2, 3]) == false
        end

        @testset "De Morgan relation to all_items" begin
            xs = [1, 2, 3, 4, 5]
            @test Collection.any_item(iseven, xs) == !Collection.all_items(x -> !iseven(x), xs)
        end

        @testset "True predicate on empty" begin
            @test Collection.any_item(_ -> true, Int[]) == false
        end
    end

    @testset "all_items" begin
        # Basic functionality tests
        @test Collection.all_items(iseven, [2, 4, 6]) == true
        @test Collection.all_items(iseven, [1, 2, 3]) == false

        @testset "Empty collection (vacuous truth)" begin
            @test Collection.all_items(iseven, Int[]) == true
        end

        @testset "True predicate" begin
            @test Collection.all_items(_ -> true, [1, 2, 3]) == true
        end

        @testset "False predicate" begin
            @test Collection.all_items(_ -> false, [1, 2, 3]) == false
        end

        @testset "False predicate on empty" begin
            @test Collection.all_items(_ -> false, Int[]) == true
        end

        @testset "De Morgan relation to any_item" begin
            xs = [1, 2, 3, 4, 5]
            @test Collection.all_items(iseven, xs) == !Collection.any_item(x -> !iseven(x), xs)
        end
    end
end
