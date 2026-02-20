# SPDX-License-Identifier: PMPL-1.0-or-later
using Test
using PolyglotFormalisms.StringOps

@testset "String Operations" begin
    @testset "concat" begin
        @test StringOps.concat("Hello", " World") == "Hello World"
        @test StringOps.concat("", "test") == "test"
        @test StringOps.concat("test", "") == "test"
        @test StringOps.concat("a", "b") == "ab"

        @testset "Associativity" begin
            @test StringOps.concat(StringOps.concat("a", "b"), "c") ==
                  StringOps.concat("a", StringOps.concat("b", "c"))
        end

        @testset "Identity element" begin
            @test StringOps.concat("test", "") == "test"
            @test StringOps.concat("", "test") == "test"
        end
    end

    @testset "length" begin
        @test StringOps.length("Hello") == 5
        @test StringOps.length("") == 0
        @test StringOps.length("🎉") == 1
        @test StringOps.length("Test 123") == 8

        @testset "Non-negativity" begin
            @test StringOps.length("") >= 0
            @test StringOps.length("test") >= 0
        end

        @testset "Concatenation property" begin
            a = "Hello"
            b = " World"
            @test StringOps.length(StringOps.concat(a, b)) == StringOps.length(a) + StringOps.length(b)
        end
    end

    @testset "substring" begin
        @test StringOps.substring("Hello World", 1, 5) == "Hello"
        @test StringOps.substring("Hello World", 7, 11) == "World"
        @test StringOps.substring("Test", 1, 1) == "T"
        @test StringOps.substring("Test", 3, 2) == ""

        @testset "Edge cases" begin
            @test StringOps.substring("Test", 2, 3) == "es"
            @test StringOps.substring("Hello", 1, 5) == "Hello"
        end
    end

    @testset "index_of" begin
        @test StringOps.index_of("Hello World", "World") == 7
        @test StringOps.index_of("Hello World", "o") == 5
        @test StringOps.index_of("Test", "xyz") == 0
        @test StringOps.index_of("Test", "") == 1

        @testset "Not found" begin
            @test StringOps.index_of("Hello", "xyz") == 0
        end

        @testset "Empty substring" begin
            @test StringOps.index_of("Test", "") == 1
            @test StringOps.index_of("", "") == 1
        end
    end

    @testset "contains" begin
        @test StringOps.contains("Hello World", "World") == true
        @test StringOps.contains("Hello World", "xyz") == false
        @test StringOps.contains("Test", "") == true
        @test StringOps.contains("", "Test") == false

        @testset "Empty substring" begin
            @test StringOps.contains("Test", "") == true
            @test StringOps.contains("", "") == true
        end

        @testset "Reflexivity" begin
            @test StringOps.contains("Hello", "Hello") == true
        end
    end

    @testset "starts_with" begin
        @test StringOps.starts_with("Hello World", "Hello") == true
        @test StringOps.starts_with("Hello World", "World") == false
        @test StringOps.starts_with("Test", "") == true
        @test StringOps.starts_with("", "Test") == false

        @testset "Empty prefix" begin
            @test StringOps.starts_with("Test", "") == true
            @test StringOps.starts_with("", "") == true
        end

        @testset "Reflexivity" begin
            @test StringOps.starts_with("Hello", "Hello") == true
        end
    end

    @testset "ends_with" begin
        @test StringOps.ends_with("Hello World", "World") == true
        @test StringOps.ends_with("Hello World", "Hello") == false
        @test StringOps.ends_with("Test", "") == true
        @test StringOps.ends_with("", "Test") == false

        @testset "Empty suffix" begin
            @test StringOps.ends_with("Test", "") == true
            @test StringOps.ends_with("", "") == true
        end

        @testset "Reflexivity" begin
            @test StringOps.ends_with("Hello", "Hello") == true
        end
    end

    @testset "to_uppercase" begin
        @test StringOps.to_uppercase("Hello World") == "HELLO WORLD"
        @test StringOps.to_uppercase("test") == "TEST"
        @test StringOps.to_uppercase("TEST") == "TEST"
        @test StringOps.to_uppercase("café") == "CAFÉ"

        @testset "Idempotence" begin
            @test StringOps.to_uppercase(StringOps.to_uppercase("test")) == StringOps.to_uppercase("test")
        end
    end

    @testset "to_lowercase" begin
        @test StringOps.to_lowercase("Hello World") == "hello world"
        @test StringOps.to_lowercase("TEST") == "test"
        @test StringOps.to_lowercase("test") == "test"
        @test StringOps.to_lowercase("CAFÉ") == "café"

        @testset "Idempotence" begin
            @test StringOps.to_lowercase(StringOps.to_lowercase("TEST")) == StringOps.to_lowercase("TEST")
        end
    end

    @testset "trim" begin
        @test StringOps.trim("  Hello World  ") == "Hello World"
        @test StringOps.trim("\n\tTest\n") == "Test"
        @test StringOps.trim("NoSpaces") == "NoSpaces"
        @test StringOps.trim("   ") == ""

        @testset "Idempotence" begin
            @test StringOps.trim(StringOps.trim("  test  ")) == StringOps.trim("  test  ")
        end

        @testset "Internal whitespace preserved" begin
            @test StringOps.trim("  Hello World  ") == "Hello World"
        end
    end

    @testset "split" begin
        @test StringOps.split("a,b,c", ",") == ["a", "b", "c"]
        @test StringOps.split("Hello World", " ") == ["Hello", "World"]
        @test StringOps.split("test", ",") == ["test"]
        @test StringOps.split("a,,b", ",") == ["a", "", "b"]

        @testset "Empty delimiter" begin
            @test StringOps.split("abc", "") == ["a", "b", "c"]
        end

        @testset "Not found" begin
            @test StringOps.split("test", ",") == ["test"]
        end
    end

    @testset "join" begin
        @test StringOps.join(["a", "b", "c"], ",") == "a,b,c"
        @test StringOps.join(["Hello", "World"], " ") == "Hello World"
        @test StringOps.join(["test"], ",") == "test"
        @test StringOps.join(String[], ",") == ""

        @testset "Empty array" begin
            @test StringOps.join(String[], ",") == ""
        end

        @testset "Single element" begin
            @test StringOps.join(["test"], ",") == "test"
        end

        @testset "Inverse of split" begin
            original = "a,b,c"
            @test StringOps.join(StringOps.split(original, ","), ",") == original
        end
    end

    @testset "replace" begin
        @test StringOps.replace("Hello World", "World", "Universe") == "Hello Universe"
        @test StringOps.replace("test test", "test", "demo") == "demo demo"
        @test StringOps.replace("Hello", "xyz", "abc") == "Hello"
        @test StringOps.replace("Hello", "l", "") == "Heo"

        @testset "Not found" begin
            @test StringOps.replace("Hello", "xyz", "abc") == "Hello"
        end

        @testset "Empty old" begin
            @test StringOps.replace("Hello", "", "x") == "Hello"
        end

        @testset "Multiple occurrences" begin
            @test StringOps.replace("test test test", "test", "demo") == "demo demo demo"
        end
    end

    @testset "is_empty" begin
        @test StringOps.is_empty("") == true
        @test StringOps.is_empty("test") == false
        @test StringOps.is_empty(" ") == false

        @testset "Equivalence to length" begin
            @test StringOps.is_empty("") == (StringOps.length("") == 0)
            @test StringOps.is_empty("test") == (StringOps.length("test") == 0)
        end
    end
end
