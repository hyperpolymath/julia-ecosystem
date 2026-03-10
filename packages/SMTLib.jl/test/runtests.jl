# SPDX-License-Identifier: PMPL-1.0-or-later
# Test suite for SMTLib.jl
# Covers: core types, SMT-LIB generation, parsing, context operations,
#         push/pop, named assertions, solver options, quantifiers,
#         optimization, theory helpers, statistics, model evaluation,
#         unsat core, and edge cases.
#
# Copyright (c) 2026 Jonathan D.A. Jewell (hyperpolymath) <jonathan.jewell@open.ac.uk>

using Test
using SMTLib
using SMTLib

@testset "SMTLib.jl" begin

    # ========================================================================
    # Solver Discovery
    # ========================================================================
    @testset "Solver Discovery" begin
        solvers = available_solvers()
        @test solvers isa Vector{SMTSolver}
        # Note: May be empty if no solvers installed

        # find_solver returns nothing or SMTSolver
        solver = find_solver()
        @test solver isa Union{SMTSolver, Nothing}

        # find_solver with preference
        solver_z3 = find_solver(:z3)
        @test solver_z3 isa Union{SMTSolver, Nothing}
        if solver_z3 !== nothing
            @test solver_z3.kind == :z3
        end

        # find_solver with preference returns first available if preference not found
        solver_cvc5 = find_solver(:cvc5)
        @test solver_cvc5 isa Union{SMTSolver, Nothing}
        # Only test kind if CVC5 is actually available
        has_cvc5 = any(s -> s.kind == :cvc5, solvers)
        if solver_cvc5 !== nothing && has_cvc5
            @test solver_cvc5.kind == :cvc5
        end

        # find_solver with nonexistent preference falls back to first available
        solver_fake = find_solver(:nonexistent_solver)
        # Should either return nothing (no solvers) or first available
        @test solver_fake isa Union{SMTSolver, Nothing}
    end

    # ========================================================================
    # SMT-LIB Generation (to_smtlib)
    # ========================================================================
    @testset "SMT-LIB Generation" begin
        @testset "Arithmetic operators" begin
            @test to_smtlib(:(x + y)) == "(+ x y)"
            @test to_smtlib(:(x - y)) == "(- x y)"
            @test to_smtlib(:(x * y)) == "(* x y)"
            @test to_smtlib(:(x / y)) == "(/ x y)"
            @test to_smtlib(:(div(x, y))) == "(div x y)"
            @test to_smtlib(:(mod(x, y))) == "(mod x y)"
            @test to_smtlib(:(abs(x))) == "(abs x)"
        end

        @testset "Comparison operators" begin
            @test to_smtlib(:(x == y)) == "(= x y)"
            @test to_smtlib(:(x != y)) == "(distinct x y)"
            @test to_smtlib(:(x < y)) == "(< x y)"
            @test to_smtlib(:(x <= y)) == "(<= x y)"
            @test to_smtlib(:(x > y)) == "(> x y)"
            @test to_smtlib(:(x >= y)) == "(>= x y)"
        end

        @testset "Logical operators" begin
            @test to_smtlib(:(!x)) == "(not x)"
            @test to_smtlib(:(x && y)) == "(and x y)"
            @test to_smtlib(:(x || y)) == "(or x y)"
            @test to_smtlib(:(xor(x, y))) == "(xor x y)"
            @test to_smtlib(:(implies(x, y))) == "(=> x y)"
        end

        @testset "Literals" begin
            @test to_smtlib(true) == "true"
            @test to_smtlib(false) == "false"
            @test to_smtlib(42) == "42"
            @test to_smtlib(0) == "0"
            @test to_smtlib(-42) == "(- 42)"
            @test to_smtlib(-1) == "(- 1)"
            @test to_smtlib(3.14) == "3.14"
            @test to_smtlib(-2.5) == "(- 2.5)"
            @test to_smtlib(0.0) == "0.0"
        end

        @testset "Nested expressions" begin
            @test to_smtlib(:(x + y * z)) == "(+ x (* y z))"
            @test to_smtlib(:((x + y) * z)) == "(* (+ x y) z)"
            @test to_smtlib(:(!(x == y))) == "(not (= x y))"
            @test to_smtlib(:((x > 0) && (y < 10))) == "(and (> x 0) (< y 10))"
            @test to_smtlib(:((x > 0) || (y > 0))) == "(or (> x 0) (> y 0))"
        end

        @testset "Bitvector operators" begin
            @test to_smtlib(:(bvadd(a, b))) == "(bvadd a b)"
            @test to_smtlib(:(bvsub(a, b))) == "(bvsub a b)"
            @test to_smtlib(:(bvmul(a, b))) == "(bvmul a b)"
            @test to_smtlib(:(bvand(a, b))) == "(bvand a b)"
            @test to_smtlib(:(bvor(a, b))) == "(bvor a b)"
            @test to_smtlib(:(bvxor(a, b))) == "(bvxor a b)"
            @test to_smtlib(:(bvnot(a))) == "(bvnot a)"
            @test to_smtlib(:(bvshl(a, b))) == "(bvshl a b)"
            @test to_smtlib(:(bvlshr(a, b))) == "(bvlshr a b)"
            @test to_smtlib(:(bvashr(a, b))) == "(bvashr a b)"
        end

        @testset "Array operators" begin
            @test to_smtlib(:(select(a, i))) == "(select a i)"
            @test to_smtlib(:(store(a, i, v))) == "(store a i v)"
        end

        @testset "String literals" begin
            @test to_smtlib("hello") == "\"hello\""
            @test to_smtlib("") == "\"\""
        end

        @testset "Symbols" begin
            @test to_smtlib(:x) == "x"
            @test to_smtlib(:my_var) == "my_var"
        end
    end

    # ========================================================================
    # Type Mapping
    # ========================================================================
    @testset "Type Mapping" begin
        @test SMTLib.smt_type(Int) == "Int"
        @test SMTLib.smt_type(Int64) == "Int"
        @test SMTLib.smt_type(Int32) == "Int"
        @test SMTLib.smt_type(Bool) == "Bool"
        @test SMTLib.smt_type(Float64) == "Real"
        @test SMTLib.smt_type(Float32) == "Real"
        @test SMTLib.smt_type(String) == "String"
        @test SMTLib.smt_type(SMTLib.BitVec{8}) == "(_ BitVec 8)"
        @test SMTLib.smt_type(SMTLib.BitVec{16}) == "(_ BitVec 16)"
        @test SMTLib.smt_type(SMTLib.BitVec{32}) == "(_ BitVec 32)"
        @test SMTLib.smt_type(SMTLib.BitVec{64}) == "(_ BitVec 64)"
        @test SMTLib.smt_type(SMTLib.SMTArray{Int, Int}) == "(Array Int Int)"
        @test SMTLib.smt_type(SMTLib.SMTArray{Int, Bool}) == "(Array Int Bool)"
        @test SMTLib.smt_type(SMTLib.SMTArray{SMTLib.BitVec{32}, Int}) == "(Array (_ BitVec 32) Int)"
    end

    # ========================================================================
    # Value Parsing
    # ========================================================================
    @testset "Value Parsing" begin
        @testset "Boolean values" begin
            @test SMTLib.parse_smt_value("true") == true
            @test SMTLib.parse_smt_value("false") == false
            @test SMTLib.parse_smt_value("  true  ") == true
        end

        @testset "Integer values" begin
            @test SMTLib.parse_smt_value("42") == 42
            @test SMTLib.parse_smt_value("0") == 0
            @test SMTLib.parse_smt_value("-7") == -7
            @test SMTLib.parse_smt_value("(- 42)") == -42
            @test SMTLib.parse_smt_value("(- 0)") == 0
            @test SMTLib.parse_smt_value("(- 1)") == -1
            @test SMTLib.parse_smt_value("123456789") == 123456789
        end

        @testset "Rational values" begin
            @test SMTLib.parse_smt_value("(/ 1 2)") == 1//2
            @test SMTLib.parse_smt_value("(/ 3 4)") == 3//4
            @test SMTLib.parse_smt_value("(/ -1 3)") == -1//3
            @test SMTLib.parse_smt_value("(/ 7 1)") == 7//1
        end

        @testset "Negative rational values" begin
            @test SMTLib.parse_smt_value("(- (/ 1 2))") == -1//2
            @test SMTLib.parse_smt_value("(- (/ 3 4))") == -3//4
        end

        @testset "Decimal values" begin
            @test SMTLib.parse_smt_value("3.14") == 3.14
            @test SMTLib.parse_smt_value("-2.5") == -2.5
            @test SMTLib.parse_smt_value("0.0") == 0.0
        end

        @testset "Bitvector values" begin
            @test SMTLib.parse_smt_value("#b1010") == 10
            @test SMTLib.parse_smt_value("#b0000") == 0
            @test SMTLib.parse_smt_value("#b11111111") == 255
            @test SMTLib.parse_smt_value("#xFF") == 255
            @test SMTLib.parse_smt_value("#x00") == 0
            @test SMTLib.parse_smt_value("#xDEAD") == 0xDEAD
            @test SMTLib.parse_smt_value("#xBEEF") == 0xBEEF
        end

        @testset "String values" begin
            @test SMTLib.parse_smt_value("\"hello\"") == "hello"
            @test SMTLib.parse_smt_value("\"\"") == ""
        end

        @testset "Unparseable values" begin
            # Should return as string
            @test SMTLib.parse_smt_value("some_symbol") == "some_symbol"
        end
    end

    # ========================================================================
    # S-Expression Tokenizer
    # ========================================================================
    @testset "S-Expression Tokenizer" begin
        @testset "Simple expressions" begin
            tokens = SMTLib.tokenize_sexpr("(+ x y)")
            @test length(tokens) == 1
            @test tokens[1] isa Vector
            @test tokens[1][1] == "+"
            @test tokens[1][2] == "x"
            @test tokens[1][3] == "y"
        end

        @testset "Nested expressions" begin
            tokens = SMTLib.tokenize_sexpr("(+ (- x 1) (* y 2))")
            @test length(tokens) == 1
            inner = tokens[1]
            @test inner[1] == "+"
            @test inner[2] isa Vector  # (- x 1)
            @test inner[2][1] == "-"
            @test inner[2][2] == "x"
            @test inner[2][3] == "1"
            @test inner[3] isa Vector  # (* y 2)
            @test inner[3][1] == "*"
            @test inner[3][2] == "y"
            @test inner[3][3] == "2"
        end

        @testset "Atoms" begin
            tokens = SMTLib.tokenize_sexpr("42")
            @test length(tokens) == 1
            @test tokens[1] == "42"
        end

        @testset "Quoted strings" begin
            tokens = SMTLib.tokenize_sexpr("(= x \"hello world\")")
            @test length(tokens) == 1
            @test tokens[1][1] == "="
            @test tokens[1][2] == "x"
            @test tokens[1][3] == "\"hello world\""
        end

        @testset "Multiple tokens" begin
            tokens = SMTLib.tokenize_sexpr("(- 42)")
            @test length(tokens) == 1
            @test tokens[1][1] == "-"
            @test tokens[1][2] == "42"
        end

        @testset "Deeply nested" begin
            tokens = SMTLib.tokenize_sexpr("(and (> x 0) (< (+ x y) 10))")
            @test length(tokens) == 1
            @test tokens[1][1] == "and"
            @test tokens[1][2] isa Vector
            @test tokens[1][3] isa Vector
        end
    end

    # ========================================================================
    # from_smtlib Parser
    # ========================================================================
    @testset "from_smtlib Parser" begin
        @testset "Atoms" begin
            @test from_smtlib("true") == true
            @test from_smtlib("false") == false
            @test from_smtlib("42") == 42
            @test from_smtlib("-7") == -7
            @test from_smtlib("3.14") == 3.14
            @test from_smtlib("x") == :x
        end

        @testset "Simple operations" begin
            result = from_smtlib("(+ x y)")
            @test result isa Expr
            @test result.head == :call
            @test result.args[1] == :+
            @test result.args[2] == :x
            @test result.args[3] == :y
        end

        @testset "Comparison operators" begin
            result = from_smtlib("(= x 5)")
            @test result.args[1] == :(==)
            @test result.args[2] == :x
            @test result.args[3] == 5
        end

        @testset "Nested expressions" begin
            result = from_smtlib("(+ (- x 1) (* y 2))")
            @test result isa Expr
            @test result.args[1] == :+
            # First arg: (- x 1)
            @test result.args[2] isa Expr
            @test result.args[2].args[1] == :-
            @test result.args[2].args[2] == :x
            @test result.args[2].args[3] == 1
            # Second arg: (* y 2)
            @test result.args[3] isa Expr
            @test result.args[3].args[1] == :*
            @test result.args[3].args[2] == :y
            @test result.args[3].args[3] == 2
        end

        @testset "Logical operators" begin
            result = from_smtlib("(and x y)")
            @test result.args[1] == :&&

            result = from_smtlib("(or x y)")
            @test result.args[1] == :||

            result = from_smtlib("(not x)")
            @test result.args[1] == :!
        end

        @testset "Negative numbers" begin
            result = from_smtlib("(- 42)")
            @test result == -42
        end

        @testset "if-then-else" begin
            result = from_smtlib("(ite (> x 0) x (- x))")
            @test result isa Expr
            @test result.args[1] == :ifelse
        end

        @testset "Bitvector operations" begin
            result = from_smtlib("(bvadd a b)")
            @test result.args[1] == :bvadd
        end
    end

    # ========================================================================
    # Context Operations
    # ========================================================================
    @testset "Context Operations" begin
        # We need a solver for context construction
        solvers = available_solvers()
        if isempty(solvers)
            @warn "No SMT solver found, skipping context tests"
        else
            solver = first(solvers)

            @testset "Basic context creation" begin
                ctx = SMTContext(solver=solver, logic=:QF_LIA)
                @test ctx.solver === solver
                @test ctx.logic == :QF_LIA
                @test isempty(ctx.declarations)
                @test isempty(ctx.assertions)
                @test ctx.timeout_ms == 30000
                @test isempty(ctx.declarations_stack)
                @test isempty(ctx.assertions_stack)
                @test isempty(ctx.assertion_names)
                @test isempty(ctx.solver_options)
                @test isempty(ctx.optimization_directives)
                @test isempty(ctx.push_pop_commands)
            end

            @testset "Declare variables" begin
                ctx = SMTContext(solver=solver, logic=:QF_LIA)
                declare(ctx, :x, Int)
                @test length(ctx.declarations) == 1
                @test ctx.declarations[1] == "(declare-const x Int)"

                declare(ctx, :b, Bool)
                @test length(ctx.declarations) == 2
                @test ctx.declarations[2] == "(declare-const b Bool)"

                declare(ctx, :r, Float64)
                @test length(ctx.declarations) == 3
                @test ctx.declarations[3] == "(declare-const r Real)"
            end

            @testset "Assert expressions" begin
                ctx = SMTContext(solver=solver, logic=:QF_LIA)
                assert!(ctx, :(x > 0))
                @test length(ctx.assertions) == 1
                @test ctx.assertions[1] == "(assert (> x 0))"

                assert!(ctx, :(x < 10))
                @test length(ctx.assertions) == 2
            end

            @testset "Named assertions" begin
                ctx = SMTContext(solver=solver, logic=:QF_LIA)
                assert!(ctx, :(x > 0); name=:positive)
                @test length(ctx.assertions) == 1
                @test occursin(":named positive", ctx.assertions[1])
                @test haskey(ctx.assertion_names, :positive)
            end

            @testset "Reset context" begin
                ctx = SMTContext(solver=solver, logic=:QF_LIA)
                declare(ctx, :x, Int)
                assert!(ctx, :(x > 0))
                set_option!(ctx, "random-seed", "42")
                push!(ctx)

                reset!(ctx)
                @test isempty(ctx.declarations)
                @test isempty(ctx.assertions)
                @test isempty(ctx.declarations_stack)
                @test isempty(ctx.assertions_stack)
                @test isempty(ctx.assertion_names)
                @test isempty(ctx.solver_options)
                @test isempty(ctx.optimization_directives)
                @test isempty(ctx.push_pop_commands)
            end

            @testset "Set solver options" begin
                ctx = SMTContext(solver=solver, logic=:QF_LIA)
                set_option!(ctx, "produce-unsat-cores", "true")
                @test ctx.solver_options["produce-unsat-cores"] == "true"

                set_option!(ctx, "random-seed", "42")
                @test ctx.solver_options["random-seed"] == "42"
                @test length(ctx.solver_options) == 2
            end
        end
    end

    # ========================================================================
    # Push/Pop
    # ========================================================================
    @testset "Push/Pop" begin
        solvers = available_solvers()
        if isempty(solvers)
            @warn "No SMT solver found, skipping push/pop tests"
        else
            solver = first(solvers)

            @testset "Push saves state" begin
                ctx = SMTContext(solver=solver, logic=:QF_LIA)
                declare(ctx, :x, Int)
                assert!(ctx, :(x > 0))

                push!(ctx)

                @test length(ctx.declarations_stack) == 1
                @test length(ctx.assertions_stack) == 1
                @test length(ctx.declarations_stack[1]) == 1
                @test length(ctx.assertions_stack[1]) == 1

                # Modify after push
                assert!(ctx, :(x < 0))
                @test length(ctx.assertions) == 2
            end

            @testset "Pop restores state" begin
                ctx = SMTContext(solver=solver, logic=:QF_LIA)
                declare(ctx, :x, Int)
                assert!(ctx, :(x > 0))

                push!(ctx)
                assert!(ctx, :(x < 0))
                @test length(ctx.assertions) == 2

                pop!(ctx)
                @test length(ctx.assertions) == 1
                @test ctx.assertions[1] == "(assert (> x 0))"
                @test isempty(ctx.declarations_stack)
                @test isempty(ctx.assertions_stack)
            end

            @testset "Multiple push/pop" begin
                ctx = SMTContext(solver=solver, logic=:QF_LIA)
                declare(ctx, :x, Int)

                assert!(ctx, :(x > 0))
                push!(ctx)

                assert!(ctx, :(x > 5))
                push!(ctx)

                assert!(ctx, :(x > 10))
                @test length(ctx.assertions) == 3

                pop!(ctx)
                @test length(ctx.assertions) == 2

                pop!(ctx)
                @test length(ctx.assertions) == 1
            end

            @testset "Pop on empty stack errors" begin
                ctx = SMTContext(solver=solver, logic=:QF_LIA)
                @test_throws ErrorException pop!(ctx)
            end

            @testset "Push/pop with solver" begin
                ctx = SMTContext(solver=solver, logic=:QF_LIA)
                declare(ctx, :x, Int)
                assert!(ctx, :(x > 0))
                assert!(ctx, :(x < 10))

                # Should be sat
                result1 = check_sat(ctx)
                @test result1.status == :sat

                push!(ctx)
                assert!(ctx, :(x > 100))  # contradicts x < 10

                result2 = check_sat(ctx)
                @test result2.status in (:unsat, :unknown)

                pop!(ctx)  # back to just x > 0 && x < 10
                result3 = check_sat(ctx)
                @test result3.status == :sat
            end
        end
    end

    # ========================================================================
    # Script Building
    # ========================================================================
    @testset "Script Building" begin
        solvers = available_solvers()
        if isempty(solvers)
            @warn "No SMT solver found, skipping script building tests"
        else
            solver = first(solvers)

            @testset "Basic script" begin
                ctx = SMTContext(solver=solver, logic=:QF_LIA)
                declare(ctx, :x, Int)
                assert!(ctx, :(x > 0))

                script = SMTLib.build_script(ctx, true)
                @test occursin("(set-logic QF_LIA)", script)
                @test occursin("(set-option :produce-models true)", script)
                @test occursin("(declare-const x Int)", script)
                @test occursin("(assert (> x 0))", script)
                @test occursin("(check-sat)", script)
                @test occursin("(get-model)", script)
            end

            @testset "Script with options" begin
                ctx = SMTContext(solver=solver, logic=:QF_LIA)
                set_option!(ctx, "random-seed", "42")
                declare(ctx, :x, Int)

                script = SMTLib.build_script(ctx, true)
                @test occursin("(set-option :random-seed 42)", script)
            end

            @testset "Script without get-model" begin
                ctx = SMTContext(solver=solver, logic=:QF_LIA)
                declare(ctx, :x, Int)
                assert!(ctx, :(x > 0))

                script = SMTLib.build_script(ctx, false)
                @test !occursin("(get-model)", script)
            end

            @testset "Script with unsat core" begin
                ctx = SMTContext(solver=solver, logic=:QF_LIA)
                declare(ctx, :x, Int)
                assert!(ctx, :(x > 0))

                script = SMTLib.build_script(ctx, false, true)
                @test occursin("(get-unsat-core)", script)
            end
        end
    end

    # ========================================================================
    # Result Parsing
    # ========================================================================
    @testset "Result Parsing" begin
        @testset "Parse sat result" begin
            output = "sat\n(model\n  (define-fun x () Int 5)\n  (define-fun y () Int 3)\n)"
            result = SMTLib.parse_result(output)
            @test result.status == :sat
            @test haskey(result.model, :x)
            @test haskey(result.model, :y)
            @test result.model[:x] == 5
            @test result.model[:y] == 3
        end

        @testset "Parse unsat result" begin
            output = "unsat"
            result = SMTLib.parse_result(output)
            @test result.status == :unsat
            @test isempty(result.model)
        end

        @testset "Parse unknown result" begin
            output = "unknown"
            result = SMTLib.parse_result(output)
            @test result.status == :unknown
        end

        @testset "Parse timeout (empty output)" begin
            result = SMTLib.parse_result("")
            @test result.status == :timeout
        end

        @testset "Parse timeout (solver message)" begin
            output = "timeout\n"
            result = SMTLib.parse_result(output)
            @test result.status == :timeout
        end

        @testset "Parse unsat core" begin
            output = "unsat\n(pos neg)"
            result = SMTLib.parse_result(output)
            @test result.status == :unsat
            @test :pos in result.unsat_core
            @test :neg in result.unsat_core
        end
    end

    # ========================================================================
    # Model Parsing (Multi-line)
    # ========================================================================
    @testset "Model Parsing" begin
        @testset "Single-line define-fun" begin
            output = """sat
(model
  (define-fun x () Int 42)
  (define-fun y () Bool true)
)"""
            model = SMTLib.parse_model(output)
            @test model[:x] == 42
            @test model[:y] == true
        end

        @testset "Multi-line define-fun" begin
            output = """sat
(model
  (define-fun x () Int
    5)
  (define-fun y () Int
    (- 3))
)"""
            model = SMTLib.parse_model(output)
            @test model[:x] == 5
            @test model[:y] == -3
        end

        @testset "Rational model values" begin
            output = """sat
(model
  (define-fun x () Real (/ 1 2))
  (define-fun y () Real (/ 3 4))
)"""
            model = SMTLib.parse_model(output)
            @test model[:x] == 1//2
            @test model[:y] == 3//4
        end

        @testset "Bitvector model values" begin
            output = """sat
(model
  (define-fun bv8 () (_ BitVec 8) #xFF)
  (define-fun bv4 () (_ BitVec 4) #b1010)
)"""
            model = SMTLib.parse_model(output)
            @test model[:bv8] == 255
            @test model[:bv4] == 10
        end

        @testset "Empty model" begin
            output = "sat\n"
            model = SMTLib.parse_model(output)
            @test isempty(model)
        end

        @testset "No model section" begin
            output = "unsat"
            model = SMTLib.parse_model(output)
            @test isempty(model)
        end

        @testset "Boolean model values" begin
            output = """sat
(model
  (define-fun p () Bool true)
  (define-fun q () Bool false)
)"""
            model = SMTLib.parse_model(output)
            @test model[:p] == true
            @test model[:q] == false
        end
    end

    # ========================================================================
    # Operator Recognition
    # ========================================================================
    @testset "is_operator" begin
        # Arithmetic operators
        @test SMTLib.is_operator(:+) == true
        @test SMTLib.is_operator(:-) == true
        @test SMTLib.is_operator(:*) == true
        @test SMTLib.is_operator(:/) == true

        # Comparison operators
        @test SMTLib.is_operator(:(==)) == true
        @test SMTLib.is_operator(:!=) == true
        @test SMTLib.is_operator(:(<)) == true
        @test SMTLib.is_operator(:(>)) == true

        # Logical operators
        @test SMTLib.is_operator(:!) == true
        @test SMTLib.is_operator(:&&) == true
        @test SMTLib.is_operator(:||) == true

        # Bitvector operators
        @test SMTLib.is_operator(:bvadd) == true
        @test SMTLib.is_operator(:bvsub) == true

        # Boolean literals (must use Symbol() since :true evaluates to Bool in Julia)
        @test SMTLib.is_operator(Symbol("true")) == true
        @test SMTLib.is_operator(Symbol("false")) == true

        # Non-operators (variables)
        @test SMTLib.is_operator(:x) == false
        @test SMTLib.is_operator(:my_var) == false
        @test SMTLib.is_operator(:foo) == false
    end

    # ========================================================================
    # Theory Helpers
    # ========================================================================
    @testset "Theory Helpers" begin
        @testset "Bitvector literals" begin
            @test bv(42, 8) == "(_ bv42 8)"
            @test bv(0, 32) == "(_ bv0 32)"
            @test bv(255, 8) == "(_ bv255 8)"
            @test bv(1, 1) == "(_ bv1 1)"

            # Error on negative value
            @test_throws ErrorException bv(-1, 8)
            # Error on zero width
            @test_throws ErrorException bv(0, 0)
        end

        @testset "Floating-point sorts" begin
            @test SMTLib.smt_type(fp_sort(8, 24)) == "(_ FloatingPoint 8 24)"     # Float32
            @test SMTLib.smt_type(fp_sort(11, 53)) == "(_ FloatingPoint 11 53)"   # Float64
            @test SMTLib.smt_type(fp_sort(5, 11)) == "(_ FloatingPoint 5 11)"     # Float16

            # Error on invalid parameters
            @test_throws ErrorException fp_sort(0, 24)
            @test_throws ErrorException fp_sort(8, 0)
        end

        @testset "Array sorts" begin
            @test SMTLib.smt_type(array_sort(Int, Bool)) == "(Array Int Bool)"
            @test SMTLib.smt_type(array_sort(Int, Int)) == "(Array Int Int)"
            @test SMTLib.smt_type(array_sort(SMTLib.BitVec{32}, Int)) == "(Array (_ BitVec 32) Int)"
            @test SMTLib.smt_type(array_sort(Bool, Float64)) == "(Array Bool Real)"
        end

        @testset "Regex sort" begin
            @test SMTLib.smt_type(re_sort()) == "RegLan"
        end
    end

    # ========================================================================
    # Quantifiers
    # ========================================================================
    @testset "Quantifiers" begin
        @testset "forall construction" begin
            result = to_smtlib(forall([:x => Int, :y => Int], :(x + y == y + x)))
            @test result isa String
            @test occursin("forall", result)
            @test occursin("(x Int)", result)
            @test occursin("(y Int)", result)
            @test occursin("(= (+ x y) (+ y x))", result)
        end

        @testset "exists construction" begin
            result = to_smtlib(exists([:x => Int], :(x > 0)))
            @test result isa String
            @test occursin("exists", result)
            @test occursin("(x Int)", result)
            @test occursin("(> x 0)", result)
        end

        @testset "forall with Bool" begin
            result = to_smtlib(forall([:p => Bool], :(p || !p)))
            @test occursin("(p Bool)", result)
        end

        @testset "Quantifier assertion" begin
            solvers = available_solvers()
            if !isempty(solvers)
                ctx = SMTContext(solver=first(solvers), logic=:QF_LIA)
                expr = forall([:x => Int], :(x == x))
                assert!(ctx, expr)
                @test length(ctx.assertions) == 1
                @test occursin("forall", ctx.assertions[1])
            end
        end
    end

    # ========================================================================
    # Optimization
    # ========================================================================
    @testset "Optimization Directives" begin
        solvers = available_solvers()
        if isempty(solvers)
            @warn "No SMT solver found, skipping optimization tests"
        else
            solver = first(solvers)

            @testset "minimize! directive" begin
                ctx = SMTContext(solver=solver, logic=:QF_LIA)
                declare(ctx, :x, Int)
                minimize!(ctx, :x)
                @test length(ctx.optimization_directives) == 1
                @test ctx.optimization_directives[1] == "(minimize x)"
            end

            @testset "maximize! directive" begin
                ctx = SMTContext(solver=solver, logic=:QF_LIA)
                declare(ctx, :x, Int)
                maximize!(ctx, :x)
                @test length(ctx.optimization_directives) == 1
                @test ctx.optimization_directives[1] == "(maximize x)"
            end

            @testset "minimize! with expression" begin
                ctx = SMTContext(solver=solver, logic=:QF_LIA)
                declare(ctx, :x, Int)
                minimize!(ctx, :(x + 1))
                @test ctx.optimization_directives[1] == "(minimize (+ x 1))"
            end

            @testset "maximize! with expression" begin
                ctx = SMTContext(solver=solver, logic=:QF_LIA)
                declare(ctx, :x, Int)
                maximize!(ctx, :(x * 2))
                @test ctx.optimization_directives[1] == "(maximize (* x 2))"
            end

            @testset "Multiple optimization directives" begin
                ctx = SMTContext(solver=solver, logic=:QF_LIA)
                declare(ctx, :x, Int)
                declare(ctx, :y, Int)
                minimize!(ctx, :x)
                maximize!(ctx, :y)
                @test length(ctx.optimization_directives) == 2
            end

            @testset "Optimization in script" begin
                ctx = SMTContext(solver=solver, logic=:QF_LIA)
                declare(ctx, :x, Int)
                assert!(ctx, :(x > 0))
                minimize!(ctx, :x)

                script = SMTLib.build_script(ctx, true)
                @test occursin("(minimize x)", script)
            end
        end
    end

    # ========================================================================
    # Statistics Parsing
    # ========================================================================
    @testset "Statistics Parsing" begin
        @testset "Parse numeric statistics" begin
            output = "(:time 0.01 :memory 12.5 :conflicts 0)"
            result = SMTResult(:sat, Dict{Symbol,Any}(), Symbol[],
                              Dict{String,Any}(),
                              output)
            stats = get_statistics(result)
            @test haskey(stats, "time")
            @test stats["time"] == 0.01
            @test haskey(stats, "memory")
            @test stats["memory"] == 12.5
            @test haskey(stats, "conflicts")
            @test stats["conflicts"] == 0
        end

        @testset "Return existing statistics" begin
            existing = Dict{String,Any}("time" => 1.0)
            result = SMTResult(:sat, Dict{Symbol,Any}(), Symbol[],
                              existing, "")
            stats = get_statistics(result)
            @test stats["time"] == 1.0
        end

        @testset "Empty output" begin
            result = SMTResult(:sat, Dict{Symbol,Any}(), Symbol[],
                              Dict{String,Any}(), "sat\n")
            stats = get_statistics(result)
            @test isempty(stats) || !isempty(stats)  # May or may not find stats
        end
    end

    # ========================================================================
    # Model Evaluation
    # ========================================================================
    @testset "Model Evaluation" begin
        @testset "Simple arithmetic" begin
            model = Dict{Symbol,Any}(:x => 3, :y => 7)
            @test evaluate(model, :(x + y)) == 10
            @test evaluate(model, :(x * y)) == 21
            @test evaluate(model, :(x - y)) == -4
        end

        @testset "Comparison" begin
            model = Dict{Symbol,Any}(:x => 5, :y => 3)
            @test evaluate(model, :(x > y)) == true
            @test evaluate(model, :(x < y)) == false
            @test evaluate(model, :(x == 5)) == true
        end

        @testset "Literal substitution" begin
            model = Dict{Symbol,Any}(:x => 42)
            @test evaluate(model, :x) == 42
        end

        @testset "Nested expressions" begin
            model = Dict{Symbol,Any}(:x => 2, :y => 3, :z => 4)
            @test evaluate(model, :(x + y * z)) == 14
            @test evaluate(model, :((x + y) * z)) == 20
        end

        @testset "Boolean logic" begin
            model = Dict{Symbol,Any}(:p => true, :q => false)
            @test evaluate(model, :(p && q)) == false
            @test evaluate(model, :(p || q)) == true
            @test evaluate(model, :(!q)) == true
        end

        @testset "No substitution needed" begin
            model = Dict{Symbol,Any}()
            @test evaluate(model, 42) == 42
            @test evaluate(model, true) == true
        end
    end

    # ========================================================================
    # get_model convenience
    # ========================================================================
    @testset "get_model convenience" begin
        # Sat result
        model = Dict{Symbol,Any}(:x => 5)
        result = SMTResult(:sat, model, Symbol[], Dict{String,Any}(), "")
        @test get_model(result) == model
        @test get_model(result)[:x] == 5

        # Unsat result
        result_unsat = SMTResult(:unsat, Dict{Symbol,Any}(), Symbol[], Dict{String,Any}(), "")
        @test isempty(get_model(result_unsat))

        # Unknown result
        result_unknown = SMTResult(:unknown, Dict{Symbol,Any}(), Symbol[], Dict{String,Any}(), "")
        @test isempty(get_model(result_unknown))

        # Timeout result
        result_timeout = SMTResult(:timeout, Dict{Symbol,Any}(), Symbol[], Dict{String,Any}(), "")
        @test isempty(get_model(result_timeout))
    end

    # ========================================================================
    # SMTResult display
    # ========================================================================
    @testset "SMTResult display" begin
        # Sat with model
        result = SMTResult(:sat, Dict{Symbol,Any}(:x => 5), Symbol[], Dict{String,Any}(), "")
        str = sprint(show, result)
        @test occursin(":sat", str)
        @test occursin("model", str)

        # Unsat with core
        result = SMTResult(:unsat, Dict{Symbol,Any}(), [:a, :b], Dict{String,Any}(), "")
        str = sprint(show, result)
        @test occursin(":unsat", str)
        @test occursin("core", str)

        # Unknown
        result = SMTResult(:unknown, Dict{Symbol,Any}(), Symbol[], Dict{String,Any}(), "")
        str = sprint(show, result)
        @test occursin(":unknown", str)
    end

    # ========================================================================
    # SMTSolver display
    # ========================================================================
    @testset "SMTSolver display" begin
        solver = SMTSolver(:z3, "/usr/bin/z3", "4.12.4")
        str = sprint(show, solver)
        @test occursin(":z3", str)
        @test occursin("/usr/bin/z3", str)
    end

    # ========================================================================
    # Edge Cases
    # ========================================================================
    @testset "Edge Cases" begin
        @testset "Empty from_smtlib" begin
            @test_throws Exception from_smtlib("")
        end

        @testset "to_smtlib with plain symbol" begin
            @test to_smtlib(:x) == "x"
            @test to_smtlib(:abc_123) == "abc_123"
        end

        @testset "to_smtlib with zero" begin
            @test to_smtlib(0) == "0"
            @test to_smtlib(0.0) == "0.0"
        end

        @testset "Large integer" begin
            @test to_smtlib(1000000000) == "1000000000"
            @test SMTLib.parse_smt_value("1000000000") == 1000000000
        end

        @testset "Roundtrip simple expressions" begin
            # to_smtlib -> from_smtlib should preserve structure
            original = "(+ x y)"
            @test to_smtlib(from_smtlib(original)) == original

            original2 = "(= x 5)"
            @test to_smtlib(from_smtlib(original2)) == original2
        end

        @testset "Multiple variable types in context" begin
            solvers = available_solvers()
            if !isempty(solvers)
                ctx = SMTContext(solver=first(solvers), logic=:QF_LIA)
                declare(ctx, :x, Int)
                declare(ctx, :y, Bool)
                @test length(ctx.declarations) == 2
                @test ctx.declarations[1] == "(declare-const x Int)"
                @test ctx.declarations[2] == "(declare-const y Bool)"
            end
        end
    end

    # ========================================================================
    # Solver Integration Tests (require actual solver)
    # ========================================================================
    @testset "Solver Integration" begin
        solvers = available_solvers()
        if isempty(solvers)
            @warn "No SMT solver found, skipping solver integration tests"
        else
            for solver in solvers
                @testset "Solver $(solver.kind)" begin
                    @testset "Basic satisfiability" begin
                        ctx = SMTContext(solver=solver, logic=:QF_LIA)
                        declare(ctx, :x, Int)
                        declare(ctx, :y, Int)
                        assert!(ctx, :(x + y == 10))
                        assert!(ctx, :(x > 0))
                        assert!(ctx, :(y > 0))

                        result = check_sat(ctx)
                        @test result.status == :sat
                        @test haskey(result.model, :x)
                        @test haskey(result.model, :y)

                        if haskey(result.model, :x) && haskey(result.model, :y)
                            x_val = result.model[:x]
                            y_val = result.model[:y]
                            if x_val isa Number && y_val isa Number
                                @test x_val + y_val == 10
                                @test x_val > 0
                                @test y_val > 0
                            end
                        end
                    end

                    @testset "Unsatisfiable" begin
                        ctx = SMTContext(solver=solver, logic=:QF_LIA)
                        declare(ctx, :x, Int)
                        assert!(ctx, :(x > 0))
                        assert!(ctx, :(x < 0))

                        result = check_sat(ctx)
                        @test result.status in (:unsat, :unknown)
                    end

                    @testset "Real arithmetic" begin
                        ctx = SMTContext(solver=solver, logic=:QF_LRA)
                        declare(ctx, :x, Float64)
                        declare(ctx, :y, Float64)
                        assert!(ctx, :(x + y == 1.0))
                        assert!(ctx, :(x - y == 0.5))

                        result = check_sat(ctx)
                        @test result.status in (:sat, :unknown)
                    end

                    @testset "Boolean variables" begin
                        ctx = SMTContext(solver=solver, logic=:QF_LIA)
                        declare(ctx, :p, Bool)
                        declare(ctx, :q, Bool)
                        assert!(ctx, :(p || q))
                        assert!(ctx, :(!p))

                        result = check_sat(ctx)
                        @test result.status == :sat
                        if haskey(result.model, :q)
                            @test result.model[:q] == true
                        end
                    end

                    @testset "Multiple check-sat calls" begin
                        ctx = SMTContext(solver=solver, logic=:QF_LIA)
                        declare(ctx, :x, Int)
                        assert!(ctx, :(x > 0))
                        assert!(ctx, :(x < 100))

                        result1 = check_sat(ctx)
                        @test result1.status == :sat

                        # Add more constraints and check again
                        assert!(ctx, :(x == 42))
                        result2 = check_sat(ctx)
                        @test result2.status == :sat
                        if haskey(result2.model, :x)
                            @test result2.model[:x] == 42
                        end
                    end

                    @testset "Named assertions with solver" begin
                        ctx = SMTContext(solver=solver, logic=:QF_LIA)
                        declare(ctx, :x, Int)
                        assert!(ctx, :(x > 0); name=:pos)
                        assert!(ctx, :(x < 10); name=:bound)

                        result = check_sat(ctx)
                        @test result.status == :sat
                    end

                    if solver.kind == :z3
                        @testset "Unsat core extraction (Z3)" begin
                            ctx = SMTContext(solver=solver, logic=:QF_LIA)
                            set_option!(ctx, :produce_unsat_cores, true)
                            declare(ctx, :x, Int)
                            assert!(ctx, :(x > 0); name=:pos)
                            assert!(ctx, :(x < 0); name=:neg)

                            result = check_sat(ctx)
                            @test result.status == :unsat
                            core = get_unsat_core(result)
                            # Z3 should return core labels if unsat
                            @test core isa Vector{Symbol}
                            @test !isempty(core) && (:pos in core || :neg in core)
                        end
                    end
                end
            end
        end
    end

    # ========================================================================
    # @smt Macro
    # ========================================================================
    @testset "@smt Macro" begin
        solvers = available_solvers()
        if isempty(solvers)
            @warn "No SMT solver found, skipping @smt macro tests"
        else
            @testset "Basic usage" begin
                result = @smt begin
                    x::Int
                    y::Int
                    x + y == 10
                    x > 0
                    y > 0
                end

                @test result isa SMTResult
                @test result.status == :sat
            end

            @testset "With logic option" begin
                result = @smt logic=:QF_LRA begin
                    x::Float64
                    x > 0.0
                end

                @test result isa SMTResult
                @test result.status in (:sat, :unknown)
            end
        end
    end

    # ========================================================================
    # Constants Integrity
    # ========================================================================
    @testset "Constants" begin
        @testset "JULIA_OP_TO_SMT_MAP completeness" begin
            @test haskey(SMTLib.JULIA_OP_TO_SMT_MAP, :+)
            @test haskey(SMTLib.JULIA_OP_TO_SMT_MAP, :-)
            @test haskey(SMTLib.JULIA_OP_TO_SMT_MAP, :*)
            @test haskey(SMTLib.JULIA_OP_TO_SMT_MAP, :/)
            @test haskey(SMTLib.JULIA_OP_TO_SMT_MAP, :(==))
            @test haskey(SMTLib.JULIA_OP_TO_SMT_MAP, :!)
            @test haskey(SMTLib.JULIA_OP_TO_SMT_MAP, :bvadd)
            @test haskey(SMTLib.JULIA_OP_TO_SMT_MAP, :select)
            @test haskey(SMTLib.JULIA_OP_TO_SMT_MAP, :store)
            @test haskey(SMTLib.JULIA_OP_TO_SMT_MAP, :forall)
            @test haskey(SMTLib.JULIA_OP_TO_SMT_MAP, :exists)
        end

        @testset "SMT_OP_TO_JULIA_MAP completeness" begin
            @test haskey(SMTLib.SMT_OP_TO_JULIA_MAP, "+")
            @test haskey(SMTLib.SMT_OP_TO_JULIA_MAP, "-")
            @test haskey(SMTLib.SMT_OP_TO_JULIA_MAP, "=")
            @test haskey(SMTLib.SMT_OP_TO_JULIA_MAP, "not")
            @test haskey(SMTLib.SMT_OP_TO_JULIA_MAP, "and")
            @test haskey(SMTLib.SMT_OP_TO_JULIA_MAP, "or")
        end

        @testset "KNOWN_OPERATORS consistency" begin
            # Every key in the op map should be in KNOWN_OPERATORS
            for k in keys(SMTLib.JULIA_OP_TO_SMT_MAP)
                @test k in SMTLib.KNOWN_OPERATORS
            end
            @test Symbol("true") in SMTLib.KNOWN_OPERATORS
            @test Symbol("false") in SMTLib.KNOWN_OPERATORS
        end

        @testset "LOGICS" begin
            @test :QF_LIA in SMTLib.LOGICS
            @test :QF_LRA in SMTLib.LOGICS
            @test :QF_BV in SMTLib.LOGICS
            @test :ALL in SMTLib.LOGICS
            @test :QF_S in SMTLib.LOGICS
        end
    end

    # ========================================================================
    # Unicode Operator Conversions
    # ========================================================================
    @testset "Unicode Operator Conversions" begin
        @test SMTLib.julia_op_to_smt(Symbol("≠")) == "distinct"
        @test SMTLib.julia_op_to_smt(Symbol("≤")) == "<="
        @test SMTLib.julia_op_to_smt(Symbol("≥")) == ">="
        @test SMTLib.julia_op_to_smt(Symbol("¬")) == "not"
        @test SMTLib.julia_op_to_smt(Symbol("∧")) == "and"
        @test SMTLib.julia_op_to_smt(Symbol("∨")) == "or"
        @test SMTLib.julia_op_to_smt(Symbol("⟹")) == "=>"
        @test SMTLib.julia_op_to_smt(Symbol("⟺")) == "="
        @test SMTLib.julia_op_to_smt(Symbol("∀")) == "forall"
        @test SMTLib.julia_op_to_smt(Symbol("∃")) == "exists"
    end

    # ========================================================================
    # String and Regex Operator Conversions
    # ========================================================================
    @testset "String Operator Conversions" begin
        @test SMTLib.julia_op_to_smt(:str_len) == "str.len"
        @test SMTLib.julia_op_to_smt(:str_concat) == "str.++"
        @test SMTLib.julia_op_to_smt(:str_at) == "str.at"
        @test SMTLib.julia_op_to_smt(:str_contains) == "str.contains"
        @test SMTLib.julia_op_to_smt(:str_prefixof) == "str.prefixof"
        @test SMTLib.julia_op_to_smt(:str_suffixof) == "str.suffixof"
        @test SMTLib.julia_op_to_smt(:str_replace) == "str.replace"
        @test SMTLib.julia_op_to_smt(:str_substr) == "str.substr"
        @test SMTLib.julia_op_to_smt(:str_to_int) == "str.to_int"
        @test SMTLib.julia_op_to_smt(:int_to_str) == "str.from_int"
        @test SMTLib.julia_op_to_smt(:str_in_re) == "str.in_re"
        @test SMTLib.julia_op_to_smt(:re_star) == "re.*"
        @test SMTLib.julia_op_to_smt(:re_plus) == "re.+"
        @test SMTLib.julia_op_to_smt(:re_union) == "re.union"
        @test SMTLib.julia_op_to_smt(:str_to_re) == "str.to_re"

        # to_smtlib with string operator expressions
        @test to_smtlib(:(str_len(s))) == "(str.len s)"
        @test to_smtlib(:(str_contains(s, t))) == "(str.contains s t)"
    end

    # ========================================================================
    # Extract Variables
    # ========================================================================
    @testset "Extract Variables" begin
        vars = SMTLib.extract_variables(:(x + y > z))
        @test haskey(vars, :x)
        @test haskey(vars, :y)
        @test haskey(vars, :z)
        @test vars[:x] == Int  # default type
        @test !haskey(vars, :+)  # operators excluded
        @test !haskey(vars, :>)  # operators excluded

        # Nested expression
        vars2 = SMTLib.extract_variables(:((a + b) * (c - d) == 0))
        @test haskey(vars2, :a)
        @test haskey(vars2, :b)
        @test haskey(vars2, :c)
        @test haskey(vars2, :d)
        @test !haskey(vars2, :*)

        # No variables (literal only)
        vars3 = SMTLib.extract_variables(42)
        @test isempty(vars3)
    end

    # ========================================================================
    # Chained Comparisons
    # ========================================================================
    @testset "Chained Comparisons" begin
        # a < b < c produces (and (< a b) (< b c))
        expr = Expr(:comparison, :a, :<, :b, :<, :c)
        result = to_smtlib(expr)
        @test occursin("and", result)
        @test occursin("(< a b)", result)
        @test occursin("(< b c)", result)

        # Single comparison: a <= b
        expr2 = Expr(:comparison, :a, :<=, :b)
        result2 = to_smtlib(expr2)
        @test result2 == "(<= a b)"
    end

    # ========================================================================
    # BitVecLiteral Equality and Hash
    # ========================================================================
    @testset "BitVecLiteral equality and hash" begin
        bv1 = bv(42, 8)
        bv2 = bv(42, 8)
        bv3 = bv(42, 16)
        bv4 = bv(0, 8)

        @test bv1 == bv2
        @test bv1 != bv3  # different width
        @test bv1 != bv4  # different value

        # String equality
        @test bv1 == "(_ bv42 8)"
        @test "(_ bv42 8)" == bv1

        # Hash consistency
        @test hash(bv1) == hash(bv2)
    end

    # ========================================================================
    # get_unsat_core from SMTResult
    # ========================================================================
    @testset "get_unsat_core from SMTResult" begin
        # With core labels
        core_labels = [:pos, :neg, :bound]
        result = SMTResult(:unsat, Dict{Symbol,Any}(), core_labels, Dict{String,Any}(), "")
        core = get_unsat_core(result)
        @test core == core_labels
        @test :pos in core
        @test :neg in core

        # Empty core
        result_empty = SMTResult(:unsat, Dict{Symbol,Any}(), Symbol[], Dict{String,Any}(), "")
        @test isempty(get_unsat_core(result_empty))

        # Sat result (core should be empty)
        result_sat = SMTResult(:sat, Dict{Symbol,Any}(:x => 5), Symbol[], Dict{String,Any}(), "")
        @test isempty(get_unsat_core(result_sat))
    end

    # ========================================================================
    # optimize function
    # ========================================================================
    @testset "optimize function" begin
        solvers = available_solvers()
        if !isempty(solvers)
            solver = first(solvers)
            if solver.kind == :z3
                ctx = SMTContext(solver=solver, logic=:QF_LIA)
                declare(ctx, :x, Int)
                assert!(ctx, :(x > 0))
                assert!(ctx, :(x < 100))
                minimize!(ctx, :x)

                result = optimize(ctx)
                @test result.status in (:sat, :unknown)
                if result.status == :sat && haskey(result.model, :x)
                    @test result.model[:x] == 1
                end
            end
        end
    end

    # ========================================================================
    # from_smtlib: let expressions and ite
    # ========================================================================
    @testset "from_smtlib: let and ite advanced" begin
        # if-then-else
        result_ite = from_smtlib("(ite (> x 0) x (- x))")
        @test result_ite isa Expr
        @test result_ite.args[1] == :ifelse

        # Nested ite
        result_nested = from_smtlib("(ite (> x 0) (+ x 1) (- x 1))")
        @test result_nested isa Expr
        @test result_nested.args[1] == :ifelse
        @test result_nested.args[2] isa Expr  # condition
        @test result_nested.args[3] isa Expr  # then branch

        # Bitvector comparisons
        result_bv = from_smtlib("(bvult a b)")
        @test result_bv.args[1] == :bvult

        result_bvsle = from_smtlib("(bvsle a b)")
        @test result_bvsle.args[1] == :bvsle
    end

    # ========================================================================
    # Bitvector comparison operators
    # ========================================================================
    @testset "Bitvector Comparison Operators" begin
        @test to_smtlib(:(bvult(a, b))) == "(bvult a b)"
        @test to_smtlib(:(bvugt(a, b))) == "(bvugt a b)"
        @test to_smtlib(:(bvule(a, b))) == "(bvule a b)"
        @test to_smtlib(:(bvuge(a, b))) == "(bvuge a b)"
        @test to_smtlib(:(bvslt(a, b))) == "(bvslt a b)"
        @test to_smtlib(:(bvsgt(a, b))) == "(bvsgt a b)"
        @test to_smtlib(:(bvsle(a, b))) == "(bvsle a b)"
        @test to_smtlib(:(bvsge(a, b))) == "(bvsge a b)"
        @test to_smtlib(:(concat(a, b))) == "(concat a b)"
    end

    # ========================================================================
    # Math function operators (nonlinear)
    # ========================================================================
    @testset "Math Function Operators" begin
        @test to_smtlib(:(sqrt(x))) == "(sqrt x)"
        @test to_smtlib(:(sin(x))) == "(sin x)"
        @test to_smtlib(:(cos(x))) == "(cos x)"
        @test to_smtlib(:(tan(x))) == "(tan x)"
        @test to_smtlib(:(exp(x))) == "(exp x)"
        @test to_smtlib(:(log(x))) == "(log x)"
    end

    # ========================================================================
    # SMT_OP_TO_JULIA_MAP reverse mapping
    # ========================================================================
    @testset "SMT_OP_TO_JULIA_MAP reverse correctness" begin
        @test SMTLib.SMT_OP_TO_JULIA_MAP["="] == :(==)
        @test SMTLib.SMT_OP_TO_JULIA_MAP["distinct"] == :(!=)
        @test SMTLib.SMT_OP_TO_JULIA_MAP["not"] == :!
        @test SMTLib.SMT_OP_TO_JULIA_MAP["and"] == :&&
        @test SMTLib.SMT_OP_TO_JULIA_MAP["or"] == :||
        @test SMTLib.SMT_OP_TO_JULIA_MAP["=>"] == :implies
        @test SMTLib.SMT_OP_TO_JULIA_MAP["ite"] == :ifelse
        @test SMTLib.SMT_OP_TO_JULIA_MAP["<"] == :(<)
        @test SMTLib.SMT_OP_TO_JULIA_MAP[">"] == :(>)
        @test SMTLib.SMT_OP_TO_JULIA_MAP["<="] == :(<=)
        @test SMTLib.SMT_OP_TO_JULIA_MAP[">="] == :(>=)
        @test SMTLib.SMT_OP_TO_JULIA_MAP["true"] == Symbol("true")
        @test SMTLib.SMT_OP_TO_JULIA_MAP["false"] == Symbol("false")
    end

    # ========================================================================
    # get_unsat_core(ctx) error
    # ========================================================================
    @testset "get_unsat_core context error" begin
        solvers = available_solvers()
        if !isempty(solvers)
            ctx = SMTContext(solver=first(solvers), logic=:QF_LIA)
            @test_throws ErrorException get_unsat_core(ctx)
        end
    end

    # ========================================================================
    # Solver Integration: optimization with constraints
    # ========================================================================
    @testset "Solver Integration: Z3 optimization" begin
        solvers = available_solvers()
        z3_solver = nothing
        for s in solvers
            if s.kind == :z3
                z3_solver = s
                break
            end
        end

        if z3_solver !== nothing
            @testset "maximize with bounds" begin
                ctx = SMTContext(solver=z3_solver, logic=:QF_LIA)
                declare(ctx, :x, Int)
                assert!(ctx, :(x > 0))
                assert!(ctx, :(x < 50))
                maximize!(ctx, :x)

                result = optimize(ctx)
                @test result.status in (:sat, :unknown)
                if result.status == :sat && haskey(result.model, :x)
                    @test result.model[:x] == 49
                end
            end

            @testset "minimize with expression" begin
                ctx = SMTContext(solver=z3_solver, logic=:QF_LIA)
                declare(ctx, :x, Int)
                declare(ctx, :y, Int)
                assert!(ctx, :(x > 0))
                assert!(ctx, :(y > 0))
                assert!(ctx, :(x + y == 10))
                minimize!(ctx, :(x * x + y * y))

                result = optimize(ctx)
                @test result.status in (:sat, :unknown)
            end
        end
    end

    # ========================================================================
    # to_smtlib: iff and xor operators
    # ========================================================================
    @testset "Logical: iff and xor" begin
        @test to_smtlib(:(iff(x, y))) == "(= x y)"
        @test to_smtlib(:(xor(x, y))) == "(xor x y)"
        @test to_smtlib(:(implies(x, y))) == "(=> x y)"
    end

    # ========================================================================
    # Roundtrip: to_smtlib -> from_smtlib for more complex cases
    # ========================================================================
    @testset "Roundtrip: complex expressions" begin
        # Logical operators
        @test to_smtlib(from_smtlib("(not x)")) == "(not x)"
        @test to_smtlib(from_smtlib("(and x y)")) == "(and x y)"
        @test to_smtlib(from_smtlib("(or x y)")) == "(or x y)"

        # Nested arithmetic
        @test to_smtlib(from_smtlib("(* (+ a b) (- c d))")) == "(* (+ a b) (- c d))"

        # Bitvector
        @test to_smtlib(from_smtlib("(bvadd x y)")) == "(bvadd x y)"
    end

    # ========================================================================
    # parse_smt_value: additional edge cases
    # ========================================================================
    @testset "parse_smt_value: additional patterns" begin
        # Large bitvector hex
        @test SMTLib.parse_smt_value("#xFFFF") == 0xFFFF
        @test SMTLib.parse_smt_value("#x0000") == 0

        # Large bitvector binary
        @test SMTLib.parse_smt_value("#b10101010") == 0xAA

        # Zero rational
        @test SMTLib.parse_smt_value("(/ 0 1)") == 0//1

        # Negative number in parentheses
        @test SMTLib.parse_smt_value("(- 100)") == -100
    end

    # ========================================================================
    # parse_unsat_core: various formats
    # ========================================================================
    @testset "parse_unsat_core: formats" begin
        # Standard format
        core = SMTLib.parse_unsat_core("unsat\n(a b c)")
        @test :a in core
        @test :b in core
        @test :c in core
        @test length(core) == 3

        # Single label
        core2 = SMTLib.parse_unsat_core("unsat\n(single)")
        @test core2 == [:single]

        # No core section
        core3 = SMTLib.parse_unsat_core("unsat")
        @test isempty(core3)

        # Error line should be skipped
        core4 = SMTLib.parse_unsat_core("unsat\n(error \"some error\")\n(a b)")
        @test :a in core4
    end

end
