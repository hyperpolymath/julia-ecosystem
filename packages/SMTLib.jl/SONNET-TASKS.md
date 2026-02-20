# SONNET-TASKS: SMTLib.jl

**Date:** 2026-02-12
**Auditor:** Claude Opus 4.6
**Honest Completion:** ~55%

The Julia source (`src/SMTLib.jl`) is a real, functional single-file library with
working solver discovery, SMT-LIB2 generation, model parsing, and a convenience
macro. The tests are meaningful and would pass given an installed solver.

However: 4 exported symbols have no implementation (`push!`, `pop!`, `get_model`,
`from_smtlib`), the docs reference features that do not exist (`solver_options`,
named assertions, `unsat_core`), the ABI/FFI layer is unmodified RSR template
boilerplate with `{{PROJECT}}` placeholders everywhere, every RSR community file
still has `{{PLACEHOLDER}}` tokens, there is no `.machine_readable/` directory,
no `.editorconfig`, no `.gitignore`, no `.bot_directives/`, the CodeQL workflow
scans for Rust (not Julia), the examples directory contains ReScript and Deno
files that have nothing to do with SMT solving, and the ROADMAP is the raw
template with "YOUR Template Repo."

---

## GROUND RULES FOR SONNET

1. Do NOT add features that are not already partially started. Fix what exists.
2. Every change MUST be verified by a runnable command or test.
3. Do NOT refactor working code. Only fix broken, missing, or misleading things.
4. Read the full file before editing -- many issues are interconnected.
5. Run `julia --project=/var/mnt/eclipse/repos/SMTLib.jl -e 'using Pkg; Pkg.test()'` after every task to confirm nothing is broken.

---

## TASK 1: Implement the 4 Exported-But-Missing Functions

**Files:**
- `/var/mnt/eclipse/repos/SMTLib.jl/src/SMTLib.jl` (lines 36, 37, 39)

**Problem:**
The module exports `push!`, `pop!`, `get_model`, and `from_smtlib` on lines
36 and 39, but none of these functions are defined anywhere in the source:

- `push!` and `pop!` for `SMTContext` -- documented in README (line 77-80),
  referenced in examples (docs/src/examples.md lines 31-37), in AGENTS.md
  (line 19). The `SMTContext` struct has no stack field to support push/pop.
- `get_model` -- exported on line 36, referenced in API docs
  (docs/src/api.md line 26), but never defined. `check_sat` already returns
  models inline via `SMTResult.model`.
- `from_smtlib` -- exported on line 39, referenced in API docs
  (docs/src/api.md line 40), but never defined. Would be the inverse of
  `to_smtlib`.

**What to do:**
1. Add a `stack_depth::Int` field to `SMTContext` (line 96-102) initialized to 0.
2. Add `declarations_stack::Vector{Int}` and `assertions_stack::Vector{Int}` to
   track push points.
3. Implement `Base.push!(ctx::SMTContext)` that records current lengths and
   emits `(push 1)` to the script.
4. Implement `Base.pop!(ctx::SMTContext)` that truncates back to last push point
   and emits `(pop 1)`.
5. Implement `get_model(result::SMTResult)` as a convenience alias returning
   `result.model`.
6. Implement `from_smtlib(s::String)` to parse an SMT-LIB2 expression string
   back into a Julia `Expr`. At minimum, handle:
   - `(+ x y)` -> `:(x + y)`
   - `(= x y)` -> `:(x == y)`
   - `(and ...)` / `(or ...)` / `(not ...)`
   - Integer and boolean literals.
7. Update `build_script` (line 430) to emit push/pop commands from the stack.

**Verification:**
```julia
julia --project=/var/mnt/eclipse/repos/SMTLib.jl -e '
using SMTLib

# Test push!/pop! exist and work structurally
ctx = SMTContext(logic=:QF_LIA)
declare(ctx, :x, Int)
assert!(ctx, :(x > 5))
push!(ctx)
assert!(ctx, :(x < 3))
pop!(ctx)
@assert length(ctx.assertions) == 1 "pop! should restore assertion count"

# Test get_model exists
r = SMTResult(:sat, Dict(:x => 42), Symbol[], Dict{String,Any}(), "")
@assert get_model(r) == Dict(:x => 42) "get_model should return model dict"

# Test from_smtlib exists and round-trips basics
@assert from_smtlib("(+ x y)") == :(x + y) "from_smtlib basic arithmetic"
@assert from_smtlib("true") == true "from_smtlib boolean"

println("TASK 1 PASSED")
'
```

---

## TASK 2: Remove Bogus Examples Directory

**Files:**
- `/var/mnt/eclipse/repos/SMTLib.jl/examples/SafeDOMExample.res`
- `/var/mnt/eclipse/repos/SMTLib.jl/examples/web-project-deno.json`

**Problem:**
The `examples/` directory contains a ReScript DOM-mounting example and a Deno
project config. Neither has anything to do with SMT solving. They are leftover
RSR template files. The SPDX header in `SafeDOMExample.res` is `AGPL-3.0-or-later`
which violates the license policy (should be PMPL-1.0-or-later or at least not AGPL).

**What to do:**
1. Delete `examples/SafeDOMExample.res`.
2. Delete `examples/web-project-deno.json`.
3. Create `examples/basic_sat.jl` with a self-contained example that checks
   satisfiability (can run without a solver by showing the generated SMT-LIB2
   script output).
4. Create `examples/incremental.jl` showing push/pop usage (after TASK 1).
5. Add SPDX header `# SPDX-License-Identifier: PMPL-1.0-or-later` to all new files.

**Verification:**
```bash
# Bogus files gone
test ! -f /var/mnt/eclipse/repos/SMTLib.jl/examples/SafeDOMExample.res && echo "PASS: res removed"
test ! -f /var/mnt/eclipse/repos/SMTLib.jl/examples/web-project-deno.json && echo "PASS: deno removed"

# New examples exist and are valid Julia
julia --project=/var/mnt/eclipse/repos/SMTLib.jl -e 'include("/var/mnt/eclipse/repos/SMTLib.jl/examples/basic_sat.jl")' && echo "PASS: basic_sat runs"
```

---

## TASK 3: Fix Unsat Core and Named Assertions (Documented but Nonexistent)

**Files:**
- `/var/mnt/eclipse/repos/SMTLib.jl/src/SMTLib.jl` (function `assert!` at line 400, `build_script` at line 430, `parse_result` at line 503)
- `/var/mnt/eclipse/repos/SMTLib.jl/docs/src/examples.md` (lines 86-97)

**Problem:**
`docs/src/examples.md` lines 86-97 show `assert!(ctx, expr, name=:c1)` with
named assertions and `check_sat(ctx, unsat_core=true)` returning
`result.unsat_core`. The `SMTResult` struct has an `unsat_core` field (line 78),
but:
- `assert!` does not accept a `name` keyword argument.
- `build_script` never emits `(set-option :produce-unsat-cores true)`.
- `build_script` never emits `(get-unsat-core)`.
- `parse_result` never parses unsat core output.
- `check_sat` does not accept an `unsat_core` keyword.

**What to do:**
1. Extend `assert!(ctx, expr; name=nothing)` to accept an optional name. When
   `name` is provided, emit `(assert (! expr :named name))`.
2. Add `unsat_core_requested::Bool` field to `SMTContext`, defaulting to `false`.
3. Extend `check_sat(ctx; get_model=true, unsat_core=false)` to accept the
   `unsat_core` keyword.
4. Update `build_script` to emit `(set-option :produce-unsat-cores true)` and
   `(get-unsat-core)` when `unsat_core=true`.
5. Update `parse_result` to parse unsat core from solver output (lines matching
   parenthesized symbol lists after "unsat").

**Verification:**
```julia
julia --project=/var/mnt/eclipse/repos/SMTLib.jl -e '
using SMTLib

ctx = SMTContext(logic=:QF_LIA)
declare(ctx, :x, Int)

# Named assertions should work
assert!(ctx, :(x > 10), name=:c1)
assert!(ctx, :(x < 5), name=:c2)

# Verify the SMT-LIB script contains named assertions
script = SMTLib.build_script(ctx, true)
@assert occursin(":named c1", script) "Named assertion c1 missing from script"
@assert occursin(":named c2", script) "Named assertion c2 missing from script"

println("TASK 3 PASSED")
'
```

---

## TASK 4: Fix solver_options References in Documentation

**Files:**
- `/var/mnt/eclipse/repos/SMTLib.jl/docs/src/solvers.md` (lines 137-146, 191)

**Problem:**
The solvers documentation references `ctx.solver_options[:timeout]`,
`ctx.solver_options[:random_seed]`, `ctx.solver_options[:finite_model_find]`,
and `ctx.solver_options[:max_memory]`. The `SMTContext` struct has no
`solver_options` field. This code will throw `ErrorException` if anyone runs it.

**What to do:**
Either:
(a) Add a `solver_options::Dict{Symbol, Any}` field to `SMTContext` and wire
    it into `build_script` to emit `(set-option ...)` commands, OR
(b) Remove the solver_options examples from `docs/src/solvers.md` and replace
    with documentation of existing features (timeout_ms constructor arg).

Option (a) is preferred since the feature is useful.

If implementing (a):
1. Add `solver_options::Dict{Symbol, Any}` to `SMTContext` struct (line 96-102),
   initialized to `Dict{Symbol, Any}()`.
2. Update the constructor (line 104-112).
3. In `build_script`, emit solver options as `(set-option :key value)` lines.

**Verification:**
```julia
julia --project=/var/mnt/eclipse/repos/SMTLib.jl -e '
using SMTLib

ctx = SMTContext(logic=:QF_LIA)

# solver_options should be accessible
ctx.solver_options[:random_seed] = 42

script = SMTLib.build_script(ctx, false)
@assert occursin("random_seed", script) || occursin("random-seed", script) "solver option not in script"

println("TASK 4 PASSED")
'
```

---

## TASK 5: Replace All {{PLACEHOLDER}} Tokens in RSR Files

**Files:**
- `/var/mnt/eclipse/repos/SMTLib.jl/SECURITY.md` ({{OWNER}}, {{REPO}}, {{PROJECT_NAME}}, etc.)
- `/var/mnt/eclipse/repos/SMTLib.jl/CONTRIBUTING.md` ({{FORGE}}, {{OWNER}}, {{REPO}})
- `/var/mnt/eclipse/repos/SMTLib.jl/CODE_OF_CONDUCT.md` ({{OWNER}}, {{REPO}}, etc.)
- `/var/mnt/eclipse/repos/SMTLib.jl/docs/CITATIONS.adoc` (wrong project name, AGPL license ref)
- `/var/mnt/eclipse/repos/SMTLib.jl/ROADMAP.adoc` (says "YOUR Template Repo Roadmap")
- `/var/mnt/eclipse/repos/SMTLib.jl/ABI-FFI-README.md` (line 1: "delete this line")

**Problem:**
Dozens of `{{OWNER}}`, `{{REPO}}`, `{{FORGE}}`, `{{PROJECT_NAME}}`,
`{{SECURITY_EMAIL}}`, `{{PGP_FINGERPRINT}}`, etc. remain unreplaced.
`ROADMAP.adoc` line 2 says "YOUR Template Repo Roadmap." `CITATIONS.adoc` cites
"rsr-template-repo" with AGPL license. The ABI-FFI-README.md line 1 still says
"delete this line."

**What to do:**
1. In SECURITY.md:
   - `{{PROJECT_NAME}}` -> `SMTLib.jl`
   - `{{OWNER}}` -> `hyperpolymath`
   - `{{REPO}}` -> `SMTLib.jl`
   - `{{SECURITY_EMAIL}}` -> `jonathan.jewell@open.ac.uk`
   - Remove the template instruction comment block (lines 3-19).
   - Remove PGP sections if not applicable.
2. In CONTRIBUTING.md:
   - `{{FORGE}}` -> `github.com`
   - `{{OWNER}}` -> `hyperpolymath`
   - `{{REPO}}` -> `SMTLib.jl`
3. In CODE_OF_CONDUCT.md:
   - Same replacements as above.
   - Remove template instruction block.
4. In CITATIONS.adoc:
   - Replace `rsr-template-repo` with `SMTLib.jl`.
   - Replace `AGPL-3.0-or-later` with `PMPL-1.0-or-later`.
   - Replace author `Polymath, Hyper` with `Jewell, Jonathan D.A.`
5. In ROADMAP.adoc:
   - Replace "YOUR Template Repo" with "SMTLib.jl".
   - Add real milestones reflecting the actual state of the project.
6. In ABI-FFI-README.md:
   - Delete line 1 (`{{~ Aditionally delete this line...}}`).
   - Replace `{{PROJECT}}` with `SMTLib` and `{{project}}` with `smtlib`.
   - Replace `{{LICENSE}}` with `PMPL-1.0-or-later`.

**Verification:**
```bash
cd /var/mnt/eclipse/repos/SMTLib.jl
grep -rn '{{' --include='*.md' --include='*.adoc' . | grep -v '.git/' | grep -v 'contractiles/' | grep -v 'ABI-FFI' | head -5
# Should return NO matches (excluding contractile templates which are allowed)
echo "---"
grep -c '{{PROJECT}}' ABI-FFI-README.md
# Should return 0
echo "---"
head -1 ROADMAP.adoc | grep -v 'YOUR'
# Should not contain "YOUR"
echo "TASK 5 PASSED (if all above are empty/0)"
```

---

## TASK 6: Replace All {{PROJECT}} Placeholders in ABI/FFI Files

**Files:**
- `/var/mnt/eclipse/repos/SMTLib.jl/src/abi/Types.idr` (lines 6, 11)
- `/var/mnt/eclipse/repos/SMTLib.jl/src/abi/Layout.idr` (lines 8, 10)
- `/var/mnt/eclipse/repos/SMTLib.jl/src/abi/Foreign.idr` (lines 9, 11, 12, 23, 35, 49, etc.)
- `/var/mnt/eclipse/repos/SMTLib.jl/ffi/zig/build.zig` (lines 1, 12, 23, 35, 36, 82)
- `/var/mnt/eclipse/repos/SMTLib.jl/ffi/zig/src/main.zig` (lines 1, 12, 54, 73, 89, etc.)
- `/var/mnt/eclipse/repos/SMTLib.jl/ffi/zig/test/integration_test.zig` (lines 1, 10-17, etc.)

**Problem:**
Every ABI and FFI file is the raw RSR template with `{{PROJECT}}` and
`{{project}}` placeholders. None of these files will compile. Additionally,
the SPDX headers in the Zig files say `AGPL-3.0-or-later` instead of
`PMPL-1.0-or-later`.

**What to do:**
1. In all `.idr` files: replace `{{PROJECT}}` with `SMTLib`.
2. In all `.zig` files: replace `{{PROJECT}}` with `SMTLib` and `{{project}}`
   with `smtlib`.
3. In all `.zig` files: change SPDX from `AGPL-3.0-or-later` to
   `PMPL-1.0-or-later`.
4. Consider whether the ABI/FFI layer makes sense for a pure-Julia SMT interface.
   If it does not, add a note to `ABI-FFI-README.md` explaining that the ABI/FFI
   layer is reserved for future native solver bindings, or remove it entirely
   and document why in the commit message.

**Verification:**
```bash
cd /var/mnt/eclipse/repos/SMTLib.jl
grep -rn '{{PROJECT}}\|{{project}}' src/abi/ ffi/ | head -5
# Should return 0 matches
grep -rn 'AGPL' src/abi/ ffi/ | head -5
# Should return 0 matches
echo "TASK 6 PASSED (if both empty)"
```

---

## TASK 7: Fix CodeQL Workflow (Scanning for Wrong Language)

**Files:**
- `/var/mnt/eclipse/repos/SMTLib.jl/.github/workflows/codeql.yml`

**Problem:**
The CodeQL workflow (line 24-25) scans for `language: rust` with
`build-mode: none`. This is a Julia repository. CodeQL does not have a Julia
language scanner, so this workflow will either fail silently or produce no
useful results. The correct approach is to scan for `actions` (workflow
security) or remove the workflow if no supported language is present.

**What to do:**
1. Change the matrix to `language: actions` (CodeQL can scan GitHub Actions
   workflows for injection vulnerabilities, which IS useful).
2. Remove `build-mode: none` (not applicable to actions scanning).
3. Update the checkout action SHA to match the standard pinned version from
   CLAUDE.md: `actions/checkout@34e114876b0b11c390a56381ad16ebd13914f8d5` (v4).
4. Update CodeQL action SHA to: `github/codeql-action@6624720a57d4c312633c7b953db2f2da5bcb4c3a` (v3).

**Verification:**
```bash
cd /var/mnt/eclipse/repos/SMTLib.jl
grep 'language:' .github/workflows/codeql.yml
# Should show "actions", not "rust"
grep 'build-mode' .github/workflows/codeql.yml
# Should return nothing
echo "TASK 7 PASSED (if actions and no build-mode)"
```

---

## TASK 8: Fix Scorecard Workflow (Unpinned Actions)

**Files:**
- `/var/mnt/eclipse/repos/SMTLib.jl/.github/workflows/scorecard.yml`

**Problem:**
Lines 18, 22, 29 use unpinned action tags (`@v4`, `@v2.3.1`, `@v3`) instead
of SHA-pinned versions. This fails OpenSSF Scorecard "Pinned-Dependencies"
check and is a supply-chain security risk.

**What to do:**
1. Pin `actions/checkout@v4` to SHA `34e114876b0b11c390a56381ad16ebd13914f8d5`.
2. Pin `ossf/scorecard-action@v2.4.0` to SHA `62b2cac7ed8198b15735ed49ab1e5cf35480ba46`.
3. Pin `github/codeql-action/upload-sarif@v3` to SHA `6624720a57d4c312633c7b953db2f2da5bcb4c3a`.

**Verification:**
```bash
cd /var/mnt/eclipse/repos/SMTLib.jl
grep -E '@v[0-9]' .github/workflows/scorecard.yml
# Should return nothing (all SHA-pinned)
echo "TASK 8 PASSED (if empty)"
```

---

## TASK 9: Add Missing RSR Infrastructure Files

**Files to create:**
- `/var/mnt/eclipse/repos/SMTLib.jl/.editorconfig`
- `/var/mnt/eclipse/repos/SMTLib.jl/.gitignore`
- `/var/mnt/eclipse/repos/SMTLib.jl/.machine_readable/STATE.scm`
- `/var/mnt/eclipse/repos/SMTLib.jl/.machine_readable/ECOSYSTEM.scm`
- `/var/mnt/eclipse/repos/SMTLib.jl/.machine_readable/META.scm`

**Problem:**
The repo is missing `.editorconfig`, `.gitignore`, and the entire
`.machine_readable/` directory with STATE.scm, ECOSYSTEM.scm, and META.scm.
The AI.a2ml file references `.machines_readable/6scm/STATE.scm` (wrong path,
wrong directory name). There is no `.bot_directives/` directory either.

**What to do:**
1. Create `.editorconfig` with Julia-appropriate settings (4-space indent,
   UTF-8, LF line endings, trim trailing whitespace).
2. Create `.gitignore` for Julia packages:
   - `/Manifest.toml` (already tracked but should be in .gitignore for libraries)
   - `*.jl.cov`, `*.jl.*.cov`, `*.jl.mem`
   - `/docs/build/`
   - `/generated/`
   - `*.smt2` (temp solver files)
3. Create `.machine_readable/STATE.scm` reflecting actual project state:
   phase=implementation, maturity=alpha, ~55% completion.
4. Create `.machine_readable/ECOSYSTEM.scm` with relationship to Axiom.jl
   (as noted in README.adoc line 85).
5. Create `.machine_readable/META.scm` with architecture decisions and license
   info.
6. Fix `AI.a2ml` to reference `.machine_readable/` (not `.machines_readable/6scm/`).

**Verification:**
```bash
cd /var/mnt/eclipse/repos/SMTLib.jl
test -f .editorconfig && echo "PASS: .editorconfig exists"
test -f .gitignore && echo "PASS: .gitignore exists"
test -f .machine_readable/STATE.scm && echo "PASS: STATE.scm exists"
test -f .machine_readable/ECOSYSTEM.scm && echo "PASS: ECOSYSTEM.scm exists"
test -f .machine_readable/META.scm && echo "PASS: META.scm exists"
grep -c 'machines_readable' AI.a2ml
# Should return 0 (fixed to machine_readable)
echo "TASK 9 PASSED"
```

---

## TASK 10: Fix Test Coverage Gaps

**Files:**
- `/var/mnt/eclipse/repos/SMTLib.jl/test/runtests.jl`

**Problem:**
The test file has decent coverage of `to_smtlib`, type mapping, and value
parsing. But several functions have zero test coverage:
- `prove()` (line 689) -- never tested.
- `extract_variables()` (line 711) -- never tested.
- `is_operator()` (line 728) -- never tested.
- `handle_let()` (line 267) -- never tested.
- `handle_chained_comparison()` (line 250) -- never tested.
- `build_script()` (line 430) -- never directly tested.
- `build_solver_command()` (line 484) -- never tested.
- `@smt` macro (line 630) -- never tested without a solver.
- `reset!()` (line 411) -- never tested.
- `SMTContext` constructor error path (line 108-109) -- never tested.

Also, `to_smtlib` for `&&` and `||` (lines 227-232) uses `expr.head == :&&`
syntax which changed across Julia versions. The tests for `to_smtlib(:(x && y))`
may fail on Julia < 1.7 because the AST representation changed.

**What to do:**
1. Add a `@testset "Script Generation"` that calls `build_script` directly and
   checks the output string contains `(set-logic ...)`, `(declare-const ...)`,
   `(assert ...)`, and `(check-sat)`.
2. Add a `@testset "Solver Commands"` that tests `build_solver_command` for
   z3, cvc5, yices, and unknown solvers.
3. Add a `@testset "Variable Extraction"` testing `extract_variables`.
4. Add a `@testset "Operator Detection"` testing `is_operator`.
5. Add a `@testset "Context Reset"` testing `reset!`.
6. Add a `@testset "Chained Comparison"` testing `handle_chained_comparison`
   with expressions like `:(1 < x < 10)`.
7. Add a `@testset "Let Bindings"` testing `handle_let`.
8. Add a `@testset "Macro Generation"` that tests `@smt` structurally without
   needing a solver (mock or check generated script).

**Verification:**
```julia
julia --project=/var/mnt/eclipse/repos/SMTLib.jl -e '
using Pkg; Pkg.test()
' 2>&1 | tail -5
# Should show all tests passing with expanded coverage
```

---

## TASK 11: Fix the `prove()` Function Type Inference

**Files:**
- `/var/mnt/eclipse/repos/SMTLib.jl/src/SMTLib.jl` (function `prove` at line 689, `extract_variables` at line 711)

**Problem:**
The `prove()` function calls `extract_variables()` which defaults every
variable to `Int` (line 720: `vars[expr] = Int`). This means:
- `prove(:(x > 0 && x < 10))` declares `x` as Int but also declares `&&` and
  `>` and `<` as Int variables (they pass `is_operator` but the check is flawed).
- Actually, `is_operator` checks for `Symbol("+")` etc., but operators in Julia
  ASTs are stored as `:+`, `:>`, `:&&` -- the `Symbol("...")` constructor creates
  them. However, `expr.args[1]` in a `:call` Expr is the operator symbol, so
  `_extract_vars!` recurses into `expr.args` including the operator. It would
  declare `:+` as a variable unless `is_operator` catches it.
- The `is_operator` list is incomplete: missing `:implies`, `:iff`, `:xor`,
  `:forall`, `:exists`, `:select`, `:store`, all `:bv*` operators, math
  functions (`:sqrt`, `:exp`, `:log`, `:sin`, `:cos`, `:tan`, `:^`).
- The comment on line 713 says "Simple heuristic - would need type inference
  in practice" which is honest, but the function is exported and documented.

**What to do:**
1. Expand `is_operator` to include all operators from `julia_op_to_smt`'s
   keys (lines 287-351). Extract the keys from the Dict and check membership.
   Better yet, define the operator set once and share it.
2. Add a `type` parameter to `prove()` so users can specify variable types:
   `prove(expr; vars=Dict(:x => Int, :y => Float64))`.
3. When `vars` is provided, skip `extract_variables` and use the provided dict.
4. Document the limitations of `extract_variables` in its docstring.

**Verification:**
```julia
julia --project=/var/mnt/eclipse/repos/SMTLib.jl -e '
using SMTLib

# Test that operators are not extracted as variables
vars = SMTLib.extract_variables(:(x + y > 0))
@assert !haskey(vars, :+) "Operator + should not be a variable"
@assert !haskey(vars, :>) "Operator > should not be a variable"
@assert haskey(vars, :x) "x should be extracted"
@assert haskey(vars, :y) "y should be extracted"
@assert !haskey(vars, 0) || true  # literals should not be variables

println("TASK 11 PASSED")
'
```

---

## TASK 12: Fix parse_model Regex (Misses Multi-line Models)

**Files:**
- `/var/mnt/eclipse/repos/SMTLib.jl/src/SMTLib.jl` (function `parse_model` at line 539, line 544)

**Problem:**
The model parser on line 544 uses a single-line regex:
```
r"\(define-fun\s+(\w+)\s+\(\)\s+\w+\s+(.+?)\)"
```
This fails for multi-line model output, which is the default format from Z3
and CVC5. A typical Z3 model looks like:
```
(model
  (define-fun x () Int
    5)
  (define-fun y () Int
    (- 3))
)
```
The regex requires the entire `define-fun` to be on one line. It also fails to
match type names with spaces like `(_ BitVec 32)`.

**What to do:**
1. Replace the single-line regex with a proper S-expression parser that handles
   nested parentheses and multi-line output.
2. At minimum, use `s` flag for dotall matching and handle the multi-line case:
   `r"\(define-fun\s+(\w+)\s+\(\)\s+[\w\s\(\)_]+\s+(.+?)\)"s`.
3. Better approach: write a minimal S-expression tokenizer that finds balanced
   `(define-fun ...)` blocks, then extracts name, type, and value.
4. Handle type names like `(_ BitVec 32)` and `(Array Int Int)`.

**Verification:**
```julia
julia --project=/var/mnt/eclipse/repos/SMTLib.jl -e '
using SMTLib

# Multi-line model output from Z3
output = """sat
(model
  (define-fun x () Int
    5)
  (define-fun y () Int
    (- 3))
)"""

model = SMTLib.parse_model(output)
@assert haskey(model, :x) "Should parse x from multi-line model"
@assert haskey(model, :y) "Should parse y from multi-line model"
@assert model[:x] == 5 "x should be 5"
@assert model[:y] == -3 "y should be -3"

println("TASK 12 PASSED")
'
```

---

## TASK 13: Fix Timeout Detection in parse_result

**Files:**
- `/var/mnt/eclipse/repos/SMTLib.jl/src/SMTLib.jl` (function `parse_result` at line 503, line 519)

**Problem:**
Line 519 checks `startswith(line, "timeout")` but:
- Z3 does not output "timeout" -- it outputs "unknown" with `(:reason-unknown timeout)` in a separate get-info response, or just outputs nothing/errors.
- CVC5 outputs "unknown" on timeout, not "timeout".
- Yices outputs "unknown" on timeout.
- No major solver outputs a bare "timeout" string.

The timeout detection is effectively dead code. Real timeout detection requires
checking the process exit code or using Julia's `timedwait`/`Timer`.

**What to do:**
1. Use Julia's `Base.process_running` and timeout mechanism. Wrap the solver
   process in a `Timer` that kills it after `timeout_ms`.
2. In `run_solver`, use `open(cmd)` with a timer instead of `read(ignorestatus(cmd))`.
3. If the process is killed due to timeout, set status to `:timeout`.
4. Remove the bogus `startswith(line, "timeout")` check.
5. Also handle solver stderr output (currently ignored) which may contain error
   messages.

**Verification:**
```julia
julia --project=/var/mnt/eclipse/repos/SMTLib.jl -e '
using SMTLib

# Test that a very short timeout results in :timeout (if solver available)
solvers = available_solvers()
if !isempty(solvers)
    ctx = SMTContext(solver=first(solvers), logic=:QF_NRA, timeout_ms=1)
    declare(ctx, :x, Float64)
    # Intentionally hard problem
    for i in 1:100
        assert!(ctx, :(x * x * x + x > $i))
    end
    result = check_sat(ctx)
    @assert result.status in (:timeout, :unknown, :sat, :unsat) "Should handle timeout gracefully"
    println("Solver timeout test: status = $(result.status)")
else
    println("No solver available, skipping timeout test")
end

# Test that parse_result handles empty output gracefully
r = SMTLib.parse_result("")
@assert r.status == :unknown "Empty output should be :unknown"

println("TASK 13 PASSED")
'
```

---

## TASK 14: Add .gitignore and Remove Tracked Manifest.toml

**Files:**
- `/var/mnt/eclipse/repos/SMTLib.jl/Manifest.toml` (currently tracked)
- `/var/mnt/eclipse/repos/SMTLib.jl/.gitignore` (does not exist)

**Problem:**
`Manifest.toml` is tracked in git. For Julia libraries (not applications),
`Manifest.toml` should NOT be committed because it pins exact dependency
versions and Julia versions (`julia_version = "1.12.2"` on line 3) that will
break for users on different Julia versions. The Julia community convention
is clear: libraries should `.gitignore` their `Manifest.toml`.

**What to do:**
1. Create `.gitignore` with standard Julia entries:
   ```
   /Manifest.toml
   *.jl.cov
   *.jl.*.cov
   *.jl.mem
   /docs/build/
   /generated/
   *.smt2
   ```
2. Remove `Manifest.toml` from git tracking: `git rm --cached Manifest.toml`.
3. Ensure `Manifest.toml` remains on disk (not deleted, just untracked).

**Verification:**
```bash
cd /var/mnt/eclipse/repos/SMTLib.jl
test -f .gitignore && echo "PASS: .gitignore exists"
grep 'Manifest.toml' .gitignore && echo "PASS: Manifest.toml in .gitignore"
git ls-files --error-unmatch Manifest.toml 2>/dev/null && echo "FAIL: still tracked" || echo "PASS: Manifest.toml untracked"
```

---

## TASK 15: Fix quality.yml TODO Scanner (Wrong File Extensions)

**Files:**
- `/var/mnt/eclipse/repos/SMTLib.jl/.github/workflows/quality.yml` (line 31)

**Problem:**
The quality workflow scans for TODOs in `*.rs`, `*.res`, `*.py`, `*.ex` files.
This is a Julia repository. It should scan `*.jl` files. None of the scanned
extensions exist in the repo.

**What to do:**
1. Change line 31 to include `*.jl` instead of/in addition to the current
   extensions:
   ```
   grep -rn "TODO\|FIXME\|HACK\|XXX" --include="*.jl" --include="*.idr" --include="*.zig" . | head -20 || echo "None found"
   ```

**Verification:**
```bash
grep 'include.*\.jl' /var/mnt/eclipse/repos/SMTLib.jl/.github/workflows/quality.yml && echo "TASK 15 PASSED"
```

---

## FINAL VERIFICATION

After completing all tasks, run this sequence:

```bash
cd /var/mnt/eclipse/repos/SMTLib.jl

echo "=== 1. Julia tests pass ==="
julia --project=. -e 'using Pkg; Pkg.test()' 2>&1 | tail -10

echo ""
echo "=== 2. No remaining template placeholders (excluding contractiles) ==="
grep -rn '{{' --include='*.md' --include='*.adoc' --include='*.idr' --include='*.zig' . | grep -v '.git/' | grep -v 'contractiles/' | wc -l
# Expected: 0

echo ""
echo "=== 3. All exports have implementations ==="
julia --project=. -e '
using SMTLib
for name in names(SMTLib)
    sym = getfield(SMTLib, name)
    println("  $name => $(typeof(sym))")
end
'

echo ""
echo "=== 4. No AGPL references in source ==="
grep -rn 'AGPL' --include='*.jl' --include='*.idr' --include='*.zig' . | grep -v '.git/' | wc -l
# Expected: 0

echo ""
echo "=== 5. RSR infrastructure files exist ==="
for f in .editorconfig .gitignore .machine_readable/STATE.scm .machine_readable/ECOSYSTEM.scm .machine_readable/META.scm; do
    test -f "$f" && echo "  OK: $f" || echo "  MISSING: $f"
done

echo ""
echo "=== 6. CodeQL scans correct language ==="
grep 'language:' .github/workflows/codeql.yml

echo ""
echo "=== 7. All workflow actions SHA-pinned ==="
grep -E 'uses:.*@v[0-9]' .github/workflows/*.yml | wc -l
# Expected: 0

echo ""
echo "=== 8. Examples are Julia files ==="
ls examples/*.jl 2>/dev/null && echo "OK" || echo "MISSING Julia examples"
ls examples/*.res 2>/dev/null && echo "FAIL: ReScript still present" || echo "OK: no ReScript"

echo ""
echo "=== 9. Manifest.toml not tracked ==="
git ls-files --error-unmatch Manifest.toml 2>/dev/null && echo "FAIL" || echo "OK"

echo ""
echo "=== FINAL VERDICT ==="
echo "If all above show OK/0/expected values, all tasks are complete."
```
