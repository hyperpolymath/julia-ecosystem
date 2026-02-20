# SONNET-TASKS.md -- Axiology.jl Completion Tasks

> **Generated:** 2026-02-12 by Opus audit
> **Purpose:** Unambiguous instructions for Sonnet to complete all stubs, TODOs, and placeholder code.
> **Honest completion before this file:** 35%

The README claims "Production Ready - 1,088 lines of code, 45/45 tests passing" and STATE.scm
claims `overall-completion 100`. Both are FALSE. The source files contain critical duplicate
orphan code blocks after every function in `fairness.jl` and several in `welfare.jl` that will
cause Julia parse errors at module load time. The test suite cannot possibly pass in its current
state. Additionally, four API functions advertised in the README do not exist, the examples
directory contains unrelated files, and `verify_value` is a trivial stub.

---

## GROUND RULES FOR SONNET

1. Read this entire file before starting any task.
2. Do tasks in order listed. Earlier tasks unblock later ones.
3. After each task, run the verification command. If it fails, fix before moving on.
4. Do NOT mark done unless verification passes.
5. Update STATE.scm with honest completion percentages after each task.
6. Commit after each task: `fix(component): complete <description>`
7. Run full test suite after every 3 tasks: `cd /var/mnt/eclipse/repos/Axiology.jl && julia --project=. -e 'using Pkg; Pkg.test()'`

---

## TASK 1: Remove duplicate orphan code blocks in fairness.jl (CRITICAL)

**Files:** `/var/mnt/eclipse/repos/Axiology.jl/src/fairness.jl`

**Problem:** Every function in this file has a complete, documented implementation that ends
with `end`, immediately followed by an orphan duplicate of the function body (without the
`function` declaration). In Julia, code after `end` at top level is executed at load time,
causing errors. There are 6 orphan blocks:

- Lines 101-119: orphan duplicate of `demographic_parity` body
- Lines 201-238: orphan duplicate of `equalized_odds` body
- Lines 313-335: orphan duplicate of `equal_opportunity` body
- Lines 406-424: orphan duplicate of `disparate_impact` body
- Lines 502-518: orphan duplicate of `individual_fairness` body (also hardcodes 0.8 instead of using `similarity_threshold` parameter)
- Lines 616-642: orphan duplicate of `satisfy(::Fairness, ::Dict)` body (also lacks `individual_fairness` metric support)

Also: duplicate SPDX header on lines 1-2 and 4-5.

**What to do:**
1. Delete line 4 (`# SPDX-License-Identifier: PMPL-1.0-or-later`) and line 5 (`# Copyright (c) 2026 Jonathan D.A. Jewell <jonathan.jewell@open.ac.uk>`) -- these are duplicates of lines 1-2.
2. Delete lines 101-119 (orphan `demographic_parity` body after the real function's `end` on line 100).
3. Delete lines 201-238 (orphan `equalized_odds` body after the real function's `end` on line 200).
4. Delete lines 313-335 (orphan `equal_opportunity` body after the real function's `end` on line 312).
5. Delete lines 406-424 (orphan `disparate_impact` body after the real function's `end` on line 405).
6. Delete lines 502-518 (orphan `individual_fairness` body after the real function's `end` on line 501).
7. Delete lines 616-642 (orphan `satisfy` body after the real function's `end` on line 615).
8. IMPORTANT: Delete from highest line numbers first to avoid line number shifts.
9. After deletion, verify there are exactly 6 `function` declarations and 1 `satisfy` method in the file.

**Verification:**
```julia
cd("/var/mnt/eclipse/repos/Axiology.jl")
using Pkg; Pkg.activate(".")
# This must not error:
include("src/types.jl")
include("src/fairness.jl")
# Verify functions work:
using Statistics
@assert demographic_parity([1,0,1,0], [:a,:b,:a,:b]) == 0.0
@assert equalized_odds([1,0,1,0], [1,0,1,0], [:a,:b,:a,:b]) == 0.0
@assert equal_opportunity([1,0,1,0], [1,0,1,0], [:a,:b,:a,:b]) == 0.0
@assert disparate_impact([1,1,0,0], [:a,:b,:a,:b]) == 1.0
println("TASK 1 PASSED")
```

---

## TASK 2: Remove duplicate orphan code blocks in welfare.jl (CRITICAL)

**Files:** `/var/mnt/eclipse/repos/Axiology.jl/src/welfare.jl`

**Problem:** Three functions have orphan duplicate bodies after their `end`:

- Lines 123-124: orphan duplicate of `rawlsian_welfare` body (just `return minimum(utilities)` and `end`)
- Lines 179-180: orphan duplicate of `egalitarian_welfare` body (just `return -var(utilities)` and `end`)
- Lines 241-257: orphan duplicate of `satisfy(::Welfare, ::Dict)` body

Also: duplicate SPDX header on lines 1-2 and 4-5.

**What to do:**
1. Delete line 4 and line 5 (duplicate SPDX header).
2. Delete lines 241-257 (orphan `satisfy(Welfare)` body after the real function's `end` on line 240).
3. Delete lines 179-180 (orphan `egalitarian_welfare` body after the real function's `end` on line 178).
4. Delete lines 123-124 (orphan `rawlsian_welfare` body after the real function's `end` on line 122).
5. Delete from highest line numbers first.

**Verification:**
```julia
cd("/var/mnt/eclipse/repos/Axiology.jl")
using Pkg; Pkg.activate(".")
include("src/types.jl")
include("src/fairness.jl")  # Must work after Task 1
include("src/welfare.jl")
# Verify functions work:
@assert utilitarian_welfare([10.0, 8.0, 12.0]) == 30.0
@assert rawlsian_welfare([10.0, 8.0, 12.0]) == 8.0
@assert egalitarian_welfare([10.0, 10.0, 10.0]) == 0.0
println("TASK 2 PASSED")
```

---

## TASK 3: Remove duplicate SPDX header in optimization.jl (LOW)

**Files:** `/var/mnt/eclipse/repos/Axiology.jl/src/optimization.jl`

**Problem:** Duplicate SPDX header on lines 1-2 and 4-5.

**What to do:**
1. Delete line 4 (`# SPDX-License-Identifier: PMPL-1.0-or-later`) and line 5 (`# Copyright (c) 2026 Jonathan D.A. Jewell <jonathan.jewell@open.ac.uk>`).

**Verification:**
```julia
cd("/var/mnt/eclipse/repos/Axiology.jl")
using Pkg; Pkg.activate(".")
# Full module load test:
using Axiology
@assert Axiology.value_score(Fairness(metric=:demographic_parity, threshold=0.1),
    Dict(:predictions => [1,0,1,0], :protected => [:a,:b,:a,:b])) == 1.0
println("TASK 3 PASSED")
```

---

## TASK 4: Verify full module loads and tests pass (CRITICAL)

**Files:** All `src/*.jl` and `test/runtests.jl`

**Problem:** After Tasks 1-3, the module should load cleanly. Run the full test suite to confirm
the 45 tests listed in `test/runtests.jl` all pass.

**What to do:**
1. Run `julia --project=. -e 'using Pkg; Pkg.test()'` from the repo root.
2. If any tests fail, fix them. The test expectations should match the documented (non-orphan) implementations.
3. If there are remaining load errors, track down and fix any additional orphan code.

**Verification:**
```julia
cd("/var/mnt/eclipse/repos/Axiology.jl")
using Pkg; Pkg.activate(".")
Pkg.test()
# All tests must pass with 0 failures, 0 errors.
println("TASK 4 PASSED")
```

---

## TASK 5: Remove unrelated example files (MEDIUM)

**Files:**
- `/var/mnt/eclipse/repos/Axiology.jl/examples/SafeDOMExample.res`
- `/var/mnt/eclipse/repos/Axiology.jl/examples/web-project-deno.json`

**Problem:** `SafeDOMExample.res` is a ReScript file for DOM manipulation that has nothing to do
with Axiology.jl. It also uses an `AGPL-3.0-or-later` SPDX header, which violates the project's
license policy. `web-project-deno.json` is an unrelated Deno project config. Both are clearly
copy-paste artifacts from the RSR template or another project.

**What to do:**
1. Delete `/var/mnt/eclipse/repos/Axiology.jl/examples/SafeDOMExample.res`.
2. Delete `/var/mnt/eclipse/repos/Axiology.jl/examples/web-project-deno.json`.
3. Create a real example file at `/var/mnt/eclipse/repos/Axiology.jl/examples/basic_usage.jl` that demonstrates:
   - Creating Fairness, Welfare, Profit, Efficiency, and Safety values
   - Using `satisfy` to check value satisfaction
   - Using `maximize` to compute value scores
   - Using `pareto_frontier` for multi-objective optimization
   - Using `weighted_score` for aggregation
4. The example file must use `# SPDX-License-Identifier: PMPL-1.0-or-later` header.
5. The example must actually run: `julia --project=. examples/basic_usage.jl`

**Verification:**
```julia
# Run the example:
cd("/var/mnt/eclipse/repos/Axiology.jl")
include("examples/basic_usage.jl")
# Must complete without errors and print meaningful output.
println("TASK 5 PASSED")
```

---

## TASK 6: Fix STATE.scm to reflect honest completion (MEDIUM)

**Files:** `/var/mnt/eclipse/repos/Axiology.jl/.machine_readable/STATE.scm`

**Problem:** STATE.scm claims `overall-completion 100` and `phase "production-ready"`. This is
dishonest. After Tasks 1-4, the core library works, but:
- `verify_value` is a trivial stub (just reads `proof[:verified]`)
- `maximize(Efficiency)` returns hardcoded `1.0` for pareto/kaldor_hicks metrics
- No ML framework integration exists
- No formal verification integration exists
- README advertises 4 functions that don't exist
- `value_score` for Welfare uses hardcoded normalization assumptions

**What to do:**
1. Change `overall-completion` from `100` to `65`.
2. Change `phase` from `"production-ready"` to `"alpha"`.
3. Update the `Type Definitions` component: change `completion` from `10` to `100` and `status` from `"minimal"` to `"complete"` (it IS complete after Tasks 1-2).
4. Update `Julia Implementation` component: change `completion` from `100` to `70`, change `description` to note that core functions work but `verify_value` is a stub, `maximize(Efficiency)` has placeholder returns, and normalization in `value_score` uses hardcoded assumptions.
5. Add a new component for "ML Integration" with `status "not-started"` and `completion 0`.
6. Add a new component for "Formal Verification" with `status "stub"` and `completion 5` (only `verify_value` exists as a trivial wrapper).
7. Update `updated` date to `"2026-02-12"`.

**Verification:**
```bash
# Check that STATE.scm is valid Scheme and contains updated values:
grep 'overall-completion 65' /var/mnt/eclipse/repos/Axiology.jl/.machine_readable/STATE.scm
grep 'phase "alpha"' /var/mnt/eclipse/repos/Axiology.jl/.machine_readable/STATE.scm
echo "TASK 6 PASSED"
```

---

## TASK 7: Fix README.adoc false claims (MEDIUM)

**Files:** `/var/mnt/eclipse/repos/Axiology.jl/README.adoc`

**Problem:** Multiple false claims and advertised-but-nonexistent API:

- Line 10: Claims "Production Ready" -- should say "Alpha" after Tasks 1-4
- Line 174: Documents `select_solution(solutions, preference)` -- this function DOES NOT EXIST
- Line 179: Documents `echidna_verify(value, system)` -- this function DOES NOT EXIST
- Line 182: Documents `fairlearn_constraint(value)` -- this function DOES NOT EXIST
- Line 185: Documents `flux_loss(value)` -- this function DOES NOT EXIST
- Lines 86-93: Example uses `train(algorithm, data, constraints=[fairness])` -- no such function
- Lines 104-113: Example uses `select_solution(solutions, preference=:balanced)` -- no such function
- Lines 119-138: Example uses `prove`, `Verified`, `verification.counterexample` -- none exist
- Lines 238-241: Development status checkboxes are wrong (Pareto IS implemented, marked unchecked)
- Lines 269-274: Claims dual MIT/PMPL license but project actually uses PMPL-1.0-or-later only

**What to do:**
1. Change line 10 status to: `> **Status**: Alpha - Core value system functional, integration APIs planned`
2. Remove the `=== Integration APIs` section (lines 177-186) entirely. These functions do not exist.
3. Fix the Usage Examples to use actual API signatures that exist in the codebase. Replace the
   fake `train()`, `select_solution()`, `prove()`, `Verified` examples with real working examples
   using `satisfy`, `maximize`, `value_score`, `weighted_score`, `pareto_frontier`.
4. Fix the Development Status checkboxes:
   - Check `[x] Multi-objective optimization` (Pareto frontier IS implemented)
   - Keep `[ ] Formal verification integration with ECHIDNA` unchecked
   - Keep `[ ] ML fairness library integration` unchecked
   - Keep `[ ] Comprehensive documentation` unchecked
5. Fix the License section: Remove mention of MIT. The license is PMPL-1.0-or-later per the
   LICENSE file and SPDX headers.
6. Update the ECHIDNA Integration section to clearly state it is PLANNED, not implemented.

**Verification:**
```bash
# Verify no mention of nonexistent functions:
! grep -q 'select_solution' /var/mnt/eclipse/repos/Axiology.jl/README.adoc
! grep -q 'echidna_verify' /var/mnt/eclipse/repos/Axiology.jl/README.adoc
! grep -q 'fairlearn_constraint' /var/mnt/eclipse/repos/Axiology.jl/README.adoc
! grep -q 'flux_loss' /var/mnt/eclipse/repos/Axiology.jl/README.adoc
! grep -q 'Production Ready' /var/mnt/eclipse/repos/Axiology.jl/README.adoc
echo "TASK 7 PASSED"
```

---

## TASK 8: Fix ROADMAP.adoc incorrect checkboxes (LOW)

**Files:** `/var/mnt/eclipse/repos/Axiology.jl/ROADMAP.adoc`

**Problem:** Line 14 marks "Multi-objective optimization (Pareto frontiers)" as unchecked `[ ]`,
but `pareto_frontier`, `dominated`, `value_score`, `weighted_score`, and `normalize_scores` are
all implemented in `optimization.jl`. Lines 61-65 mark all fairness metrics as unchecked, but
`demographic_parity`, `equalized_odds`, `equal_opportunity`, `disparate_impact`, and
`individual_fairness` are all implemented in `fairness.jl`.

**What to do:**
1. Line 14: Change `* [ ] Multi-objective optimization (Pareto frontiers)` to `* [x] Multi-objective optimization (Pareto frontiers)`
2. Lines 61-65: Check all five fairness metrics that are implemented:
   - `* [x] Demographic parity (group fairness)`
   - `* [x] Equalized odds (conditional fairness)`
   - `* [ ] Predictive parity (calibration)` (keep unchecked, not implemented)
   - `* [x] Individual fairness (Lipschitz continuity)`
   - `* [ ] Counterfactual fairness (causal)` (keep unchecked, not implemented)
3. Under v0.2.0, check "Implement Pareto frontier algorithm for value tradeoffs".

**Verification:**
```bash
grep '\[x\] Multi-objective optimization' /var/mnt/eclipse/repos/Axiology.jl/ROADMAP.adoc
grep '\[x\] Demographic parity' /var/mnt/eclipse/repos/Axiology.jl/ROADMAP.adoc
grep '\[x\] Equalized odds' /var/mnt/eclipse/repos/Axiology.jl/ROADMAP.adoc
grep '\[x\] Individual fairness' /var/mnt/eclipse/repos/Axiology.jl/ROADMAP.adoc
echo "TASK 8 PASSED"
```

---

## TASK 9: Implement real verify_value logic (MEDIUM)

**Files:** `/var/mnt/eclipse/repos/Axiology.jl/src/welfare.jl` (lines 730-733)

**Problem:** `verify_value(value::Value, proof::Dict)::Bool` is a trivial stub that only reads
`proof[:verified]`. It does no actual verification. The docstring (lines 694-729) describes
checking for a `:prover` key and `:details` key, but the function ignores them entirely.

**What to do:**
1. Expand `verify_value` to:
   - Require `proof[:verified]` to be a `Bool` (error if missing or wrong type)
   - If `value isa Safety` and `value.critical == true`, also require that `proof` contains
     a `:prover` key (it should be verified by a named prover, not just asserted)
   - Log/return the prover name and details if present
   - Return `proof[:verified]` as before for the boolean result, but with the added validation
2. Add a `verify_value` method specifically for `Safety` that enforces critical safety proofs
   must have a `:prover` field.
3. Add tests in `test/runtests.jl` for the new validation behavior:
   - Test that a critical Safety value with no `:prover` in proof returns `false` or errors
   - Test that a non-critical Safety value without `:prover` still works
   - Test that other Value types work as before

**Verification:**
```julia
cd("/var/mnt/eclipse/repos/Axiology.jl")
using Pkg; Pkg.activate(".")
using Axiology

# Non-critical safety: no prover needed
s_noncrit = Safety(invariant="test", critical=false)
@assert verify_value(s_noncrit, Dict(:verified => true)) == true

# Critical safety: prover should be required
s_crit = Safety(invariant="test", critical=true)
@assert verify_value(s_crit, Dict(:verified => true, :prover => :Lean)) == true

# Other value types still work:
f = Fairness(metric=:demographic_parity, threshold=0.05)
@assert verify_value(f, Dict(:verified => true)) == true
@assert verify_value(f, Dict(:verified => false)) == false

println("TASK 9 PASSED")
```

---

## TASK 10: Improve maximize(Efficiency) for pareto/kaldor_hicks (LOW)

**Files:** `/var/mnt/eclipse/repos/Axiology.jl/src/welfare.jl` (lines 560-574)

**Problem:** `maximize(value::Efficiency, initial_state::Dict)` returns a hardcoded `1.0` for
`:pareto` and `:kaldor_hicks` metrics (line 570). The docstring on line 540 explicitly
acknowledges this is a "placeholder". For `:pareto`, it should return `1.0` if
`state[:is_pareto_efficient]` is `true` and `0.0` otherwise. For `:kaldor_hicks`, it should
return `state[:net_gain]`.

**What to do:**
1. For `:pareto` metric: read `initial_state[:is_pareto_efficient]` and return `1.0` if true, `0.0` if false. Error if key is missing.
2. For `:kaldor_hicks` metric: read `initial_state[:net_gain]` and return it directly. Error if key is missing.
3. Update the docstring to remove the "placeholder" language.
4. Add tests:
   - `maximize(Efficiency(metric=:pareto), Dict(:is_pareto_efficient => true))` returns `1.0`
   - `maximize(Efficiency(metric=:pareto), Dict(:is_pareto_efficient => false))` returns `0.0`
   - `maximize(Efficiency(metric=:kaldor_hicks, target=100.0), Dict(:net_gain => 150.0))` returns `150.0`

**Verification:**
```julia
cd("/var/mnt/eclipse/repos/Axiology.jl")
using Pkg; Pkg.activate(".")
using Axiology

@assert maximize(Efficiency(metric=:pareto), Dict(:is_pareto_efficient => true)) == 1.0
@assert maximize(Efficiency(metric=:pareto), Dict(:is_pareto_efficient => false)) == 0.0
@assert maximize(Efficiency(metric=:kaldor_hicks, target=100.0), Dict(:net_gain => 150.0)) == 150.0
@assert maximize(Efficiency(metric=:computation_time), Dict(:computation_time => 0.5)) == -0.5

println("TASK 10 PASSED")
```

---

## TASK 11: Add equal_opportunity to test suite (LOW)

**Files:** `/var/mnt/eclipse/repos/Axiology.jl/test/runtests.jl`

**Problem:** The test suite tests `demographic_parity`, `disparate_impact`, and `individual_fairness`
(via `satisfy`), but never directly tests `equalized_odds` or `equal_opportunity` metric
functions. `individual_fairness` is also only tested indirectly through `satisfy`. Add direct
tests for completeness.

**What to do:**
1. Add a `@testset "Equalized Odds"` block inside "Fairness Metrics" that:
   - Tests perfect equalized odds (same TPR and FPR across groups) returns 0.0
   - Tests a case with known unequal TPR/FPR and verifies disparity > 0.0
2. Add a `@testset "Equal Opportunity"` block inside "Fairness Metrics" that:
   - Tests perfect equal opportunity (same TPR across groups) returns 0.0
   - Tests a case with unequal TPR
3. Add a `@testset "Individual Fairness"` block inside "Fairness Metrics" that:
   - Tests with a similarity matrix where similar individuals get similar predictions (returns ~0.0)
   - Tests with a similarity matrix where similar individuals get different predictions (returns > 0)

**Verification:**
```julia
cd("/var/mnt/eclipse/repos/Axiology.jl")
using Pkg; Pkg.activate(".")
Pkg.test()
# Verify the new testsets appear in output and pass
println("TASK 11 PASSED")
```

---

## TASK 12: Add edge case tests (LOW)

**Files:** `/var/mnt/eclipse/repos/Axiology.jl/test/runtests.jl`

**Problem:** No tests for edge cases or error conditions:
- Empty vectors
- Single-group protected attributes
- Zero-weight values in weighted_score
- Invalid metric symbols (should error)
- Missing state keys (should error)

**What to do:**
1. Add a `@testset "Edge Cases"` block with subtests for:
   - `demographic_parity` with single group returns 0.0
   - `disparate_impact` with single group returns 1.0
   - `utilitarian_welfare` with empty vector returns 0.0
   - `rawlsian_welfare` with empty vector throws error
   - `egalitarian_welfare` with single element returns 0.0
   - `normalize_scores` with identical scores returns all 1.0
   - `normalize_scores` with empty vector throws ArgumentError
   - `Fairness` constructor with invalid metric throws AssertionError
   - `Safety` constructor with empty invariant throws AssertionError
   - `weighted_score` with all zero weights returns 0.0
   - `satisfy(Fairness, Dict())` without `:predictions` throws error

**Verification:**
```julia
cd("/var/mnt/eclipse/repos/Axiology.jl")
using Pkg; Pkg.activate(".")
Pkg.test()
println("TASK 12 PASSED")
```

---

## FINAL VERIFICATION

After all tasks are complete, run the following sequence:

```bash
cd /var/mnt/eclipse/repos/Axiology.jl

# 1. Module loads without errors
julia --project=. -e 'using Axiology; println("Module loaded successfully")'

# 2. Full test suite passes
julia --project=. -e 'using Pkg; Pkg.test()'

# 3. No orphan code blocks remain (no bare @assert at top level outside functions)
julia --project=. -e '
    for f in ["src/fairness.jl", "src/welfare.jl", "src/optimization.jl"]
        content = read(f, String)
        # Count function declarations vs end statements
        funcs = count(r"^function ", content)
        println("$f: $funcs function declarations")
    end
    println("Source structure check passed")
'

# 4. No remaining stubs or placeholders
grep -rn "placeholder\|not implemented\|TODO\|FIXME\|HACK\|XXX\|STUB" src/ || echo "No stubs found"

# 5. No AGPL headers (wrong license)
grep -rn "AGPL" . --include="*.jl" --include="*.res" --include="*.json" || echo "No AGPL found"

# 6. Example runs
julia --project=. examples/basic_usage.jl

# 7. STATE.scm is honest
grep 'overall-completion 65' .machine_readable/STATE.scm && echo "STATE.scm is honest"

echo "ALL VERIFICATION PASSED"
```
