# SONNET-TASKS.md — Cliometrics.jl Completion Tasks

> **Generated:** 2026-02-12 by Opus audit
> **Purpose:** Unambiguous instructions for Sonnet to complete all stubs, TODOs, and placeholder code.
> **Honest completion before this file:** 35%

The Julia source code (`src/Cliometrics.jl`) has 7 implemented functions out of 11 exported symbols. Four exported functions have NO implementation at all: `interpolate_missing_years`, `quantify_institutions`, `counterfactual_scenario`, and `estimate_treatment_effect`. The README claims features (sigma-convergence, outlier detection, cross-country alignment, long-run trend analysis) that have zero code behind them. The entire RSR template layer (Idris2 ABI, Zig FFI, contractiles, SCM files) is uncustomized boilerplate with `{{PROJECT}}` placeholders throughout. The SCM directory is misspelled (`.machines_readable/6scm/` instead of `.machine_readable/`). Multiple files still use AGPL-3.0-or-later instead of PMPL-1.0-or-later.

---

## GROUND RULES FOR SONNET

1. Read this entire file before starting any task.
2. Do tasks in order listed. Earlier tasks unblock later ones.
3. After each task, run the verification command. If it fails, fix before moving on.
4. Do NOT mark done unless verification passes.
5. Update `.machines_readable/6scm/STATE.scm` with honest completion percentages after each task.
6. Commit after each task: `fix(component): complete <description>`
7. Run full test suite after every 3 tasks: `cd /var/mnt/eclipse/repos/Cliometrics.jl && julia --project=. -e 'using Pkg; Pkg.test()'`

---

## TASK 1: Implement `interpolate_missing_years` (CRITICAL)

**Files:** `/var/mnt/eclipse/repos/Cliometrics.jl/src/Cliometrics.jl`

**Problem:** The function `interpolate_missing_years` is exported on line 44 but has NO implementation anywhere in the codebase. Any call to it will throw `UndefVarError`.

**What to do:**
1. Add the function implementation after the `clean_historical_series` function (after line 361), before the `compare_historical_trajectories` function.
2. The function should accept a `DataFrame` with a `:year` column and a `variable::Symbol` column.
3. It should identify gaps in the year sequence (e.g., years 1950, 1952 missing 1951).
4. It should insert rows for missing years and linearly interpolate the specified variable's values.
5. Return the expanded DataFrame with no year gaps.
6. Add a proper docstring following the existing style (see lines 63-81 for reference).

**Implementation signature:**
```julia
function interpolate_missing_years(data::DataFrame, variable::Symbol; method::Symbol=:linear)
```

**Verification:**
```julia
using Cliometrics, DataFrames
df = DataFrame(year=[2000, 2002, 2005], gdp=[100.0, 110.0, 130.0])
result = interpolate_missing_years(df, :gdp)
@assert nrow(result) == 6 "Expected 6 rows (2000-2005), got $(nrow(result))"
@assert result.year == 2000:2005 "Years should be continuous 2000:2005"
@assert result.gdp[2] ≈ 105.0 atol=1e-6 "Year 2001 should interpolate to 105.0"
@assert result.gdp[4] ≈ (110.0 + (130.0-110.0)*1/3) atol=1e-6 "Year 2003 should interpolate correctly"
println("TASK 1 PASSED")
```

---

## TASK 2: Implement `quantify_institutions` (CRITICAL)

**Files:** `/var/mnt/eclipse/repos/Cliometrics.jl/src/Cliometrics.jl`

**Problem:** The function `quantify_institutions` is exported on line 52 but has NO implementation anywhere. This is distinct from `institutional_quality_index` which IS implemented (lines 285-308). `quantify_institutions` should provide a different capability: measuring institutional change over time, not just a static composite index.

**What to do:**
1. Add the function after `institutional_quality_index` (after line 308).
2. It should measure how institutional indicators change over time for a given entity (country/region).
3. Accept a panel DataFrame with `:year`, `:entity`, and multiple indicator columns.
4. For each entity, calculate: rate of institutional change per year, volatility of change, direction (improving/deteriorating).
5. Return a DataFrame with entity-level summary statistics.
6. Add a proper docstring.

**Implementation signature:**
```julia
function quantify_institutions(data::DataFrame, entity::Symbol, indicators::Vector{Symbol};
                               period::Union{Tuple{Int,Int},Nothing}=nothing)
```

**Verification:**
```julia
using Cliometrics, DataFrames, Statistics
df = DataFrame(
    year = repeat(2000:2004, 2),
    country = repeat(["A", "B"], inner=5),
    rule_of_law = [0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 0.78, 0.76, 0.74, 0.72],
    corruption = [0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.58, 0.55, 0.52, 0.50]
)
result = quantify_institutions(df, :country, [:rule_of_law, :corruption])
@assert nrow(result) == 2 "Should have 2 rows (one per country)"
@assert "country" in names(result) "Should have entity column"
@assert "avg_change_rate" in names(result) "Should have avg_change_rate column"
# Country A is improving (positive change), B is deteriorating (negative change)
row_a = result[result.country .== "A", :]
row_b = result[result.country .== "B", :]
@assert row_a.avg_change_rate[1] > 0 "Country A should show positive institutional change"
@assert row_b.avg_change_rate[1] < 0 "Country B should show negative institutional change"
println("TASK 2 PASSED")
```

---

## TASK 3: Implement `counterfactual_scenario` (CRITICAL)

**Files:** `/var/mnt/eclipse/repos/Cliometrics.jl/src/Cliometrics.jl`

**Problem:** The function `counterfactual_scenario` is exported on line 60 but has NO implementation. The README (line 16) claims "Counterfactual Modeling" as a feature. Zero code exists for it.

**What to do:**
1. Add the function before the closing `end # module Cliometrics` (before line 411).
2. It should create a counterfactual time series by modifying a parameter at a specific point in time.
3. Accept: the actual historical data, a `break_year` (when the counterfactual diverges), a `variable` to modify, and an `adjustment` (multiplicative factor or additive shift).
4. From `break_year` onward, apply the adjustment and propagate forward using the original growth rates.
5. Return a DataFrame with both actual and counterfactual series for comparison.
6. Add a proper docstring.

**Implementation signature:**
```julia
function counterfactual_scenario(data::DataFrame, variable::Symbol, break_year::Int;
                                 adjustment::Float64=1.0,
                                 method::Symbol=:multiplicative)
```

**Verification:**
```julia
using Cliometrics, DataFrames
df = DataFrame(year=2000:2004, gdp=[100.0, 110.0, 121.0, 133.1, 146.41])
result = counterfactual_scenario(df, :gdp, 2002, adjustment=0.9, method=:multiplicative)
@assert "actual" in names(result) "Should have actual column"
@assert "counterfactual" in names(result) "Should have counterfactual column"
@assert nrow(result) == 5 "Should have same number of rows"
@assert result.actual[1] ≈ 100.0 "Actual should be unchanged"
@assert result.counterfactual[1] ≈ 100.0 "Before break_year, counterfactual equals actual"
@assert result.counterfactual[2] ≈ 110.0 "Year 2001 (before break) unchanged"
@assert result.counterfactual[3] ≈ 121.0 * 0.9 atol=1e-6 "Break year gets adjustment"
# After break year, growth rates from actual applied to counterfactual base
println("TASK 3 PASSED")
```

---

## TASK 4: Implement `estimate_treatment_effect` (CRITICAL)

**Files:** `/var/mnt/eclipse/repos/Cliometrics.jl/src/Cliometrics.jl`

**Problem:** The function `estimate_treatment_effect` is exported on line 61 but has NO implementation. This is the second part of the counterfactual modeling feature claimed in the README.

**What to do:**
1. Add the function after `counterfactual_scenario`.
2. Implement a simple difference-in-differences (DiD) estimator, which is the standard cliometric method for estimating treatment effects in historical data.
3. Accept: a DataFrame with `:year`, a group indicator (`:treated` boolean), a `:variable` column, and a `treatment_year`.
4. Calculate the DiD estimate: (post_treated - pre_treated) - (post_control - pre_control).
5. Return a NamedTuple with the treatment effect, pre/post means, and a simple t-statistic.
6. Add a proper docstring referencing the DiD methodology.

**Implementation signature:**
```julia
function estimate_treatment_effect(data::DataFrame, variable::Symbol,
                                   group::Symbol, treatment_year::Int)
```

**Verification:**
```julia
using Cliometrics, DataFrames, Statistics
df = DataFrame(
    year = repeat(1990:1999, 2),
    country = repeat(["treated", "control"], inner=10),
    treated = repeat([true, false], inner=10),
    gdp = vcat(
        [100, 102, 104, 106, 108, 115, 120, 125, 130, 135],  # treated: jump at 1995
        [100, 102, 104, 106, 108, 110, 112, 114, 116, 118]   # control: steady
    ) .* 1.0
)
result = estimate_treatment_effect(df, :gdp, :treated, 1995)
@assert haskey(result, :treatment_effect) "Must return treatment_effect"
@assert result.treatment_effect > 0 "Treatment effect should be positive (treated grew faster)"
@assert haskey(result, :pre_treatment_diff) "Must return pre_treatment_diff"
@assert haskey(result, :post_treatment_diff) "Must return post_treatment_diff"
println("TASK 4 PASSED")
```

---

## TASK 5: Add tests for the four new functions (HIGH)

**Files:** `/var/mnt/eclipse/repos/Cliometrics.jl/test/runtests.jl`

**Problem:** The test file only tests the 7 originally implemented functions. The 4 new functions from Tasks 1-4 have no test coverage.

**What to do:**
1. Add a `@testset "Interpolate Missing Years"` block after the "Historical Series Cleaning" testset (after line 107).
2. Add a `@testset "Quantify Institutions"` block after the "Institutional Quality Index" testset (after line 93).
3. Add a `@testset "Counterfactual Scenario"` block after the "Compare Historical Trajectories" testset (after line 150).
4. Add a `@testset "Estimate Treatment Effect"` block after the counterfactual testset.
5. Each testset should have at least 3 `@test` assertions covering: normal case, edge case, and expected properties.
6. Use the verification code from Tasks 1-4 as a starting point but convert assertions to `@test` macros.

**Verification:**
```julia
cd("/var/mnt/eclipse/repos/Cliometrics.jl")
using Pkg; Pkg.test()
# All tests including the 4 new testsets must pass
```

---

## TASK 6: Fix SPDX license headers — replace AGPL-3.0-or-later with PMPL-1.0-or-later (HIGH)

**Files:**
- `/var/mnt/eclipse/repos/Cliometrics.jl/.machines_readable/6scm/STATE.scm` (line 1)
- `/var/mnt/eclipse/repos/Cliometrics.jl/.machines_readable/6scm/META.scm` (line 1)
- `/var/mnt/eclipse/repos/Cliometrics.jl/.machines_readable/6scm/ECOSYSTEM.scm` (line 1)
- `/var/mnt/eclipse/repos/Cliometrics.jl/.gitignore` (line 1)
- `/var/mnt/eclipse/repos/Cliometrics.jl/.gitattributes` (line 1)
- `/var/mnt/eclipse/repos/Cliometrics.jl/ffi/zig/build.zig` (line 2)
- `/var/mnt/eclipse/repos/Cliometrics.jl/ffi/zig/src/main.zig` (line 6)
- `/var/mnt/eclipse/repos/Cliometrics.jl/ffi/zig/test/integration_test.zig` (line 2)
- `/var/mnt/eclipse/repos/Cliometrics.jl/examples/SafeDOMExample.res` (line 1)
- `/var/mnt/eclipse/repos/Cliometrics.jl/docs/CITATIONS.adoc` (line 13, inside bibtex block)

**Problem:** These files use `AGPL-3.0-or-later` which is the OLD license. Per CLAUDE.md, the primary license is PMPL-1.0-or-later and AGPL-3.0 must NEVER be used.

**What to do:**
1. In each file listed above, replace `AGPL-3.0-or-later` with `PMPL-1.0-or-later`.
2. For `docs/CITATIONS.adoc` line 13, also update the bibtex `license` field value.
3. Do NOT change the SPDX headers in `src/Cliometrics.jl` or `test/runtests.jl` (they already use PMPL-1.0-or-later correctly).

**Verification:**
```bash
cd /var/mnt/eclipse/repos/Cliometrics.jl && grep -rn "AGPL-3.0" --include="*.scm" --include="*.zig" --include="*.res" --include="*.adoc" . | grep -v ".git/"
# Should return zero lines
```

---

## TASK 7: Replace `{{PROJECT}}` / `{{REPO}}` / `{{OWNER}}` / `{{FORGE}}` template placeholders (HIGH)

**Files:**
- `/var/mnt/eclipse/repos/Cliometrics.jl/src/abi/Types.idr` (lines 6, 7, 11)
- `/var/mnt/eclipse/repos/Cliometrics.jl/src/abi/Layout.idr` (lines 8, 10)
- `/var/mnt/eclipse/repos/Cliometrics.jl/src/abi/Foreign.idr` (lines 9, 11, 12, and all `{{project}}` on lines 23, 35, 49, 72, 77, 98, 125, 152, 164, 185, 211)
- `/var/mnt/eclipse/repos/Cliometrics.jl/ffi/zig/build.zig` (lines 1, 12, 23, 35, 36, 82)
- `/var/mnt/eclipse/repos/Cliometrics.jl/ffi/zig/src/main.zig` (lines 1, 12, and all `{{project}}_` function names)
- `/var/mnt/eclipse/repos/Cliometrics.jl/ffi/zig/test/integration_test.zig` (line 1 and all `{{project}}_` references)
- `/var/mnt/eclipse/repos/Cliometrics.jl/ABI-FFI-README.md` (all `{{PROJECT}}` and `{{project}}` occurrences)
- `/var/mnt/eclipse/repos/Cliometrics.jl/CODE_OF_CONDUCT.md` (lines 9, 10, 313)
- `/var/mnt/eclipse/repos/Cliometrics.jl/CONTRIBUTING.md` (lines 2, 3, 9, 10, 20, 89-92)
- `/var/mnt/eclipse/repos/Cliometrics.jl/SECURITY.md` (lines 9, 10, 43, 206, 325, 374, 386, 387)
- `/var/mnt/eclipse/repos/Cliometrics.jl/0-AI-MANIFEST.a2ml` (line 7, 56)

**Problem:** The entire RSR template layer was never customized. Every `{{PROJECT}}`, `{{project}}`, `{{OWNER}}`, `{{REPO}}`, and `{{FORGE}}` placeholder is still present, making the Idris2 ABI, Zig FFI, and community files non-functional.

**What to do:**
1. Replace `{{PROJECT}}` with `Cliometrics` (capitalized, for module/display names).
2. Replace `{{project}}` with `cliometrics` (lowercase, for C symbols and file names).
3. Replace `{{OWNER}}` with `hyperpolymath`.
4. Replace `{{REPO}}` with `Cliometrics.jl`.
5. Replace `{{FORGE}}` with `github.com`.
6. Replace `[YOUR-REPO-NAME]` with `Cliometrics.jl` in `0-AI-MANIFEST.a2ml`.
7. Replace `{{SECURITY_EMAIL}}` with `jonathan.jewell@open.ac.uk` in SECURITY.md if present.

**Verification:**
```bash
cd /var/mnt/eclipse/repos/Cliometrics.jl && grep -rn '{{PROJECT}}\|{{project}}\|{{OWNER}}\|{{REPO}}\|{{FORGE}}\|\[YOUR-REPO-NAME\]' . --include="*.idr" --include="*.zig" --include="*.md" --include="*.a2ml" --include="*.adoc" | grep -v ".git/" | grep -v "SONNET-TASKS"
# Should return zero lines
```

---

## TASK 8: Update SCM files to reflect Cliometrics.jl accurately (MEDIUM)

**Files:**
- `/var/mnt/eclipse/repos/Cliometrics.jl/.machines_readable/6scm/STATE.scm`
- `/var/mnt/eclipse/repos/Cliometrics.jl/.machines_readable/6scm/ECOSYSTEM.scm`
- `/var/mnt/eclipse/repos/Cliometrics.jl/.machines_readable/6scm/META.scm`

**Problem:** All three SCM files still reference `rsr-template-repo` (STATE.scm lines 5, 11, 12; ECOSYSTEM.scm lines 6, 7; META.scm line 5). STATE.scm claims 5% completion (line 22) and empty tech-stack (line 17). ECOSYSTEM.scm has `[TODO: Add specific description]` on line 24.

**What to do:**

1. **STATE.scm:**
   - Line 5: Change `rsr-template-repo` to `Cliometrics.jl`
   - Line 11: Change `"rsr-template-repo"` to `"Cliometrics.jl"`
   - Line 12: Change `"hyperpolymath/rsr-template-repo"` to `"hyperpolymath/Cliometrics.jl"`
   - Line 15: Change `"rsr-template-repo"` to `"Cliometrics.jl"`
   - Line 16: Set tagline to `"Quantitative economic history analysis in Julia"`
   - Line 17: Set tech-stack to `("Julia" "Statistics" "DataFrames" "CSV")`
   - Line 22: Update overall-completion to an honest percentage based on work done
   - Add working features list: `("load_historical_data" "calculate_growth_rates" "solow_residual" "decompose_growth" "convergence_analysis" "institutional_quality_index" "clean_historical_series" "compare_historical_trajectories")`

2. **ECOSYSTEM.scm:**
   - Line 6: Change name to `"Cliometrics.jl"`
   - Line 24: Replace `"[TODO: Add specific description]"` with `"A Julia library for quantitative economic history analysis, providing growth accounting, convergence testing, and institutional analysis tools."`

3. **META.scm:**
   - Line 5: Change `rsr-template-repo` to `Cliometrics.jl`

**Verification:**
```bash
cd /var/mnt/eclipse/repos/Cliometrics.jl && grep -c "rsr-template-repo" .machines_readable/6scm/STATE.scm .machines_readable/6scm/ECOSYSTEM.scm .machines_readable/6scm/META.scm
# All three should show 0
grep "TODO" .machines_readable/6scm/ECOSYSTEM.scm
# Should return nothing
```

---

## TASK 9: Update ROADMAP.adoc from template to project-specific content (MEDIUM)

**Files:** `/var/mnt/eclipse/repos/Cliometrics.jl/ROADMAP.adoc`

**Problem:** The ROADMAP.adoc is still the raw template text. Line 2 says `= YOUR Template Repo Roadmap`. All milestones are generic placeholders (`Core functionality`, `Basic documentation`). There is no mention of Cliometrics.jl or any of its actual features.

**What to do:**
1. Change the title on line 2 to `= Cliometrics.jl Roadmap`.
2. Update the current status section to reflect actual state: 7 core functions implemented, 4 pending (or complete after Tasks 1-4).
3. Replace v0.1.0 milestone items with actual Cliometrics.jl features:
   - Growth accounting (done)
   - Convergence analysis (done)
   - Institutional quality index (done)
   - Data cleaning and interpolation (done/in-progress)
   - Counterfactual modeling (done/in-progress)
4. Add a v0.2.0 milestone with planned features:
   - Sigma-convergence testing (claimed in README but not implemented)
   - Long-run growth trend analysis (claimed in README but not implemented)
   - Outlier detection and handling (claimed in README but not implemented)
   - Cross-country data alignment (claimed in README but not implemented)
5. Keep the SPDX header on line 1.

**Verification:**
```bash
cd /var/mnt/eclipse/repos/Cliometrics.jl && head -5 ROADMAP.adoc | grep -c "YOUR Template"
# Should return 0
grep -c "Cliometrics" ROADMAP.adoc
# Should return at least 1
```

---

## TASK 10: Update docs/CITATIONS.adoc from template to project-specific (MEDIUM)

**Files:** `/var/mnt/eclipse/repos/Cliometrics.jl/docs/CITATIONS.adoc`

**Problem:** The entire citations file references `rsr-template-repo` and uses author `Polymath, Hyper` instead of the correct `Jewell, Jonathan D.A.`. The year is 2025 instead of 2026. The license field says AGPL-3.0-or-later.

**What to do:**
1. Replace the title on line 1: `RSR-template-repo` to `Cliometrics.jl`.
2. Update the BibTeX block (lines 8-15):
   - `author`: `{Jewell, Jonathan D.A.}`
   - `title`: `{Cliometrics.jl: Quantitative Economic History in Julia}`
   - `year`: `{2026}`
   - `url`: `{https://github.com/hyperpolymath/Cliometrics.jl}`
   - `license`: `{PMPL-1.0-or-later}`
3. Update Harvard, OSCOLA, MLA, and APA sections similarly:
   - Author: `Jewell, J.D.A.` / `Jonathan D.A. Jewell`
   - Title: `Cliometrics.jl`
   - Year: `2026`
   - URL: `github.com/hyperpolymath/Cliometrics.jl`

**Verification:**
```bash
cd /var/mnt/eclipse/repos/Cliometrics.jl && grep -c "rsr-template-repo\|RSR-template-repo\|Polymath, Hyper" docs/CITATIONS.adoc
# Should return 0
grep -c "Cliometrics.jl" docs/CITATIONS.adoc
# Should return at least 4
grep -c "Jewell" docs/CITATIONS.adoc
# Should return at least 4
```

---

## TASK 11: Remove irrelevant RSR template example files (MEDIUM)

**Files:**
- `/var/mnt/eclipse/repos/Cliometrics.jl/examples/SafeDOMExample.res`
- `/var/mnt/eclipse/repos/Cliometrics.jl/examples/web-project-deno.json`

**Problem:** These are ReScript/Deno web project examples from the RSR template. They have nothing to do with a Julia cliometrics library. `SafeDOMExample.res` is a ReScript DOM mounting example. `web-project-deno.json` is a Deno configuration file. Neither is relevant.

**What to do:**
1. Delete both files.
2. Create a new `examples/growth_decomposition.jl` example file that demonstrates the core Cliometrics.jl workflow (loading data, calculating growth rates, decomposing growth, convergence analysis).
3. Add SPDX header `# SPDX-License-Identifier: PMPL-1.0-or-later` and author line.
4. The example should be runnable (use synthetic data since we have no bundled CSV files).

**Verification:**
```bash
cd /var/mnt/eclipse/repos/Cliometrics.jl && test ! -f examples/SafeDOMExample.res && test ! -f examples/web-project-deno.json && test -f examples/growth_decomposition.jl && echo "PASS" || echo "FAIL"
```
```julia
# Verify the example is valid Julia
include("/var/mnt/eclipse/repos/Cliometrics.jl/examples/growth_decomposition.jl")
println("TASK 11 PASSED")
```

---

## TASK 12: Fix `clean_historical_series` to handle `missing` values correctly (MEDIUM)

**Files:** `/var/mnt/eclipse/repos/Cliometrics.jl/src/Cliometrics.jl`

**Problem:** On line 329, `float.(data)` will fail on a `Vector{Union{Float64, Missing}}` because `float(missing)` throws a `MethodError`. The test on lines 97-98 passes `[100.0, 105.0, missing, 115.0, 120.0]` which creates a `Vector{Union{Float64, Missing}}`. The function signature on line 327 accepts `data::Vector` which includes this type, but `float.(data)` on line 329 cannot convert `missing` to a float.

Additionally, on line 334, `ismissing(cleaned[i])` after `float.(data)` is inconsistent -- if `float.` succeeded (which it would not with missing), the values would all be Float64 and `ismissing` would never be true.

**What to do:**
1. On line 329, replace `cleaned = float.(data)` with a version that preserves missing values:
   ```julia
   cleaned = Vector{Union{Float64,Missing}}(data)
   ```
2. On line 334, the `isnan` check should also handle the case where `cleaned[i]` is missing. Use `ismissing(cleaned[i]) || (!ismissing(cleaned[i]) && isnan(cleaned[i]))` or restructure the condition.
3. At the end (line 360), convert the result to `Vector{Float64}` by replacing any remaining missing values with NaN, or by requiring all missing values were filled.
4. Update the return type in the docstring (line 311) to clarify behavior.

**Verification:**
```julia
using Cliometrics
# Test with missing values
data = [100.0, 105.0, missing, 115.0, 120.0]
cleaned = clean_historical_series(data, method=:linear)
@assert length(cleaned) == 5
@assert !any(ismissing, cleaned) "No missing values should remain"
@assert cleaned[3] ≈ 110.0 atol=1e-6
# Test with NaN values
data2 = [100.0, 105.0, NaN, 115.0, 120.0]
cleaned2 = clean_historical_series(data2, method=:linear)
@assert !any(isnan, cleaned2) "No NaN values should remain"
# Test forward fill with missing
data3 = [100.0, missing, missing, 115.0, 120.0]
cleaned3 = clean_historical_series(data3, method=:forward_fill)
@assert cleaned3[2] ≈ 100.0
@assert cleaned3[3] ≈ 100.0
println("TASK 12 PASSED")
```

---

## TASK 13: Fix `compare_historical_trajectories` to handle `push!` correctly (LOW)

**Files:** `/var/mnt/eclipse/repos/Cliometrics.jl/src/Cliometrics.jl`

**Problem:** On line 389, `results = DataFrame()` creates an empty DataFrame with no columns. Then on line 398, `push!(results, (...))` tries to push a NamedTuple into an empty DataFrame. In DataFrames.jl, `push!` on an empty DataFrame with a NamedTuple does work (it creates columns from the NamedTuple field names), BUT only if the DataFrame truly has no columns. This is fragile and version-dependent. A more robust approach initializes the DataFrame with column types.

**What to do:**
1. Replace line 389 with a properly typed empty DataFrame:
   ```julia
   results = DataFrame(
       region = String[],
       initial_level = Float64[],
       final_level = Float64[],
       avg_growth = Float64[],
       std_growth = Float64[],
       cumulative_growth = Float64[]
   )
   ```
2. This makes the function robust across DataFrames.jl versions.

**Verification:**
```julia
using Cliometrics, DataFrames
data = DataFrame(
    year = repeat(1950:1960, 2),
    region = repeat(["Europe", "Asia"], inner=11),
    gdp_per_capita = vcat(
        [1000, 1100, 1210, 1331, 1464, 1610, 1771, 1948, 2143, 2357, 2593],
        [500, 525, 551, 579, 608, 638, 670, 703, 738, 775, 814]
    ) .* 1.0
)
result = compare_historical_trajectories(data, ["Europe", "Asia"])
@assert nrow(result) == 2
@assert eltype(result.region) <: AbstractString
@assert eltype(result.avg_growth) <: AbstractFloat
println("TASK 13 PASSED")
```

---

## TASK 14: Add `:spline` method to `clean_historical_series` (LOW)

**Files:** `/var/mnt/eclipse/repos/Cliometrics.jl/src/Cliometrics.jl`

**Problem:** The docstring on line 313 lists `:spline` as a valid method option, but the implementation (lines 327-361) only handles `:linear` and `:forward_fill`. If a user calls `clean_historical_series(data, method=:spline)`, it silently returns the uncleaned data since neither branch matches.

**What to do:**
1. Either implement a simple spline interpolation (cubic), or
2. Add an `else` clause that throws an informative error: `error("Unknown method: $method. Use :linear, :spline, or :forward_fill")`
3. If implementing spline: use a natural cubic spline between known points. You may use `StatsBase` or implement a basic version. Given that the package already depends on `StatsBase`, check if it provides interpolation utilities.
4. If spline is too complex to implement cleanly, remove `:spline` from the docstring on line 313 and add the error clause.

**Verification:**
```julia
using Cliometrics
# If spline is implemented:
data = [100.0, 105.0, NaN, NaN, 120.0]
cleaned = clean_historical_series(data, method=:spline)
@assert length(cleaned) == 5
@assert all(isfinite.(cleaned))
println("TASK 14 PASSED")

# OR if spline is removed, verify error:
try
    clean_historical_series([1.0, 2.0], method=:spline)
    error("Should have thrown")
catch e
    @assert occursin("Unknown method", e.msg) "Should throw informative error"
    println("TASK 14 PASSED (spline removed, error added)")
end
```

---

## TASK 15: Fix Dustfile and Intentfile SPDX typo (LOW)

**Files:**
- `/var/mnt/eclipse/repos/Cliometrics.jl/contractiles/dust/Dustfile` (line 1)
- `/var/mnt/eclipse/repos/Cliometrics.jl/contractiles/lust/Intentfile` (line 1)
- `/var/mnt/eclipse/repos/Cliometrics.jl/contractiles/must/Mustfile` (line 1)
- `/var/mnt/eclipse/repos/Cliometrics.jl/contractiles/trust/Trustfile.hs` (line 1)

**Problem:** These files use `PLMP-1.0-or-later` which is a typo. The correct identifier is `PMPL-1.0-or-later` (Palimpsest License).

**What to do:**
1. In each file, replace `PLMP-1.0-or-later` with `PMPL-1.0-or-later` on line 1.

**Verification:**
```bash
cd /var/mnt/eclipse/repos/Cliometrics.jl && grep -rn "PLMP" contractiles/
# Should return zero lines
grep -rn "PMPL" contractiles/dust/Dustfile contractiles/lust/Intentfile contractiles/must/Mustfile contractiles/trust/Trustfile.hs
# Should return 4 lines, one per file
```

---

## TASK 16: Add README claim reconciliation — remove or implement claimed features (LOW)

**Files:** `/var/mnt/eclipse/repos/Cliometrics.jl/README.md`

**Problem:** The README.md (lines 69-84) claims the following features that have NO implementation:
- "Sigma-convergence testing" (line 71) -- only beta-convergence exists
- "Conditional convergence estimation" (line 72) -- not implemented
- "Long-run growth trend analysis" (line 67) -- not implemented
- "Institutional change measurement" (line 78) -- partially addressed by Task 2
- "Outlier detection and handling" (line 83) -- not implemented
- "Cross-country data alignment" (line 84) -- not implemented

**What to do:**
1. For features completed by Tasks 1-4, verify they are accurately described.
2. For features NOT implemented (sigma-convergence, conditional convergence, long-run trends, outlier detection, cross-country alignment), either:
   a. Mark them as "Planned" or "Coming in v0.2.0" in the README, or
   b. Remove them from the feature list entirely.
3. Do NOT claim features that do not exist in the code.

**Verification:**
```bash
cd /var/mnt/eclipse/repos/Cliometrics.jl
# Manually verify: every feature listed in README.md has a corresponding exported function
grep -oP '(?<=- ).*' README.md | head -20
# Cross-reference with:
grep "^    " src/Cliometrics.jl | head -20  # exported functions
```

---

## FINAL VERIFICATION

After all tasks are complete, run the following sequence:

```bash
cd /var/mnt/eclipse/repos/Cliometrics.jl

# 1. Full test suite
julia --project=. -e 'using Pkg; Pkg.test()'

# 2. No AGPL references remain
grep -rn "AGPL-3.0" . --include="*.scm" --include="*.zig" --include="*.res" --include="*.adoc" --include="*.jl" | grep -v ".git/" | grep -v "SONNET-TASKS"

# 3. No template placeholders remain
grep -rn '{{PROJECT}}\|{{project}}\|{{OWNER}}\|{{REPO}}\|{{FORGE}}\|\[YOUR-REPO-NAME\]' . --include="*.idr" --include="*.zig" --include="*.md" --include="*.a2ml" --include="*.adoc" | grep -v ".git/" | grep -v "SONNET-TASKS"

# 4. No PLMP typos remain
grep -rn "PLMP" . --include="*" | grep -v ".git/" | grep -v "SONNET-TASKS"

# 5. No rsr-template-repo references in SCM files
grep -rn "rsr-template-repo" .machines_readable/

# 6. All exported functions have implementations
julia --project=. -e '
using Cliometrics
for name in names(Cliometrics)
    fn = getfield(Cliometrics, name)
    if fn isa Function
        println("OK: $name is defined")
    end
end
'

# 7. Verify the example runs
julia --project=. examples/growth_decomposition.jl
```

All 7 checks must pass with zero errors. If any fail, fix the root cause before declaring completion.
