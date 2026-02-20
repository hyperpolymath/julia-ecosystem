# SONNET-TASKS.md -- ZeroProb.jl

**Repo:** `hyperpolymath/ZeroProb.jl`
**Audit date:** 2026-02-12
**Auditor model:** claude-opus-4-6
**Honest completion:** ~62%

STATE.scm claims 100% -- this is FALSE. The core types, measures, paradoxes,
and applications code is real and functional. But the README advertises an API
that does not exist (4 functions mentioned in docs are never implemented),
visualization has zero tests, there are phantom dependencies in Project.toml,
examples/ has nothing to do with ZeroProb.jl, license headers are wrong in
multiple template files, the `DiscreteZeroProbEvent` has no `relevance()`
dispatch, `handles_zero_prob_event` has a stub fallthrough that silently
returns true, `hausdorff_measure` only handles dimensions 0 and 1 (trivial),
and `plot_black_swan_impact` is defined but not exported.

---

## GROUND RULES FOR SONNET

1. Read this file top-to-bottom before starting.
2. Do ONE task at a time. Verify each before moving on.
3. Do NOT modify SONNET-TASKS.md itself (except to check off items if instructed).
4. Run all verification commands and confirm they pass.
5. Commit after each task with a descriptive message.
6. If a task says "line N", confirm the line number -- file may have shifted.
7. Do NOT introduce new dependencies unless the task explicitly says to.
8. Update `.machine_readable/STATE.scm` after all tasks are done to reflect honest completion.

---

## TASK 1: Implement the 4 missing functions advertised in README.adoc

**Files:**
- `/var/mnt/eclipse/repos/ZeroProb.jl/README.adoc` (lines 29, 76-89, 105-109, 115-128)
- `/var/mnt/eclipse/repos/ZeroProb.jl/src/ZeroProb.jl` (export list)
- `/var/mnt/eclipse/repos/ZeroProb.jl/src/measures.jl` (add implementations)

**Problem:**
The README.adoc advertises these functions in code examples and the API reference:

1. `density_ratio(dist, x, baseline)` -- a 3-argument form comparing density at x vs baseline point. The actual implementation (`measures.jl` line 46) is `density_ratio(event::ContinuousZeroProbEvent)` which takes a single event, not (dist, x, baseline).

2. `hausdorff_dimension(set)` -- mentioned at README line 88 (`dimension = hausdorff_dimension(cantor_set)`). Never implemented anywhere.

3. `construct_cantor_set(iterations=10)` -- mentioned at README line 87. Never implemented anywhere.

4. `estimate_convergence_rate(ε_values, probs)` -- mentioned at README lines 108-109 and API reference at line 128. Never implemented anywhere.

5. `epsilon_neighborhood_prob(dist, x, ε)` -- the README uses this name (lines 47, 105, 124) but the actual function is `epsilon_neighborhood(event, ε)`. Different name AND different signature (takes event, not dist+x).

**What to do:**

Option A (recommended): Implement the missing functions in `src/measures.jl`:
- `density_ratio(dist::Distribution, x::Real, baseline::Real)` -- returns `pdf(dist, x) / pdf(dist, baseline)` as a true ratio
- `hausdorff_dimension(points; method=:box_counting)` -- box-counting dimension estimation
- `construct_cantor_set(; iterations::Int=10)` -- returns the Cantor set as a vector of intervals
- `estimate_convergence_rate(ε_values::Vector, prob_values::Vector)` -- log-log linear regression for convergence rate
- `epsilon_neighborhood_prob` as an alias for `epsilon_neighborhood` with (dist, x, ε) signature

Add all 5 to the export list in `src/ZeroProb.jl`.

Option B: Fix the README to match the actual API. Only do this if Option A is too much work.

**Verification:**
```julia
using ZeroProb, Distributions

# density_ratio 3-arg form
dist = Normal(0, 1)
r = density_ratio(dist, 0.0, 3.0)
@assert r > 1.0  # Center is denser than tail
@assert r isa Float64

# hausdorff_dimension
cantor = construct_cantor_set(iterations=8)
dim = hausdorff_dimension(cantor)
@assert 0.5 < dim < 0.7  # Should be ~log(2)/log(3) ≈ 0.631

# estimate_convergence_rate
event = ContinuousZeroProbEvent(Normal(0,1), 0.0, :epsilon)
εs = [0.1, 0.01, 0.001, 0.0001]
probs = [epsilon_neighborhood(event, ε) for ε in εs]
rate = estimate_convergence_rate(εs, probs)
@assert rate isa Float64
@assert rate > 0.0

# epsilon_neighborhood_prob alias
p = epsilon_neighborhood_prob(Normal(0,1), 0.0, 0.1)
@assert p > 0.0
```

---

## TASK 2: Add `relevance()` dispatch for DiscreteZeroProbEvent

**Files:**
- `/var/mnt/eclipse/repos/ZeroProb.jl/src/measures.jl`
- `/var/mnt/eclipse/repos/ZeroProb.jl/test/test_measures.jl`

**Problem:**
`relevance()` is only implemented for `ContinuousZeroProbEvent` (measures.jl line 140). There is no `relevance()` method for `DiscreteZeroProbEvent`. Calling `relevance(DiscreteZeroProbEvent(...))` will throw a MethodError at runtime.

The `DiscreteZeroProbEvent` struct does not have a `relevance_measure` field, so the dispatch needs a different approach -- it should return the pdf value (which is 0 by construction, since the constructor asserts `p == 0.0`).

**What to do:**
Add to `measures.jl`:
```julia
function relevance(event::DiscreteZeroProbEvent{T}; kwargs...) where T
    # For discrete zero-prob events, relevance is based on the probability mass
    # which is 0 by construction. Return 0.0 to be consistent.
    return 0.0
end
```

Also add `relevance_score` for `DiscreteZeroProbEvent`:
```julia
function relevance_score(event::DiscreteZeroProbEvent{T}, application::Symbol) where T
    return 0.0  # Discrete zero-prob events have no relevance by construction
end
```

Add tests to `test_measures.jl`:
```julia
@testset "DiscreteZeroProbEvent relevance" begin
    dist = Geometric(0.5)
    event = DiscreteZeroProbEvent(dist, -1)
    @test relevance(event) == 0.0
    @test relevance_score(event, :black_swan) == 0.0
end
```

**Verification:**
```julia
using ZeroProb, Distributions
dist = Geometric(0.5)
event = DiscreteZeroProbEvent(dist, -1)
@assert relevance(event) == 0.0
@assert relevance_score(event, :black_swan) == 0.0
@assert relevance_score(event, :betting) == 0.0
```

---

## TASK 3: Export and test `plot_black_swan_impact`

**Files:**
- `/var/mnt/eclipse/repos/ZeroProb.jl/src/ZeroProb.jl` (lines 90-91 export list)
- `/var/mnt/eclipse/repos/ZeroProb.jl/src/visualization.jl` (line 216, `plot_black_swan_impact`)

**Problem:**
`plot_black_swan_impact` is defined in `visualization.jl` (line 216) but is NOT in the export list in `ZeroProb.jl` (lines 90-91). The export list only exports:
```
plot_zero_probability, plot_continuum_paradox,
plot_density_vs_probability, plot_epsilon_neighborhood
```

Missing: `plot_black_swan_impact`.

**What to do:**
Add `plot_black_swan_impact` to the export list on line 91 of `src/ZeroProb.jl`.

**Verification:**
```julia
using ZeroProb
@assert isdefined(ZeroProb, :plot_black_swan_impact)
@assert hasmethod(plot_black_swan_impact, Tuple{BlackSwanEvent})
```

---

## TASK 4: Add visualization tests

**Files:**
- `/var/mnt/eclipse/repos/ZeroProb.jl/test/runtests.jl`
- Create `/var/mnt/eclipse/repos/ZeroProb.jl/test/test_visualization.jl`

**Problem:**
There are ZERO tests for any of the 5 visualization functions:
- `plot_zero_probability`
- `plot_continuum_paradox`
- `plot_density_vs_probability`
- `plot_epsilon_neighborhood`
- `plot_black_swan_impact`

The visualization module `using Plots` at the top of `visualization.jl`, but Plots is listed as a dependency in Project.toml, so it should load. However, none of the plotting functions are tested to verify they at least return a plot object without crashing.

**What to do:**
Create `test/test_visualization.jl`:
```julia
# SPDX-License-Identifier: PMPL-1.0-or-later

using Plots

@testset "Visualization" begin
    @testset "plot_zero_probability" begin
        dist = Normal(0, 1)
        event = ContinuousZeroProbEvent(dist, 0.0, :density)
        p = plot_zero_probability(event)
        @test p isa Plots.Plot
    end

    @testset "plot_continuum_paradox" begin
        dist = Normal(0, 1)
        p = plot_continuum_paradox(dist, num_points=5)
        @test p isa Plots.Plot
    end

    @testset "plot_density_vs_probability" begin
        event = ContinuousZeroProbEvent(Normal(0, 1), 0.0, :density)
        p = plot_density_vs_probability(event, ε_max=1.0)
        @test p isa Plots.Plot
    end

    @testset "plot_epsilon_neighborhood" begin
        event = ContinuousZeroProbEvent(Normal(0, 1), 0.0, :epsilon)
        p = plot_epsilon_neighborhood(event, ε=0.5)
        @test p isa Plots.Plot
    end

    @testset "plot_black_swan_impact" begin
        crash = MarketCrashEvent(severity=:catastrophic)
        p = plot_black_swan_impact(crash, samples=100)
        @test p isa Plots.Plot
    end
end
```

Add `include("test_visualization.jl")` to `test/runtests.jl`.

**Verification:**
```bash
cd /var/mnt/eclipse/repos/ZeroProb.jl
julia --project=. -e 'using Pkg; Pkg.test()'
```
All 5 visualization tests should pass.

---

## TASK 5: Remove phantom dependencies from Project.toml

**Files:**
- `/var/mnt/eclipse/repos/ZeroProb.jl/Project.toml` (lines 8, 11)

**Problem:**
Project.toml lists two dependencies that are never used anywhere in the source code:

1. `Makie = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"` (line 8) -- grep confirms Makie is never imported or used in any `.jl` file. The visualization module uses `Plots`, not `Makie`.

2. `Zstd_jll = "3161d3a3-bdf6-5164-811a-617609db77b4"` (line 11) -- grep confirms Zstd_jll is never imported or used anywhere. This is a compression library JLL wrapper with no connection to probability theory.

These add unnecessary compile-time overhead and dependency weight.

**What to do:**
1. Remove the `Makie` line from `[deps]`
2. Remove the `Zstd_jll` line from `[deps]`
3. Remove the `Makie = "0.20"` line from `[compat]`
4. Remove the `Zstd_jll = "1.5.7"` line from `[compat]`
5. Run `julia --project=. -e 'using Pkg; Pkg.resolve()'` to regenerate Manifest.toml

**Verification:**
```bash
cd /var/mnt/eclipse/repos/ZeroProb.jl
julia --project=. -e 'using Pkg; Pkg.resolve(); using ZeroProb; println("OK")'
```
Should load without error and without Makie/Zstd_jll.

---

## TASK 6: Fix the `handles_zero_prob_event` stub fallthrough

**Files:**
- `/var/mnt/eclipse/repos/ZeroProb.jl/src/applications.jl` (lines 339-341)

**Problem:**
The catch-all branch at the bottom of `handles_zero_prob_event` (line 339-341):
```julia
    else
        @warn "No specific `handles_zero_prob_event` implementation for type $(typeof(event)). Returning true as a stub."
        return true
    end
```

This silently returns `true` for any unknown event type, which is dangerous. If someone creates a new `ZeroProbEvent` subtype, `handles_zero_prob_event` will claim the model handles it when it was never actually tested.

**What to do:**
Change line 341 from `return true` to `return false` and update the warning message:
```julia
    else
        @warn "No specific `handles_zero_prob_event` implementation for type $(typeof(event)). Returning false (unverified)."
        return false
    end
```

Also add a test for this behavior in `test/test_applications.jl`:
```julia
@testset "handles_zero_prob_event unknown type" begin
    # Create a custom ZeroProbEvent subtype
    struct TestZeroProbEvent <: ZeroProbEvent end
    model = x -> x
    # Unknown types should return false (not silently true)
    @test handles_zero_prob_event(model, TestZeroProbEvent()) == false
end
```

**Verification:**
```julia
using ZeroProb
struct MyCustomEvent <: ZeroProbEvent end
model = x -> x
@assert handles_zero_prob_event(model, MyCustomEvent()) == false
```

---

## TASK 7: Replace examples/ with actual ZeroProb.jl examples

**Files:**
- `/var/mnt/eclipse/repos/ZeroProb.jl/examples/SafeDOMExample.res` (DELETE)
- `/var/mnt/eclipse/repos/ZeroProb.jl/examples/web-project-deno.json` (DELETE)
- Create `/var/mnt/eclipse/repos/ZeroProb.jl/examples/basic_usage.jl`
- Create `/var/mnt/eclipse/repos/ZeroProb.jl/examples/black_swan_analysis.jl`

**Problem:**
The examples/ directory contains two files that have NOTHING to do with ZeroProb.jl:
- `SafeDOMExample.res` -- a ReScript file about DOM mounting with AGPL license header
- `web-project-deno.json` -- a Deno project config for a web project

These are leftover RSR template files. A Julia probability library should have Julia examples.

**What to do:**
1. Delete `SafeDOMExample.res` and `web-project-deno.json`
2. Create `examples/basic_usage.jl` demonstrating:
   - Creating ContinuousZeroProbEvent instances
   - Computing probability (always 0) vs relevance (non-zero)
   - All three relevance measures (density, hausdorff, epsilon)
   - The continuum paradox function
3. Create `examples/black_swan_analysis.jl` demonstrating:
   - Creating a MarketCrashEvent
   - Computing probability and expected impact
   - Using handles_black_swan to test a model
   - BettingEdgeCase expected value computation

Both files must have:
- `# SPDX-License-Identifier: PMPL-1.0-or-later` header
- `# Copyright (c) 2026 Jonathan D.A. Jewell <jonathan.jewell@open.ac.uk>`
- Comments explaining each step
- Must be runnable: `julia --project=.. examples/basic_usage.jl`

**Verification:**
```bash
cd /var/mnt/eclipse/repos/ZeroProb.jl
julia --project=. examples/basic_usage.jl
julia --project=. examples/black_swan_analysis.jl
```
Both should run without error and produce output.

---

## TASK 8: Fix AGPL license headers in template files

**Files:**
- `/var/mnt/eclipse/repos/ZeroProb.jl/.gitignore` (line 1: `# SPDX-License-Identifier: AGPL-3.0-or-later`)
- `/var/mnt/eclipse/repos/ZeroProb.jl/.gitattributes` (line 1: `# SPDX-License-Identifier: AGPL-3.0-or-later`)
- `/var/mnt/eclipse/repos/ZeroProb.jl/ffi/zig/build.zig` (line 2: `// SPDX-License-Identifier: AGPL-3.0-or-later`)
- `/var/mnt/eclipse/repos/ZeroProb.jl/ffi/zig/src/main.zig` (line 6: `// SPDX-License-Identifier: AGPL-3.0-or-later`)
- `/var/mnt/eclipse/repos/ZeroProb.jl/ffi/zig/test/integration_test.zig` (line 2: `// SPDX-License-Identifier: AGPL-3.0-or-later`)
- `/var/mnt/eclipse/repos/ZeroProb.jl/docs/CITATIONS.adoc` (line 13: `license = {AGPL-3.0-or-later}`)
- `/var/mnt/eclipse/repos/ZeroProb.jl/RSR_OUTLINE.adoc` (lines 72, 160 mention AGPL)

**Problem:**
Per CLAUDE.md, AGPL-3.0 is the OLD license and must NEVER be used. All hyperpolymath original code must use `PMPL-1.0-or-later`. These are leftover RSR template headers that were never updated.

**What to do:**
Replace `AGPL-3.0-or-later` with `PMPL-1.0-or-later` in all files listed above.

For `docs/CITATIONS.adoc`, also fix the author from `Polymath, Hyper` to `Jewell, Jonathan D.A.`, the title from `RSR-template-repo` to `ZeroProb.jl`, the year from `2025` to `2026`, and the URL to `https://github.com/hyperpolymath/ZeroProb.jl`.

For `RSR_OUTLINE.adoc`, update line 72 and 160 to reference PMPL-1.0-or-later instead of AGPL.

**Verification:**
```bash
cd /var/mnt/eclipse/repos/ZeroProb.jl
grep -r "AGPL" --include="*.jl" --include="*.zig" --include="*.adoc" --include=".git*" . | grep -v ".git/"
```
Should return NO matches (zero lines).

---

## TASK 9: Make `hausdorff_measure` non-trivial (support dimensions > 1)

**Files:**
- `/var/mnt/eclipse/repos/ZeroProb.jl/src/measures.jl` (lines 70-78)

**Problem:**
The current `hausdorff_measure` implementation is trivial:
```julia
function hausdorff_measure(event::ContinuousZeroProbEvent{T}, dimension::Int=0) where T
    if dimension == 0
        return 1.0  # Single point has unit 0-dimensional Hausdorff measure
    elseif dimension == 1
        return 0.0  # But zero 1-dimensional measure
    else
        throw(ArgumentError("Only dimensions 0 and 1 currently supported"))
    end
end
```

This throws on any dimension other than 0 or 1. For a point set, the Hausdorff measure of dimension d > 0 is always 0 (a single point has zero d-dimensional measure for any d > 0). Dimension 0 is always 1 (counting measure). So the function should handle all non-negative integer dimensions, not just 0 and 1.

**What to do:**
Replace the function body:
```julia
function hausdorff_measure(event::ContinuousZeroProbEvent{T}, dimension::Int=0) where T
    @assert dimension >= 0 "Hausdorff dimension must be non-negative"
    if dimension == 0
        return 1.0  # Single point has unit 0-dimensional Hausdorff measure (counting measure)
    else
        return 0.0  # Single point has zero d-dimensional Hausdorff measure for all d > 0
    end
end
```

Update tests in `test_measures.jl` to cover higher dimensions:
```julia
@testset "hausdorff_measure higher dimensions" begin
    dist = Normal(0, 1)
    event = ContinuousZeroProbEvent(dist, 0.0, :hausdorff)
    @test hausdorff_measure(event, 0) == 1.0
    @test hausdorff_measure(event, 1) == 0.0
    @test hausdorff_measure(event, 2) == 0.0
    @test hausdorff_measure(event, 10) == 0.0
    @test_throws AssertionError hausdorff_measure(event, -1)
end
```

**Verification:**
```julia
using ZeroProb, Distributions
event = ContinuousZeroProbEvent(Normal(0,1), 0.0, :hausdorff)
@assert hausdorff_measure(event, 0) == 1.0
@assert hausdorff_measure(event, 2) == 0.0
@assert hausdorff_measure(event, 5) == 0.0
# Should NOT throw:
hausdorff_measure(event, 100)
```

---

## TASK 10: Fix STATE.scm to reflect honest completion

**Files:**
- `/var/mnt/eclipse/repos/ZeroProb.jl/.machine_readable/STATE.scm`

**Problem:**
STATE.scm line 22 claims `(overall-completion 100)` and line 21 claims `(phase "complete")`. This is false. After completing tasks 1-9, update STATE.scm to reflect the true state.

Before tasks 1-9 are done, the honest completion is ~62%:
- Types: 100% (solid)
- Measures: 70% (missing DiscreteZeroProbEvent dispatch, hausdorff trivial, advertised functions missing)
- Paradoxes: 95% (functional, well-documented)
- Applications: 80% (stub fallthrough, placeholder comments)
- Visualization: 50% (implemented but untested, one function not exported)
- Examples: 0% (wrong language, wrong project)
- Documentation: 60% (README advertises vaporware API)
- License compliance: 70% (AGPL remnants throughout)

After tasks 1-9 are done, it should be ~90%.

**What to do:**
Update STATE.scm:
- `(phase "active")` (or "complete" only after all tasks done)
- `(overall-completion NN)` -- set to the actual number after completing other tasks
- Update `(updated ...)` to today's date
- Add honest component completion percentages
- Add blockers if any remain
- Add session history entry documenting this audit

**Verification:**
```bash
cat /var/mnt/eclipse/repos/ZeroProb.jl/.machine_readable/STATE.scm
# Verify: no "100" completion unless everything is actually done
# Verify: updated date is 2026-02-12 or later
# Verify: phase is not "complete" unless all tasks are done
```

---

## FINAL VERIFICATION

After ALL tasks are completed, run this full verification sequence:

```bash
cd /var/mnt/eclipse/repos/ZeroProb.jl

# 1. Full test suite passes
julia --project=. -e 'using Pkg; Pkg.test()'

# 2. No AGPL references remain
grep -r "AGPL" --include="*.jl" --include="*.zig" --include="*.adoc" --include=".git*" . | grep -v ".git/" | wc -l
# Expected: 0

# 3. No phantom dependencies
julia --project=. -e 'using Pkg; deps = keys(Pkg.project().dependencies); @assert !("Makie" in deps); @assert !("Zstd_jll" in deps); println("No phantom deps")'

# 4. All advertised functions exist
julia --project=. -e '
using ZeroProb, Distributions

# Core types
@assert isdefined(ZeroProb, :ContinuousZeroProbEvent)
@assert isdefined(ZeroProb, :DiscreteZeroProbEvent)
@assert isdefined(ZeroProb, :AlmostSureEvent)
@assert isdefined(ZeroProb, :SureEvent)

# Measures
@assert isdefined(ZeroProb, :probability)
@assert isdefined(ZeroProb, :relevance)
@assert isdefined(ZeroProb, :density_ratio)
@assert isdefined(ZeroProb, :hausdorff_measure)
@assert isdefined(ZeroProb, :epsilon_neighborhood)
@assert isdefined(ZeroProb, :relevance_score)

# Visualization
@assert isdefined(ZeroProb, :plot_zero_probability)
@assert isdefined(ZeroProb, :plot_continuum_paradox)
@assert isdefined(ZeroProb, :plot_density_vs_probability)
@assert isdefined(ZeroProb, :plot_epsilon_neighborhood)
@assert isdefined(ZeroProb, :plot_black_swan_impact)

# Applications
@assert isdefined(ZeroProb, :BlackSwanEvent)
@assert isdefined(ZeroProb, :MarketCrashEvent)
@assert isdefined(ZeroProb, :BettingEdgeCase)
@assert isdefined(ZeroProb, :handles_black_swan)
@assert isdefined(ZeroProb, :handles_zero_prob_events)

println("All exports verified")
'

# 5. Examples run
julia --project=. examples/basic_usage.jl
julia --project=. examples/black_swan_analysis.jl

# 6. DiscreteZeroProbEvent relevance works
julia --project=. -e '
using ZeroProb, Distributions
event = DiscreteZeroProbEvent(Geometric(0.5), -1)
@assert relevance(event) == 0.0
println("DiscreteZeroProbEvent relevance OK")
'

# 7. Unknown event type returns false (not true)
julia --project=. -e '
using ZeroProb
struct TestEvent <: ZeroProbEvent end
@assert handles_zero_prob_event(x->x, TestEvent()) == false
println("Unknown event fallthrough returns false OK")
'

# 8. hausdorff_measure handles arbitrary dimensions
julia --project=. -e '
using ZeroProb, Distributions
event = ContinuousZeroProbEvent(Normal(0,1), 0.0, :hausdorff)
@assert hausdorff_measure(event, 5) == 0.0
println("hausdorff_measure arbitrary dims OK")
'

# 9. STATE.scm is honest
grep "overall-completion 100" .machine_readable/STATE.scm
# Expected: no output (should NOT claim 100% unless everything is done)
```

If all 9 verification steps pass with expected output, the audit tasks are complete.
