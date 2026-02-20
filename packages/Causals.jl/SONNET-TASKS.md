# SONNET-TASKS.md — Causals.jl Completion Tasks

> **Generated:** 2026-02-12 by Opus audit
> **Purpose:** Unambiguous instructions for Sonnet to complete all stubs, TODOs, and placeholder code.
> **Honest completion before this file:** 42%

The README claims "production-ready" and "complete with comprehensive test coverage." The ROADMAP.md claims v1.0 is done with "7 causal inference methods" (some of which do not even exist in the codebase -- Bayesian Networks, Fuzzy Logic, Evidential Reasoning, Belief Propagation, Conditional Probability, Probabilistic Logic are listed in the ROADMAP but not implemented). The Project.toml says version `1.0.0`. In reality:

- 5 out of 7 source modules contain placeholder/stub code
- `d_separation` always returns `true` (hardcoded)
- `frontdoor_criterion` always returns `true` (hardcoded)
- `confounding_adjustment` always returns `0.0` (hardcoded)
- `do_calculus_rules` returns its input unchanged (no-op)
- `counterfactual` returns `nothing` (completely unimplemented)
- `propensity_score` ignores covariates entirely (returns constant)
- `doubly_robust` just calls IPW (defeats the purpose)
- `granger_test` uses a hardcoded critical F-value instead of the Distributions.jl F-distribution that is already a dependency
- Both example files have severe API mismatches and will not run
- No tests exist for DoCalculus, Counterfactuals, matching, stratification, or doubly_robust
- 6 of 7 docs pages referenced in `docs/make.jl` do not exist
- The ABI/FFI Idris2/Zig files are unmodified templates with `{{PROJECT}}` placeholders
- The `examples/` directory contains two unrelated non-Julia files (SafeDOMExample.res, web-project-deno.json) that should not be there

---

## GROUND RULES FOR SONNET

1. Read this entire file before starting any task.
2. Do tasks in order listed. Earlier tasks unblock later ones.
3. After each task, run the verification command. If it fails, fix before moving on.
4. Do NOT mark done unless verification passes.
5. Update STATE.scm with honest completion percentages after each task.
6. Commit after each task: `fix(component): complete <description>`
7. Run full test suite after every 3 tasks: `cd /var/mnt/eclipse/repos/Causals.jl && julia --project=. -e 'using Pkg; Pkg.test()'`

---

## TASK 1: Fix d_separation to use proper Bayes-Ball algorithm (CRITICAL)

**Files:** `/var/mnt/eclipse/repos/Causals.jl/src/CausalDAG.jl`

**Problem:** The `d_separation` function at lines 60-70 is a complete stub. It always returns `true` regardless of input. The comment at line 62 says "Simplified implementation" and line 69 says `true  # Placeholder`. This makes all downstream d-separation-dependent code (backdoor criterion, frontdoor criterion, do-calculus identification) unreliable.

**What to do:**

1. Replace the body of `d_separation` (lines 60-70) with a proper implementation using the Bayes-Ball algorithm (Shachter 1998) or the reachability-based algorithm:
   - Build the "ancestral graph" of X, Y, and Z
   - Moralize the ancestral graph (add edges between parents of common children)
   - Remove edges involving Z nodes
   - Check if X and Y are still connected in the resulting undirected graph
2. The function signature stays the same: `d_separation(g::CausalGraph, X::Set{Symbol}, Y::Set{Symbol}, Z::Set{Symbol}) -> Bool`
3. Return `true` if X and Y are d-separated given Z, `false` otherwise.
4. Handle edge cases: empty Z set, X or Y being singletons, X == Y (should return true trivially).

**Verification:**
```julia
cd("/var/mnt/eclipse/repos/Causals.jl")
using Pkg; Pkg.activate(".")
using Causals
using Causals.CausalDAG: add_edge!

# Chain: X -> M -> Y -- X _|_ Y | M should be true
g1 = CausalGraph([:X, :M, :Y])
add_edge!(g1, :X, :M)
add_edge!(g1, :M, :Y)
@assert d_separation(g1, Set([:X]), Set([:Y]), Set([:M])) == true "Chain: X _|_ Y | M"
@assert d_separation(g1, Set([:X]), Set([:Y]), Set{Symbol}()) == false "Chain: X NOT _|_ Y | {}"

# Fork: X <- C -> Y -- X _|_ Y | C should be true
g2 = CausalGraph([:X, :C, :Y])
add_edge!(g2, :C, :X)
add_edge!(g2, :C, :Y)
@assert d_separation(g2, Set([:X]), Set([:Y]), Set([:C])) == true "Fork: X _|_ Y | C"
@assert d_separation(g2, Set([:X]), Set([:Y]), Set{Symbol}()) == false "Fork: X NOT _|_ Y | {}"

# Collider: X -> M <- Y -- X _|_ Y | {} should be true, X NOT _|_ Y | M
g3 = CausalGraph([:X, :M, :Y])
add_edge!(g3, :X, :M)
add_edge!(g3, :Y, :M)
@assert d_separation(g3, Set([:X]), Set([:Y]), Set{Symbol}()) == true "Collider: X _|_ Y | {}"
@assert d_separation(g3, Set([:X]), Set([:Y]), Set([:M])) == false "Collider: X NOT _|_ Y | M"

println("TASK 1 PASSED: d_separation works correctly")
```

---

## TASK 2: Fix frontdoor_criterion stub (CRITICAL)

**Files:** `/var/mnt/eclipse/repos/Causals.jl/src/CausalDAG.jl`

**Problem:** The `frontdoor_criterion` function at lines 188-194 always returns `true`. The body is `true` with a comment saying "Simplified implementation". The three conditions listed in the docstring (lines 184-186) are never checked.

**What to do:**

1. Replace the body of `frontdoor_criterion` (lines 188-194) with a proper implementation that checks all three conditions:
   - Condition 1: M intercepts all directed paths from X to Y (every directed path from X to Y passes through some node in M)
   - Condition 2: There are no unblocked backdoor paths from X to any node in M (i.e., no unconfounded relationship X->M)
   - Condition 3: All backdoor paths from M to Y are blocked by X
2. For condition 1, enumerate directed paths from X to Y (DFS on the directed graph) and verify each passes through at least one node in M.
3. For condition 2, check `backdoor_criterion(g, X, m, Set{Symbol}())` or equivalent for each m in M (after removing X from the graph conceptually).
4. For condition 3, check d-separation conditions with X as the conditioning set.

**Verification:**
```julia
cd("/var/mnt/eclipse/repos/Causals.jl")
using Pkg; Pkg.activate(".")
using Causals
using Causals.CausalDAG: add_edge!

# Classic frontdoor: X -> M -> Y, U -> X, U -> Y (U unobserved but in graph)
g = CausalGraph([:X, :M, :Y, :U])
add_edge!(g, :X, :M)
add_edge!(g, :M, :Y)
add_edge!(g, :U, :X)
add_edge!(g, :U, :Y)
@assert frontdoor_criterion(g, :X, :Y, Set([:M])) == true "Frontdoor should hold for M"

# Invalid frontdoor: M does not intercept all paths (direct X->Y edge exists)
g2 = CausalGraph([:X, :M, :Y, :U])
add_edge!(g2, :X, :M)
add_edge!(g2, :M, :Y)
add_edge!(g2, :X, :Y)  # Direct path bypasses M
add_edge!(g2, :U, :X)
add_edge!(g2, :U, :Y)
@assert frontdoor_criterion(g2, :X, :Y, Set([:M])) == false "Frontdoor should fail: direct X->Y bypasses M"

println("TASK 2 PASSED: frontdoor_criterion works correctly")
```

---

## TASK 3: Implement proper p-value computation in granger_test using Distributions.jl (HIGH)

**Files:** `/var/mnt/eclipse/repos/Causals.jl/src/Granger.jl`

**Problem:** Lines 49-50 compute the p-value using a hardcoded critical value `critical_F = 3.0` and return fake p-values (`0.01` or `0.1`). The package already depends on `Distributions.jl` (listed in Project.toml line 11) but it is not used in this module.

**What to do:**

1. Add `using Distributions` at the top of the Granger module (after line 11).
2. Replace lines 49-52 with proper F-distribution computation:
   ```julia
   df1 = k  # numerator degrees of freedom
   df2 = n_obs - 2 * best_lag - 1  # denominator degrees of freedom
   f_dist = FDist(df1, df2)
   p_value = 1.0 - cdf(f_dist, F_stat)
   causes = p_value < α
   ```
3. Remove the `critical_F` variable entirely.
4. Ensure the `α` keyword argument is actually used (currently it is accepted but ignored because the hardcoded critical value is used instead).

**Verification:**
```julia
cd("/var/mnt/eclipse/repos/Causals.jl")
using Pkg; Pkg.activate(".")
using Causals

# Known causal relationship
n = 200
x = randn(n)
y = zeros(n)
for t in 2:n
    y[t] = 0.5 * y[t-1] + 0.8 * x[t-1] + 0.05 * randn()
end

causes, F_stat, p_value, lag = granger_test(x, y, 5)
@assert p_value >= 0.0 && p_value <= 1.0 "p-value must be in [0,1], got $p_value"
@assert p_value < 0.05 "Strong causal signal should have p < 0.05, got $p_value"
@assert causes == true "Should detect Granger causality"

# Independent series should not show causality
x_indep = randn(200)
y_indep = randn(200)
causes_indep, _, p_indep, _ = granger_test(x_indep, y_indep, 5)
@assert p_indep > 0.0 "Independent series p-value should not be zero"

println("TASK 3 PASSED: granger_test uses proper F-distribution p-values")
```

---

## TASK 4: Implement real propensity_score using logistic regression (HIGH)

**Files:** `/var/mnt/eclipse/repos/Causals.jl/src/PropensityScore.jl`

**Problem:** The `propensity_score` function at lines 22-31 completely ignores the `covariates` argument. It computes `p = sum(treatment) / n` (the marginal treatment probability) and returns `fill(p, n)` -- a vector of identical values. This makes all downstream methods (matching, IPW, stratification) meaningless because every unit gets the same propensity score.

**What to do:**

1. Implement logistic regression using iteratively reweighted least squares (IRLS):
   - Initialize coefficients beta to zeros
   - Iterate: compute predictions p = logistic(X*beta), weights W = diag(p.*(1-p)), update beta = beta + inv(X'*W*X) * X' * (treatment - p)
   - Converge when max change in beta < 1e-8 or after 25 iterations
2. Add an intercept column to covariates internally: `X = hcat(ones(n), covariates)`
3. Clip propensity scores to [0.01, 0.99] to avoid division by zero in IPW.
4. Return the fitted propensity scores.

**Verification:**
```julia
cd("/var/mnt/eclipse/repos/Causals.jl")
using Pkg; Pkg.activate(".")
using Causals
using Random; Random.seed!(42)

n = 200
x1 = randn(n)
x2 = randn(n)
# Treatment depends on covariates
logit = 0.5 .* x1 .+ 0.3 .* x2 .- 0.2
p_true = 1.0 ./ (1.0 .+ exp.(-logit))
treatment = rand(n) .< p_true

ps = propensity_score(treatment, hcat(x1, x2))
@assert length(ps) == n
@assert !all(ps .== ps[1]) "Propensity scores must vary across observations, not be constant"
@assert all(0.0 .<= ps .<= 1.0) "All scores must be in [0,1]"
@assert cor(ps, p_true) > 0.5 "Estimated scores should correlate with true propensity"

println("TASK 4 PASSED: propensity_score uses actual logistic regression")
```

---

## TASK 5: Implement confounding_adjustment (HIGH)

**Files:** `/var/mnt/eclipse/repos/Causals.jl/src/DoCalculus.jl`

**Problem:** The `confounding_adjustment` function at lines 90-103 always returns `0.0` (line 102 says `0.0  # Placeholder`). The function is supposed to compute the backdoor-adjusted causal effect E[Y|do(X=1)] - E[Y|do(X=0)] using stratification over confounders.

**What to do:**

1. Replace the body of `confounding_adjustment` (lines 90-103) with a real implementation:
   - The `data` dictionary maps variable names to Float64 vectors.
   - Binarize or stratify by the confounder values (use quantile-based binning for continuous confounders).
   - For each stratum z of confounders: compute E[Y|X=1,Z=z] and E[Y|X=0,Z=z], weighted by P(Z=z).
   - Return the weighted sum: sum_z (E[Y|X=1,Z=z] - E[Y|X=0,Z=z]) * P(Z=z).
2. Handle cases where a stratum has no treated or no control units (skip that stratum).
3. Handle single vs multiple confounders by creating combined strata.

**Verification:**
```julia
cd("/var/mnt/eclipse/repos/Causals.jl")
using Pkg; Pkg.activate(".")
using Causals
using Random; Random.seed!(42)

n = 500
z = randn(n)
x = (z .+ randn(n)) .> 0  # Treatment depends on confounder
x_float = Float64.(x)
y = 2.0 .* x_float .+ 1.5 .* z .+ randn(n)  # True causal effect of X is 2.0

data = Dict(
    :X => x_float,
    :Y => y,
    :Z => z
)

effect = confounding_adjustment(:X, :Y, Set([:Z]), data)
@assert abs(effect) > 0.0 "Effect must not be zero placeholder"
@assert abs(effect - 2.0) < 1.0 "Adjusted effect should be near 2.0, got $effect"

println("TASK 5 PASSED: confounding_adjustment computes real adjusted effects")
```

---

## TASK 6: Implement do_calculus_rules (MEDIUM)

**Files:** `/var/mnt/eclipse/repos/Causals.jl/src/DoCalculus.jl`

**Problem:** The `do_calculus_rules` function at lines 78-82 is a no-op that returns its input query unchanged. The docstring describes Pearl's three rules of do-calculus but none are implemented.

**What to do:**

1. Define a proper query representation type. At minimum, a tagged union or struct that can represent:
   - `P(Y | do(X), Z)` -- conditional with intervention
   - `P(Y | X, Z)` -- standard conditional
2. Implement Rule 1 (insertion/deletion of observations): P(Y | do(X), Z, W) = P(Y | do(X), Z) if Y _|_ W | X, Z in G_overbar_X
3. Implement Rule 2 (action/observation exchange): P(Y | do(X), do(Z), W) = P(Y | do(X), Z, W) if Y _|_ Z | X, W in G_overbar_X_underbar_Z
4. Implement Rule 3 (insertion/deletion of actions): P(Y | do(X), do(Z), W) = P(Y | do(X), W) if Y _|_ Z | X, W in G_overbar_X_overbar_Z(W)
5. If full implementation is too complex, implement at least Rules 1 and 2 with proper graph manipulation and d-separation checks (which will work after Task 1).
6. Update the function signature to accept a `CausalGraph` and return a simplified query or `:cannot_simplify`.

**Verification:**
```julia
cd("/var/mnt/eclipse/repos/Causals.jl")
using Pkg; Pkg.activate(".")
using Causals
using Causals.CausalDAG: add_edge!
using Causals.DoCalculus

# Build a simple graph
g = CausalGraph([:X, :Y, :Z])
add_edge!(g, :X, :Y)
add_edge!(g, :Z, :X)
add_edge!(g, :Z, :Y)

# At minimum, the function should not be a no-op
result = do_calculus_rules(g, (:Y, :do_X))
@assert result !== (:Y, :do_X) || typeof(result) != typeof((:Y, :do_X)) "do_calculus_rules must not be a no-op"

println("TASK 6 PASSED: do_calculus_rules implements at least basic simplification")
```

---

## TASK 7: Implement counterfactual function with structural equations (HIGH)

**Files:** `/var/mnt/eclipse/repos/Causals.jl/src/Counterfactuals.jl`

**Problem:** The `counterfactual` function at lines 37-54 returns `nothing` (line 53). The three-step process described in the docstring (Abduction, Action, Prediction) is outlined in comments but not implemented. `U` is an empty Dict (line 45) and the function terminates with `nothing`.

**What to do:**

1. Extend the `counterfactual` function to accept structural equations. Add an optional `equations` parameter of type `Dict{Symbol, Function}` where each function takes a Dict of parent values and noise and returns the variable's value.
2. Implement the three-step process:
   - **Abduction**: Given observations and structural equations, infer noise terms U by solving the equations backwards.
   - **Action**: Apply the intervention (set X=x, remove incoming edges to X in the graph).
   - **Prediction**: Evaluate structural equations forward in topological order using the inferred U values and the intervention.
3. Return the counterfactual value of the outcome variable as a `Dict{Symbol, Any}` mapping variable names to their counterfactual values.
4. If no equations are provided, return `nothing` with a warning (preserve backward compatibility).

**Verification:**
```julia
cd("/var/mnt/eclipse/repos/Causals.jl")
using Pkg; Pkg.activate(".")
using Causals
using Causals.CausalDAG: add_edge!
using Causals.Counterfactuals

# Simple SCM: X -> Y, Y = 2*X + noise
g = CausalGraph([:X, :Y])
add_edge!(g, :X, :Y)

# Structural equations: Y = 2X + U_Y
equations = Dict(
    :X => (parents, noise) -> get(noise, :U_X, 0.0),
    :Y => (parents, noise) -> 2.0 * parents[:X] + get(noise, :U_Y, 0.0)
)

# Observed: X=3, Y=6.5 (so U_Y = 0.5)
observations = Dict(:X => 3.0, :Y => 6.5)

result = counterfactual(g, :Y, :X => 5.0, observations; equations=equations)
@assert result !== nothing "counterfactual must not return nothing"
# Counterfactual Y when X=5: 2*5 + 0.5 = 10.5
@assert abs(result - 10.5) < 0.01 "Counterfactual Y should be 10.5, got $result"

println("TASK 7 PASSED: counterfactual computes actual counterfactual values")
```

---

## TASK 8: Implement doubly_robust estimator properly (MEDIUM)

**Files:** `/var/mnt/eclipse/repos/Causals.jl/src/PropensityScore.jl`

**Problem:** The `doubly_robust` function at lines 145-155 just calls `inverse_probability_weighting` and returns its result. The comment on line 151 says "Simplified implementation". This defeats the entire purpose of the doubly robust estimator, which should be consistent if EITHER the propensity model OR the outcome model is correct.

**What to do:**

1. Replace the body of `doubly_robust` (lines 145-155) with the proper Augmented IPW (AIPW) formula:
   ```
   ATE = (1/n) * sum_i [
       (treatment_i * outcome_i / propensity_i) - ((treatment_i - propensity_i) / propensity_i) * outcome_model_1(x_i)
       - ((1-treatment_i) * outcome_i / (1-propensity_i)) + ((treatment_i - propensity_i) / (1-propensity_i)) * outcome_model_0(x_i)
   ]
   ```
2. The `outcome_model` parameter should accept predicted outcomes for both treated and control groups. Update the function signature if needed: accept `outcome_model_1` and `outcome_model_0` vectors (predicted outcomes under treatment and control), or a single function that takes covariates and treatment indicator.
3. Compute proper standard errors using the influence function.

**Verification:**
```julia
cd("/var/mnt/eclipse/repos/Causals.jl")
using Pkg; Pkg.activate(".")
using Causals
using Random; Random.seed!(42)

n = 200
treatment = rand(Bool, n)
outcome = Float64.(treatment) .* 2.0 .+ randn(n)
propensity = fill(0.5, n)

# Simple outcome model: predict mean outcome per group
mean_treated = mean(outcome[treatment])
mean_control = mean(outcome[.!treatment])
outcome_model = (x) -> treatment .* mean_treated .+ (1 .- treatment) .* mean_control

dr_ate = doubly_robust(treatment, outcome, propensity, outcome_model)
@assert !isnan(dr_ate) "DR estimate must not be NaN"
@assert abs(dr_ate - 2.0) < 1.5 "DR estimate should be near true effect 2.0, got $dr_ate"

# Verify it gives different result than plain IPW
ipw_ate, _ = inverse_probability_weighting(treatment, outcome, propensity)
# They may be similar but should not be identical in general
println("DR ATE: $dr_ate, IPW ATE: $ipw_ate")

println("TASK 8 PASSED: doubly_robust uses proper AIPW estimator")
```

---

## TASK 9: Fix identify_effect to try frontdoor criterion (MEDIUM)

**Files:** `/var/mnt/eclipse/repos/Causals.jl/src/DoCalculus.jl`

**Problem:** The `identify_effect` function at lines 41-51 only tries the backdoor criterion and immediately gives up if it fails (line 49-50: "Simplified: return false if backdoor fails"). The comment at lines 47-49 says it should try frontdoor but does not.

**What to do:**

1. After the backdoor check fails, enumerate possible mediator sets M (subsets of non-treatment, non-outcome nodes).
2. For each candidate set M, call `frontdoor_criterion(g, X, Y, M)`.
3. If any M satisfies the frontdoor criterion, return `(true, :frontdoor, M)`.
4. Only return `(false, :unidentifiable, Set{Symbol}())` if BOTH backdoor and frontdoor criteria fail for all candidate adjustment/mediator sets.
5. For the backdoor case with empty Z: also try non-empty subsets of valid adjustment sets (non-descendants of X).

**Verification:**
```julia
cd("/var/mnt/eclipse/repos/Causals.jl")
using Pkg; Pkg.activate(".")
using Causals
using Causals.CausalDAG: add_edge!
using Causals.DoCalculus

# Frontdoor-identifiable graph: X -> M -> Y, U -> X, U -> Y
g = CausalGraph([:X, :M, :Y, :U])
add_edge!(g, :X, :M)
add_edge!(g, :M, :Y)
add_edge!(g, :U, :X)
add_edge!(g, :U, :Y)

identifiable, method, set = identify_effect(g, :X, :Y)
@assert identifiable == true "Effect should be identifiable via frontdoor"
@assert method == :frontdoor "Should identify via frontdoor, got $method"
@assert :M in set "Mediator set should contain M"

# Also test backdoor still works
g2 = CausalGraph([:X, :Y, :Z])
add_edge!(g2, :X, :Y)
add_edge!(g2, :Z, :X)
add_edge!(g2, :Z, :Y)

identifiable2, method2, set2 = identify_effect(g2, :X, :Y, Set([:Z]))
@assert identifiable2 == true "Effect should be identifiable via backdoor"
@assert method2 == :backdoor "Should identify via backdoor"

println("TASK 9 PASSED: identify_effect tries frontdoor criterion")
```

---

## TASK 10: Fix example 01_basic_usage.jl API mismatches (HIGH)

**Files:** `/var/mnt/eclipse/repos/Causals.jl/examples/01_basic_usage.jl`

**Problem:** The example will not run because it uses APIs that do not match the actual source code. Specific mismatches:

- Line 30-31: `MassAssignment(Dict(...))` -- constructor requires TWO args: `MassAssignment(frame, masses_dict)`, not just a dict.
- Line 36-38: Same issue for `evidence2`.
- Line 51: `pignistic_transform(combined, frame)` -- function only takes 1 argument: `pignistic_transform(m::MassAssignment)`.
- Line 75-79: `assess_causality(assessment)` returns a tuple `(verdict, confidence)`, not a scalar. The code assigns it to `causality_score` and calls it as if it is a number, then also calls `strength_of_evidence` separately.
- Line 91: `CausalGraph(DiGraph(4))` -- constructor takes `Vector{Symbol}`, not a `DiGraph`. Should be `CausalGraph([:Education, :Income, :Health, :Exercise])`.
- Lines 94-97: `CausalDAG.add_edge!(cg, 1, 2)` -- uses integer indices, but `add_edge!` takes `Symbol` arguments.
- Line 105: `d_separation(cg, [1], [4], [2])` -- takes `Set{Symbol}`, not `Vector{Int}`.
- Lines 112-113: `ancestors(cg, 3)` and `descendants(cg, 1)` -- take `Symbol`, not `Int`.
- Line 117: `backdoor_criterion(cg, 2, 3, [1])` -- takes `Symbol, Symbol, Set{Symbol}`, not integers and vector.

**What to do:**

1. Fix all constructor calls to use `MassAssignment(frame, masses)` (two arguments).
2. Fix `pignistic_transform` call to pass only one argument.
3. Fix `assess_causality` return value handling (it returns a tuple).
4. Replace `CausalGraph(DiGraph(4))` with `CausalGraph([:Education, :Income, :Health, :Exercise])`.
5. Replace all integer-based API calls with Symbol-based calls.
6. Replace `d_separation(cg, [1], [4], [2])` with `d_separation(cg, Set([:Education]), Set([:Exercise]), Set([:Income]))`.
7. Fix `ancestors`/`descendants` calls to use Symbols.
8. Fix `backdoor_criterion` call to use Symbols and Set.
9. Remove `using Graphs` import (no longer needed after fixing CausalGraph constructor).

**Verification:**
```julia
cd("/var/mnt/eclipse/repos/Causals.jl")
using Pkg; Pkg.activate(".")
include("/var/mnt/eclipse/repos/Causals.jl/examples/01_basic_usage.jl")
println("TASK 10 PASSED: example 01 runs without errors")
```

---

## TASK 11: Fix example 02_advanced_analysis.jl API mismatches (HIGH)

**Files:** `/var/mnt/eclipse/repos/Causals.jl/examples/02_advanced_analysis.jl`

**Problem:** The example will not run because it uses APIs that do not match the actual source code. Specific mismatches:

- Line 47: `granger_test(x, y; max_lag=5)` uses keyword arg syntax. The actual function has `max_lag` as a positional argument: `granger_test(x, y, 5)`.
- Lines 48-50: `result.f_stat`, `result.p_value`, `result.causes` -- `granger_test` returns a tuple `(causes, F_stat, p_value, best_lag)`, not a named struct.
- Line 54: `optimal_lag(x, y; max_lag=8)` uses keyword arg. Actual signature: `optimal_lag(x, y, 8)`.
- Line 88: `matching(treatment, propensity_scores)` -- missing `outcome` argument. Actual signature: `matching(treatment, outcome, propensity; method=:nearest, caliper=0.1)`, returns `(matches, ate, se)`.
- Lines 93-94: Tries to index matches as `pair[1]`, `pair[2]` -- but matches are `Tuple{Int,Int}` which is indexed this way, so this may work if matching returns pairs.
- Line 108: `CausalGraph(DiGraph(3))` -- should be `CausalGraph([:Z, :X, :Y])` with Symbols.
- Lines 109-111: `CausalDAG.add_edge!(confounded_graph, 1, 2)` -- uses integers, should use Symbols.
- Line 117: `identify_effect(confounded_graph, 2, 3)` -- uses integers, should use Symbols.
- Line 123: `confounding_adjustment(confounded_graph, 2, 3)` -- completely wrong signature. The actual function takes `(treatment::Symbol, outcome::Symbol, confounders::Set{Symbol}, data::Dict{...})`, not a graph with integers.
- Lines 135-158: The counterfactual section uses `counterfactual(structural_equations, observed_values, Dict(:X => 5.0))` -- completely wrong API. Actual signature: `counterfactual(g::CausalGraph, outcome::Symbol, intervention::Pair{Symbol, Any}, observations::Dict{Symbol, Any})`.

**What to do:**

1. Fix `granger_test` call to use positional `max_lag` and destructure the tuple return.
2. Fix `optimal_lag` call to use positional argument.
3. Fix `matching` call to include all required arguments: `matching(treatment, observed_outcomes, propensity_scores)`.
4. Fix `CausalGraph` constructor to use Symbols.
5. Fix all `add_edge!` calls to use Symbols.
6. Fix `identify_effect` call to use Symbols.
7. Fix `confounding_adjustment` call to use the correct signature.
8. Fix the counterfactual section to use the actual API.
9. Remove `using Graphs` import (no longer needed).

**Verification:**
```julia
cd("/var/mnt/eclipse/repos/Causals.jl")
using Pkg; Pkg.activate(".")
include("/var/mnt/eclipse/repos/Causals.jl/examples/02_advanced_analysis.jl")
println("TASK 11 PASSED: example 02 runs without errors")
```

---

## TASK 12: Remove non-Julia junk files from examples/ (LOW)

**Files:** `/var/mnt/eclipse/repos/Causals.jl/examples/SafeDOMExample.res`, `/var/mnt/eclipse/repos/Causals.jl/examples/web-project-deno.json`

**Problem:** The `examples/` directory contains two files that have nothing to do with Causals.jl:
- `SafeDOMExample.res` -- a ReScript DOM mounting example (from a different project entirely)
- `web-project-deno.json` -- a Deno project config for ReScript web projects

These are RSR template leftovers that were never cleaned up.

**What to do:**

1. Delete `/var/mnt/eclipse/repos/Causals.jl/examples/SafeDOMExample.res`
2. Delete `/var/mnt/eclipse/repos/Causals.jl/examples/web-project-deno.json`
3. Verify only `01_basic_usage.jl` and `02_advanced_analysis.jl` remain in `examples/`.

**Verification:**
```julia
files = readdir("/var/mnt/eclipse/repos/Causals.jl/examples/")
@assert files == ["01_basic_usage.jl", "02_advanced_analysis.jl"] "examples/ should only contain Julia files, got $files"
println("TASK 12 PASSED: examples directory is clean")
```

---

## TASK 13: Add missing tests for DoCalculus, Counterfactuals, matching, stratification, doubly_robust (HIGH)

**Files:** `/var/mnt/eclipse/repos/Causals.jl/test/runtests.jl`

**Problem:** The test file has no test sections for:
- DoCalculus (no `@testset "DoCalculus"` at all)
- Counterfactuals (no `@testset "Counterfactuals"` at all)
- `matching` (the function is not tested despite being exported)
- `stratification` (not tested)
- `doubly_robust` (not tested)
- `d_separation` (not tested despite being a critical function)
- `frontdoor_criterion` (not tested)
- `markov_blanket` (not tested)

**What to do:**

1. Add a `@testset "D-Separation"` block with tests for chains, forks, and colliders (similar to Task 1 verification).
2. Add a `@testset "Frontdoor Criterion"` block with positive and negative cases.
3. Add a `@testset "Markov Blanket"` block verifying parents, children, and co-parents are included.
4. Add a `@testset "Propensity Score Matching"` block testing the `matching` function.
5. Add a `@testset "Stratification"` block testing the `stratification` function.
6. Add a `@testset "Doubly Robust"` block testing `doubly_robust`.
7. Add a `@testset "DoCalculus"` block testing `do_intervention`, `identify_effect`, `adjustment_formula`, and `confounding_adjustment`.
8. Add a `@testset "Counterfactuals"` block testing `counterfactual`, `twin_network`, `probability_of_necessity`, `probability_of_sufficiency`, and `probability_of_necessity_and_sufficiency`.
9. Each test should verify both normal operation and edge cases.

**Verification:**
```julia
cd("/var/mnt/eclipse/repos/Causals.jl")
using Pkg; Pkg.activate(".")
Pkg.test()
# Should report all test sets passing with 0 failures
```

---

## TASK 14: Create missing documentation pages referenced in docs/make.jl (MEDIUM)

**Files:** `/var/mnt/eclipse/repos/Causals.jl/docs/src/`

**Problem:** The `docs/make.jl` file at lines 13-25 references 9 documentation pages, but only 1 exists (`index.md`). Missing pages:
- `dempster_shafer.md`
- `bradford_hill.md`
- `causal_dag.md`
- `granger.md`
- `propensity.md`
- `do_calculus.md`
- `counterfactuals.md`
- `examples.md`
- `api.md`

This means `makedocs` will fail with `checkdocs = :exports`.

**What to do:**

1. Create each missing `.md` file in `/var/mnt/eclipse/repos/Causals.jl/docs/src/`.
2. Each module page should contain:
   - A brief description of the module
   - Key concepts
   - API reference using Documenter.jl `@docs` blocks for all exported functions/types
3. `examples.md` should reference the two example files with brief descriptions.
4. `api.md` should be a comprehensive API reference listing all exported symbols with `@docs` blocks.
5. Verify that `makedocs` can at least parse the pages without errors (full build requires all deps).

**Verification:**
```julia
cd("/var/mnt/eclipse/repos/Causals.jl")
expected = ["index.md", "dempster_shafer.md", "bradford_hill.md", "causal_dag.md",
            "granger.md", "propensity.md", "do_calculus.md", "counterfactuals.md",
            "examples.md", "api.md"]
actual = readdir("docs/src")
for page in expected
    @assert page in actual "Missing docs page: $page"
end
println("TASK 14 PASSED: all documentation pages exist")
```

---

## TASK 15: Fix version inconsistency between Project.toml and Manifest.toml (LOW)

**Files:** `/var/mnt/eclipse/repos/Causals.jl/Project.toml`, `/var/mnt/eclipse/repos/Causals.jl/Manifest.toml`

**Problem:** `Project.toml` line 5 says `version = "1.0.0"` but `Manifest.toml` line 33 says `version = "0.1.0"`. Additionally, the git tags show both `v0.1.0` and `v1.0.0` exist. Given the actual state of the code (many stubs and placeholders), `1.0.0` is dishonest. The version should be `0.2.0` at most until all stubs are implemented.

**What to do:**

1. Change `Project.toml` line 5 to `version = "0.2.0"`.
2. Delete the `Manifest.toml` file and regenerate it: `cd /var/mnt/eclipse/repos/Causals.jl && julia --project=. -e 'using Pkg; Pkg.resolve(); Pkg.instantiate()'`
3. The regenerated Manifest.toml will have the correct version.

**Verification:**
```julia
cd("/var/mnt/eclipse/repos/Causals.jl")
toml = read("Project.toml", String)
@assert occursin("version = \"0.2.0\"", toml) "Project.toml should have version 0.2.0"
println("TASK 15 PASSED: version is honest")
```

---

## TASK 16: Fix ROADMAP.md claims about non-existent modules (LOW)

**Files:** `/var/mnt/eclipse/repos/Causals.jl/ROADMAP.md`

**Problem:** Lines 7-13 of ROADMAP.md claim "Production-ready implementation of 7 causal inference methods" and list: Dempster-Shafer, Conditional Probability, Probabilistic Logic, Bayesian Networks, Fuzzy Logic, Evidential Reasoning, Belief Propagation. In reality, the codebase has 7 DIFFERENT modules: DempsterShafer, BradfordHill, CausalDAG, Granger, PropensityScore, DoCalculus, Counterfactuals. The ROADMAP lists modules that do not exist.

Line 14 claims "Complete with comprehensive test coverage (66 tests)" -- the actual test file has far fewer distinct assertions, and zero tests for 3 of the 7 modules.

**What to do:**

1. Replace lines 5-14 of ROADMAP.md with accurate module listing:
   - DempsterShafer (mostly complete)
   - BradfordHill (complete)
   - CausalDAG (d_separation and frontdoor_criterion are stubs)
   - Granger (p-value computation is fake)
   - PropensityScore (propensity_score and doubly_robust are stubs)
   - DoCalculus (confounding_adjustment and do_calculus_rules are stubs)
   - Counterfactuals (counterfactual function is stub)
2. Update the status line to say "Alpha" or "In development" instead of "Production-ready".
3. Update test count to actual number.

**Verification:**
```julia
cd("/var/mnt/eclipse/repos/Causals.jl")
roadmap = read("ROADMAP.md", String)
@assert !occursin("Bayesian Networks", roadmap) "ROADMAP should not claim Bayesian Networks exist"
@assert !occursin("Fuzzy Logic", roadmap) "ROADMAP should not claim Fuzzy Logic exists"
@assert !occursin("Production-ready", roadmap) "ROADMAP should not claim production-ready"
println("TASK 16 PASSED: ROADMAP is honest")
```

---

## TASK 17: Fix CITATIONS.adoc template placeholders (LOW)

**Files:** `/var/mnt/eclipse/repos/Causals.jl/docs/CITATIONS.adoc`

**Problem:** The entire file is an unmodified RSR template. Line 1 says `= RSR-template-repo - Citation Guide`, line 8 says `rsr-template-repo_2025`, and line 14 references `AGPL-3.0-or-later` (banned license). All references point to `hyperpolymath/RSR-template-repo` instead of `hyperpolymath/Causals.jl`.

**What to do:**

1. Replace all occurrences of `RSR-template-repo` with `Causals.jl`.
2. Replace `rsr-template-repo` with `causals_jl`.
3. Replace `AGPL-3.0-or-later` with `PMPL-1.0-or-later`.
4. Replace `Polymath, Hyper` / `Hyper Polymath` with `Jewell, Jonathan D.A.` / `Jonathan D.A. Jewell`.
5. Update the year to 2026 if applicable.
6. Update URLs to point to `hyperpolymath/Causals.jl`.

**Verification:**
```julia
cd("/var/mnt/eclipse/repos/Causals.jl")
citations = read("docs/CITATIONS.adoc", String)
@assert !occursin("RSR-template-repo", citations) "Should not reference template repo"
@assert !occursin("AGPL", citations) "Should not reference AGPL license"
@assert occursin("Causals.jl", citations) "Should reference Causals.jl"
@assert occursin("PMPL", citations) "Should reference PMPL license"
println("TASK 17 PASSED: CITATIONS.adoc is properly customized")
```

---

## TASK 18: Fix ROADMAP.adoc template placeholders (LOW)

**Files:** `/var/mnt/eclipse/repos/Causals.jl/ROADMAP.adoc`

**Problem:** The entire file is the unmodified RSR template. Line 2 says `= YOUR Template Repo Roadmap`. All milestones are generic placeholders (`Core functionality`, `Basic documentation`, `Full feature set`).

**What to do:**

1. Replace the title with `= Causals.jl Roadmap`.
2. Replace the generic milestones with actual Causals.jl milestones (can be based on ROADMAP.md content, corrected per Task 16).
3. Or delete this file entirely since ROADMAP.md already exists and serves the same purpose. Having both is confusing.

**Verification:**
```julia
cd("/var/mnt/eclipse/repos/Causals.jl")
if isfile("ROADMAP.adoc")
    roadmap = read("ROADMAP.adoc", String)
    @assert !occursin("YOUR Template Repo", roadmap) "Should not be template placeholder"
    @assert occursin("Causals", roadmap) "Should reference Causals.jl"
end
println("TASK 18 PASSED: ROADMAP.adoc is resolved")
```

---

## TASK 19: Fix AI.a2ml to reference correct directory and project (LOW)

**Files:** `/var/mnt/eclipse/repos/Causals.jl/AI.a2ml`

**Problem:** Line 5 says `rsr-template-repo is treated as a Rhodium Standard Repository` -- should reference Causals.jl. Line 5 also references `.machines_readable/6scm/` (wrong path -- should be `.machine_readable/`). Lines 9-10 reference `.machines_readable/6scm/STATE.scm` and `.machines_readable/6scm/AGENTIC.scm` with the wrong directory name and a nonexistent `6scm` subdirectory. There is no `.machine_readable/` directory at all in this repo.

**What to do:**

1. Replace `rsr-template-repo` with `Causals.jl` on line 5.
2. Fix directory references from `.machines_readable/6scm/` to `.machine_readable/`.
3. Create the `.machine_readable/` directory with at minimum `STATE.scm`, `ECOSYSTEM.scm`, and `META.scm` files.
4. Update all path references throughout the file.

**Verification:**
```julia
cd("/var/mnt/eclipse/repos/Causals.jl")
ai = read("AI.a2ml", String)
@assert !occursin("rsr-template-repo", ai) "Should not reference template repo"
@assert occursin("Causals.jl", ai) "Should reference Causals.jl"
@assert isdir(".machine_readable") ".machine_readable directory must exist"
println("TASK 19 PASSED: AI.a2ml is properly configured")
```

---

## FINAL VERIFICATION

After completing all tasks, run the following to verify the entire package is working:

```julia
cd("/var/mnt/eclipse/repos/Causals.jl")
using Pkg
Pkg.activate(".")

# 1. Full test suite
Pkg.test()

# 2. Run both examples
include("examples/01_basic_usage.jl")
include("examples/02_advanced_analysis.jl")

# 3. Verify no placeholder code remains
for f in ["src/CausalDAG.jl", "src/DoCalculus.jl", "src/Counterfactuals.jl",
          "src/Granger.jl", "src/PropensityScore.jl"]
    content = read(f, String)
    @assert !occursin("# Placeholder", content) "Placeholder found in $f"
    @assert !occursin("# Simplified", content) || occursin("# Simplified standard error", content) "Stub found in $f"
end

# 4. Verify version
toml = read("Project.toml", String)
@assert occursin("version = \"0.2.0\"", toml) "Version should be 0.2.0"

# 5. Verify docs pages exist
expected_docs = ["index.md", "dempster_shafer.md", "bradford_hill.md", "causal_dag.md",
                 "granger.md", "propensity.md", "do_calculus.md", "counterfactuals.md",
                 "examples.md", "api.md"]
actual_docs = readdir("docs/src")
for page in expected_docs
    @assert page in actual_docs "Missing doc: $page"
end

# 6. Verify no junk files in examples
example_files = readdir("examples")
@assert all(endswith.(example_files, ".jl")) "Only .jl files should be in examples/"

println("=" ^ 60)
println("ALL VERIFICATION PASSED - Causals.jl is genuinely complete")
println("=" ^ 60)
```
