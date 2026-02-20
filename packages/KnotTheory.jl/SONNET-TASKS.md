# SONNET-TASKS.md -- KnotTheory.jl Completion Tasks

> **Generated:** 2026-02-12 by Opus audit
> **Purpose:** Unambiguous instructions for Sonnet to complete all stubs, TODOs, and placeholder code.
> **Honest completion before this file:** 62%

The Julia knot-theory library (`src/KnotTheory.jl`) is a genuine, working module with
real data structures, invariants, and tests that pass. However, there are
significant issues: the Alexander polynomial is a self-described "placeholder"
that produces wrong results, the `to_polynomial` helper crashes on negative
exponents (which `jones_polynomial` routinely produces), several RSR template
files still have unreplaced `{{PLACEHOLDER}}` markers, SPDX headers use
the banned `AGPL-3.0-or-later` in multiple files, the `.machine_readable/`
directory is entirely missing, the `examples/` directory contains files
irrelevant to knot theory (a ReScript SafeDOM example and a Deno JSON config),
the version in `Project.toml` (1.0.0) contradicts `Manifest.toml` (0.1.0),
and three Idris2 proofs are holes (`?`-prefixed).

---

## GROUND RULES FOR SONNET

1. Read this entire file before starting any task.
2. Do tasks in order listed. Earlier tasks unblock later ones.
3. After each task, run the verification command. If it fails, fix before moving on.
4. Do NOT mark done unless verification passes.
5. Update STATE.scm with honest completion percentages after each task.
6. Commit after each task: `fix(component): complete <description>`
7. Run full test suite after every 3 tasks: `cd /var/mnt/eclipse/repos/KnotTheory.jl && julia --project=. -e 'using Pkg; Pkg.test()'`

---

## TASK 1: Fix `to_polynomial` crash on negative exponents (CRITICAL)

**Problem:** `jones_polynomial` returns a `Dict{Int,Int}` where keys are often
negative (e.g., `Dict(-5 => 1, -3 => -1, -1 => 1)`). The `to_polynomial`
function at line 442-452 of `src/KnotTheory.jl` computes `max_exp + 1` and
builds a `coeffs` array indexed `exp + 1`. When `exp` is negative, this
produces an out-of-bounds index. This means Jones polynomial results cannot be
converted to `Polynomials.Polynomial` objects.

**File:** `/var/mnt/eclipse/repos/KnotTheory.jl/src/KnotTheory.jl`, lines 442-452

**Fix:** Rewrite `to_polynomial` to handle negative exponents by using
`min_exp` as an offset. Shift all exponents so the lowest becomes index 1.
The resulting `Polynomial` should carry correct coefficients; callers can
interpret the minimum exponent separately.

**Exact function to replace:**
```julia
function to_polynomial(dict::Dict{Int, Int})
    if isempty(dict)
        return Polynomial([0])
    end
    max_exp = maximum(collect(keys(dict)))
    coeffs = zeros(Int, max_exp + 1)
    for (exp, coeff) in dict
        coeffs[exp + 1] = coeff
    end
    Polynomial(coeffs)
end
```

**Replace with a version that:**
1. Computes `min_exp = minimum(keys(dict))` and `max_exp = maximum(keys(dict))`.
2. Allocates `coeffs` of length `max_exp - min_exp + 1`.
3. Uses `coeffs[exp - min_exp + 1] = coeff` for placement.
4. Returns a tuple `(Polynomial(coeffs), min_exp)` so the caller knows the
   leading exponent offset. Alternatively, keep the return type as just
   `Polynomial` but document that the variable represents `t^min_exp * poly(t)`.
   Choose the tuple approach since it is unambiguous.
5. Update the export list and docstring accordingly.

**Verification:**
```bash
cd /var/mnt/eclipse/repos/KnotTheory.jl && julia --project=. -e '
using KnotTheory
pd = KnotTheory.pdcode([(1,2,3,4,1)])
j = KnotTheory.jones_polynomial(pd; wr=1)
println("Jones dict: ", j)
poly, offset = KnotTheory.to_polynomial(j)
println("Polynomial: ", poly, " with offset t^", offset)
@assert poly isa KnotTheory.Polynomials.Polynomial
println("PASS: to_polynomial handles negative exponents")
'
```

---

## TASK 2: Fix the Alexander polynomial placeholder (HIGH)

**Problem:** The `alexander_polynomial` function at lines 263-306 of
`src/KnotTheory.jl` is explicitly marked as a "crude expansion" and a
"placeholder". It builds a Seifert matrix `V` using `(a % n) + 1` and
`(c % n) + 1` as indices, which is mathematically unjustified. It then
evaluates `det(V)` and `det(V - V')` as an approximation. This does not
compute the actual Alexander polynomial for any non-trivial knot.

**File:** `/var/mnt/eclipse/repos/KnotTheory.jl/src/KnotTheory.jl`, lines 263-306

**Fix:** Implement a correct Alexander polynomial computation. The standard
approach for a planar diagram:

1. Build the actual Seifert matrix from the Seifert surface: for each pair
   of Seifert circles (i, j), compute the linking number contribution from
   each crossing that connects circles i and j.
2. Compute `det(V - t * V')` as a polynomial in `t` using the `Polynomials`
   package directly. The matrix `V - t * V'` is a matrix of `Polynomial`
   entries; take its determinant symbolically.
3. For small matrices (say n <= 10), a cofactor expansion or Bareiss-like
   algorithm over polynomials works. For the scope of this library (up to
   20 crossings), this is adequate.
4. Normalize the result so the polynomial is symmetric and evaluates to 1
   at t=1 (standard normalization).
5. Return a `Dict{Int,Int}` mapping exponent to coefficient, consistent
   with the existing API.

**Keep the signature:** `alexander_polynomial(pd::PlanarDiagram) -> Dict{Int,Int}`

**Verification:**
```bash
cd /var/mnt/eclipse/repos/KnotTheory.jl && julia --project=. -e '
using KnotTheory
# Trefoil knot PD code: standard positive trefoil
# PD notation: X[1,4,2,5], X[3,6,4,1], X[5,2,6,3] with all positive crossings
pd = KnotTheory.pdcode([
    (1, 4, 2, 5, 1),
    (3, 6, 4, 1, 1),
    (5, 2, 6, 3, 1)
])
alex = KnotTheory.alexander_polynomial(pd)
println("Alexander poly of trefoil: ", alex)
# The Alexander polynomial of the trefoil is t^-1 - 1 + t  (up to normalization)
# Verify it is non-trivial (not just {0 => 1})
@assert length(alex) > 1 || (length(alex) == 1 && !haskey(alex, 0))
println("PASS: Alexander polynomial is non-trivial for trefoil")
'
```

---

## TASK 3: Fix version mismatch between Project.toml and Manifest.toml (HIGH)

**Problem:** `Project.toml` line 4 says `version = "1.0.0"` but
`Manifest.toml` line 101 says `version = "0.1.0"`. The README.md and
ROADMAP.md refer to this as "v1.0" and "Production-ready". This is an
early-stage library with placeholder algorithms; it is not v1.0.

**Files:**
- `/var/mnt/eclipse/repos/KnotTheory.jl/Project.toml`, line 4
- `/var/mnt/eclipse/repos/KnotTheory.jl/ROADMAP.md`, line 3 and line 14

**Fix:**
1. Change `Project.toml` version to `"0.1.0"`.
2. Update `ROADMAP.md` to say "v0.1.0" / "Current State (v0.1.0)" and
   remove the claim "Production-ready" (replace with "Early development").
3. Regenerate `Manifest.toml` by running `julia --project=. -e 'using Pkg; Pkg.resolve()'`.

**Verification:**
```bash
cd /var/mnt/eclipse/repos/KnotTheory.jl && julia --project=. -e '
using Pkg
ctx = Pkg.Types.Context()
proj = ctx.env.project
println("Project version: ", proj.version)
@assert string(proj.version) == "0.1.0" "Version should be 0.1.0, got $(proj.version)"
println("PASS: version is 0.1.0")
'
```

---

## TASK 4: Fix SPDX license headers -- replace AGPL-3.0-or-later with PMPL-1.0-or-later (HIGH)

**Problem:** Five files use the banned `AGPL-3.0-or-later` SPDX identifier:
1. `/var/mnt/eclipse/repos/KnotTheory.jl/ffi/zig/build.zig` (line 2)
2. `/var/mnt/eclipse/repos/KnotTheory.jl/ffi/zig/src/main.zig` (line 6)
3. `/var/mnt/eclipse/repos/KnotTheory.jl/ffi/zig/test/integration_test.zig` (line 2)
4. `/var/mnt/eclipse/repos/KnotTheory.jl/examples/SafeDOMExample.res` (line 1)
5. `/var/mnt/eclipse/repos/KnotTheory.jl/docs/CITATIONS.adoc` (line 13, in BibTeX block)

Per CLAUDE.md: "NEVER use AGPL-3.0".

**Fix:** In each file, replace `AGPL-3.0-or-later` with `PMPL-1.0-or-later`.

**Verification:**
```bash
cd /var/mnt/eclipse/repos/KnotTheory.jl && \
  ! grep -r "AGPL" --include="*.zig" --include="*.res" --include="*.adoc" . && \
  echo "PASS: No AGPL references remain"
```

---

## TASK 5: Replace all unreplaced `{{PLACEHOLDER}}` template markers (HIGH)

**Problem:** Ten files still contain unreplaced `{{PROJECT}}`, `{{project}}`,
`{{OWNER}}`, `{{REPO}}`, `{{FORGE}}`, `{{SECURITY_EMAIL}}`, `{{PGP_FINGERPRINT}}`,
`{{PGP_KEY_URL}}`, `{{WEBSITE}}`, `{{CURRENT_YEAR}}`, `{{PROJECT_NAME}}`,
`{{CONDUCT_EMAIL}}`, `{{CONDUCT_TEAM}}`, `{{RESPONSE_TIME}}`, `{{MAIN_BRANCH}}`,
and `{{LICENSE}}` markers from the RSR template. These files are:

1. `/var/mnt/eclipse/repos/KnotTheory.jl/ffi/zig/build.zig` -- `{{PROJECT}}`, `{{project}}`
2. `/var/mnt/eclipse/repos/KnotTheory.jl/ffi/zig/src/main.zig` -- `{{PROJECT}}`, `{{project}}`
3. `/var/mnt/eclipse/repos/KnotTheory.jl/ffi/zig/test/integration_test.zig` -- `{{PROJECT}}`, `{{project}}`
4. `/var/mnt/eclipse/repos/KnotTheory.jl/src/abi/Types.idr` -- `{{PROJECT}}`
5. `/var/mnt/eclipse/repos/KnotTheory.jl/src/abi/Layout.idr` -- `{{PROJECT}}`
6. `/var/mnt/eclipse/repos/KnotTheory.jl/src/abi/Foreign.idr` -- `{{PROJECT}}`, `{{project}}`
7. `/var/mnt/eclipse/repos/KnotTheory.jl/ABI-FFI-README.md` -- `{{PROJECT}}`, `{{project}}`, `{{LICENSE}}`
8. `/var/mnt/eclipse/repos/KnotTheory.jl/SECURITY.md` -- `{{OWNER}}`, `{{REPO}}`, `{{SECURITY_EMAIL}}`, `{{PGP_FINGERPRINT}}`, `{{PGP_KEY_URL}}`, `{{WEBSITE}}`, `{{CURRENT_YEAR}}`, `{{PROJECT_NAME}}`
9. `/var/mnt/eclipse/repos/KnotTheory.jl/CODE_OF_CONDUCT.md` -- `{{PROJECT_NAME}}`, `{{OWNER}}`, `{{REPO}}`, `{{CONDUCT_EMAIL}}`, `{{CONDUCT_TEAM}}`, `{{RESPONSE_TIME}}`, `{{CURRENT_YEAR}}`, `{{FORGE}}`
10. `/var/mnt/eclipse/repos/KnotTheory.jl/CONTRIBUTING.md` -- `{{FORGE}}`, `{{OWNER}}`, `{{REPO}}`, `{{MAIN_BRANCH}}`

**Fix:** Replace with these values:
- `{{PROJECT}}` -> `KnotTheory`
- `{{project}}` -> `knottheory`
- `{{OWNER}}` -> `hyperpolymath`
- `{{REPO}}` -> `KnotTheory.jl`
- `{{FORGE}}` -> `github.com`
- `{{SECURITY_EMAIL}}` -> `jonathan.jewell@open.ac.uk`
- `{{PGP_FINGERPRINT}}` -> (remove the PGP section or leave a note to fill in)
- `{{PGP_KEY_URL}}` -> (remove the PGP section or leave a note to fill in)
- `{{WEBSITE}}` -> `https://github.com/hyperpolymath`
- `{{CURRENT_YEAR}}` -> `2026`
- `{{PROJECT_NAME}}` -> `KnotTheory.jl`
- `{{CONDUCT_EMAIL}}` -> `jonathan.jewell@open.ac.uk`
- `{{CONDUCT_TEAM}}` -> `KnotTheory.jl Maintainers`
- `{{RESPONSE_TIME}}` -> `48 hours`
- `{{MAIN_BRANCH}}` -> `main`
- `{{LICENSE}}` -> `PMPL-1.0-or-later`

Also delete the HTML comment blocks that say "TEMPLATE INSTRUCTIONS (delete this block before publishing)" from SECURITY.md (lines 3-19) and CODE_OF_CONDUCT.md (lines 3-21).

**Verification:**
```bash
cd /var/mnt/eclipse/repos/KnotTheory.jl && \
  ! grep -r '{{' --include="*.md" --include="*.zig" --include="*.idr" --include="*.adoc" . && \
  echo "PASS: No template placeholders remain"
```

---

## TASK 6: Create `.machine_readable/` directory with STATE.scm, META.scm, ECOSYSTEM.scm (HIGH)

**Problem:** The `.machine_readable/` directory is entirely absent. Per
CLAUDE.md, SCM files MUST be in `.machine_readable/` only, never in root.
There are no SCM files anywhere in the repo.

**Fix:** Create the directory and populate three files:

### `.machine_readable/STATE.scm`
```scheme
(define state
  `((metadata
     (project . "KnotTheory.jl")
     (version . "0.1.0")
     (updated . "2026-02-12"))
    (project-context
     (description . "Julia toolkit for knot theory: planar diagrams, invariants, polynomials")
     (language . "Julia")
     (category . "mathematics"))
    (current-position
     (phase . implementation)
     (maturity . alpha)
     (completion-percentage . 62))
    (route-to-mvp
     (milestones
       ((name . "correct-alexander")
        (status . incomplete)
        (description . "Replace Alexander polynomial placeholder with correct Seifert matrix algorithm"))
       ((name . "homfly-pt")
        (status . not-started)
        (description . "Implement HOMFLY-PT two-variable polynomial"))
       ((name . "knot-table")
        (status . incomplete)
        (description . "Expand knot table beyond 3 entries"))
       ((name . "reidemeister-ii-iii")
        (status . not-started)
        (description . "Implement Reidemeister II and III moves"))))
    (blockers-and-issues
     ((blocker . "Alexander polynomial is a placeholder producing wrong results")
      (severity . high))
     ((blocker . "to_polynomial crashes on negative exponents from Jones polynomial")
      (severity . critical)))
    (critical-next-actions
     ("Fix to_polynomial negative exponent handling"
      "Implement correct Alexander polynomial"
      "Add Reidemeister II and III simplifications"
      "Expand knot table with Rolfsen data"))))
```

### `.machine_readable/META.scm`
```scheme
(define meta
  `((architecture-decisions
     ((id . "ADR-001")
      (title . "Dict-based polynomial representation")
      (status . accepted)
      (rationale . "Using Dict{Int,Int} for polynomial coefficients allows negative exponents and sparse representation"))
     ((id . "ADR-002")
      (title . "Kauffman bracket for Jones polynomial")
      (status . accepted)
      (rationale . "State-sum expansion is correct for small crossing numbers; limited to 20 crossings")))
    (development-practices
     (testing . "julia --project=. -e 'using Pkg; Pkg.test()'")
     (language . "Julia 1.9+")
     (license . "PMPL-1.0-or-later"))))
```

### `.machine_readable/ECOSYSTEM.scm`
```scheme
(define ecosystem
  `((version . "1.0")
    (name . "KnotTheory.jl")
    (type . "julia-package")
    (purpose . "Knot theory computations: planar diagrams, invariants, polynomial invariants")
    (position-in-ecosystem . "standalone-library")
    (related-projects
     ((name . "Graphs.jl")
      (relationship . dependency)
      (description . "Used for graph representation of planar diagrams"))
     ((name . "Polynomials.jl")
      (relationship . dependency)
      (description . "Used for polynomial arithmetic"))
     ((name . "CairoMakie")
      (relationship . optional-dependency)
      (description . "Used for knot diagram plotting via package extension")))))
```

**Verification:**
```bash
cd /var/mnt/eclipse/repos/KnotTheory.jl && \
  test -f .machine_readable/STATE.scm && \
  test -f .machine_readable/META.scm && \
  test -f .machine_readable/ECOSYSTEM.scm && \
  echo "PASS: .machine_readable/ directory exists with all 3 SCM files"
```

---

## TASK 7: Remove irrelevant example files (MEDIUM)

**Problem:** The `examples/` directory contains two files that have nothing to
do with knot theory:

1. `examples/SafeDOMExample.res` -- A ReScript SafeDOM example from the RSR
   template. Not relevant to a Julia knot theory library.
2. `examples/web-project-deno.json` -- A Deno project config for ReScript web
   projects. Not relevant.

**Fix:**
1. Delete `examples/SafeDOMExample.res`.
2. Delete `examples/web-project-deno.json`.
3. Create `examples/basic_usage.jl` with actual knot theory examples:

```julia
# SPDX-License-Identifier: PMPL-1.0-or-later
# Basic usage examples for KnotTheory.jl

using KnotTheory

# --- Create knots from the built-in table ---
k = trefoil()
println("Trefoil crossing number: ", crossing_number(k))
println("Trefoil DT code: ", dtcode(k).code)

fe = figure_eight()
println("Figure-eight crossing number: ", crossing_number(fe))

# --- Build a knot from PD code ---
# Single positive crossing
pd = pdcode([(1, 2, 3, 4, 1)])
sample = Knot(:sample, pd, nothing)
println("Sample writhe: ", writhe(sample))

# --- Compute invariants ---
println("Seifert circles: ", seifert_circles(pd))
println("Braid index estimate: ", braid_index_estimate(pd))

# --- Jones polynomial ---
jones = jones_polynomial(pd; wr=1)
println("Jones polynomial (dict): ", jones)

# --- Simplification ---
# A crossing with repeated arcs (R1 reducible)
pd_loop = pdcode([(1, 1, 2, 2, 1)])
reduced = r1_simplify(pd_loop)
println("Before R1: ", length(pd_loop.crossings), " crossings")
println("After R1: ", length(reduced.crossings), " crossings")

# --- JSON round-trip ---
path = tempname() * ".json"
write_knot_json(path, sample)
loaded = read_knot_json(path)
println("Round-tripped knot name: ", loaded.name)
rm(path)

# --- Graph conversion ---
using Graphs
g = to_graph(pd)
println("Graph vertices: ", nv(g), ", edges: ", ne(g))

# --- Knot table ---
table = knot_table()
for (name, entry) in table
    println("  ", name, ": ", entry.crossings, " crossings")
end
```

**Verification:**
```bash
cd /var/mnt/eclipse/repos/KnotTheory.jl && \
  test ! -f examples/SafeDOMExample.res && \
  test ! -f examples/web-project-deno.json && \
  test -f examples/basic_usage.jl && \
  julia --project=. examples/basic_usage.jl && \
  echo "PASS: examples/ contains only relevant knot theory examples and they run"
```

---

## TASK 8: Fix CITATIONS.adoc to reference KnotTheory.jl instead of RSR-template-repo (MEDIUM)

**Problem:** `/var/mnt/eclipse/repos/KnotTheory.jl/docs/CITATIONS.adoc`
still references `rsr-template-repo` everywhere and uses
`AGPL-3.0-or-later` for the license. It also attributes authorship to
`Polymath, Hyper` instead of the correct `Jewell, Jonathan D.A.`.

**File:** `/var/mnt/eclipse/repos/KnotTheory.jl/docs/CITATIONS.adoc`

**Fix:** Replace all occurrences of:
- `rsr-template-repo` -> `KnotTheory.jl`
- `RSR-template-repo` -> `KnotTheory.jl`
- `Polymath, Hyper` / `Hyper Polymath` -> `Jewell, Jonathan D.A.` / `Jonathan D.A. Jewell`
- `AGPL-3.0-or-later` -> `PMPL-1.0-or-later`
- `2025` -> `2026` (year)
- Fix the title line to say `= KnotTheory.jl - Citation Guide`
- Fix URLs to point to `https://github.com/hyperpolymath/KnotTheory.jl`

**Verification:**
```bash
cd /var/mnt/eclipse/repos/KnotTheory.jl && \
  ! grep -i "rsr-template" docs/CITATIONS.adoc && \
  ! grep "AGPL" docs/CITATIONS.adoc && \
  grep "KnotTheory.jl" docs/CITATIONS.adoc > /dev/null && \
  grep "Jewell" docs/CITATIONS.adoc > /dev/null && \
  echo "PASS: CITATIONS.adoc correctly references KnotTheory.jl"
```

---

## TASK 9: Add Reidemeister II and III simplification (MEDIUM)

**Problem:** Only Reidemeister I simplification is implemented (`r1_simplify`
at line 244). The `simplify_pd` function (line 257) delegates exclusively to
`r1_simplify`. Reidemeister II (removing two crossings that cancel) and
Reidemeister III (triangle move) are listed in the ROADMAP as "SHOULD" for
v1.1 but are needed for basic diagram simplification to work on any
non-trivial knot.

**File:** `/var/mnt/eclipse/repos/KnotTheory.jl/src/KnotTheory.jl`

**Fix:**
1. Add `r2_simplify(pd::PlanarDiagram)::PlanarDiagram` -- detect pairs of
   crossings where two arcs connect the same two crossings with opposite
   signs and can be removed (bigon removal).
2. Add `r3_simplify(pd::PlanarDiagram)::PlanarDiagram` -- detect a triangle
   of three crossings where a strand can be slid across (this is a
   topology-preserving move, not a reduction, so it is optional for
   simplification but should be available).
3. Update `simplify_pd` to iterate R1 and R2 moves until no further
   reduction occurs (a fixed-point loop).
4. Export `r2_simplify` and `r3_simplify`.
5. Add tests for each move.

**Verification:**
```bash
cd /var/mnt/eclipse/repos/KnotTheory.jl && julia --project=. -e '
using KnotTheory
# Test R2: two crossings that form a bigon should cancel
# Construct a PD with two crossings that form an R2 pair
pd = KnotTheory.pdcode([
    (1, 2, 3, 4, 1),
    (3, 4, 1, 2, -1)
])
reduced = KnotTheory.r2_simplify(pd)
println("R2: $(length(pd.crossings)) -> $(length(reduced.crossings)) crossings")
@assert length(reduced.crossings) < length(pd.crossings) "R2 should reduce crossing count"
println("PASS: Reidemeister II simplification works")
'
```

---

## TASK 10: Add tests for edge cases and improve test coverage (MEDIUM)

**Problem:** The test file (`test/runtests.jl`) has only 8 test sets with
basic happy-path tests. Missing coverage includes:

1. No test for `linking_number` on actual multi-component links.
2. No test for `to_dowker` correctness.
3. No test for `jones_polynomial` on a known knot with a known answer.
4. No test for the CairoMakie extension (even a conditional test).
5. No edge case tests (empty PD, single crossing, very large arc labels).
6. No test for `to_polynomial` with negative exponents (will be needed
   after Task 1).
7. No test that the Alexander polynomial produces correct results for known
   knots (will be needed after Task 2).

**File:** `/var/mnt/eclipse/repos/KnotTheory.jl/test/runtests.jl`

**Fix:** Add the following test sets:

1. `@testset "Linking Number"` -- Create a Hopf link with known linking
   number +/-1 and verify.
2. `@testset "Dowker Code"` -- Verify `to_dowker` produces correct codes
   for trefoil PD.
3. `@testset "Jones Known Values"` -- Verify Jones polynomial of trefoil PD
   matches the known result (up to normalization).
4. `@testset "Edge Cases"` -- Empty PD, unknot, single crossing with repeated
   arcs.
5. `@testset "to_polynomial negative exponents"` -- Verify the tuple return
   from Task 1 works correctly.
6. `@testset "Alexander Known Values"` -- After Task 2, verify Alexander
   polynomial of trefoil matches `t^-1 - 1 + t`.

**Verification:**
```bash
cd /var/mnt/eclipse/repos/KnotTheory.jl && julia --project=. -e '
using Pkg; Pkg.test()
' 2>&1 | tail -5
```

---

## TASK 11: Expand knot table beyond 3 entries (LOW)

**Problem:** `knot_table()` at lines 524-530 only contains unknot, trefoil,
and figure-eight. The ROADMAP lists "Knot table integration" with "10K+ knots
up to 16 crossings" as a v1.1 goal. For a minimal viable library, at least
the prime knots through 7 crossings should be present (15 knots total).

**File:** `/var/mnt/eclipse/repos/KnotTheory.jl/src/KnotTheory.jl`, lines 524-530

**Fix:** Expand `knot_table()` to include all prime knots through 7 crossings
using their DT codes. The standard Rolfsen table entries are:

- 0_1 (unknot): DT=[]
- 3_1 (trefoil): DT=[4,6,2]
- 4_1 (figure-eight): DT=[4,6,8,2]
- 5_1: DT=[6,8,10,2,4]
- 5_2: DT=[4,8,10,2,6]
- 6_1: DT=[4,8,12,2,10,6]
- 6_2: DT=[4,8,10,12,2,6]
- 6_3: DT=[4,8,10,2,12,6]
- 7_1: DT=[8,10,12,14,2,4,6]
- 7_2: DT=[4,10,14,12,2,8,6]
- 7_3: DT=[4,12,10,14,2,8,6]
- 7_4: DT=[6,10,14,12,2,4,8]
- 7_5: DT=[6,10,14,8,2,4,12]
- 7_6: DT=[4,10,14,8,2,12,6]
- 7_7: DT=[4,12,14,8,2,10,6]

Also update `lookup_knot` to handle the symbol names (e.g., `Symbol("5_1")`).
Add convenience constructors for each (e.g., `knot_5_1()`), or at minimum
ensure `lookup_knot(Symbol("5_1"))` works.

**Verification:**
```bash
cd /var/mnt/eclipse/repos/KnotTheory.jl && julia --project=. -e '
using KnotTheory
table = knot_table()
@assert length(table) >= 15 "Expected at least 15 knots, got $(length(table))"
entry = lookup_knot(Symbol("5_1"))
@assert entry !== nothing "5_1 should be in the table"
@assert entry.crossings == 5
println("PASS: Knot table has $(length(table)) entries including 5_1")
'
```

---

## TASK 12: Fix the intro notebook to be a useful tutorial (LOW)

**Problem:** `tutorials/intro.ipynb` contains only two cells: a markdown
title and a single Julia cell `using KnotTheory; trefoil()`. This is not
a useful tutorial.

**File:** `/var/mnt/eclipse/repos/KnotTheory.jl/tutorials/intro.ipynb`

**Fix:** Expand the notebook to include cells demonstrating:
1. Installing/loading KnotTheory.jl
2. Creating knots from the table (unknot, trefoil, figure-eight)
3. Computing crossing number and writhe
4. Building a PD code from scratch
5. Computing Alexander and Jones polynomials
6. Simplifying diagrams with Reidemeister moves
7. JSON import/export round-trip
8. Graph conversion

Each section should have a markdown cell explaining the concept and a code
cell demonstrating it.

**Verification:**
```bash
cd /var/mnt/eclipse/repos/KnotTheory.jl && julia --project=. -e '
using JSON3
nb = JSON3.read(read("tutorials/intro.ipynb", String))
cells = nb["cells"]
println("Notebook has $(length(cells)) cells")
@assert length(cells) >= 10 "Tutorial should have at least 10 cells, got $(length(cells))"
println("PASS: Tutorial notebook has sufficient content")
'
```

---

## TASK 13: Fix Idris2 proof holes in Layout.idr (LOW)

**Problem:** `/var/mnt/eclipse/repos/KnotTheory.jl/src/abi/Layout.idr`
contains three unfinished proof holes (Idris2 `?`-prefixed metavariables):

1. Line 138: `?fieldsAlignedProof` in `checkCABI`
2. Line 159: `?exampleFieldsAligned` in `exampleLayoutValid`
3. Line 176: `?offsetInBoundsProof` in `offsetInBounds`

These are admitted proofs that the Idris2 compiler will accept but are not
actually proven.

**File:** `/var/mnt/eclipse/repos/KnotTheory.jl/src/abi/Layout.idr`

**Fix:** Either:
(a) Complete the proofs properly using Idris2 proof tactics, OR
(b) If the ABI/FFI layer is not actually used by this Julia package (it is
    RSR template boilerplate), add a comment `-- NOTE: Template proof hole;
    KnotTheory.jl does not use the Idris2 ABI layer` to each hole to make
    the incomplete status explicit.

Option (b) is recommended since KnotTheory.jl is a pure Julia package and
the Idris2/Zig ABI-FFI layer is RSR template scaffolding that has no
functional connection to the knot theory code.

**Verification:**
```bash
cd /var/mnt/eclipse/repos/KnotTheory.jl && \
  grep -c "NOTE: Template proof hole" src/abi/Layout.idr | \
  xargs -I{} test {} -ge 3 && \
  echo "PASS: All proof holes annotated"
```

---

## TASK 14: Update CI workflow to include Julia 1.12 and fix checkout SHA (LOW)

**Problem:** The CI workflow at `.github/workflows/ci.yml` tests Julia
versions `['1.9', '1.10', '1.11']` but the `Manifest.toml` was generated
with Julia 1.12.2 (line 3: `julia_version = "1.12.2"`). The checkout action
SHA `b4ffde65f46336ab88eb53be808477a3936bae11` does not match the SHA
listed in CLAUDE.md for `actions/checkout@v4` which is
`34e114876b0b11c390a56381ad16ebd13914f8d5`.

**File:** `/var/mnt/eclipse/repos/KnotTheory.jl/.github/workflows/ci.yml`

**Fix:**
1. Add `'1.12'` to the Julia version matrix.
2. Replace checkout SHA with the one from CLAUDE.md: `34e114876b0b11c390a56381ad16ebd13914f8d5`.
3. Add `permissions: read-all` at the workflow level (per CLAUDE.md checklist).

**Verification:**
```bash
cd /var/mnt/eclipse/repos/KnotTheory.jl && \
  grep "1.12" .github/workflows/ci.yml > /dev/null && \
  grep "34e114876b0b11c390a56381ad16ebd13914f8d5" .github/workflows/ci.yml > /dev/null && \
  grep "permissions:" .github/workflows/ci.yml > /dev/null && \
  echo "PASS: CI workflow updated"
```

---

## TASK 15: Update docs/src/index.md with actual API documentation (LOW)

**Problem:** `docs/src/index.md` is a 14-line stub with a single trivial
example. It does not document any of the exported functions.

**File:** `/var/mnt/eclipse/repos/KnotTheory.jl/docs/src/index.md`

**Fix:** Expand to document all exported symbols with their signatures,
parameters, return types, and brief descriptions. Group by category:

1. **Types:** `EdgeOrientation`, `Crossing`, `PlanarDiagram`, `DTCode`, `Knot`, `Link`
2. **Constructors:** `pdcode`, `unknot`, `trefoil`, `figure_eight`
3. **Invariants:** `crossing_number`, `writhe`, `linking_number`, `seifert_circles`, `braid_index_estimate`
4. **Polynomials:** `alexander_polynomial`, `jones_polynomial`
5. **Simplification:** `r1_simplify`, `simplify_pd` (and `r2_simplify`, `r3_simplify` after Task 9)
6. **Code Conversion:** `dtcode`, `to_dowker`
7. **I/O:** `write_knot_json`, `read_knot_json`
8. **Table:** `knot_table`, `lookup_knot`
9. **Utilities:** `to_graph`, `to_polynomial`, `plot_pd`

**Verification:**
```bash
cd /var/mnt/eclipse/repos/KnotTheory.jl && \
  wc -l docs/src/index.md | awk '{if ($1 >= 80) print "PASS: index.md has sufficient content ("$1" lines)"; else {print "FAIL: only "$1" lines"; exit 1}}'
```

---

## FINAL VERIFICATION

After all tasks are complete, run this comprehensive check:

```bash
cd /var/mnt/eclipse/repos/KnotTheory.jl && \
echo "=== 1. Full test suite ===" && \
julia --project=. -e 'using Pkg; Pkg.test()' && \
echo "=== 2. No AGPL references ===" && \
! grep -r "AGPL" --include="*.jl" --include="*.zig" --include="*.idr" --include="*.res" --include="*.adoc" . && \
echo "=== 3. No template placeholders ===" && \
! grep -r '{{' --include="*.md" --include="*.zig" --include="*.idr" --include="*.adoc" . && \
echo "=== 4. .machine_readable/ exists ===" && \
test -d .machine_readable && \
test -f .machine_readable/STATE.scm && \
test -f .machine_readable/META.scm && \
test -f .machine_readable/ECOSYSTEM.scm && \
echo "=== 5. Version is 0.1.0 ===" && \
grep 'version = "0.1.0"' Project.toml > /dev/null && \
echo "=== 6. Examples are relevant ===" && \
test ! -f examples/SafeDOMExample.res && \
test -f examples/basic_usage.jl && \
echo "=== 7. to_polynomial handles negative exponents ===" && \
julia --project=. -e '
using KnotTheory
pd = KnotTheory.pdcode([(1,2,3,4,1)])
j = KnotTheory.jones_polynomial(pd; wr=1)
result = KnotTheory.to_polynomial(j)
@assert result isa Tuple
println("to_polynomial returns tuple correctly")
' && \
echo "=== 8. Knot table has >= 15 entries ===" && \
julia --project=. -e '
using KnotTheory
@assert length(knot_table()) >= 15
' && \
echo "" && \
echo "============================================" && \
echo "  ALL FINAL VERIFICATION CHECKS PASSED" && \
echo "============================================"
```
