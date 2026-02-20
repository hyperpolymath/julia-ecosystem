# SONNET-TASKS.md --- Exnovation.jl Completion Tasks

> **Generated:** 2026-02-12 by Opus audit
> **Purpose:** Unambiguous instructions for Sonnet to complete all stubs, TODOs, and placeholder code.
> **Honest completion before this file:** 72%

The Julia core library (`src/Exnovation.jl`) is genuinely complete: 14 exported functions,
12 exported types, 2 enums, all with implementations and docstrings, and a test suite with
29+ assertions. However, the repo is littered with uncustomized RSR template files,
wrong license headers, a missing documentation page, a missing `.machine_readable/`
directory, a version mismatch, and a `debiasing_actions` gap. The ABI/FFI scaffolding
(Idris2 + Zig) is entirely boilerplate with `{{PROJECT}}` placeholders -- these template
files add no value for a pure-Julia package.

---

## GROUND RULES FOR SONNET

1. Read this entire file before starting any task.
2. Do tasks in order listed. Earlier tasks unblock later ones.
3. After each task, run the verification command. If it fails, fix before moving on.
4. Do NOT mark done unless verification passes.
5. Update STATE.scm with honest completion percentages after each task.
6. Commit after each task: `fix(component): complete <description>`
7. Run full test suite after every 3 tasks: `cd /var/mnt/eclipse/repos/Exnovation.jl && julia --project=. -e 'using Pkg; Pkg.test()'`

---

## TASK 1: Fix Version Mismatch Between Project.toml and Manifest.toml (HIGH)

**Files:** `/var/mnt/eclipse/repos/Exnovation.jl/Project.toml` (line 4), `/var/mnt/eclipse/repos/Exnovation.jl/Manifest.toml` (line 16)

**Problem:** `Project.toml` declares `version = "1.0.0"` but `Manifest.toml` records the
package as `version = "0.1.0"`. The `Manifest.toml` is stale from an older `Project.toml`
version. This means anyone running `Pkg.instantiate()` will get a mismatched manifest.

**What to do:**
1. Delete `Manifest.toml` entirely. It will be regenerated from `Project.toml`.
2. Run `julia --project=. -e 'using Pkg; Pkg.instantiate()'` to regenerate it.
3. Verify the regenerated `Manifest.toml` shows `version = "1.0.0"` for Exnovation.

**Verification:**
```julia
cd("/var/mnt/eclipse/repos/Exnovation.jl")
toml = Pkg.TOML.parsefile("Project.toml")
manifest = Pkg.TOML.parsefile("Manifest.toml")
@assert toml["version"] == "1.0.0" "Project.toml version must be 1.0.0"
# Find Exnovation entry in manifest deps
exnov_entry = manifest["deps"]["Exnovation"][1]
@assert exnov_entry["version"] == "1.0.0" "Manifest must match Project.toml version"
println("PASS: versions match")
```

---

## TASK 2: Add Missing `Political` Case to `debiasing_actions` (HIGH)

**Files:** `/var/mnt/eclipse/repos/Exnovation.jl/src/Exnovation.jl` (lines 234-249)

**Problem:** The `debiasing_actions` function handles `Cognitive`, `Emotional`, `Behavioral`,
and `Structural` barriers but silently ignores `Political` barriers. The `BarrierType` enum
(line 19) includes `Political`, and `barrier_templates()` (line 342) creates `Political`
barriers. Passing a `Political` barrier to `debiasing_actions` produces an empty result
with no warning.

**What to do:**
1. Add an `elseif barrier.kind == Political` branch after the `Structural` branch (after line 245).
2. Push two debiasing actions for Political barriers:
   - `"Map stakeholder influence and build coalition support for the transition."`
   - `"Communicate reputational benefits and de-risk through staged rollout."`
3. Add a test in `test/runtests.jl` that passes a `Political` barrier and asserts the
   result is non-empty.

**Verification:**
```julia
using Exnovation
political_barriers = [Barrier(Political, 0.5, "Stakeholder resistance")]
actions = debiasing_actions(political_barriers)
@assert length(actions) >= 1 "Political barriers must produce debiasing actions"
@assert any(contains(a, "stakeholder") || contains(a, "Stakeholder") for a in actions) "Must mention stakeholders"
println("PASS: Political debiasing actions work")
```

---

## TASK 3: Create Missing `docs/src/api.md` (HIGH)

**Files:** `/var/mnt/eclipse/repos/Exnovation.jl/docs/make.jl` (line 12), `/var/mnt/eclipse/repos/Exnovation.jl/docs/src/api.md` (MISSING)

**Problem:** `docs/make.jl` line 12 declares `pages = ["Home" => "index.md", "API" => "api.md"]`.
The file `docs/src/api.md` does not exist. Running `makedocs()` will fail or produce a broken
documentation site.

**What to do:**
1. Create `/var/mnt/eclipse/repos/Exnovation.jl/docs/src/api.md`.
2. Add an SPDX header comment (use HTML comment since it is Markdown).
3. Add a title `# API Reference`.
4. Use Documenter.jl `@autodocs` or `@docs` blocks to auto-generate documentation
   from the docstrings in `src/Exnovation.jl`. Include all 14 exported functions
   and 12 exported types. Example:

```markdown
# API Reference

## Types

```@docs
ExnovationItem
Driver
Barrier
DecisionCriteria
ExnovationAssessment
ExnovationSummary
IntelligentFailureCriteria
FailureAssessment
FailureSummary
RiskGovernance
ExnovationCase
DecisionReport
ImpactModel
PortfolioItem
StageGate
```

## Enums

```@docs
BarrierType
FailureType
```

## Functions

```@docs
sunk_cost_bias_index
exnovation_score
recommendation
debiasing_actions
intelligent_failure_score
failure_summary
decision_pipeline
write_report_json
barrier_templates
run_stage_gates
portfolio_scores
allocate_budget
```
```

**Verification:**
```julia
@assert isfile("/var/mnt/eclipse/repos/Exnovation.jl/docs/src/api.md") "api.md must exist"
content = read("/var/mnt/eclipse/repos/Exnovation.jl/docs/src/api.md", String)
@assert contains(content, "ExnovationItem") "Must document ExnovationItem"
@assert contains(content, "exnovation_score") "Must document exnovation_score"
@assert contains(content, "allocate_budget") "Must document allocate_budget"
println("PASS: api.md exists and documents key exports")
```

---

## TASK 4: Update `docs/src/index.md` Placeholder (MEDIUM)

**Files:** `/var/mnt/eclipse/repos/Exnovation.jl/docs/src/index.md` (line 17)

**Problem:** Line 17 says `# Examples coming soon` -- this is a placeholder. The repo
has two complete examples in `examples/` and a full Quick Start in `README.md`.

**What to do:**
1. Replace `# Examples coming soon` with an actual Quick Start code example, adapted
   from the `README.md` Quick Start section (lines 48-81).
2. Add a brief description of the package (1-2 sentences from `README.md` lines 8-10).
3. Mention the two example files: `examples/01_basic_usage.jl` and
   `examples/02_portfolio_management.jl`.

**Verification:**
```julia
content = read("/var/mnt/eclipse/repos/Exnovation.jl/docs/src/index.md", String)
@assert !contains(content, "coming soon") "Must remove 'coming soon' placeholder"
@assert contains(content, "ExnovationItem") "Must include actual code example"
println("PASS: index.md updated with real content")
```

---

## TASK 5: Fix AGPL-3.0 License Headers (Must Be PMPL-1.0-or-later) (HIGH)

**Files:**
- `/var/mnt/eclipse/repos/Exnovation.jl/ffi/zig/build.zig` (line 2)
- `/var/mnt/eclipse/repos/Exnovation.jl/ffi/zig/src/main.zig` (line 6)
- `/var/mnt/eclipse/repos/Exnovation.jl/ffi/zig/test/integration_test.zig` (line 2)
- `/var/mnt/eclipse/repos/Exnovation.jl/examples/SafeDOMExample.res` (line 1)
- `/var/mnt/eclipse/repos/Exnovation.jl/docs/CITATIONS.adoc` (line 13)

**Problem:** Five files use `SPDX-License-Identifier: AGPL-3.0-or-later`. Per
CLAUDE.md license policy: "NEVER use AGPL-3.0 (old license, replaced by PMPL-1.0-or-later)".
The `docs/CITATIONS.adoc` also says `license = {AGPL-3.0-or-later}` in the BibTeX block.

**What to do:**
1. In each of the 5 files listed, replace `AGPL-3.0-or-later` with `PMPL-1.0-or-later`.
2. In `docs/CITATIONS.adoc`, also fix the project name from `rsr-template-repo` to
   `Exnovation.jl`, the author from `Polymath, Hyper` to `Jewell, Jonathan D.A.`, the
   year to `2026`, and the URL to `https://github.com/hyperpolymath/Exnovation.jl`.

**Verification:**
```bash
cd /var/mnt/eclipse/repos/Exnovation.jl
count=$(grep -r "AGPL-3.0" --include="*.zig" --include="*.res" --include="*.adoc" . | wc -l)
if [ "$count" -eq 0 ]; then echo "PASS: no AGPL-3.0 references remain"; else echo "FAIL: $count AGPL-3.0 references found"; exit 1; fi
```

---

## TASK 6: Remove or Customize Boilerplate ABI/FFI Template Files (MEDIUM)

**Files:**
- `/var/mnt/eclipse/repos/Exnovation.jl/src/abi/Types.idr`
- `/var/mnt/eclipse/repos/Exnovation.jl/src/abi/Layout.idr`
- `/var/mnt/eclipse/repos/Exnovation.jl/src/abi/Foreign.idr`
- `/var/mnt/eclipse/repos/Exnovation.jl/ffi/zig/build.zig`
- `/var/mnt/eclipse/repos/Exnovation.jl/ffi/zig/src/main.zig`
- `/var/mnt/eclipse/repos/Exnovation.jl/ffi/zig/test/integration_test.zig`
- `/var/mnt/eclipse/repos/Exnovation.jl/ABI-FFI-README.md`

**Problem:** Exnovation.jl is a pure-Julia package. It has no C FFI, no Zig build, and no
Idris2 ABI. All 7 files above contain raw `{{PROJECT}}` / `{{project}}` template
placeholders that have never been customized. They are non-functional boilerplate from
`rsr-template-repo` and will confuse users.

**What to do:**
1. Delete all 7 files listed above.
2. Remove the empty directories `src/abi/`, `ffi/zig/src/`, `ffi/zig/test/`, `ffi/zig/`,
   and `ffi/` if they become empty.
3. In `README.adoc`, remove or rewrite the ABI/FFI section (lines 5-34 and lines 36-101)
   so it describes Exnovation.jl instead of RSR template boilerplate. Replace the entire
   file content with a brief pointer: `= Exnovation.jl` followed by `See README.md for
   full documentation.`

**Verification:**
```bash
cd /var/mnt/eclipse/repos/Exnovation.jl
if [ -d "src/abi" ]; then echo "FAIL: src/abi/ still exists"; exit 1; fi
if [ -d "ffi" ]; then echo "FAIL: ffi/ still exists"; exit 1; fi
if [ -f "ABI-FFI-README.md" ]; then echo "FAIL: ABI-FFI-README.md still exists"; exit 1; fi
count=$(grep -r '{{PROJECT}}\|{{project}}' --include="*.idr" --include="*.zig" --include="*.md" . 2>/dev/null | wc -l)
if [ "$count" -eq 0 ]; then echo "PASS: no template placeholders remain in code"; else echo "FAIL: $count template placeholders found"; exit 1; fi
```

---

## TASK 7: Customize RSR Template Placeholders in Markdown Files (MEDIUM)

**Files:**
- `/var/mnt/eclipse/repos/Exnovation.jl/CONTRIBUTING.md`
- `/var/mnt/eclipse/repos/Exnovation.jl/CODE_OF_CONDUCT.md`
- `/var/mnt/eclipse/repos/Exnovation.jl/SECURITY.md`
- `/var/mnt/eclipse/repos/Exnovation.jl/ROADMAP.adoc`
- `/var/mnt/eclipse/repos/Exnovation.jl/RSR_OUTLINE.adoc`

**Problem:** These files are raw RSR template copies with `{{FORGE}}`, `{{OWNER}}`,
`{{REPO}}`, `{{PROJECT_NAME}}`, `{{SECURITY_EMAIL}}`, `{{CONDUCT_EMAIL}}`,
`{{CONDUCT_TEAM}}`, `{{RESPONSE_TIME}}`, `{{CURRENT_YEAR}}`, `{{PGP_FINGERPRINT}}`,
`{{PGP_KEY_URL}}`, `{{WEBSITE}}`, `{{MAIN_BRANCH}}` placeholders.

`ROADMAP.adoc` is also a generic template that conflicts with the real `ROADMAP.md`.

**What to do:**
1. In `CONTRIBUTING.md`, replace:
   - `{{FORGE}}` with `github.com`
   - `{{OWNER}}` with `hyperpolymath`
   - `{{REPO}}` with `Exnovation.jl`
   - `{{MAIN_BRANCH}}` with `main`
2. In `CODE_OF_CONDUCT.md`, replace:
   - `{{PROJECT_NAME}}` with `Exnovation.jl`
   - `{{OWNER}}` with `hyperpolymath`
   - `{{REPO}}` with `Exnovation.jl`
   - `{{CONDUCT_EMAIL}}` with `jonathan.jewell@open.ac.uk`
   - `{{CONDUCT_TEAM}}` with `Exnovation.jl Maintainers`
   - `{{RESPONSE_TIME}}` with `72 hours`
   - `{{CURRENT_YEAR}}` with `2026`
   - `{{FORGE}}` with `github.com`
3. In `SECURITY.md`, replace:
   - `{{PROJECT_NAME}}` with `Exnovation.jl`
   - `{{OWNER}}` with `hyperpolymath`
   - `{{REPO}}` with `Exnovation.jl`
   - `{{SECURITY_EMAIL}}` with `jonathan.jewell@open.ac.uk`
   - `{{PGP_FINGERPRINT}}` with `(not yet configured)`
   - `{{PGP_KEY_URL}}` with `(not yet configured)`
   - `{{WEBSITE}}` with `https://github.com/hyperpolymath/Exnovation.jl`
   - `{{CURRENT_YEAR}}` with `2026`
4. Delete `ROADMAP.adoc` (the real roadmap is `ROADMAP.md`).
5. In `RSR_OUTLINE.adoc`, replace the title on line 1 from `= RSR Template Repository`
   to `= Exnovation.jl RSR Outline`. Replace the SPDX identifier on line 212 from
   `PMPL-1.0-or-later-or-later` (typo with doubled suffix) to `PMPL-1.0-or-later`.

**Verification:**
```bash
cd /var/mnt/eclipse/repos/Exnovation.jl
count=$(grep -r '{{[A-Z_]*}}' CONTRIBUTING.md CODE_OF_CONDUCT.md SECURITY.md RSR_OUTLINE.adoc 2>/dev/null | wc -l)
if [ "$count" -eq 0 ]; then echo "PASS: no template placeholders in docs"; else echo "FAIL: $count placeholders remain"; exit 1; fi
if [ -f "ROADMAP.adoc" ]; then echo "FAIL: ROADMAP.adoc should be deleted"; exit 1; fi
echo "PASS: all template docs customized"
```

---

## TASK 8: Create `.machine_readable/` Directory with SCM Files (MEDIUM)

**Files:** `/var/mnt/eclipse/repos/Exnovation.jl/.machine_readable/` (MISSING)

**Problem:** Per CLAUDE.md and the project's own `AI.djot`, every hyperpolymath repo must
have `.machine_readable/STATE.scm`, `.machine_readable/ECOSYSTEM.scm`, and
`.machine_readable/META.scm`. This directory does not exist.

**What to do:**
1. Create directory `/var/mnt/eclipse/repos/Exnovation.jl/.machine_readable/`.
2. Create `STATE.scm` with:
   - `(metadata (project . "Exnovation.jl") (updated . "2026-02-12"))`
   - `(position (phase . maintenance) (maturity . production))`
   - `(completion-percentage . 85)`
   - `(blockers . ())`
   - Current status note: core library complete, docs and template cleanup remaining.
3. Create `ECOSYSTEM.scm` with:
   - Name: `Exnovation.jl`
   - Type: `julia-package`
   - Purpose: `Exnovation decision framework for phase-out analysis`
   - Related projects: `(related-projects ((name . "BowtieRisk.jl") (relationship . "potential-consumer")))`
4. Create `META.scm` with:
   - License: `PMPL-1.0-or-later`
   - Author: `Jonathan D.A. Jewell`
   - Architecture decision: single-module Julia package, no FFI needed.

**Verification:**
```bash
cd /var/mnt/eclipse/repos/Exnovation.jl
for f in STATE.scm ECOSYSTEM.scm META.scm; do
  if [ ! -f ".machine_readable/$f" ]; then echo "FAIL: .machine_readable/$f missing"; exit 1; fi
done
echo "PASS: .machine_readable/ directory with all SCM files"
```

---

## TASK 9: Remove Unrelated Example Files (LOW)

**Files:**
- `/var/mnt/eclipse/repos/Exnovation.jl/examples/SafeDOMExample.res`
- `/var/mnt/eclipse/repos/Exnovation.jl/examples/web-project-deno.json`

**Problem:** These are RSR template examples for ReScript web projects. They have nothing
to do with Exnovation.jl (a Julia decision-framework package). `SafeDOMExample.res` is
a ReScript file demonstrating DOM mounting. `web-project-deno.json` is a Deno configuration
for ReScript projects. Both are confusing detritus.

**What to do:**
1. Delete `examples/SafeDOMExample.res`.
2. Delete `examples/web-project-deno.json`.
3. Verify that `examples/01_basic_usage.jl` and `examples/02_portfolio_management.jl`
   remain untouched.

**Verification:**
```bash
cd /var/mnt/eclipse/repos/Exnovation.jl
if [ -f "examples/SafeDOMExample.res" ]; then echo "FAIL: SafeDOMExample.res should be deleted"; exit 1; fi
if [ -f "examples/web-project-deno.json" ]; then echo "FAIL: web-project-deno.json should be deleted"; exit 1; fi
if [ ! -f "examples/01_basic_usage.jl" ]; then echo "FAIL: 01_basic_usage.jl must exist"; exit 1; fi
if [ ! -f "examples/02_portfolio_management.jl" ]; then echo "FAIL: 02_portfolio_management.jl must exist"; exit 1; fi
echo "PASS: only Julia examples remain"
```

---

## TASK 10: Customize `docs/CITATIONS.adoc` (LOW)

**Files:** `/var/mnt/eclipse/repos/Exnovation.jl/docs/CITATIONS.adoc`

**Problem:** This file is a raw RSR template copy. It references `rsr-template-repo`,
uses author `Polymath, Hyper`, year `2025`, and the wrong URL. It also references
non-existent `CITATION.cff` and `codemeta.json` files.

**What to do:**
1. Replace all instances of `rsr-template-repo` / `RSR-template-repo` with `Exnovation.jl`.
2. Replace author `Polymath, Hyper` with `Jewell, Jonathan D.A.` (BibTeX last-first) and
   `Hyper Polymath` with `Jonathan D.A. Jewell`.
3. Replace year `2025` with `2026`.
4. Replace the URL with `https://github.com/hyperpolymath/Exnovation.jl`.
5. Fix the license from `AGPL-3.0-or-later` to `PMPL-1.0-or-later` (if not done in Task 5).
6. Remove the "See Also" section referencing `CITATION.cff` and `codemeta.json` (they
   do not exist), or create those files.

**Verification:**
```bash
cd /var/mnt/eclipse/repos/Exnovation.jl
content=$(cat docs/CITATIONS.adoc)
if echo "$content" | grep -q "rsr-template-repo"; then echo "FAIL: still references rsr-template-repo"; exit 1; fi
if echo "$content" | grep -q "AGPL"; then echo "FAIL: still references AGPL"; exit 1; fi
if echo "$content" | grep -q "Polymath, Hyper"; then echo "FAIL: wrong author name"; exit 1; fi
echo "PASS: CITATIONS.adoc customized"
```

---

## TASK 11: Pin Unpinned GitHub Actions in release.yml (LOW)

**Files:** `/var/mnt/eclipse/repos/Exnovation.jl/.github/workflows/release.yml` (lines 46, 94, 108)

**Problem:** Three action references use tag-only pins (`@v4`) instead of SHA pins:
- Line 46: `actions/upload-artifact@v4`
- Line 94: `actions/upload-artifact@v4`
- Line 108: `actions/download-artifact@v4`

Per CLAUDE.md workflow standards, all actions must be SHA-pinned.

**What to do:**
1. Replace `actions/upload-artifact@v4` on lines 46 and 94 with a SHA-pinned version.
   Use a current v4 SHA (e.g., `actions/upload-artifact@ea165f8d65b6db9a8b22b984b926f09f6cef9ab8`
   or look up the latest v4 tag SHA on the actions/upload-artifact repo).
2. Replace `actions/download-artifact@v4` on line 108 with a SHA-pinned version
   (e.g., `actions/download-artifact@d3f86a106a0bac45b974a628896c90dbdf5c8093`
   or look up the latest v4 tag SHA).

**Verification:**
```bash
cd /var/mnt/eclipse/repos/Exnovation.jl
count=$(grep -E 'uses:.*@v[0-9]+\s*$' .github/workflows/release.yml | wc -l)
if [ "$count" -eq 0 ]; then echo "PASS: all actions SHA-pinned"; else echo "FAIL: $count actions not SHA-pinned"; exit 1; fi
```

---

## TASK 12: Fix `AI.a2ml` Template References (LOW)

**Files:** `/var/mnt/eclipse/repos/Exnovation.jl/AI.a2ml`

**Problem:** This file references `rsr-template-repo` (line 5) and paths like
`.machines_readable/6scm/STATE.scm` (line 9) and `.machines_readable/6scm/AGENTIC.scm`
(line 10). The correct path per CLAUDE.md is `.machine_readable/` (no `s`, no `6scm/`
subdirectory). The file also does not mention Exnovation.jl at all.

**What to do:**
1. Replace `rsr-template-repo` with `Exnovation.jl` on line 5.
2. Replace `.machines_readable/6scm/` with `.machine_readable/` throughout (lines 9-10).
3. Update the description to mention exnovation decision-making.

**Verification:**
```bash
cd /var/mnt/eclipse/repos/Exnovation.jl
content=$(cat AI.a2ml)
if echo "$content" | grep -q "rsr-template-repo"; then echo "FAIL: still says rsr-template-repo"; exit 1; fi
if echo "$content" | grep -q "machines_readable"; then echo "FAIL: wrong directory name (has extra s)"; exit 1; fi
if echo "$content" | grep -q "6scm"; then echo "FAIL: references 6scm subdirectory"; exit 1; fi
echo "PASS: AI.a2ml references corrected"
```

---

## TASK 13: Add Input Validation to Public API Functions (LOW)

**Files:** `/var/mnt/eclipse/repos/Exnovation.jl/src/Exnovation.jl`

**Problem:** The `DecisionCriteria` struct (lines 52-57) expects weights in 0..1 but
no validation is performed. Negative weights or weights > 1 are silently accepted.
Similarly, `Driver` and `Barrier` weights have no validation. The `_clamp01` function
clamps individual weights during scoring, but the raw structs allow nonsensical values
like `-5.0` or `100.0` to be constructed without any warning.

**What to do:**
1. Add a constructor function `DecisionCriteria(sw, sfw, pw, rw)` that warns (via
   `@warn`) if any weight is outside [0, 1]. Do NOT throw -- just warn. The clamping
   in scoring already handles the math, but users should know their inputs are unusual.
2. Alternatively, add a `validate(criteria::DecisionCriteria)` exported function that
   returns a vector of warning strings. This is less intrusive.
3. Add a test that constructs `DecisionCriteria` with out-of-range weights and verifies
   that `validate()` returns warnings, or that scoring still works correctly.

**Verification:**
```julia
using Exnovation
# Extreme values should not crash
bad_criteria = DecisionCriteria(-1.0, 2.0, 0.5, 0.5)
item = ExnovationItem(:test, "test", "test")
drivers = [Driver(:d, 0.5, "test")]
barriers = Barrier[]
a = ExnovationAssessment(item, drivers, barriers, bad_criteria, 100.0, 50.0, 0.0, 0.5, 0.5, 0.5)
s = exnovation_score(a)
@assert isfinite(s.total_score) "Score must be finite even with bad inputs"
println("PASS: out-of-range weights handled gracefully")
```

---

## TASK 14: Add `permissions: read-all` to CI Workflow (LOW)

**Files:** `/var/mnt/eclipse/repos/Exnovation.jl/.github/workflows/ci.yml`

**Problem:** Per CLAUDE.md workflow validation checklist item 4: "`permissions: read-all`
at workflow level". The CI workflow has no `permissions` block at all.

**What to do:**
1. Add `permissions: read-all` after the `on:` block (after line 6) and before `jobs:`.

**Verification:**
```bash
cd /var/mnt/eclipse/repos/Exnovation.jl
if grep -q "permissions:" .github/workflows/ci.yml; then echo "PASS: permissions block exists"; else echo "FAIL: no permissions block"; exit 1; fi
```

---

## FINAL VERIFICATION

After all tasks are complete, run:

```bash
cd /var/mnt/eclipse/repos/Exnovation.jl

echo "=== 1. Julia tests ==="
julia --project=. -e 'using Pkg; Pkg.test()'

echo "=== 2. No AGPL references ==="
count=$(grep -r "AGPL" --include="*.jl" --include="*.zig" --include="*.idr" --include="*.res" --include="*.adoc" . 2>/dev/null | wc -l)
[ "$count" -eq 0 ] && echo "PASS" || echo "FAIL: $count AGPL references"

echo "=== 3. No raw template placeholders in code ==="
count=$(grep -rn '{{[A-Za-z_]*}}' --include="*.jl" --include="*.zig" --include="*.idr" --include="*.yml" . 2>/dev/null | wc -l)
[ "$count" -eq 0 ] && echo "PASS" || echo "FAIL: $count placeholders"

echo "=== 4. Machine-readable directory exists ==="
[ -f ".machine_readable/STATE.scm" ] && [ -f ".machine_readable/META.scm" ] && [ -f ".machine_readable/ECOSYSTEM.scm" ] && echo "PASS" || echo "FAIL"

echo "=== 5. No stale ABI/FFI boilerplate ==="
[ ! -d "src/abi" ] && [ ! -d "ffi" ] && [ ! -f "ABI-FFI-README.md" ] && echo "PASS" || echo "FAIL"

echo "=== 6. api.md exists ==="
[ -f "docs/src/api.md" ] && echo "PASS" || echo "FAIL"

echo "=== 7. Version consistency ==="
julia --project=. -e '
  using Pkg
  p = Pkg.TOML.parsefile("Project.toml")
  m = Pkg.TOML.parsefile("Manifest.toml")
  pv = p["version"]
  mv = m["deps"]["Exnovation"][1]["version"]
  @assert pv == mv "Version mismatch: Project=$pv Manifest=$mv"
  println("PASS: versions match ($pv)")
'

echo "=== 8. Political debiasing actions ==="
julia --project=. -e '
  using Exnovation
  actions = debiasing_actions([Barrier(Political, 0.5, "test")])
  @assert length(actions) >= 1 "Political barriers must produce actions"
  println("PASS: $(length(actions)) actions for Political barriers")
'

echo "=== AUDIT COMPLETE ==="
```
