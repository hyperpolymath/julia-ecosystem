# SONNET-TASKS.md — HackenbushGames.jl Completion Tasks

> **Generated:** 2026-02-12 by Opus audit
> **Purpose:** Unambiguous instructions for Sonnet to complete all stubs, TODOs, and placeholder code.
> **Honest completion before this file:** 62%

The Julia core (`src/HackenbushGames.jl` + `test/runtests.jl`) is genuinely functional with
real implementations. However, the repository is cloned from `rsr-template-repo` and most
surrounding files still contain `{{PLACEHOLDER}}` template tokens, the ABI/FFI layer is
entirely generic template code unrelated to Hackenbush, the Documenter.jl setup references
a missing `api.md` page, docs say "coming soon", the `examples/` directory contains
irrelevant ReScript/Deno template files, the `ROADMAP.adoc` is still the RSR template,
`CITATIONS.adoc` references the template repo with wrong author/license, there is no
`.machine_readable/` directory with SCM files, and the Manifest.toml version (0.1.0)
contradicts Project.toml version (1.0.0).

---

## GROUND RULES FOR SONNET

1. Read this entire file before starting any task.
2. Do tasks in order listed. Earlier tasks unblock later ones.
3. After each task, run the verification command. If it fails, fix before moving on.
4. Do NOT mark done unless verification passes.
5. Update STATE.scm with honest completion percentages after each task.
6. Commit after each task: `fix(component): complete <description>`
7. Run full test suite after every 3 tasks: `cd /var/mnt/eclipse/repos/HackenbushGames.jl && julia --project=. -e 'using Pkg; Pkg.test()'`

---

## TASK 1: Fix Manifest.toml version mismatch (HIGH)

**Files:** `/var/mnt/eclipse/repos/HackenbushGames.jl/Manifest.toml`

**Problem:** Line 9 says `version = "0.1.0"` but `Project.toml` line 4 says
`version = "1.0.0"`. The Manifest is machine-generated and stale.

**What to do:**
1. Delete `Manifest.toml` entirely.
2. Regenerate it: `cd /var/mnt/eclipse/repos/HackenbushGames.jl && julia --project=. -e 'using Pkg; Pkg.instantiate()'`
3. Verify the regenerated `Manifest.toml` shows `version = "1.0.0"` for HackenbushGames.

**Verification:**
```bash
cd /var/mnt/eclipse/repos/HackenbushGames.jl && grep 'version = "1.0.0"' Manifest.toml
```

---

## TASK 2: Create .machine_readable/ directory with SCM files (HIGH)

**Files:**
- `/var/mnt/eclipse/repos/HackenbushGames.jl/.machine_readable/STATE.scm` (create)
- `/var/mnt/eclipse/repos/HackenbushGames.jl/.machine_readable/ECOSYSTEM.scm` (create)
- `/var/mnt/eclipse/repos/HackenbushGames.jl/.machine_readable/META.scm` (create)

**Problem:** The repo has no `.machine_readable/` directory at all. This is mandatory for
every hyperpolymath repo per CLAUDE.md. The AI.a2ml file on line 9 references
`.machines_readable/6scm/STATE.scm` which is both a different path AND does not exist.

**What to do:**
1. Create the directory: `mkdir -p /var/mnt/eclipse/repos/HackenbushGames.jl/.machine_readable`
2. Create `STATE.scm` with content reflecting actual project state:
   - Phase: implementation
   - Maturity: beta (core Julia code works, but template artifacts are unfinished)
   - Completion: 62%
   - Blockers: template placeholders, missing docs
3. Create `ECOSYSTEM.scm` placing HackenbushGames.jl in the hyperpolymath Julia ecosystem,
   related to combinatorial game theory.
4. Create `META.scm` with architecture decisions (pure Julia, no deps, dyadic rationals,
   Grundy numbers).
5. Use `language-bridges` repo as reference for SCM file format.

**Verification:**
```bash
ls /var/mnt/eclipse/repos/HackenbushGames.jl/.machine_readable/STATE.scm \
   /var/mnt/eclipse/repos/HackenbushGames.jl/.machine_readable/ECOSYSTEM.scm \
   /var/mnt/eclipse/repos/HackenbushGames.jl/.machine_readable/META.scm
```

---

## TASK 3: Replace all {{PLACEHOLDER}} tokens in CONTRIBUTING.md (HIGH)

**Files:** `/var/mnt/eclipse/repos/HackenbushGames.jl/CONTRIBUTING.md`

**Problem:** Lines 2, 3, 9, 10, 20, 89, 90, 91, 92 all contain raw `{{FORGE}}`,
`{{OWNER}}`, `{{REPO}}` template placeholders.

**What to do:**
1. Replace every `{{FORGE}}` with `github.com`
2. Replace every `{{OWNER}}` with `hyperpolymath`
3. Replace every `{{REPO}}` with `HackenbushGames.jl`
4. Review the entire file to ensure no `{{` tokens remain.

**Verification:**
```bash
grep -c '{{' /var/mnt/eclipse/repos/HackenbushGames.jl/CONTRIBUTING.md && echo "FAIL: placeholders remain" || echo "PASS"
```

---

## TASK 4: Replace all {{PLACEHOLDER}} tokens in CODE_OF_CONDUCT.md (HIGH)

**Files:** `/var/mnt/eclipse/repos/HackenbushGames.jl/CODE_OF_CONDUCT.md`

**Problem:** Lines 7-14, 313 contain `{{PLACEHOLDER}}`, `{{OWNER}}`, `{{REPO}}`,
`{{FORGE}}`, `{{PROJECT_NAME}}`, `{{CONDUCT_EMAIL}}`, `{{CONDUCT_TEAM}}`,
`{{RESPONSE_TIME}}`, `{{CURRENT_YEAR}}` template tokens.

**What to do:**
1. Delete the template instruction comment block (lines 3-20 approximately).
2. Replace `{{PROJECT_NAME}}` with `HackenbushGames.jl`
3. Replace `{{OWNER}}` with `hyperpolymath`
4. Replace `{{REPO}}` with `HackenbushGames.jl`
5. Replace `{{FORGE}}` with `github.com`
6. Replace `{{CONDUCT_EMAIL}}` with `jonathan.jewell@open.ac.uk`
7. Replace `{{CONDUCT_TEAM}}` with `Project Maintainers`
8. Replace `{{RESPONSE_TIME}}` with `48 hours`
9. Replace `{{CURRENT_YEAR}}` with `2026`

**Verification:**
```bash
grep -c '{{' /var/mnt/eclipse/repos/HackenbushGames.jl/CODE_OF_CONDUCT.md && echo "FAIL: placeholders remain" || echo "PASS"
```

---

## TASK 5: Replace all {{PLACEHOLDER}} tokens in SECURITY.md (HIGH)

**Files:** `/var/mnt/eclipse/repos/HackenbushGames.jl/SECURITY.md`

**Problem:** Lines 7-16, 43, 61-73, 206, 325, 374, 386-387, 402, 406 contain
`{{OWNER}}`, `{{REPO}}`, `{{PROJECT_NAME}}`, `{{SECURITY_EMAIL}}`,
`{{PGP_FINGERPRINT}}`, `{{PGP_KEY_URL}}`, `{{WEBSITE}}`, `{{CURRENT_YEAR}}`
template tokens.

**What to do:**
1. Delete the template instruction comment block (lines 3-19).
2. Replace `{{PROJECT_NAME}}` with `HackenbushGames.jl`
3. Replace `{{OWNER}}` with `hyperpolymath`
4. Replace `{{REPO}}` with `HackenbushGames.jl`
5. Replace `{{SECURITY_EMAIL}}` with `jonathan.jewell@open.ac.uk`
6. Remove the PGP section entirely (lines 60-74) since there is no PGP key configured.
7. Replace `{{CURRENT_YEAR}}` with `2026`
8. Remove `{{WEBSITE}}` references or replace with `https://github.com/hyperpolymath`

**Verification:**
```bash
grep -c '{{' /var/mnt/eclipse/repos/HackenbushGames.jl/SECURITY.md && echo "FAIL: placeholders remain" || echo "PASS"
```

---

## TASK 6: Fix CITATIONS.adoc (HIGH)

**Files:** `/var/mnt/eclipse/repos/HackenbushGames.jl/docs/CITATIONS.adoc`

**Problem:** The entire file (lines 1-36) references `rsr-template-repo` instead of
`HackenbushGames.jl`. The author is listed as `Polymath, Hyper` instead of
`Jewell, Jonathan D.A.`. The license says `AGPL-3.0-or-later` instead of
`PMPL-1.0-or-later`. It references `CITATION.cff` and `codemeta.json` which do not exist.

**What to do:**
1. Replace every occurrence of `rsr-template-repo` and `RSR-template-repo` with `HackenbushGames.jl`
2. Replace every occurrence of `RSR-template-repo` in URLs with `HackenbushGames.jl`
3. Replace author `Polymath, Hyper` with `Jewell, Jonathan D.A.` in all citation formats
4. Replace author `Hyper Polymath` with `Jonathan D.A. Jewell` in OSCOLA format
5. Replace `AGPL-3.0-or-later` with `PMPL-1.0-or-later`
6. Replace year `2025` with `2026`
7. Remove the `See Also` section referencing non-existent `CITATION.cff` and `codemeta.json`,
   OR create those files (preferred: remove the references).

**Verification:**
```bash
grep -c 'rsr-template-repo\|RSR-template-repo\|AGPL\|Polymath, Hyper' /var/mnt/eclipse/repos/HackenbushGames.jl/docs/CITATIONS.adoc && echo "FAIL" || echo "PASS"
```

---

## TASK 7: Fix Documenter.jl — missing api.md page (MEDIUM)

**Files:**
- `/var/mnt/eclipse/repos/HackenbushGames.jl/docs/make.jl` (line 12)
- `/var/mnt/eclipse/repos/HackenbushGames.jl/docs/src/api.md` (create)
- `/var/mnt/eclipse/repos/HackenbushGames.jl/docs/src/index.md` (line 17)

**Problem:** `docs/make.jl` line 12 references `"API" => "api.md"` but `docs/src/api.md`
does not exist. Also, `docs/src/index.md` line 17 says `# Examples coming soon` which is
a placeholder.

**What to do:**
1. Create `/var/mnt/eclipse/repos/HackenbushGames.jl/docs/src/api.md` with proper
   Documenter.jl autodoc blocks for all exported symbols:
   - `EdgeColor`, `Edge`, `HackenbushGraph`, `GameForm`
   - `Blue`, `Red`, `Green`
   - `prune_disconnected`, `cut_edge`, `moves`, `game_sum`
   - `simplest_dyadic_between`, `stalk_value`
   - `mex`, `nim_sum`, `green_stalk_nimber`, `green_grundy`
   - `simple_stalk`, `to_graphviz`, `to_ascii`
   - `canonical_game`, `simplify_game`, `game_value`
   Use `@docs` blocks, e.g.:
   ```markdown
   # API Reference

   ## Types

   ```@docs
   EdgeColor
   Edge
   HackenbushGraph
   GameForm
   ```

   ## Graph Operations

   ```@docs
   prune_disconnected
   cut_edge
   moves
   game_sum
   simple_stalk
   ```

   (etc. for all exported symbols)
   ```
2. Replace `# Examples coming soon` in `docs/src/index.md` with actual examples matching
   the README.md Quick Start section (lines 44-66).

**Verification:**
```bash
test -f /var/mnt/eclipse/repos/HackenbushGames.jl/docs/src/api.md && echo "PASS: api.md exists" || echo "FAIL"
grep -c 'coming soon' /var/mnt/eclipse/repos/HackenbushGames.jl/docs/src/index.md && echo "FAIL: placeholder remains" || echo "PASS"
```

---

## TASK 8: Remove irrelevant example files (MEDIUM)

**Files:**
- `/var/mnt/eclipse/repos/HackenbushGames.jl/examples/SafeDOMExample.res` (delete)
- `/var/mnt/eclipse/repos/HackenbushGames.jl/examples/web-project-deno.json` (delete)

**Problem:** These are RSR template boilerplate files for a ReScript web project. They have
nothing to do with a Julia Hackenbush game theory library. `SafeDOMExample.res` line 1 also
has `SPDX-License-Identifier: AGPL-3.0-or-later` (wrong license).

**What to do:**
1. Delete `examples/SafeDOMExample.res`
2. Delete `examples/web-project-deno.json`
3. Create `examples/basic_usage.jl` with working examples derived from README.md:
   - Stalk value computation
   - Green Grundy number
   - Graph sum
   - Canonical game form
   - GraphViz and ASCII output
4. Add SPDX header `# SPDX-License-Identifier: PMPL-1.0-or-later` to the new file.

**Verification:**
```bash
test ! -f /var/mnt/eclipse/repos/HackenbushGames.jl/examples/SafeDOMExample.res && echo "PASS: res deleted" || echo "FAIL"
test ! -f /var/mnt/eclipse/repos/HackenbushGames.jl/examples/web-project-deno.json && echo "PASS: json deleted" || echo "FAIL"
cd /var/mnt/eclipse/repos/HackenbushGames.jl && julia --project=. examples/basic_usage.jl
```

---

## TASK 9: Replace ROADMAP.adoc template content (MEDIUM)

**Files:** `/var/mnt/eclipse/repos/HackenbushGames.jl/ROADMAP.adoc`

**Problem:** The entire file is the RSR template boilerplate (`YOUR Template Repo Roadmap`,
`Core functionality`, `To be determined`). `ROADMAP.md` exists with real content but
`ROADMAP.adoc` is still the template.

**What to do:**
1. Delete `ROADMAP.adoc` entirely (the real roadmap is `ROADMAP.md`).
   OR replace its content with an AsciiDoc version of `ROADMAP.md`.
   Preferred: delete `ROADMAP.adoc` since `ROADMAP.md` is the authoritative file.

**Verification:**
```bash
test ! -f /var/mnt/eclipse/repos/HackenbushGames.jl/ROADMAP.adoc && echo "PASS: template roadmap removed" || echo "FAIL"
test -f /var/mnt/eclipse/repos/HackenbushGames.jl/ROADMAP.md && echo "PASS: real roadmap exists" || echo "FAIL"
```

---

## TASK 10: Remove or customize ABI/FFI template files (MEDIUM)

**Files:**
- `/var/mnt/eclipse/repos/HackenbushGames.jl/src/abi/Types.idr`
- `/var/mnt/eclipse/repos/HackenbushGames.jl/src/abi/Layout.idr`
- `/var/mnt/eclipse/repos/HackenbushGames.jl/src/abi/Foreign.idr`
- `/var/mnt/eclipse/repos/HackenbushGames.jl/ffi/zig/build.zig`
- `/var/mnt/eclipse/repos/HackenbushGames.jl/ffi/zig/src/main.zig`
- `/var/mnt/eclipse/repos/HackenbushGames.jl/ffi/zig/test/integration_test.zig`
- `/var/mnt/eclipse/repos/HackenbushGames.jl/ABI-FFI-README.md`

**Problem:** All 7 files are unmodified RSR template boilerplate with `{{PROJECT}}` and
`{{project}}` placeholders throughout (hundreds of occurrences). They define generic
`Handle`, `Result`, `ExampleStruct` types that have nothing to do with Hackenbush.
The Zig files have `SPDX-License-Identifier: AGPL-3.0-or-later` (wrong license).
This is a pure Julia library with zero FFI needs.

**What to do:**
1. Delete the entire `src/abi/` directory (3 Idris2 template files).
2. Delete the entire `ffi/zig/` directory tree (build.zig, src/main.zig, test/integration_test.zig).
3. Delete `ABI-FFI-README.md`.
4. These are template scaffolding for projects that need C FFI. A pure Julia library
   with no dependencies does not need them.

**Verification:**
```bash
test ! -d /var/mnt/eclipse/repos/HackenbushGames.jl/src/abi && echo "PASS: abi removed" || echo "FAIL"
test ! -d /var/mnt/eclipse/repos/HackenbushGames.jl/ffi && echo "PASS: ffi removed" || echo "FAIL"
test ! -f /var/mnt/eclipse/repos/HackenbushGames.jl/ABI-FFI-README.md && echo "PASS: abi readme removed" || echo "FAIL"
cd /var/mnt/eclipse/repos/HackenbushGames.jl && julia --project=. -e 'using Pkg; Pkg.test()'
```

---

## TASK 11: Fix AI.a2ml to reference correct paths (LOW)

**Files:** `/var/mnt/eclipse/repos/HackenbushGames.jl/AI.a2ml`

**Problem:** Line 1 says this is `rsr-template-repo`. Line 6 says obey Rhodium policies
and keep `.machines_readable/6scm/` authoritative (wrong path -- should be
`.machine_readable/` per CLAUDE.md). Line 9 references `.machines_readable/6scm/STATE.scm`
(wrong path). Line 10 references `.machines_readable/6scm/AGENTIC.scm` (does not exist).

**What to do:**
1. Replace the title/description to reference HackenbushGames.jl, not rsr-template-repo.
2. Replace `.machines_readable/6scm/` with `.machine_readable/` everywhere.
3. Remove reference to `AGENTIC.scm` (does not exist in this repo, not mandatory).
4. Update the workflow section to reflect actual project structure.

**Verification:**
```bash
grep -c 'rsr-template-repo\|machines_readable\|6scm' /var/mnt/eclipse/repos/HackenbushGames.jl/AI.a2ml && echo "FAIL" || echo "PASS"
```

---

## TASK 12: Fix CodeQL workflow language matrix (LOW)

**Files:** `/var/mnt/eclipse/repos/HackenbushGames.jl/.github/workflows/codeql.yml`

**Problem:** Line 24-25 configures CodeQL to scan `javascript-typescript` language. This is
a pure Julia repository with no JavaScript or TypeScript files. CodeQL does not support
Julia, so this workflow should either scan `actions` (workflow files only) or be removed.

**What to do:**
1. Change the language matrix on line 24 from `javascript-typescript` to `actions`.
2. Keep `build-mode: none` since actions analysis does not need building.

**Verification:**
```bash
grep 'javascript-typescript' /var/mnt/eclipse/repos/HackenbushGames.jl/.github/workflows/codeql.yml && echo "FAIL" || echo "PASS"
grep 'actions' /var/mnt/eclipse/repos/HackenbushGames.jl/.github/workflows/codeql.yml && echo "PASS" || echo "FAIL"
```

---

## TASK 13: Fix quality.yml TODO scanner to include Julia files (LOW)

**Files:** `/var/mnt/eclipse/repos/HackenbushGames.jl/.github/workflows/quality.yml`

**Problem:** Line 31 scans for TODOs in `*.rs`, `*.res`, `*.py`, `*.ex` files only. This
is a Julia project; it should scan `*.jl` files.

**What to do:**
1. On line 31, add `--include="*.jl"` to the grep command.
2. Optionally remove `*.rs`, `*.res`, `*.py`, `*.ex` includes since those languages are
   not present in this repo.

**Verification:**
```bash
grep '\.jl' /var/mnt/eclipse/repos/HackenbushGames.jl/.github/workflows/quality.yml && echo "PASS" || echo "FAIL"
```

---

## TASK 14: Add more comprehensive tests (LOW)

**Files:** `/var/mnt/eclipse/repos/HackenbushGames.jl/test/runtests.jl`

**Problem:** The test suite has only 7 test cases. Key behaviors are untested:
- `prune_disconnected` is never directly tested.
- `simplest_dyadic_between` is never directly tested.
- `nim_sum` is never tested.
- `mex` is never tested.
- `green_stalk_nimber` is never tested.
- `game_sum` with empty graphs is not tested.
- `game_value` returning `nothing` for non-numeric positions is not tested.
- `cut_edge` with invalid index is not tested.
- `moves` with Green edges (both players can move) is not tested.
- Edge cases: empty graph, single-edge graph, multi-branch graphs.

**What to do:**
1. Add a `@testset "Prune Disconnected"` block testing that floating edges are removed.
2. Add a `@testset "Simplest Dyadic"` block verifying:
   - `simplest_dyadic_between(0//1, 1//1) == 1//2` (or the simplest integer 0 if 0 is between)
   - `simplest_dyadic_between(-1//1, 1//1) == 0//1`
   - Error on `l >= r`
3. Add a `@testset "Nim Sum"` block: `nim_sum([3, 5]) == 6` (3 XOR 5).
4. Add a `@testset "Mex"` block: `mex([0, 1, 3]) == 2`, `mex(Int[]) == 0`.
5. Add a `@testset "Green Stalk Nimber"` block: `green_stalk_nimber(5) == 5`.
6. Add a `@testset "Green Moves"` block: Green edges allow both left and right moves.
7. Add a `@testset "Empty Graph"` block: empty graph has no moves, value 0//1.
8. Add a `@testset "Game Value Nothing"` testing a position where `game_value` returns `nothing`.

**Verification:**
```julia
cd /var/mnt/eclipse/repos/HackenbushGames.jl && julia --project=. -e 'using Pkg; Pkg.test()'
```

---

## TASK 15: Fix README.adoc template content (LOW)

**Files:** `/var/mnt/eclipse/repos/HackenbushGames.jl/README.adoc`

**Problem:** This is the RSR template README (`see RSR_OUTLINE.adoc in root`, `This is
your repo - don't forget to rename me!`, SafeDOM examples). It conflicts with the real
README.md. A Julia Hackenbush library does not need ReScript web dependency instructions.

**What to do:**
1. Delete `README.adoc` entirely. The authoritative README is `README.md`.
   OR convert it to an AsciiDoc version of `README.md` with Hackenbush-specific content.
   Preferred: delete it to avoid confusion.

**Verification:**
```bash
test ! -f /var/mnt/eclipse/repos/HackenbushGames.jl/README.adoc && echo "PASS" || echo "FAIL"
test -f /var/mnt/eclipse/repos/HackenbushGames.jl/README.md && echo "PASS" || echo "FAIL"
```

---

## TASK 16: Fix RSR_OUTLINE.adoc template content (LOW)

**Files:** `/var/mnt/eclipse/repos/HackenbushGames.jl/RSR_OUTLINE.adoc`

**Problem:** This is a generic RSR standard description file not specific to this project.
It references `RSR-template-repo`, `139 repos`, `justfile`, `guix.scm`, `STATE.scm` in
root (violates SCM path rules), and includes a `Cookbook generation` section. None of this
is specific to HackenbushGames.jl. Line 212 says `SPDX-License-Identifier: PMPL-1.0-or-later-or-later` (doubled suffix).

**What to do:**
1. Either delete `RSR_OUTLINE.adoc` (it is RSR framework docs, not project docs),
   or customize it to describe this project's RSR compliance status.
   Preferred: delete it.

**Verification:**
```bash
test ! -f /var/mnt/eclipse/repos/HackenbushGames.jl/RSR_OUTLINE.adoc && echo "PASS" || echo "FAIL"
```

---

## TASK 17: Clean up ROADMAP.md false claims (LOW)

**Files:** `/var/mnt/eclipse/repos/HackenbushGames.jl/ROADMAP.md`

**Problem:** Lines 5-13 claim "Production-ready" and "Complete with security hardening and
comprehensive test coverage." It also claims "Game comparison (>, <, =, ||)" (line 9) which
is NOT implemented anywhere in the source. The code has no `>`, `<`, `==`, or `||` game
comparison operators. It also claims "Basic game operations (negation, addition)" (line 10)
-- `game_sum` exists but there is no negation function.

**What to do:**
1. Change "Production-ready" to "Functional beta"
2. Remove or mark as TODO: "Game comparison (>, <, =, ||)" -- not implemented
3. Change "Basic game operations (negation, addition)" to "Basic game operations (addition via game_sum)"
4. Change "Complete with security hardening and comprehensive test coverage" to
   "Core algorithms implemented. Test coverage is basic (7 test cases)."

**Verification:**
```bash
grep -c 'Production-ready\|Game comparison' /var/mnt/eclipse/repos/HackenbushGames.jl/ROADMAP.md && echo "FAIL: false claims remain" || echo "PASS"
```

---

## FINAL VERIFICATION

After all 17 tasks are complete, run the following sequence to confirm everything works:

```bash
# 1. Full test suite
cd /var/mnt/eclipse/repos/HackenbushGames.jl && julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.test()'

# 2. No template placeholders remain in any file
grep -r '{{' /var/mnt/eclipse/repos/HackenbushGames.jl --include='*.md' --include='*.adoc' --include='*.yml' --include='*.idr' --include='*.zig' --include='*.a2ml' --include='*.djot' --include='*.res' --include='*.json' | grep -v '.git/' | grep -v 'node_modules' && echo "FAIL: template tokens remain" || echo "PASS: no template tokens"

# 3. No AGPL references remain (old license, replaced by PMPL)
grep -r 'AGPL' /var/mnt/eclipse/repos/HackenbushGames.jl --include='*.jl' --include='*.zig' --include='*.idr' --include='*.res' --include='*.adoc' | grep -v '.git/' && echo "FAIL: AGPL references remain" || echo "PASS: no AGPL"

# 4. SCM files exist in correct location
ls /var/mnt/eclipse/repos/HackenbushGames.jl/.machine_readable/STATE.scm \
   /var/mnt/eclipse/repos/HackenbushGames.jl/.machine_readable/ECOSYSTEM.scm \
   /var/mnt/eclipse/repos/HackenbushGames.jl/.machine_readable/META.scm && echo "PASS: SCM files exist" || echo "FAIL"

# 5. No irrelevant template files remain
test ! -f /var/mnt/eclipse/repos/HackenbushGames.jl/examples/SafeDOMExample.res && \
test ! -d /var/mnt/eclipse/repos/HackenbushGames.jl/src/abi && \
test ! -d /var/mnt/eclipse/repos/HackenbushGames.jl/ffi && \
echo "PASS: template artifacts removed" || echo "FAIL"

# 6. Docs build check (optional, needs Documenter.jl installed)
cd /var/mnt/eclipse/repos/HackenbushGames.jl && julia --project=docs -e '
  using Pkg
  Pkg.develop(PackageSpec(path="."))
  Pkg.instantiate()
  include("docs/make.jl")
' 2>&1 | tail -5

# 7. Example runs without error
cd /var/mnt/eclipse/repos/HackenbushGames.jl && julia --project=. examples/basic_usage.jl
```
