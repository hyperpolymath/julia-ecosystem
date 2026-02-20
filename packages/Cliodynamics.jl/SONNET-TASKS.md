# SONNET-TASKS.md — Cliodynamics.jl Completion Tasks

> **Generated:** 2026-02-12 by Opus audit
> **Purpose:** Unambiguous instructions for Sonnet to complete all stubs, TODOs, and placeholder code.
> **Honest completion before this file:** 35%

The Julia source code (`src/Cliodynamics.jl`) and tests (`test/runtests.jl`) are genuinely
complete -- all 16 exported functions have real implementations, all types are defined, and
the test suite covers every public API. That part is solid.

However, the repository was cloned from `rsr-template-repo` and the vast majority of
infrastructure files were never customized. SCM files describe "rsr-template-repo" at 5%
completion. ABI/FFI files are raw `{{project}}` templates. Examples are ReScript/Deno
leftovers from the template, not Julia. The AI manifest, ROADMAP, and README.adoc still
have placeholder text. SPDX headers in SCM files are wrong. The SCM directory is named
incorrectly. A phantom `Plots` dependency sits in Project.toml unused by source code.

---

## GROUND RULES FOR SONNET

1. Read this entire file before starting any task.
2. Do tasks in order listed. Earlier tasks unblock later ones.
3. After each task, run the verification command. If it fails, fix before moving on.
4. Do NOT mark done unless verification passes.
5. Update `.machine_readable/STATE.scm` with honest completion percentages after each task.
6. Commit after each task: `fix(component): complete <description>`
7. Run full test suite after every 3 tasks: `cd /var/mnt/eclipse/repos/Cliodynamics.jl && julia --project=. -e 'using Pkg; Pkg.test()'`

---

## TASK 1: Fix SCM directory name (CRITICAL)

**Files:**
- `/var/mnt/eclipse/repos/Cliodynamics.jl/.machines_readable/` (entire directory)

**Problem:** The SCM directory is named `.machines_readable/6scm/` but the standard requires
`.machine_readable/` (no trailing "s", no `6scm/` subdirectory). The AI manifest at line 16
and CLAUDE.md both state SCM files MUST be in `.machine_readable/` ONLY.

**What to do:**
1. Create directory `.machine_readable/` in repository root.
2. Move all 6 `.scm` files from `.machines_readable/6scm/` to `.machine_readable/`:
   - `STATE.scm`
   - `META.scm`
   - `ECOSYSTEM.scm`
   - `AGENTIC.scm`
   - `NEUROSYM.scm`
   - `PLAYBOOK.scm`
3. Remove the empty `.machines_readable/6scm/` directory.
4. Remove the empty `.machines_readable/` directory.
5. Update `.gitignore` if it references the old path.

**Verification:**
```bash
# All 6 SCM files exist in correct location
test -f /var/mnt/eclipse/repos/Cliodynamics.jl/.machine_readable/STATE.scm && \
test -f /var/mnt/eclipse/repos/Cliodynamics.jl/.machine_readable/META.scm && \
test -f /var/mnt/eclipse/repos/Cliodynamics.jl/.machine_readable/ECOSYSTEM.scm && \
test -f /var/mnt/eclipse/repos/Cliodynamics.jl/.machine_readable/AGENTIC.scm && \
test -f /var/mnt/eclipse/repos/Cliodynamics.jl/.machine_readable/NEUROSYM.scm && \
test -f /var/mnt/eclipse/repos/Cliodynamics.jl/.machine_readable/PLAYBOOK.scm && \
! test -d /var/mnt/eclipse/repos/Cliodynamics.jl/.machines_readable && \
echo "PASS" || echo "FAIL"
```

---

## TASK 2: Fix SPDX headers in all SCM files (CRITICAL)

**Files:**
- `/var/mnt/eclipse/repos/Cliodynamics.jl/.machine_readable/STATE.scm` (line 1)
- `/var/mnt/eclipse/repos/Cliodynamics.jl/.machine_readable/META.scm` (line 1)
- `/var/mnt/eclipse/repos/Cliodynamics.jl/.machine_readable/ECOSYSTEM.scm` (line 1)

**Problem:** STATE.scm, META.scm, and ECOSYSTEM.scm all have `AGPL-3.0-or-later` as their
SPDX identifier. Per CLAUDE.md license policy, AGPL-3.0 must NEVER be used. All hyperpolymath
original code uses `PMPL-1.0-or-later`. AGENTIC.scm, NEUROSYM.scm, and PLAYBOOK.scm already
have the correct header.

**What to do:**
1. In `STATE.scm` line 1: change `AGPL-3.0-or-later` to `PMPL-1.0-or-later`.
2. In `META.scm` line 1: change `AGPL-3.0-or-later` to `PMPL-1.0-or-later`.
3. In `ECOSYSTEM.scm` line 1: change `AGPL-3.0-or-later` to `PMPL-1.0-or-later`.

**Verification:**
```bash
cd /var/mnt/eclipse/repos/Cliodynamics.jl/.machine_readable
grep -c "PMPL-1.0-or-later" STATE.scm META.scm ECOSYSTEM.scm AGENTIC.scm NEUROSYM.scm PLAYBOOK.scm | \
  awk -F: '{sum += $2} END {if (sum == 6) print "PASS"; else print "FAIL: only " sum " of 6 files have correct SPDX"}'
```

---

## TASK 3: Fix SPDX headers in Zig and Idris2 template files (CRITICAL)

**Files:**
- `/var/mnt/eclipse/repos/Cliodynamics.jl/ffi/zig/src/main.zig` (line 6)
- `/var/mnt/eclipse/repos/Cliodynamics.jl/ffi/zig/build.zig` (line 2)
- `/var/mnt/eclipse/repos/Cliodynamics.jl/ffi/zig/test/integration_test.zig` (line 2)

**Problem:** All three Zig files have `SPDX-License-Identifier: AGPL-3.0-or-later`.
Must be `PMPL-1.0-or-later`.

**What to do:**
1. In `ffi/zig/src/main.zig` line 6: change `AGPL-3.0-or-later` to `PMPL-1.0-or-later`.
2. In `ffi/zig/build.zig` line 2: change `AGPL-3.0-or-later` to `PMPL-1.0-or-later`.
3. In `ffi/zig/test/integration_test.zig` line 2: change `AGPL-3.0-or-later` to `PMPL-1.0-or-later`.

**Verification:**
```bash
cd /var/mnt/eclipse/repos/Cliodynamics.jl
grep -r "AGPL" ffi/ src/abi/ .machine_readable/ && echo "FAIL: AGPL references remain" || echo "PASS: no AGPL references"
```

---

## TASK 4: Rewrite STATE.scm for Cliodynamics.jl (HIGH)

**Files:**
- `/var/mnt/eclipse/repos/Cliodynamics.jl/.machine_readable/STATE.scm`

**Problem:** The entire file (lines 1-65) still describes "rsr-template-repo" with 5%
completion and generic milestones. The actual Julia source code is complete with 16 exported
functions and a full test suite.

**What to do:**
1. Replace the entire contents of STATE.scm with a file that accurately describes Cliodynamics.jl.
2. Set `project` to `"Cliodynamics.jl"`.
3. Set `repo` to `"hyperpolymath/Cliodynamics.jl"`.
4. Set `tech-stack` to `("Julia" "DifferentialEquations.jl" "DataFrames.jl" "Optim.jl")`.
5. Set `overall-completion` to `90` (core Julia code is done, but infrastructure/metadata needs cleanup).
6. List working features: Malthusian model, DST model, elite overproduction index, PSI,
   secular cycle analysis, phase detection, state capacity model, collective action problem,
   utility functions (moving average, detrend, normalize, carrying capacity, crisis threshold,
   instability events, conflict intensity, population pressure).
7. List remaining items: Julia examples needed, Project.toml cleanup, ABI/FFI template
   customization, documentation polish.
8. Set `phase` to `"beta"`.
9. Keep the helper functions at the bottom.

**Verification:**
```bash
cd /var/mnt/eclipse/repos/Cliodynamics.jl/.machine_readable
grep -q "Cliodynamics" STATE.scm && \
grep -q "90" STATE.scm && \
! grep -q "rsr-template-repo" STATE.scm && \
echo "PASS" || echo "FAIL"
```

---

## TASK 5: Rewrite META.scm for Cliodynamics.jl (HIGH)

**Files:**
- `/var/mnt/eclipse/repos/Cliodynamics.jl/.machine_readable/META.scm`

**Problem:** Lines 1-47 still describe "rsr-template-repo" with generic RSR-focused ADRs
and development practices. Should describe Cliodynamics.jl architectural decisions.

**What to do:**
1. Change `define-meta` name from `rsr-template-repo` to `Cliodynamics.jl`.
2. Replace ADR-001 with a decision about using Julia + DifferentialEquations.jl for
   cliodynamic modeling.
3. Add ADR-002 about the single-file module design (`src/Cliodynamics.jl`).
4. Update development practices to reference Julia conventions (docstrings, `@testset`, Pkg.test).
5. Update design rationale to explain why cliodynamics models benefit from Julia's ODE solvers.
6. Remove references to ReScript/Rust/Gleam in code-style (this is a Julia project).

**Verification:**
```bash
cd /var/mnt/eclipse/repos/Cliodynamics.jl/.machine_readable
grep -q "Cliodynamics" META.scm && \
! grep -q "rsr-template-repo" META.scm && \
grep -q "Julia" META.scm && \
echo "PASS" || echo "FAIL"
```

---

## TASK 6: Rewrite ECOSYSTEM.scm for Cliodynamics.jl (HIGH)

**Files:**
- `/var/mnt/eclipse/repos/Cliodynamics.jl/.machine_readable/ECOSYSTEM.scm`

**Problem:** Lines 1-29 still describe "rsr-template-repo" with `[TODO: Add specific description]`
on line 24. Must describe Cliodynamics.jl's position in the ecosystem.

**What to do:**
1. Change `name` from `"rsr-template-repo"` to `"Cliodynamics.jl"`.
2. Set `type` to `"library"`.
3. Set `purpose` to describe cliodynamic modeling and historical dynamics analysis.
4. Update `position-in-ecosystem` to describe this as a Julia scientific computing library.
5. Add `related-projects`: sibling `Cliometrics.jl`, dependency `DifferentialEquations.jl`,
   inspiration `Seshat Global History Databank`.
6. Replace the `[TODO: Add specific description]` in `what-this-is`.
7. Update `what-this-is-not` to clarify it is not a general-purpose statistics library.

**Verification:**
```bash
cd /var/mnt/eclipse/repos/Cliodynamics.jl/.machine_readable
grep -q "Cliodynamics" ECOSYSTEM.scm && \
! grep -q "rsr-template-repo" ECOSYSTEM.scm && \
! grep -q "TODO" ECOSYSTEM.scm && \
echo "PASS" || echo "FAIL"
```

---

## TASK 7: Remove unused Plots dependency from Project.toml (MEDIUM)

**Files:**
- `/var/mnt/eclipse/repos/Cliodynamics.jl/Project.toml` (lines 12, 20)

**Problem:** `Plots` is listed as a dependency (line 12, UUID `91a5bcdd-55d7-5caf-9e0b-520d859cae80`)
and in compat (line 20), but it is never `using Plots` in `src/Cliodynamics.jl`. Plots is a
heavy dependency (~100+ transitive packages) and should not be a hard dependency. The README.md
shows Plots in example code, but that is user-side usage, not library code.

**What to do:**
1. Remove the line `Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"` from `[deps]` (line 12).
2. Remove the line `Plots = "1"` from `[compat]` (line 20).
3. Do NOT add Plots to `[extras]` -- it is only used in README examples, not tests.

**Verification:**
```julia
# Run from repo root
cd("/var/mnt/eclipse/repos/Cliodynamics.jl")
toml = read("Project.toml", String)
@assert !occursin("Plots", toml) "FAIL: Plots still in Project.toml"
println("PASS: Plots removed from Project.toml")
```

---

## TASK 8: Add DataFrames and Statistics to test dependencies (MEDIUM)

**Files:**
- `/var/mnt/eclipse/repos/Cliodynamics.jl/Project.toml` (lines 22-26)

**Problem:** `test/runtests.jl` uses `using DataFrames` (line 5) and `using Statistics`
(line 6), but neither is listed in `[extras]` or `[targets]`. Currently only `Test` is in
extras. While they are regular deps, best practice for Julia packages is to also list test
dependencies in `[extras]` if they are used in tests beyond the main package deps. However,
since DataFrames and Statistics ARE already in `[deps]`, they will be available during testing.
This task is about ensuring the test target is correct.

Actually, the current setup works because `[deps]` packages are available during testing.
**Skip this task -- no changes needed.** The existing `[extras]` and `[targets]` are correct.

**Verification:**
```julia
cd("/var/mnt/eclipse/repos/Cliodynamics.jl")
using Pkg
Pkg.activate(".")
Pkg.test()
```

---

## TASK 9: Remove template ReScript/Deno examples (MEDIUM)

**Files:**
- `/var/mnt/eclipse/repos/Cliodynamics.jl/examples/SafeDOMExample.res` (entire file)
- `/var/mnt/eclipse/repos/Cliodynamics.jl/examples/web-project-deno.json` (entire file)

**Problem:** These files are leftover from `rsr-template-repo`. A Julia cliodynamics library
has no use for ReScript DOM mounting examples or Deno project config. They are confusing and
irrelevant.

**What to do:**
1. Delete `examples/SafeDOMExample.res`.
2. Delete `examples/web-project-deno.json`.
3. Create `examples/basic_usage.jl` with a runnable Julia script demonstrating:
   - Malthusian model simulation
   - Demographic-structural model simulation
   - Elite overproduction index calculation
   - Political stress indicator calculation
   - Secular cycle analysis
   Use the examples from the module docstring (lines 48-66 of `src/Cliodynamics.jl`) and
   the README.md Quick Start section as a guide.
4. Create `examples/historical_analysis.jl` demonstrating phase detection, instability events,
   and conflict intensity with synthetic data.
5. Add `# SPDX-License-Identifier: PMPL-1.0-or-later` as the first line of each new file.

**Verification:**
```bash
cd /var/mnt/eclipse/repos/Cliodynamics.jl
! test -f examples/SafeDOMExample.res && \
! test -f examples/web-project-deno.json && \
test -f examples/basic_usage.jl && \
test -f examples/historical_analysis.jl && \
head -1 examples/basic_usage.jl | grep -q "PMPL" && \
echo "PASS" || echo "FAIL"
```

```julia
# Verify examples are syntactically valid
cd("/var/mnt/eclipse/repos/Cliodynamics.jl")
include("examples/basic_usage.jl")
include("examples/historical_analysis.jl")
println("PASS: examples run without error")
```

---

## TASK 10: Customize 0-AI-MANIFEST.a2ml (MEDIUM)

**Files:**
- `/var/mnt/eclipse/repos/Cliodynamics.jl/0-AI-MANIFEST.a2ml`

**Problem:** Lines 7-8 still say `[YOUR-REPO-NAME]`. Lines 51-67 have generic placeholder
structure. Lines 112-114 have `[DATE]`, `[YOUR-NAME/ORG]`. The manifest does not describe
the actual Cliodynamics.jl repository.

**What to do:**
1. Replace all `[YOUR-REPO-NAME]` with `Cliodynamics.jl` (lines 7, 56).
2. Replace the repository structure section (lines 55-68) with the actual structure:
   ```
   Cliodynamics.jl/
   ├── 0-AI-MANIFEST.a2ml
   ├── README.md
   ├── Project.toml
   ├── src/
   │   ├── Cliodynamics.jl        # Main module (all code)
   │   └── abi/                    # Idris2 ABI definitions (template)
   ├── test/
   │   └── runtests.jl             # Test suite
   ├── examples/                   # Usage examples
   ├── ffi/zig/                    # Zig FFI (template)
   ├── .machine_readable/          # SCM files (6 files)
   └── .bot_directives/            # Bot instructions
   ```
3. Set `[DATE]` to `2026-02-07` (line 112).
4. Set `[YOUR-NAME/ORG]` to `Jonathan D.A. Jewell / hyperpolymath` (line 113).

**Verification:**
```bash
cd /var/mnt/eclipse/repos/Cliodynamics.jl
! grep -q "\[YOUR-REPO-NAME\]" 0-AI-MANIFEST.a2ml && \
! grep -q "\[DATE\]" 0-AI-MANIFEST.a2ml && \
! grep -q "\[YOUR-NAME/ORG\]" 0-AI-MANIFEST.a2ml && \
grep -q "Cliodynamics" 0-AI-MANIFEST.a2ml && \
echo "PASS" || echo "FAIL"
```

---

## TASK 11: Replace ROADMAP.adoc with Cliodynamics.jl-specific content (MEDIUM)

**Files:**
- `/var/mnt/eclipse/repos/Cliodynamics.jl/ROADMAP.adoc`

**Problem:** Line 2 says "YOUR Template Repo Roadmap". All milestone items are generic
placeholders. The actual Julia code is at v0.1.0 with all core features implemented.

**What to do:**
1. Replace the entire file with a roadmap specific to Cliodynamics.jl.
2. Add `// SPDX-License-Identifier: PMPL-1.0-or-later` as line 1 (already present).
3. Mark v0.1.0 milestones as complete:
   - Core population dynamics models
   - Elite dynamics analysis
   - Political stress indicators
   - Secular cycle analysis
   - State formation models
   - Utility functions
   - Comprehensive test suite
4. Add v0.2.0 planned milestones:
   - Empirical dataset integration (Seshat, CrisisDB)
   - Plotting recipes for Plots.jl
   - Model fitting to historical data
   - Parameter estimation with Optim.jl
5. Add v1.0.0 goals:
   - Bayesian inference support (Turing.jl integration)
   - Spatial cliodynamic models
   - Interactive documentation (Documenter.jl)
   - Publication-quality examples

**Verification:**
```bash
cd /var/mnt/eclipse/repos/Cliodynamics.jl
! grep -q "YOUR Template" ROADMAP.adoc && \
grep -q "Cliodynamics" ROADMAP.adoc && \
grep -q "v0.1.0" ROADMAP.adoc && \
echo "PASS" || echo "FAIL"
```

---

## TASK 12: Replace README.adoc with Cliodynamics.jl-specific content (MEDIUM)

**Files:**
- `/var/mnt/eclipse/repos/Cliodynamics.jl/README.adoc`

**Problem:** The entire file (134 lines) is the RSR template README describing ReScript,
SafeDOM, Deno, and the ABI/FFI standard. It has nothing to do with cliodynamic modeling.
Line 1: "RSR template repo". Line 43: "Update `[YOUR-REPO-NAME]` placeholders".
Lines 79-133: ReScript SafeDOM documentation.

**What to do:**
1. Replace the entire file with a brief AsciiDoc version of the Cliodynamics.jl description.
2. Since `README.md` already has the full project description, `README.adoc` should be a
   concise pointer that says "See README.md for full documentation" plus a brief summary.
3. Alternatively, DELETE `README.adoc` entirely -- GitHub renders `README.md` by default,
   and having both is confusing. If you keep it, make it Cliodynamics-specific.
4. Recommended: Delete `README.adoc` and let `README.md` be the single README.

**Verification:**
```bash
cd /var/mnt/eclipse/repos/Cliodynamics.jl
# If README.adoc was deleted:
! test -f README.adoc && echo "PASS: README.adoc deleted" || \
# If README.adoc was kept:
(! grep -q "RSR template" README.adoc && grep -q "Cliodynamics" README.adoc && echo "PASS: README.adoc customized")
```

---

## TASK 13: Customize ABI Idris2 files for Cliodynamics.jl (LOW)

**Files:**
- `/var/mnt/eclipse/repos/Cliodynamics.jl/src/abi/Types.idr` (lines 11, 172-175, 198-202)
- `/var/mnt/eclipse/repos/Cliodynamics.jl/src/abi/Layout.idr` (line 8)
- `/var/mnt/eclipse/repos/Cliodynamics.jl/src/abi/Foreign.idr` (lines 9, 23, 35, 49, 77, 98, 125, 152, 164, 185, 211)

**Problem:** Every Idris2 file has `{{PROJECT}}` and `{{project}}` placeholders throughout.
Module names are `{{PROJECT}}.ABI.Types`, etc. FFI declarations reference `lib{{project}}`.
These files will not compile.

**What to do:**
1. In all three `.idr` files, replace `{{PROJECT}}` with `Cliodynamics` (uppercase for module names).
2. Replace `{{project}}` with `cliodynamics` (lowercase for library names and function prefixes).
3. In `Types.idr`: Replace `ExampleStruct` with a cliodynamics-relevant struct, e.g.,
   `SimulationResult` with fields `time : Double`, `population : Double`, `elites : Double`.
   Update the size proofs accordingly.
4. In `Foreign.idr`: Update the FFI function declarations to reflect cliodynamics operations
   (e.g., `cliodynamics_init`, `cliodynamics_free`, `cliodynamics_process`).

**Verification:**
```bash
cd /var/mnt/eclipse/repos/Cliodynamics.jl
! grep -r "{{PROJECT}}" src/abi/ && \
! grep -r "{{project}}" src/abi/ && \
grep -q "Cliodynamics" src/abi/Types.idr && \
grep -q "Cliodynamics" src/abi/Layout.idr && \
grep -q "Cliodynamics" src/abi/Foreign.idr && \
echo "PASS" || echo "FAIL"
```

---

## TASK 14: Customize Zig FFI files for Cliodynamics.jl (LOW)

**Files:**
- `/var/mnt/eclipse/repos/Cliodynamics.jl/ffi/zig/src/main.zig` (lines 1, 12, 54, 73, 89, 113, 135, 148, 184, 198, 203, 215, 245, 256, 259, 263, 266, 271, 272)
- `/var/mnt/eclipse/repos/Cliodynamics.jl/ffi/zig/build.zig` (lines 1, 12, 34, 37)
- `/var/mnt/eclipse/repos/Cliodynamics.jl/ffi/zig/test/integration_test.zig` (lines 1, 10-17, 24-25, 31-32, 34, 39, 48-49, 51, 56, 65-66, 68-69, 75, 84, 96-97, 99, 110, 118, 129-130, 131, 138-139, 143, 145-146, 149, 158-159, 168, 174)

**Problem:** Every Zig file has `{{project}}` and `{{PROJECT}}` template placeholders.
Function names like `{{project}}_init()`, library name `"{{project}}"`, etc. These files
will not compile.

**What to do:**
1. In all three `.zig` files, replace `{{project}}` with `cliodynamics` (lowercase).
2. Replace `{{PROJECT}}` with `Cliodynamics` (where used as display name).
3. In `build.zig` line 34: The header reference `include/{{project}}.h` should become
   `include/cliodynamics.h`. Note: this header file does not exist yet -- that is acceptable
   for template infrastructure.

**Verification:**
```bash
cd /var/mnt/eclipse/repos/Cliodynamics.jl
! grep -r "{{project}}" ffi/ && \
! grep -r "{{PROJECT}}" ffi/ && \
grep -q "cliodynamics" ffi/zig/src/main.zig && \
grep -q "cliodynamics" ffi/zig/build.zig && \
grep -q "cliodynamics" ffi/zig/test/integration_test.zig && \
echo "PASS" || echo "FAIL"
```

---

## TASK 15: Update AGENTIC.scm with Cliodynamics.jl specifics (LOW)

**Files:**
- `/var/mnt/eclipse/repos/Cliodynamics.jl/.machine_readable/AGENTIC.scm`

**Problem:** Line 7 references `claude-opus-4-5-20251101` (outdated model ID). The `languages`
constraint on line 15 is empty. The file should reflect Julia-specific patterns.

**What to do:**
1. Update model to `"claude-opus-4-6"` (current model per system info).
2. Set `languages` to `("julia")`.
3. Add constraint `(primary-runtime . "julia")`.
4. Keep `banned` languages list as is.

**Verification:**
```bash
cd /var/mnt/eclipse/repos/Cliodynamics.jl/.machine_readable
grep -q "julia" AGENTIC.scm && \
echo "PASS" || echo "FAIL"
```

---

## TASK 16: Update PLAYBOOK.scm with Julia procedures (LOW)

**Files:**
- `/var/mnt/eclipse/repos/Cliodynamics.jl/.machine_readable/PLAYBOOK.scm`

**Problem:** Lines 7-9 reference `just build`, `just test`, `just release` but there is no
`justfile` in the repository. The correct Julia commands should be used.

**What to do:**
1. Change build procedure to `"julia --project=. -e 'using Pkg; Pkg.instantiate()'"`.
2. Change test procedure to `"julia --project=. -e 'using Pkg; Pkg.test()'"`.
3. Change release procedure to `"julia --project=. -e 'using Pkg; Pkg.build()'"`.

**Verification:**
```bash
cd /var/mnt/eclipse/repos/Cliodynamics.jl/.machine_readable
grep -q "Pkg.test" PLAYBOOK.scm && \
! grep -q "just " PLAYBOOK.scm && \
echo "PASS" || echo "FAIL"
```

---

## TASK 17: Remove sync report file (LOW)

**Files:**
- `/var/mnt/eclipse/repos/Cliodynamics.jl/sync_report_20260210_160611.txt`

**Problem:** This appears to be a generated sync report that should not be tracked in git.
It is not part of the project.

**What to do:**
1. Delete `sync_report_20260210_160611.txt`.
2. Add `sync_report_*.txt` to `.gitignore` to prevent future occurrences.

**Verification:**
```bash
cd /var/mnt/eclipse/repos/Cliodynamics.jl
! test -f sync_report_20260210_160611.txt && \
grep -q "sync_report" .gitignore && \
echo "PASS" || echo "FAIL"
```

---

## TASK 18: Add `.claude/CLAUDE.md` for project-specific instructions (LOW)

**Files:**
- `/var/mnt/eclipse/repos/Cliodynamics.jl/.claude/CLAUDE.md` (new file)

**Problem:** No project-specific CLAUDE.md exists. This file should describe how to work
with this Julia package.

**What to do:**
1. Create `.claude/` directory.
2. Create `.claude/CLAUDE.md` with:
   - Project description: Julia package for cliodynamic modeling
   - Build command: `julia --project=. -e 'using Pkg; Pkg.instantiate()'`
   - Test command: `julia --project=. -e 'using Pkg; Pkg.test()'`
   - Code style: Julia conventions, docstrings on all exports, `@testset` structure
   - Architecture note: single-file module in `src/Cliodynamics.jl`
   - Dependencies: DifferentialEquations.jl, DataFrames.jl, Optim.jl, Statistics, LinearAlgebra

**Verification:**
```bash
cd /var/mnt/eclipse/repos/Cliodynamics.jl
test -f .claude/CLAUDE.md && \
grep -q "Cliodynamics" .claude/CLAUDE.md && \
grep -q "Pkg.test" .claude/CLAUDE.md && \
echo "PASS" || echo "FAIL"
```

---

## FINAL VERIFICATION

After all tasks are complete, run this comprehensive check:

```bash
cd /var/mnt/eclipse/repos/Cliodynamics.jl

echo "=== 1. SCM directory structure ==="
ls -la .machine_readable/ && ! test -d .machines_readable && echo "OK" || echo "FAIL"

echo ""
echo "=== 2. No AGPL references ==="
grep -r "AGPL" .machine_readable/ ffi/ src/abi/ && echo "FAIL" || echo "OK"

echo ""
echo "=== 3. No template placeholders ==="
grep -r "{{project}}\|{{PROJECT}}\|\[YOUR-REPO-NAME\]\|\[TODO\]" \
  .machine_readable/ src/abi/ ffi/ 0-AI-MANIFEST.a2ml ROADMAP.adoc && echo "FAIL" || echo "OK"

echo ""
echo "=== 4. No rsr-template-repo references in SCM ==="
grep -r "rsr-template-repo" .machine_readable/ && echo "FAIL" || echo "OK"

echo ""
echo "=== 5. No ReScript/Deno examples ==="
! test -f examples/SafeDOMExample.res && ! test -f examples/web-project-deno.json && echo "OK" || echo "FAIL"

echo ""
echo "=== 6. Julia examples exist ==="
test -f examples/basic_usage.jl && test -f examples/historical_analysis.jl && echo "OK" || echo "FAIL"

echo ""
echo "=== 7. No Plots in Project.toml ==="
! grep -q "Plots" Project.toml && echo "OK" || echo "FAIL"

echo ""
echo "=== 8. No sync report ==="
! test -f sync_report_20260210_160611.txt && echo "OK" || echo "FAIL"

echo ""
echo "=== 9. CLAUDE.md exists ==="
test -f .claude/CLAUDE.md && echo "OK" || echo "FAIL"
```

```julia
# Full Julia test suite
cd("/var/mnt/eclipse/repos/Cliodynamics.jl")
using Pkg
Pkg.activate(".")
Pkg.instantiate()
Pkg.test()
println("ALL JULIA TESTS PASSED")
```

After final verification passes, update `.machine_readable/STATE.scm` to set
`overall-completion` to `95` (the remaining 5% is for Documenter.jl setup,
CI/CD Julia workflow, and package registration).
