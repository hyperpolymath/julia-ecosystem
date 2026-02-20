# SONNET-TASKS.md — PolyglotFormalisms.jl Completion Tasks

> **Generated:** 2026-02-12 by Opus audit
> **Purpose:** Unambiguous instructions for Sonnet to complete all stubs, TODOs, and placeholder code.
> **Honest completion before this file:** 62%

The code that exists (Arithmetic, Comparison, Logical, StringOps) is functional and well-documented.
However: two modules are entirely missing (Collection, Conditional), the String module has a typo-bug,
STATE.scm claims String is 0% complete when it is actually implemented and tested,
the README.md is stale (claims modules are "Planned" that already exist),
the ABI/FFI layer is entirely unsubstituted template placeholders,
and there is a SPDX license violation in three files.

---

## GROUND RULES FOR SONNET

1. Read this entire file before starting any task.
2. Do tasks in order listed. Earlier tasks unblock later ones.
3. After each task, run the verification command. If it fails, fix before moving on.
4. Do NOT mark done unless verification passes.
5. Update `.machine_readable/STATE.scm` with honest completion percentages after each task.
6. Commit after each task: `fix(component): complete <description>`
7. Run full test suite after every 3 tasks.

---

## TASK 1: Fix StringOps module closing comment typo (CRITICAL)

**File:** `/var/mnt/eclipse/repos/PolyglotFormalisms.jl/src/string.jl`
**Line:** 390

The module closing comment reads `end # module StringOpsOps` but the module is named `StringOps`.
This is a copy-paste typo that will confuse tooling and developers.

**Action:** Change line 390 from:
```julia
end # module StringOpsOps
```
to:
```julia
end # module StringOps
```

**Verification:**
```bash
cd /var/mnt/eclipse/repos/PolyglotFormalisms.jl
grep -n "StringOpsOps" src/string.jl
# Expected: no output (zero matches)
grep -n "end # module StringOps$" src/string.jl
# Expected: exactly one match at line 390
```

---

## TASK 2: Fix SPDX license headers in ABI/FFI template files (CRITICAL)

Three files use `AGPL-3.0-or-later` instead of `PMPL-1.0-or-later`. Per CLAUDE.md policy,
AGPL-3.0 is the OLD license and must NEVER be used.

**Files to fix:**
1. `/var/mnt/eclipse/repos/PolyglotFormalisms.jl/ffi/zig/src/main.zig` — line 6
2. `/var/mnt/eclipse/repos/PolyglotFormalisms.jl/ffi/zig/build.zig` — line 2
3. `/var/mnt/eclipse/repos/PolyglotFormalisms.jl/ffi/zig/test/integration_test.zig` — line 2

**Also fix:**
4. `/var/mnt/eclipse/repos/PolyglotFormalisms.jl/examples/SafeDOMExample.res` — line 1

**Action:** In each file, replace:
```
SPDX-License-Identifier: AGPL-3.0-or-later
```
with:
```
SPDX-License-Identifier: PMPL-1.0-or-later
```

**Verification:**
```bash
cd /var/mnt/eclipse/repos/PolyglotFormalisms.jl
grep -rn "AGPL" ffi/ examples/ src/abi/
# Expected: no output (zero matches)
grep -rn "PMPL-1.0-or-later" ffi/ examples/
# Expected: 4 matches (one per file)
```

---

## TASK 3: Substitute ABI/FFI template placeholders (HIGH)

All three Idris2 ABI files and all three Zig FFI files contain unsubstituted `{{PROJECT}}`
and `{{project}}` placeholders from rsr-template-repo. These files cannot compile.

**Files with `{{PROJECT}}` or `{{project}}` placeholders:**
1. `/var/mnt/eclipse/repos/PolyglotFormalisms.jl/src/abi/Types.idr` — line 11 (`{{PROJECT}}.ABI.Types`)
2. `/var/mnt/eclipse/repos/PolyglotFormalisms.jl/src/abi/Layout.idr` — lines 8, 10 (`{{PROJECT}}.ABI.Layout`, `{{PROJECT}}.ABI.Types`)
3. `/var/mnt/eclipse/repos/PolyglotFormalisms.jl/src/abi/Foreign.idr` — lines 9, 11, 12, 23, 35, 49, 72, 77, 98, 125, 152, 164, 185, 211 (many instances)
4. `/var/mnt/eclipse/repos/PolyglotFormalisms.jl/ffi/zig/src/main.zig` — lines 1, 12, 54, 73, 89, 113, 135, 148, 184, 198, 203, 215, 246, 256, 263, 267, 271
5. `/var/mnt/eclipse/repos/PolyglotFormalisms.jl/ffi/zig/build.zig` — lines 1, 13, 23, 34, 82
6. `/var/mnt/eclipse/repos/PolyglotFormalisms.jl/ffi/zig/test/integration_test.zig` — lines 1, 10-17, 24, 25, 31, 32, 34, 39, 48, 49, 51, 56, etc.

**Action:** In all six files, perform global find-and-replace:
- `{{PROJECT}}` -> `PolyglotFormalisms`
- `{{project}}` -> `polyglot_formalisms`

**Verification:**
```bash
cd /var/mnt/eclipse/repos/PolyglotFormalisms.jl
grep -rn '{{PROJECT}}\|{{project}}' src/abi/ ffi/
# Expected: no output (zero matches)
grep -c "PolyglotFormalisms" src/abi/Types.idr
# Expected: at least 1
grep -c "polyglot_formalisms" ffi/zig/src/main.zig
# Expected: at least 10
```

---

## TASK 4: Create Collection module (HIGH)

**Why:** The main module at `src/PolyglotFormalisms.jl` line 54 exports `Collection` and line 60
has `# include("collection.jl")` commented out. The module docstring (line 24) documents it.
STATE.scm lists it at 0%. The spec requires `map`, `filter`, `fold`, `contains`.

**File to create:** `/var/mnt/eclipse/repos/PolyglotFormalisms.jl/src/collection.jl`

**Required functions (matching aggregate-library spec):**

```julia
module Collection
export map_items, filter_items, fold_items, contains_item

# map_items: Apply function to each element
# Signature: map_items(f::Function, items::Vector) -> Vector
# Properties: map(id, xs) == xs; map(f . g, xs) == map(f, map(g, xs))

# filter_items: Keep elements matching predicate
# Signature: filter_items(pred::Function, items::Vector) -> Vector
# Properties: filter(const_true, xs) == xs; filter(const_false, xs) == []

# fold_items: Reduce collection to single value (left fold)
# Signature: fold_items(f::Function, init, items::Vector) -> Any
# Properties: fold(f, z, []) == z; fold(f, z, [x]) == f(z, x)

# contains_item: Check if element is in collection
# Signature: contains_item(items::Vector, item) -> Bool
# Properties: contains([], x) == false; contains([x, ...], x) == true
end
```

**Important:** Use `map_items`, `filter_items`, `fold_items`, `contains_item` to avoid
shadowing Base functions `map`, `filter`, `foldl`, `in`.

Each function MUST have:
- Full docstring in the same style as `src/arithmetic.jl` (interface signature, behavioral semantics, mathematical properties, examples, edge cases)
- SPDX header: `# SPDX-License-Identifier: PMPL-1.0-or-later`
- Implementation using Julia standard library (`Base.map`, `Base.filter`, `Base.foldl`, `Base.in`)

**Also uncomment in `src/PolyglotFormalisms.jl` line 60:**
```julia
include("collection.jl")
```

**Verification:**
```bash
cd /var/mnt/eclipse/repos/PolyglotFormalisms.jl
julia --project=. -e '
using PolyglotFormalisms
@assert Collection.map_items(x -> x * 2, [1, 2, 3]) == [2, 4, 6]
@assert Collection.filter_items(x -> x > 2, [1, 2, 3, 4]) == [3, 4]
@assert Collection.fold_items(+, 0, [1, 2, 3]) == 6
@assert Collection.contains_item([1, 2, 3], 2) == true
@assert Collection.contains_item([1, 2, 3], 5) == false
println("Collection module OK")
'
```

---

## TASK 5: Create Collection module tests (HIGH)

**File to create:** `/var/mnt/eclipse/repos/PolyglotFormalisms.jl/test/collection_tests.jl`

**Required test sets (minimum 30 tests):**
- `map_items`: basic mapping, identity function, composition, empty collection, type preservation
- `filter_items`: basic filtering, always-true predicate, always-false predicate, empty collection
- `fold_items`: sum, product, string concatenation, empty collection returns init, single-element
- `contains_item`: present element, absent element, empty collection, first/last element
- Property tests: map preserves length, filter result is subset, fold over empty returns init

**Style:** Match existing test files exactly (SPDX header, docstring, `@testset` blocks).

**Also add to `test/runtests.jl`:**
```julia
include("collection_tests.jl")
```

**Verification:**
```bash
cd /var/mnt/eclipse/repos/PolyglotFormalisms.jl
julia --project=. -e 'using Pkg; Pkg.test()'
# Expected: all tests pass, including new collection tests
```

---

## TASK 6: Create Conditional module (HIGH)

**Why:** The main module at `src/PolyglotFormalisms.jl` line 54 exports `Conditional` and line 61
has `# include("conditional.jl")` commented out. The module docstring (line 25) documents it.
STATE.scm lists it at 0%. The spec requires `if_then_else`.

**File to create:** `/var/mnt/eclipse/repos/PolyglotFormalisms.jl/src/conditional.jl`

**Required function:**

```julia
module Conditional
export if_then_else

# if_then_else: Conditional evaluation
# Signature: if_then_else(condition::Bool, then_val, else_val) -> Any
# Behavioral semantics:
#   - If condition is true, returns then_val
#   - If condition is false, returns else_val
# Properties:
#   - if_then_else(true, a, b) == a
#   - if_then_else(false, a, b) == b
#   - if_then_else(c, a, a) == a  (constant case)
end
```

Each function MUST have:
- Full docstring in the same style as `src/arithmetic.jl`
- SPDX header
- Implementation: `if_then_else(cond::Bool, then_val, else_val) = cond ? then_val : else_val`

**Also uncomment in `src/PolyglotFormalisms.jl` line 61:**
```julia
include("conditional.jl")
```

**Verification:**
```bash
cd /var/mnt/eclipse/repos/PolyglotFormalisms.jl
julia --project=. -e '
using PolyglotFormalisms
@assert Conditional.if_then_else(true, "yes", "no") == "yes"
@assert Conditional.if_then_else(false, "yes", "no") == "no"
@assert Conditional.if_then_else(true, 42, 0) == 42
@assert Conditional.if_then_else(false, 42, 0) == 0
println("Conditional module OK")
'
```

---

## TASK 7: Create Conditional module tests (HIGH)

**File to create:** `/var/mnt/eclipse/repos/PolyglotFormalisms.jl/test/conditional_tests.jl`

**Required test sets (minimum 15 tests):**
- `if_then_else`: true condition, false condition, integer values, string values, nothing/missing values
- Property tests: constant case (`if_then_else(c, a, a) == a`), true branch, false branch
- Type tests: works with mixed types, works with collections, works with functions as values
- Edge cases: nested if_then_else

**Also add to `test/runtests.jl`:**
```julia
include("conditional_tests.jl")
```

**Verification:**
```bash
cd /var/mnt/eclipse/repos/PolyglotFormalisms.jl
julia --project=. -e 'using Pkg; Pkg.test()'
# Expected: all tests pass, including new conditional tests
```

---

## TASK 8: Update README.md to reflect actual status (MEDIUM)

**File:** `/var/mnt/eclipse/repos/PolyglotFormalisms.jl/README.md`

The README has multiple stale claims:

1. **Lines 62-66**: Lists Comparison, Logical, String as `*(Planned)*` — they are all IMPLEMENTED.
   Fix: Remove `*(Planned)*` from Comparison, Logical, String. Keep `*(Planned)*` only on
   Collection and Conditional until Tasks 4-7 are done, then update those too.

2. **Line 164**: Says "Current: Arithmetic module complete with 59 passing tests" — this is the
   v0.1.0 status. After Tasks 4-7 there will be 6 modules complete. Update to actual test count.

3. **Line 166**: Says "Planned: Comparison, Logical, String, Collection, Conditional modules" —
   three of those are done. Update.

4. **Lines 57**: The "(Properties are proven when Axiom.jl integration is complete...)" note is
   accurate but should be kept.

**Action:** Update the Modules section to show actual completion status:
```markdown
## Modules

- **Arithmetic**: `add`, `subtract`, `multiply`, `divide`, `modulo`
- **Comparison**: `less_than`, `greater_than`, `equal`, `not_equal`, `less_equal`, `greater_equal`
- **Logical**: `and`, `or`, `not`
- **StringOps**: `concat`, `length`, `substring`, `index_of`, `contains`, `starts_with`, `ends_with`, `to_uppercase`, `to_lowercase`, `trim`, `split`, `join`, `replace`, `is_empty`
- **Collection**: `map_items`, `filter_items`, `fold_items`, `contains_item`
- **Conditional**: `if_then_else`
```

Update the Status section to reflect the actual test count (run tests first to get exact number).

**Verification:**
```bash
cd /var/mnt/eclipse/repos/PolyglotFormalisms.jl
grep -c "Planned" README.md
# Expected: 0 (after all modules are implemented)
grep "Current" README.md
# Expected: reflects actual status with correct test count
```

---

## TASK 9: Update STATE.scm to reflect actual completion (MEDIUM)

**File:** `/var/mnt/eclipse/repos/PolyglotFormalisms.jl/.machine_readable/STATE.scm`

STATE.scm has these inaccuracies:

1. **Line 21**: `(overall-completion 50)` — after Tasks 4-7, this should be `100` (all 6 modules).
2. **Lines 32-39**: String listed as `(completion . 0) (status . "planned")` — but `src/string.jl`
   exists with 14 functions and `test/string_tests.jl` has 89 tests. This should be `(completion . 100) (status . "complete")`.
3. **Lines 35-39**: Collection and Conditional listed as 0% — update to 100% after Tasks 4-7.
4. **Line 9**: `(updated "2026-01-23T20:00:00Z")` — update to today's date.
5. **Line 45**: Test count `198` needs updating to include string tests (89) and new module tests.
6. **Milestone-3**: Should be marked complete after Tasks 4-7.

**Action:** Update all completion values, statuses, dates, and test counts to reflect reality.

**Verification:**
```bash
cd /var/mnt/eclipse/repos/PolyglotFormalisms.jl
grep "overall-completion" .machine_readable/STATE.scm
# Expected: (overall-completion 100)
grep -A2 '"String"' .machine_readable/STATE.scm
# Expected: (completion . 100) (status . "complete")
```

---

## TASK 10: Update ROADMAP.scm to reflect actual progress (MEDIUM)

**File:** `/var/mnt/eclipse/repos/PolyglotFormalisms.jl/.machine_readable/ROADMAP.scm`

The ROADMAP.scm was last updated `2025-01-23` and still shows v0.2.0 as "planned" when
it was released on 2026-01-23. After Tasks 4-7:

1. Mark v0.2.0 as `(status . "released")`
2. Mark v0.3.0 as `(status . "released")` (String + Collection done)
3. Update Conditional to reflect completion
4. Update the themes section milestones with checkmarks

**Verification:**
```bash
cd /var/mnt/eclipse/repos/PolyglotFormalisms.jl
grep -A3 '"0.2.0"' .machine_readable/ROADMAP.scm
# Expected: (status . "released")
grep -A3 '"0.3.0"' .machine_readable/ROADMAP.scm
# Expected: (status . "released") or (status . "complete")
```

---

## TASK 11: Update CHANGELOG.scm with new release entry (MEDIUM)

**File:** `/var/mnt/eclipse/repos/PolyglotFormalisms.jl/.machine_readable/CHANGELOG.scm`

Add entries for the work done in Tasks 1-7:

1. Add a v0.3.0 release entry documenting:
   - String module (14 operations, 89 tests)
   - Collection module (4 operations)
   - Conditional module (1 operation)
   - ABI/FFI template placeholder substitution
   - SPDX license fixes

**Verification:**
```bash
cd /var/mnt/eclipse/repos/PolyglotFormalisms.jl
grep '"0.3.0"' .machine_readable/CHANGELOG.scm
# Expected: at least one match
```

---

## TASK 12: Update CrossLanguageStatus.md (LOW)

**File:** `/var/mnt/eclipse/repos/PolyglotFormalisms.jl/docs/CrossLanguageStatus.md`

Line 9 claims version `0.3.0` with `287/287` tests and "Complete" status. After Tasks 4-7,
the test count will increase. Update line 72 with actual total test count.

Also update line 9 if the version changes.

**Verification:**
```bash
cd /var/mnt/eclipse/repos/PolyglotFormalisms.jl
# Run actual test count
julia --project=. -e 'using Pkg; Pkg.test()' 2>&1 | tail -5
# Compare test count with what CrossLanguageStatus.md claims
```

---

## TASK 13: Fix Manifest.toml stale package name reference (LOW)

**File:** `/var/mnt/eclipse/repos/PolyglotFormalisms.jl/Manifest.toml`

Lines 52-55 contain a stale reference to the old package name `aLib`:
```toml
[[deps.aLib]]
path = "."
uuid = "8fd979ee-625c-447d-87f1-33af4d789de5"
version = "0.1.0"
```

The package was renamed to `PolyglotFormalisms` (Project.toml says `name = "PolyglotFormalisms"`
at version `1.0.0`). The Manifest.toml still references the old name at version `0.1.0`.

**Action:** Regenerate Manifest.toml:
```bash
cd /var/mnt/eclipse/repos/PolyglotFormalisms.jl
julia --project=. -e 'using Pkg; Pkg.resolve()'
```

If that does not fix the `aLib` reference, manually rename `aLib` to `PolyglotFormalisms`
and update version to `1.0.0` in the Manifest.toml.

**Verification:**
```bash
cd /var/mnt/eclipse/repos/PolyglotFormalisms.jl
grep "aLib" Manifest.toml
# Expected: no output (zero matches)
grep "PolyglotFormalisms" Manifest.toml
# Expected: at least one match
```

---

## TASK 14: Reconcile version number confusion (LOW)

There is a version inconsistency across the repository:

| Location | Version claimed |
|----------|----------------|
| `Project.toml` | `1.0.0` |
| Git tags | `v0.1.0`, `v0.1.1`, `v0.2.0`, `v1.0.0` |
| `STATE.scm` metadata | `"1.0"` |
| `META.scm` | `"0.1.0"` |
| `Manifest.toml` deps.aLib | `"0.1.0"` |
| `CHANGELOG.scm` latest release | `"0.2.0"` |
| `CrossLanguageStatus.md` | `"0.3.0"` |
| `ffi/zig/src/main.zig` VERSION | `"0.1.0"` |
| `ffi/zig/build.zig` lib.version | `0.1.0` |

The Project.toml says `1.0.0` and there is a `v1.0.0` git tag, but the code is clearly NOT
at 1.0.0 maturity (missing modules, template placeholders, no Axiom.jl integration).

**Action:** Decide on the correct version. Given the state after Tasks 1-13, `0.3.0` seems
appropriate (all 6 core modules complete, no formal verification yet). Update all files to
use a consistent version.

If keeping `1.0.0`, update all references. If reverting to `0.3.0`:
- `Project.toml`: `version = "0.3.0"`
- `META.scm`: `(version . "0.3.0")`
- `ffi/zig` VERSION constant: `"0.3.0"`
- `ffi/zig` build.zig version: `.{ .major = 0, .minor = 3, .patch = 0 }`

**Verification:**
```bash
cd /var/mnt/eclipse/repos/PolyglotFormalisms.jl
# All version references should be consistent
grep -rn 'version.*"0\.' Project.toml .machine_readable/META.scm ffi/zig/src/main.zig
# Expected: all show the same version
```

---

## FINAL VERIFICATION

After completing all tasks, run this full verification sequence:

```bash
cd /var/mnt/eclipse/repos/PolyglotFormalisms.jl

# 1. Full test suite passes
julia --project=. -e 'using Pkg; Pkg.test()'

# 2. All modules load without error
julia --project=. -e '
using PolyglotFormalisms
println("Arithmetic: ", methods(Arithmetic.add))
println("Comparison: ", methods(Comparison.less_than))
println("Logical: ", methods(Logical.and))
println("StringOps: ", methods(StringOps.concat))
println("Collection: ", methods(Collection.map_items))
println("Conditional: ", methods(Conditional.if_then_else))
println("All 6 modules loaded successfully")
'

# 3. No template placeholders remain
grep -rn '{{PROJECT}}\|{{project}}\|{{PLACEHOLDER}}\|{{REPO}}\|{{OWNER}}\|{{FORGE}}' \
  src/ ffi/ test/ examples/
# Expected: no output

# 4. No AGPL license references
grep -rn 'AGPL' src/ ffi/ test/ examples/
# Expected: no output

# 5. No StringOpsOps typo
grep -rn 'StringOpsOps' src/
# Expected: no output

# 6. STATE.scm shows 100% completion
grep 'overall-completion' .machine_readable/STATE.scm
# Expected: (overall-completion 100)

# 7. No stale "Planned" markers for implemented modules in README
grep 'Planned' README.md
# Expected: 0 matches (or only for future features like Axiom.jl)

# 8. Consistent version across all files
echo "=== Version check ==="
grep '^version' Project.toml
grep 'version.*"[0-9]' .machine_readable/META.scm
```
