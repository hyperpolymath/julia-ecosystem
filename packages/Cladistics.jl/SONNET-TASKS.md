# SONNET-TASKS.md -- Cladistics.jl Completion Tasks

> **Generated:** 2026-02-12 by Opus audit
> **Purpose:** Unambiguous instructions for Sonnet to complete all stubs, TODOs, and placeholder code.
> **Honest completion before this file:** 55%

The package has a solid single-file implementation (`src/Cladistics.jl`) with working distance
matrix calculations, UPGMA, Neighbor-Joining, bootstrap support, clade identification,
Robinson-Foulds distance, and Newick export. However, it contains three critical bugs that
will cause runtime crashes, one fully stubbed function (`root_tree`), one exported symbol
with no implementation (`maximum_parsimony`), and extensive uncustomized RSR template
boilerplate across all metadata files. The Project.toml UUID is fabricated (not generated
by Julia's `Pkg.generate`).

---

## GROUND RULES FOR SONNET

1. Read this entire file before starting any task.
2. Do tasks in order listed. Earlier tasks unblock later ones.
3. After each task, run the verification command. If it fails, fix before moving on.
4. Do NOT mark done unless verification passes.
5. Update `.machines_readable/6scm/STATE.scm` with honest completion percentages after each task.
6. Commit after each task: `fix(component): complete <description>`
7. Run full test suite after every 3 tasks: `cd /var/mnt/eclipse/repos/Cladistics.jl && julia --project=. -e 'using Pkg; Pkg.test()'`

---

## TASK 1: Fix `fitch_score` inconsistent return type (CRITICAL)

**Files:** `/var/mnt/eclipse/repos/Cladistics.jl/src/Cladistics.jl` lines 474-492

**Problem:** `fitch_score` has an inconsistent return type that will crash at runtime.

- Line 478: Terminal nodes return `Set{Char}` (a single value).
- Line 488: Internal nodes return `(Set{Char}, Int)` (a tuple).
- Line 490: Internal nodes return `(Set{Char}, Int)` (a tuple).

When `calculate_parsimony_score` (line 467) calls `fitch_score`, it does
`score = fitch_score(...)` and then `total_score += score`. But `score` will be either
a `Set{Char}` or a `Tuple{Set{Char}, Int}`, neither of which can be added to an `Int`.

Additionally, line 482 calls `fitch_score` recursively on children and stores results in
`child_sets`. When children are leaf nodes, `child_sets` contains `Set{Char}` values.
When children are internal nodes, `child_sets` contains `(Set{Char}, Int)` tuples. Then
line 485 calls `reduce(intersect, child_sets)` which will fail because you cannot
`intersect` tuples.

**What to do:**

1. Rewrite `fitch_score` to return a `Tuple{Set{Char}, Int}` consistently. Terminal nodes
   should return `(Set([char_column[idx]]), 0)`.
2. Internal nodes must unpack the tuple from each child: extract the `Set{Char}` part for
   the intersection/union logic, and sum the `Int` parts for the cumulative score.
3. The function should return `(result_set, child_score_sum + local_cost)` where
   `local_cost` is 0 if intersection is non-empty, 1 if union was needed.
4. Update `calculate_parsimony_score` (lines 462-472) to unpack the tuple:
   `(_, score) = fitch_score(tree.root, char_matrix[:, j], tree.taxa)` and then
   `total_score += score`.

**Verification:**
```julia
cd("/var/mnt/eclipse/repos/Cladistics.jl")
using Pkg; Pkg.activate(".")
using Cladistics

seqs = ["ATCG", "ATCG", "TTCG", "TTCC"]
dmat = distance_matrix(seqs, method=:hamming)
tree = upgma(dmat, taxa_names=["A", "B", "C", "D"])
cm = character_state_matrix(seqs)
score = calculate_parsimony_score(tree, cm)
@assert score isa Int "Score must be an Int, got $(typeof(score))"
@assert score >= 0 "Score must be non-negative"
@assert score <= 8 "Score suspiciously high for 4 taxa, 4 chars"
println("TASK 1 PASS: parsimony score = $score")
```

---

## TASK 2: Fix Kimura 2-parameter operator precedence bug (CRITICAL)

**Files:** `/var/mnt/eclipse/repos/Cladistics.jl/src/Cladistics.jl` line 187

**Problem:** Line 187 reads:

```julia
(1.0 - 2P - Q) <= 0 || (1.0 - 2Q) <= 0 && return Inf
```

In Julia, `&&` binds tighter than `||`. So this parses as:

```julia
(1.0 - 2P - Q) <= 0 || ((1.0 - 2Q) <= 0 && return Inf)
```

This means:
- If `(1.0 - 2P - Q) <= 0` is true but `(1.0 - 2Q) > 0`, the function does NOT return
  `Inf`. It continues to line 189 and computes `log` of a non-positive number, producing
  `NaN` or throwing a `DomainError`.
- The fix must ensure that if EITHER condition is non-positive, `Inf` is returned.

Additionally, `2P` and `2Q` in Julia are parsed as `2 * P` and `2 * Q` only because Julia
supports coefficient syntax, but this is fragile and unconventional for a scientific
package. Use explicit `2.0 * P` and `2.0 * Q` for clarity.

**What to do:**

1. Replace line 187 with:
   ```julia
   if (1.0 - 2.0 * P - Q) <= 0.0 || (1.0 - 2.0 * Q) <= 0.0
       return Inf
   end
   ```
2. Also update line 189 to use explicit multiplication for consistency:
   ```julia
   return -0.5 * log((1.0 - 2.0 * P - Q) * sqrt(1.0 - 2.0 * Q))
   ```

**Verification:**
```julia
cd("/var/mnt/eclipse/repos/Cladistics.jl")
using Pkg; Pkg.activate(".")
using Cladistics

# Case where (1-2P-Q) <= 0 but (1-2Q) > 0 -- would have been a bug
# P=0.4, Q=0.1 => 1-2(0.4)-0.1 = 0.1 > 0, OK
# P=0.45, Q=0.1 => 1-2(0.45)-0.1 = 0.0 <= 0, should be Inf
# Use sequences that produce high transition ratio
seqs_saturated = ["AAAA", "GGGG"]  # All transitions A<->G
dmat = distance_matrix(seqs_saturated, method=:k2p)
@assert isinf(dmat[1,2]) "Saturated transitions must give Inf, got $(dmat[1,2])"

# Normal case still works
seqs_normal = ["ATCG", "GTCG"]  # One transition
dmat_normal = distance_matrix(seqs_normal, method=:k2p)
@assert isfinite(dmat_normal[1,2]) "Normal case must be finite"
@assert dmat_normal[1,2] > 0 "Normal case must be positive"
println("TASK 2 PASS: K2P operator precedence fixed")
```

---

## TASK 3: Implement `root_tree` (stub removal) (HIGH)

**Files:** `/var/mnt/eclipse/repos/Cladistics.jl/src/Cladistics.jl` lines 665-673

**Problem:** `root_tree` is exported and documented but is a stub. Lines 670-672:

```julia
# Implementation would reroot the tree structure
# Simplified version returns the original tree
return tree
```

It finds the outgroup node but then ignores it and returns the original tree unchanged.
This is silently wrong -- callers expect a rerooted tree but get the original.

**What to do:**

1. Implement proper midpoint rerooting on the outgroup branch. The algorithm:
   a. Find the outgroup leaf node by name.
   b. Get its parent (the node it is attached to).
   c. Create a new root node at the midpoint of the outgroup's branch.
   d. One child of the new root is the outgroup (with half its original branch length).
   e. The other child is the rest of the tree (with the other half of the branch length).
   f. Unlink the outgroup from its old parent by removing it from `children`.
   g. The old parent becomes the child of the new root on the ingroup side.
   h. Fix all `parent` references.
2. Return a new `PhylogeneticTree` with `method` preserved from the input tree.
3. Handle edge case: if the outgroup is already a direct child of the root, just
   rebalance the branch lengths.

**Verification:**
```julia
cd("/var/mnt/eclipse/repos/Cladistics.jl")
using Pkg; Pkg.activate(".")
using Cladistics

dmat = [0.0 0.2 0.4 0.5;
        0.2 0.0 0.3 0.4;
        0.4 0.3 0.0 0.2;
        0.5 0.4 0.2 0.0]
tree = neighbor_joining(dmat, taxa_names=["A", "B", "C", "Outgroup"])
rooted = root_tree(tree, "Outgroup")

# The rooted tree must NOT be the same object as the input
@assert rooted !== tree || rooted.root !== tree.root "root_tree must not return input unchanged"

# The outgroup must be a direct child (or grandchild) of the new root
function find_leaf(node, name)
    node.name == name && isempty(node.children) && return true
    any(c -> find_leaf(c, name), node.children)
end
@assert find_leaf(rooted.root, "Outgroup") "Outgroup must be in rooted tree"

# All original taxa must still be present
descendants = String[]
function collect_leaves(node)
    if isempty(node.children)
        push!(descendants, node.name)
    else
        for c in node.children; collect_leaves(c); end
    end
end
collect_leaves(rooted.root)
@assert sort(descendants) == ["A", "B", "C", "Outgroup"] "All taxa must be preserved"

println("TASK 3 PASS: root_tree properly reroots on outgroup")
```

---

## TASK 4: Implement `maximum_parsimony` (exported but missing) (HIGH)

**Files:** `/var/mnt/eclipse/repos/Cladistics.jl/src/Cladistics.jl` line 57

**Problem:** `maximum_parsimony` is listed in the `export` statement on line 57 but has
no function definition anywhere in the file. Calling `maximum_parsimony(...)` will throw
`UndefVarError`.

**What to do:**

1. Add a `maximum_parsimony` function that performs a heuristic search for the most
   parsimonious tree. Since exhaustive search is NP-hard, implement a stepwise addition
   heuristic:
   a. Start with a tree of the first 3 taxa (only one unrooted topology exists).
   b. For each remaining taxon, try inserting it at every branch of the current tree.
   c. Keep the insertion that gives the lowest parsimony score (using the now-fixed
      `calculate_parsimony_score`).
   d. Return the final tree.
2. Signature: `maximum_parsimony(sequences::Vector{String}; taxa_names=nothing) -> PhylogeneticTree`
3. Add a proper docstring following the style of the existing functions.
4. Set `method = :parsimony` on the returned `PhylogeneticTree`.

**Verification:**
```julia
cd("/var/mnt/eclipse/repos/Cladistics.jl")
using Pkg; Pkg.activate(".")
using Cladistics

seqs = ["ATCGATCG", "ATCGATCG", "TTCGTTCG", "TTCCTTCC", "AACGAACG"]
tree = maximum_parsimony(seqs, taxa_names=["A", "B", "C", "D", "E"])

@assert tree isa Cladistics.PhylogeneticTree "Must return PhylogeneticTree"
@assert tree.method == :parsimony "Method must be :parsimony"
@assert length(tree.taxa) == 5 "Must have 5 taxa"
@assert !isempty(tree.root.children) "Root must have children"

# Parsimony score should be computable and reasonable
cm = character_state_matrix(seqs)
score = calculate_parsimony_score(tree, cm)
@assert score isa Int
@assert score >= 0
println("TASK 4 PASS: maximum_parsimony returns valid tree with score=$score")
```

---

## TASK 5: Generate valid Project.toml UUID (MEDIUM)

**Files:** `/var/mnt/eclipse/repos/Cladistics.jl/Project.toml` line 2

**Problem:** The UUID `9e3g4f80-5d7c-6f0d-b4e2-3g9f0d1c2e3f` is invalid. UUIDs use
hexadecimal digits (0-9, a-f) only. This UUID contains `g` characters, which are not valid
hex digits. Julia's package manager will reject this.

**What to do:**

1. Generate a valid UUID by running:
   ```julia
   using UUIDs; println(uuid4())
   ```
2. Replace the `uuid` line in `Project.toml` with the generated UUID.
3. Do NOT change any other field in `Project.toml`.

**Verification:**
```julia
cd("/var/mnt/eclipse/repos/Cladistics.jl")
toml_text = read("Project.toml", String)
uuid_match = match(r"uuid = \"([^\"]+)\"", toml_text)
uuid_str = uuid_match.captures[1]
@assert all(c -> c in "0123456789abcdef-", uuid_str) "UUID must be valid hex, got: $uuid_str"
@assert length(uuid_str) == 36 "UUID must be 36 chars (with dashes)"
@assert count(==('-'), uuid_str) == 4 "UUID must have 4 dashes"
println("TASK 5 PASS: UUID is valid: $uuid_str")
```

---

## TASK 6: Fix SPDX license headers -- replace all AGPL-3.0-or-later with PMPL-1.0-or-later (MEDIUM)

**Files:** All files listed below that still contain `AGPL-3.0-or-later`:
- `.machines_readable/6scm/STATE.scm` line 1
- `.machines_readable/6scm/META.scm` line 1
- `.machines_readable/6scm/ECOSYSTEM.scm` line 1
- `.gitignore` line 1
- `.gitattributes` line 1
- `ffi/zig/build.zig` line 2
- `ffi/zig/src/main.zig` line 6
- `ffi/zig/test/integration_test.zig` line 2
- `examples/SafeDOMExample.res` line 1
- `docs/CITATIONS.adoc` line 13

**Problem:** Per CLAUDE.md, the AGPL-3.0-or-later license has been replaced by
PMPL-1.0-or-later for all hyperpolymath original code. These files still use the old
license identifier.

**What to do:**

1. In every file listed above, replace `AGPL-3.0-or-later` with `PMPL-1.0-or-later`.
2. Do not change anything else in these files.
3. Verify no other files still contain `AGPL-3.0-or-later` (except `RSR_OUTLINE.adoc`
   which discusses licensing policy historically and should not be changed).

**Verification:**
```julia
cd("/var/mnt/eclipse/repos/Cladistics.jl")
for f in readdir(".", join=true)
    isfile(f) || continue
    endswith(f, ".adoc") && basename(f) == "RSR_OUTLINE.adoc" && continue
    content = read(f, String)
    if occursin("AGPL-3.0-or-later", content)
        error("AGPL still found in: $f")
    end
end
# Check subdirectories
for (root, dirs, files) in walkdir(".")
    startswith(root, "./.git") && continue
    for f in files
        path = joinpath(root, f)
        endswith(path, "RSR_OUTLINE.adoc") && continue
        content = read(path, String)
        if occursin("AGPL-3.0-or-later", content)
            error("AGPL still found in: $path")
        end
    end
end
println("TASK 6 PASS: No AGPL-3.0-or-later headers remain (except RSR_OUTLINE.adoc)")
```

---

## TASK 7: Customize SCM files -- replace template placeholders with Cladistics.jl content (MEDIUM)

**Files:**
- `.machines_readable/6scm/STATE.scm` (entire file)
- `.machines_readable/6scm/META.scm` (entire file)
- `.machines_readable/6scm/ECOSYSTEM.scm` (entire file)
- `.machines_readable/6scm/AGENTIC.scm` (entire file)
- `.machines_readable/6scm/NEUROSYM.scm` (entire file)
- `.machines_readable/6scm/PLAYBOOK.scm` (entire file)

**Problem:** All six SCM files are unmodified RSR template copies. Every reference says
`rsr-template-repo` instead of `Cladistics.jl`. STATE.scm claims 5% completion and has
no Cladistics-specific milestones. ECOSYSTEM.scm says `[TODO: Add specific description]`.
META.scm has no architecture decisions relevant to a Julia phylogenetics package.

**What to do:**

1. **STATE.scm:**
   - Replace all `rsr-template-repo` with `Cladistics.jl`.
   - Set `overall-completion` to the honest value after all preceding tasks are done
     (should be around 80-85% at this point).
   - Add real milestones: "v0.1.0 - Core algorithms" (done), "v0.2.0 - Maximum parsimony
     and rerooting" (done after tasks 3-4), "v1.0.0 - Package registration and Newick I/O"
     (todo).
   - Set `tech-stack` to `("Julia" "LinearAlgebra" "Graphs" "Clustering")`.
   - Set `working-features` to list what actually works.
   - Update `repo` to `hyperpolymath/Cladistics.jl`.

2. **ECOSYSTEM.scm:**
   - Replace `rsr-template-repo` with `Cladistics.jl`.
   - Set `type` to `"library"`.
   - Set `purpose` to a real description of a Julia phylogenetics package.
   - Remove `[TODO: Add specific description]` and write a real description.
   - Add related projects: `(related "PhyloNetworks.jl")`, `(related "BioJulia")`.

3. **META.scm:**
   - Replace `rsr-template-repo` with `Cladistics.jl`.
   - Add an ADR for "Use Fitch algorithm for parsimony scoring".
   - Add an ADR for "Support four distance metrics (Hamming, p-distance, JC69, K2P)".
   - Update `code-style` to mention Julia conventions.

4. **AGENTIC.scm:** Replace `rsr-template-repo` with `Cladistics.jl` in the comment.
   Add `"julia"` to the languages list.

5. **NEUROSYM.scm:** Replace `rsr-template-repo` with `Cladistics.jl` in the comment.

6. **PLAYBOOK.scm:** Replace `rsr-template-repo` with `Cladistics.jl` in the comment.
   Update build/test commands to use Julia: `"test" . "julia --project=. -e 'using Pkg; Pkg.test()'"`.

**Verification:**
```julia
cd("/var/mnt/eclipse/repos/Cladistics.jl")
for f in readdir(".machines_readable/6scm", join=true)
    content = read(f, String)
    if occursin("rsr-template-repo", content)
        error("Template placeholder 'rsr-template-repo' still in: $f")
    end
    if occursin("[TODO", content)
        error("TODO placeholder still in: $f")
    end
end
state = read(".machines_readable/6scm/STATE.scm", String)
@assert occursin("Cladistics", state) "STATE.scm must reference Cladistics"
@assert occursin("Julia", state) || occursin("julia", state) "STATE.scm must mention Julia"
println("TASK 7 PASS: All SCM files customized for Cladistics.jl")
```

---

## TASK 8: Customize AI manifest and docs -- replace template placeholders (LOW)

**Files:**
- `/var/mnt/eclipse/repos/Cladistics.jl/0-AI-MANIFEST.a2ml` (lines 7, 51, 56-57, 112-114)
- `/var/mnt/eclipse/repos/Cladistics.jl/AI.a2ml` (line 5 and throughout)
- `/var/mnt/eclipse/repos/Cladistics.jl/docs/CITATIONS.adoc` (entire file)
- `/var/mnt/eclipse/repos/Cladistics.jl/ROADMAP.adoc` (entire file)

**Problem:**

- `0-AI-MANIFEST.a2ml` line 7 says `[YOUR-REPO-NAME]`, lines 56-57 show `[YOUR-REPO-NAME]/`
  as directory name, lines 112-114 have `[DATE]`, `[YOUR-NAME/ORG]` placeholders.
- `AI.a2ml` line 5 refers to `rsr-template-repo` instead of `Cladistics.jl`.
- `docs/CITATIONS.adoc` references `RSR-template-repo` with author "Polymath, Hyper" and
  year 2025. Should reference `Cladistics.jl` with author "Jewell, Jonathan D.A." and
  year 2026.
- `ROADMAP.adoc` line 2 says `YOUR Template Repo Roadmap` and has no Cladistics-specific
  content.

**What to do:**

1. In `0-AI-MANIFEST.a2ml`:
   - Replace `[YOUR-REPO-NAME]` with `Cladistics.jl` (3 occurrences).
   - Replace `[DATE]` with `2026-02-12`.
   - Replace `[YOUR-NAME/ORG]` with `Jonathan D.A. Jewell / hyperpolymath`.
   - Update the repository structure section to show the actual `src/`, `test/` layout.

2. In `AI.a2ml`:
   - Replace `rsr-template-repo` with `Cladistics.jl` on line 5.

3. In `docs/CITATIONS.adoc`:
   - Replace all `RSR-template-repo` with `Cladistics.jl`.
   - Replace `Polymath, Hyper` / `Hyper Polymath` with `Jewell, Jonathan D.A.`.
   - Replace year `2025` with `2026`.
   - Replace author in BibTeX `author` field with `{Jewell, Jonathan D.A.}`.
   - Update URL to `https://github.com/hyperpolymath/Cladistics.jl`.
   - Replace `license = {AGPL-3.0-or-later}` with `license = {PMPL-1.0-or-later}`.

4. In `ROADMAP.adoc`:
   - Replace `YOUR Template Repo Roadmap` with `Cladistics.jl Roadmap`.
   - Replace the generic milestones with real ones:
     - v0.1.0: Core distance metrics and tree-building algorithms (done).
     - v0.2.0: Maximum parsimony search and tree rerooting (done after tasks 3-4).
     - v0.3.0: Newick parser (read trees from strings), tree visualization.
     - v1.0.0: Julia General registry submission, full API docs, benchmarks.

**Verification:**
```julia
cd("/var/mnt/eclipse/repos/Cladistics.jl")
manifest = read("0-AI-MANIFEST.a2ml", String)
@assert !occursin("[YOUR-REPO-NAME]", manifest) "Manifest still has placeholder"
@assert !occursin("[DATE]", manifest) "Manifest still has [DATE]"
@assert !occursin("[YOUR-NAME/ORG]", manifest) "Manifest still has [YOUR-NAME/ORG]"
@assert occursin("Cladistics.jl", manifest) "Manifest must reference Cladistics.jl"

ai = read("AI.a2ml", String)
@assert !occursin("rsr-template-repo", ai) "AI.a2ml still references template"

citations = read("docs/CITATIONS.adoc", String)
@assert occursin("Jewell", citations) "Citations must credit Jewell"
@assert !occursin("Polymath, Hyper", citations) "Citations must not use old author"

roadmap = read("ROADMAP.adoc", String)
@assert !occursin("YOUR Template", roadmap) "Roadmap still has template title"
@assert occursin("Cladistics", roadmap) "Roadmap must reference Cladistics"

println("TASK 8 PASS: All template placeholders replaced")
```

---

## TASK 9: Add test for `maximum_parsimony` and fix test for `calculate_parsimony_score` (MEDIUM)

**Files:** `/var/mnt/eclipse/repos/Cladistics.jl/test/runtests.jl`

**Problem:** The existing test suite has no test for `maximum_parsimony` (which did not
exist before Task 4). The existing test for `calculate_parsimony_score` (which would be
in the "Comparing Alternative Hypotheses" example in the README but is not in the test
file) is absent. The existing tests will also fail until Task 1 is completed because
`calculate_parsimony_score` crashes.

**What to do:**

1. Add a new `@testset "Maximum Parsimony Search"` block after the existing
   `"Parsimony Informative Sites"` testset. Test:
   - Returns a `PhylogeneticTree` with `method == :parsimony`.
   - Contains all input taxa.
   - Parsimony score is computable and reasonable.
   - With identical sequences, parsimony score should be 0.

2. Add a new `@testset "Calculate Parsimony Score"` block that explicitly tests
   `calculate_parsimony_score` with known inputs:
   - 4 identical sequences should give score 0.
   - 4 sequences with known differences should give a predictable non-zero score.

3. Add a test for `maximum_parsimony` with `taxa_names` omitted (default naming).

**Verification:**
```julia
cd("/var/mnt/eclipse/repos/Cladistics.jl")
using Pkg; Pkg.activate(".")
Pkg.test()
println("TASK 9 PASS: All tests pass including new ones")
```

---

## TASK 10: Add Newick parser (`parse_newick`) (LOW)

**Files:** `/var/mnt/eclipse/repos/Cladistics.jl/src/Cladistics.jl`

**Problem:** The package can export trees to Newick format (`tree_to_newick`) but cannot
read them back. A phylogenetics package without a Newick parser is incomplete. Users
cannot import trees from other tools (FigTree, MEGA, RAxML, etc.).

**What to do:**

1. Implement `parse_newick(newick_str::String) -> PhylogeneticTree` that parses a standard
   Newick format string into a `PhylogeneticTree`.
2. Support:
   - Named and unnamed internal nodes.
   - Branch lengths (`:0.123` notation).
   - Nested parentheses for subtrees.
   - Trailing semicolon.
3. Add `parse_newick` to the `export` list on line 57.
4. Add a proper docstring.
5. Add tests in `test/runtests.jl`:
   - Round-trip test: `parse_newick(tree_to_newick(tree))` should produce a tree with the
     same taxa.
   - Parse a known Newick string and verify structure.

**Verification:**
```julia
cd("/var/mnt/eclipse/repos/Cladistics.jl")
using Pkg; Pkg.activate(".")
using Cladistics

# Round-trip test
dmat = [0.0 0.2 0.4; 0.2 0.0 0.3; 0.4 0.3 0.0]
original = upgma(dmat, taxa_names=["A", "B", "C"])
newick = tree_to_newick(original)
parsed = parse_newick(newick)

original_leaves = sort(original.taxa)
parsed_leaves = sort(parsed.taxa)
@assert original_leaves == parsed_leaves "Round-trip must preserve taxa: $original_leaves vs $parsed_leaves"

# Parse known string
tree2 = parse_newick("((A:0.1,B:0.2):0.3,C:0.4);")
@assert "A" in tree2.taxa
@assert "B" in tree2.taxa
@assert "C" in tree2.taxa
@assert length(tree2.taxa) == 3

println("TASK 10 PASS: Newick parser works with round-trip")
```

---

## FINAL VERIFICATION

After all tasks are complete, run this comprehensive check:

```julia
cd("/var/mnt/eclipse/repos/Cladistics.jl")
using Pkg; Pkg.activate(".")

# 1. Full test suite must pass
Pkg.test()

# 2. All exports must be callable
using Cladistics
seqs = ["ATCGATCG", "ATCGATCG", "TTCGTTCG", "TTCCTTCC"]
taxa = ["A", "B", "C", "D"]

dmat = distance_matrix(seqs, method=:hamming)
@assert dmat isa Matrix{Float64}

tree1 = upgma(dmat, taxa_names=taxa)
@assert tree1 isa Cladistics.PhylogeneticTree

tree2 = neighbor_joining(dmat, taxa_names=taxa)
@assert tree2 isa Cladistics.PhylogeneticTree

tree3 = maximum_parsimony(seqs, taxa_names=taxa)
@assert tree3 isa Cladistics.PhylogeneticTree
@assert tree3.method == :parsimony

cm = character_state_matrix(seqs)
score = calculate_parsimony_score(tree1, cm)
@assert score isa Int && score >= 0

sites = parsimony_informative_sites(cm)
@assert sites isa Vector{Int}

support = bootstrap_support(seqs, replicates=10)
@assert support isa Dict

clades = identify_clades(tree1, 0.5)
@assert clades isa Vector{Set{String}}

rf = tree_distance(tree1, tree2)
@assert rf isa Int && rf >= 0

rooted = root_tree(tree2, "D")
# Must not be a no-op stub

newick = tree_to_newick(tree1)
@assert endswith(newick, ";")

parsed = parse_newick(newick)
@assert sort(parsed.taxa) == sort(taxa)

# 3. No template placeholders remain
for (root, dirs, files) in walkdir(".")
    startswith(root, "./.git") && continue
    for f in files
        path = joinpath(root, f)
        content = try read(path, String) catch; continue end
        if occursin("[YOUR-REPO-NAME]", content)
            error("Template placeholder in: $path")
        end
    end
end

println("\n=== ALL FINAL VERIFICATION CHECKS PASSED ===")
```
