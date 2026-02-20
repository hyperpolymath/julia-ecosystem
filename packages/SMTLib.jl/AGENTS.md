# Repository Guidelines

## Project Structure & Module Organization
- `src/SMTLib.jl` contains the main `SMTLib` module and public API.
- `test/runtests.jl` holds the test suite driven by Julia’s `Test` stdlib.
- `Project.toml` defines package metadata and dependencies.
- `README.md` documents features, usage, and solver prerequisites.

## Build, Test, and Development Commands
- `julia --project=. -e 'using Pkg; Pkg.instantiate()'` installs dependencies for this project environment.
- `julia --project=. -e 'using Pkg; Pkg.precompile()'` precompiles for faster local runs.
- `julia --project=. -e 'using Pkg; Pkg.test()'` runs the test suite.

This package is pure Julia, so there is no separate build step beyond precompilation.

## User Options & Configuration
- Prefer `find_solver(:z3)` or `find_solver(:cvc5)` when you need a specific backend.
- `SMTContext` accepts `solver`, `logic`, and `timeout_ms` (milliseconds).
- `check_sat(ctx; get_model=false)` skips model parsing when you only need status.
- The `@smt` macro mirrors `SMTContext` options: `@smt solver=:z3 logic=:QF_LRA timeout=10000 begin ... end`.

## Coding Style & Naming Conventions
- Follow Julia conventions: 4-space indentation, CamelCase for types (`SMTContext`), lowercase with underscores for functions (`available_solvers`), and `!` suffix for mutating functions (`reset!`, `assert!`).
- Keep public API exported from `src/SMTLib.jl` and add docstrings for new public functions.
- Prefer explicit types for public structs and use `Symbol` for SMT identifiers.

## Testing Guidelines
- Tests live in `test/runtests.jl` and use `Test.@testset` blocks.
- Add new tests near related functionality and keep them deterministic.
- Solver-dependent tests should gracefully skip when no SMT solver is installed.

## CI & Solver Detection
- CI should install at least one solver and ensure it is on `PATH` (e.g., `apt-get install z3` on Ubuntu runners).
- Expect solver-backed tests to skip or return `:unknown` if no solver is detected; document this in CI logs or PR notes.
- For a solver matrix, run separate CI jobs with only one solver on `PATH` to validate backend-specific behavior.
- If adding a new solver, extend `available_solvers()` and keep `README.md` and this guide in sync.

## Commit & Pull Request Guidelines
- Current history is minimal; use clear, imperative commit subjects (e.g., "Add model parsing for bitvectors").
- PRs should describe the change, list commands run (e.g., `Pkg.test()`), and note solver prerequisites when relevant.
- If behavior changes are user-visible, update `README.md` examples or API descriptions.

## Solver Prerequisites
- At least one SMT solver (Z3, CVC5, Yices, or MathSAT) must be installed to run solver-backed tests and examples.
- If adding solver-specific features, document them in `README.md` and guard for missing executables.
