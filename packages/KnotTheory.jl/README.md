# KnotTheory.jl

[![Project Topology](https://img.shields.io/badge/Project-Topology-9558B2)](TOPOLOGY.md)
[![Completion Status](https://img.shields.io/badge/Completion-95%25-green)](TOPOLOGY.md)
[![License](https://img.shields.io/badge/License-PMPL--1.0-blue.svg)](LICENSE)

A comprehensive Julia toolkit for computational knot theory: planar diagram
data structures, classical invariants, polynomial invariants, Seifert theory,
Reidemeister simplification, braid word interop, and import/export helpers.

## Installation

### From Julia REPL
```julia
using Pkg
Pkg.add("KnotTheory")
```

### From Git (Development)
```julia
using Pkg
Pkg.add(url="https://github.com/hyperpolymath/KnotTheory.jl")
```

## Quick Start

```julia
using KnotTheory

k = trefoil()
println(crossing_number(k))       # 3
println(alexander_polynomial(k))   # t^-1 - 1 + t
println(jones_polynomial(k))       # -t^-4 + t^-3 + t^-1
```

## Features

- **Planar diagram model** with oriented crossings and multi-component links.
- **Code representations**: PD code, DT/Dowker code, JSON serialization.
- **Classical invariants**: crossing number, writhe, linking number, signature, determinant.
- **Polynomial invariants**: Alexander, Jones, Conway, and HOMFLY-PT.
- **Seifert theory**: Seifert circles, Seifert matrix, braid index estimate.
- **Reidemeister simplification**: R1, R2, R3 moves and combined simplifier.
- **Braid word interop**: convert between planar diagrams and braid words (TANGLE compatibility).
- **Knot table**: built-in catalogue with named knots up to standard tables.
- **Graph conversion**: Graphs.jl integration via `to_graph`.
- **Polynomial helpers**: Polynomials.jl conversion via `to_polynomial`.
- **Optional plotting**: CairoMakie-based diagram rendering via package extension.

## API Reference

### Types

| Type | Description |
|------|-------------|
| `EdgeOrientation` | Enum for edge direction (`Over`, `Under`) |
| `Crossing` | Single crossing with strand indices and orientation |
| `PlanarDiagram` | Full planar diagram with crossings and components |
| `DTCode` | Dowker-Thistlethwaite code representation |
| `Knot` | Named knot wrapper (e.g. `trefoil()`) |
| `Link` | Named link wrapper for multi-component objects |

### Constructors

| Function | Description |
|----------|-------------|
| `unknot()` | The unknot (zero crossings) |
| `trefoil()` | Trefoil knot (3_1) |
| `figure_eight()` | Figure-eight knot (4_1) |
| `cinquefoil()` | Cinquefoil knot (5_1) |
| `knot_table(name)` | Look up knot by standard name |
| `lookup_knot(property, value)` | Search knot table by invariant value |

### Classical Invariants

| Function | Description |
|----------|-------------|
| `crossing_number(k)` | Minimum crossing number |
| `writhe(pd)` | Sum of crossing signs |
| `linking_number(pd)` | Linking number for two-component links |
| `signature(k)` | Knot signature (from Seifert matrix) |
| `determinant(k)` | Knot determinant (\|det(V + V^T)\|) |

### Polynomial Invariants

| Function | Description |
|----------|-------------|
| `alexander_polynomial(k)` | Alexander polynomial via Seifert matrix |
| `jones_polynomial(k)` | Jones polynomial via skein relation |
| `conway_polynomial(k)` | Conway polynomial (substitution from Alexander) |
| `homfly_polynomial(k)` | HOMFLY-PT two-variable polynomial |

### Seifert Theory

| Function | Description |
|----------|-------------|
| `seifert_circles(pd)` | Seifert circle decomposition |
| `seifert_circles_with_map(pd)` | Seifert circles with strand-to-circle mapping |
| `seifert_matrix(pd)` | Seifert matrix computation |
| `braid_index_estimate(pd)` | Lower bound on braid index from Seifert circles |

### Codes & Serialization

| Function | Description |
|----------|-------------|
| `pdcode(k)` | Planar diagram code (list of crossing tuples) |
| `dtcode(k)` | Dowker-Thistlethwaite code |
| `to_dowker(pd)` | Convert planar diagram to Dowker notation |
| `write_knot_json(file, k)` | Serialize knot data to JSON |
| `read_knot_json(file)` | Deserialize knot data from JSON |

### Simplification

| Function | Description |
|----------|-------------|
| `simplify_pd(pd)` | Apply all Reidemeister moves until stable |
| `r1_simplify(pd)` | Reidemeister I: remove kinks |
| `r2_simplify(pd)` | Reidemeister II: cancel opposing crossings |
| `r3_simplify(pd)` | Reidemeister III: triangle move |

### Braid Words (TANGLE Interop)

| Function | Description |
|----------|-------------|
| `from_braid_word(word)` | Construct planar diagram from braid word |
| `to_braid_word(pd)` | Convert planar diagram to braid word |

### Utilities

| Function | Description |
|----------|-------------|
| `to_graph(pd)` | Convert to Graphs.jl graph structure |
| `to_polynomial(expr)` | Convert to Polynomials.jl polynomial |
| `plot_pd(pd)` | Render diagram (requires CairoMakie extension) |

## Development

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. -e 'using Pkg; Pkg.test()'
```

285 tests across 25 test sets covering all exported functions, invariant
consistency, and known values from knot tables.

## Docs & Tutorials

- `docs/README.md` for documentation drafts.
- `tutorials/intro.ipynb` for a minimal notebook scaffold.

## References & Bibliography

### Textbooks

- Adams, C.C. _The Knot Book: An Elementary Introduction to the Mathematical Theory of Knots_. American Mathematical Society, 2004. — Accessible introduction to knot theory.
- Lickorish, W.B.R. _An Introduction to Knot Theory_. Graduate Texts in Mathematics 175, Springer, 1997. — Graduate-level treatment of knot invariants.
- Rolfsen, D. _Knots and Links_. AMS Chelsea Publishing, 1976/2003. — Classic reference for knot tables and enumeration.
- Murasugi, K. _Knot Theory and Its Applications_. Birkhauser, 1996. — Applied knot theory with connections to biology and chemistry.
- Kauffman, L.H. _Knots and Physics_. 3rd ed., World Scientific, 2001. — Jones polynomial, bracket polynomial, and physical applications.
- Cromwell, P.R. _Knots and Links_. Cambridge University Press, 2004. — Modern computational approach to knot theory.

### Key Papers

- Fox, R.H. "Free differential calculus. I: Derivation in the free group ring." _Annals of Mathematics_ 57(3), 1953, pp. 547-560. — Fox calculus underlying Alexander polynomial computation.
- Freyd, P., Yetter, D., Hoste, J., Lickorish, W.B.R., Millett, K., & Ocneanu, A. "A new polynomial invariant of knots and links." _Bulletin of the AMS_ 12(2), 1985, pp. 239-246. — HOMFLY-PT polynomial.
- Seifert, H. "Uber das Geschlecht von Knoten." _Mathematische Annalen_ 110, 1935, pp. 571-592. — Seifert surfaces and matrix construction.
- Jones, V.F.R. "A polynomial invariant for knots via von Neumann algebras." _Bulletin of the AMS_ 12(1), 1985, pp. 103-111. — Jones polynomial.

## License

Palimpsest-MPL License v1.0 (PMPL-1.0-or-later) -- see [LICENSE](LICENSE).
