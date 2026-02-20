# HackenbushGames.jl Development Roadmap

## Current State (v1.0)

Functional beta Hackenbush game implementation:
- Red, Blue, Green (neutral) edges
- Conway's surreal number evaluation
- Canonical form computation
- Basic game operations (addition via game_sum)
- Simplification algorithms

**Status:** Core algorithms implemented. Test coverage is basic (35 test cases).

---

## v1.0 → v1.2 Roadmap (Near-term)

### v1.1 - Game Analysis & Visualization (3-6 months)

**MUST:**
- [ ] **Interactive game viewer** - Makie.jl/GraphMakie.jl visualization of game positions
- [ ] **Move recommendation engine** - Suggest optimal moves based on game value
- [ ] **Game position database** - Library of solved positions with canonical forms
- [ ] **Proof verification** - Validate game value calculations with step-by-step proofs

**SHOULD:**
- [ ] **AI opponent** - Minimax/alpha-beta pruning for computer play
- [ ] **Opening book** - Pre-computed optimal strategies for common starting positions
- [ ] **Thermography** - Temperature analysis for combinatorial game theory research
- [ ] **LaTeX export** - Generate publication-quality game diagrams and surreal number expressions

**COULD:**
- [ ] **Web-based game player** - Franklin.jl/Genie.jl interactive Hackenbush app
- [ ] **Mobile app** - Touch-friendly Hackenbush game (via Julia + web frontend)
- [ ] **Tutorial mode** - Guided lessons on surreal numbers and game theory

### v1.2 - Advanced Game Theory & Extensions (6-12 months)

**MUST:**
- [ ] **Loopy Hackenbush** - Support for games with cycles (infinite game analysis)
- [ ] **Mis�re Hackenbush** - Last-to-move-loses variant
- [ ] **Team Hackenbush** - Multi-player cooperative/competitive variants
- [ ] **Integration with Graphs.jl** - Advanced graph algorithms for position analysis

**SHOULD:**
- [ ] **Game fuzzing** - Generate random positions for testing edge cases
- [ ] **Symmetry detection** - Identify isomorphic game positions for performance
- [ ] **Decomposition algorithms** - Break complex games into independent subgames
- [ ] **Integration with Axiom.jl** - Formal verification of surreal number arithmetic

**COULD:**
- [ ] **3D Hackenbush** - Extend to spatial graphs (nodes in 3D, edges as rods)
- [ ] **Quantum Hackenbush** - Superposition of game states (research exploration)
- [ ] **Hackenbush variants** - Domineering, Nim, Chomp (expand to other combinatorial games)

---

## v1.3+ Roadmap (Speculative)

### Research Frontiers

**Advanced Combinatorial Game Theory:**
- Infinitesimal game analysis (tiny, miny, infinitesimals)
- Transfinite Hackenbush (ordinal-valued positions)
- Non-deterministic Hackenbush (dice, cards, hidden information)
- Partizan game complexity (computational hardness proofs)

**AI & Machine Learning:**
- Deep reinforcement learning (AlphaGo-style neural networks for Hackenbush)
- Symbolic regression (discover new game theory theorems from data)
- Generative models (create interesting Hackenbush positions for puzzles)
- Neural surreal number arithmetic (learn surreal operations end-to-end)

**Formal Verification:**
- Coq/Lean formalization of Conway's construction
- Certified game solver (verified correctness of value computation)
- Proof-producing game analysis (generate human-readable proofs of optimality)

**Educational Technology:**
- Interactive textbook (Pluto.jl + Hackenbush animations)
- Gamified learning platform (earn surreal number badges)
- Research collaboration tool (shared game database for CGT community)

### Ecosystem Integration

- **Symbolics.jl:** Symbolic manipulation of surreal number expressions
- **JuMP.jl:** Optimization-based game solving (LP formulation of Hackenbush)
- **Graphs.jl:** Leverage advanced graph algorithms (matchings, flows, cuts)
- **DataFrames.jl:** Tabulate game databases with rich metadata

### Ambitious Features

- **Combinatorial game theory foundation model** - Pre-trained on all known solved games
- **Automated theorem discovery** - AI that proposes and proves new CGT conjectures
- **Virtual CGT conference** - Online platform for sharing positions, proofs, and puzzles
- **Hackenbush Olympics** - Annual tournament with cash prizes for best players/solvers

---

## Future Horizons (v2.0+)

### Surreal Economic & Financial Models
- [ ] **Game-Theoretic Asset Pricing**: Apply surreal number theory to model complex financial derivatives where payoffs are better represented as game positions than real numbers.
- [ ] **Incentive Alignment Verification**: Use Hackenbush game values to formally verify that a mechanism design (e.g., a multi-agent auction) is "Winning" for all participants.

### Biological & Metabolic Hackenbush
- [ ] **Metabolic Game Theory**: Model metabolic pathways as Hackenbush graphs where enzyme reactions are "cuts" and steady-state fluxes are "Winning Strategies" for cellular survival.
- [ ] **Gene Regulatory Games**: Represent gene activation/inhibition as a partizan game to understand the robustness of biological switch networks.

### Combinatorial Logic & Computing
- [ ] **Surreal Logic Gates**: Implement standard computing primitives (AND, OR, NOT) where the operands are Hackenbush game positions, enabling a new form of "Game-Based Computation".
- [ ] **Surreal Algebra System**: Full symbolic system for manipulating surreal numbers, including multiplication and division (which are notoriously difficult to implement for games).

### AI Reasoning Benchmarks
- [ ] **Hackenbush AI Arena**: A specialized benchmark for testing the logical and mathematical reasoning of Large Language Models (LLMs) through interactive Hackenbush play.
- [ ] **Neuro-Symbolic Surrealism**: Train neural networks to "Intuiter" the game value of complex Hackenbush positions without exhaustive search.

---

## Migration Path

**v1.0 → v1.1:** Backward compatible (visualization and AI features are additive)
**v1.1 → v1.2:** Mostly compatible (loopy games may require new data structures)
**v1.2 → v1.3+:** Breaking changes likely (transfinite/quantum variants need fundamental redesign)

## Community Goals

- **Adoption by CGT researchers** (Siegel, Albert, Nowakowski) by v1.2
- **Publication in Integers or similar** (CGT journal) by v1.2
- **Workshop at Combinatorial Game Theory Colloquium** by v1.2
- **1000 solved positions** in public database by v1.2
