# Causals.jl Development Roadmap

## Current State (v0.2 Alpha)

In-development implementation of 7 causal inference modules:
- **DempsterShafer**: Evidence combination (mostly complete)
- **BradfordHill**: Causal criteria assessment (complete)
- **CausalDAG**: Directed acyclic graphs (d-separation, backdoor, frontdoor criteria working)
- **Granger**: Time series causality (complete with proper F-distribution)
- **PropensityScore**: Observational study methods (propensity scores, matching, IPW, stratification, doubly robust all working)
- **DoCalculus**: Interventional queries (do-intervention, effect identification, confounding adjustment, do-calculus rules implemented)
- **Counterfactuals**: "What if" reasoning (counterfactual function with structural equations working)

**Status:** Alpha release with 105 passing tests, working examples, comprehensive documentation. All core algorithms implemented and verified.

---

## v1.0 → v1.2 Roadmap (Near-term)

### v1.1 - Performance & Usability (3-6 months)

**MUST:**
- [ ] **Performance benchmarking suite** - Baseline all 7 methods against synthetic datasets (10³, 10⁴, 10⁵ elements)
- [ ] **Memoization for Dempster-Shafer** - Cache intermediate combination results to avoid recomputation
- [ ] **Sparse matrix support in Causal DAGs** - Use SparseArrays.jl for large conditional probability tables
- [ ] **Progress indicators** - Add @showprogress for long-running computations (>1s expected runtime)
- [ ] **Input validation helpers** - `validate_mass_function()`, `validate_cpt()` convenience functions

**SHOULD:**
- [ ] **Parallel belief propagation** - Use Threads.@threads for independent message passing in large graphs
- [ ] **JSON export/import** - Serialize causal models to JSON for interoperability
- [ ] **Visualization integration** - GraphMakie.jl support for rendering Bayesian networks
- [ ] **Uncertainty quantification** - Add confidence intervals to all inference outputs

**COULD:**
- [ ] **Interactive tutorial notebook** - Pluto.jl walkthrough of all 7 methods with live examples
- [ ] **Domain-specific presets** - Medical diagnosis, fault detection, risk assessment templates
- [ ] **Model comparison metrics** - AIC/BIC for Bayesian networks, conflict metrics for Dempster-Shafer

### v1.2 - Advanced Methods & Integration (6-12 months)

**MUST:**
- [ ] **Causal discovery algorithms** - PC algorithm, Fast Causal Inference for structure learning
- [ ] **Intervention modeling** - do-calculus support for causal effect estimation
- [ ] **Missing data handling** - EM algorithm for incomplete observations in Bayesian networks
- [ ] **Model validation suite** - Cross-validation, holdout testing, bootstrap confidence intervals

**SHOULD:**
- [ ] **Temporal causal models** - Dynamic Bayesian networks for time-series causality
- [ ] **Counterfactual reasoning** - Pearl's structural causal model framework
- [ ] **Sensitivity analysis** - Robustness testing for prior distributions and model parameters
- [ ] **Integration with BowtieRisk.jl** - Bidirectional causal pathway analysis

**COULD:**
- [ ] **GPU acceleration** - CUDA.jl support for matrix operations in large networks
- [ ] **Federated learning** - Privacy-preserving causal inference across distributed datasets
- [ ] **AutoML for structure learning** - Hyperparameter tuning for causal discovery algorithms

---

## v1.3+ Roadmap (Speculative)

### Research Frontiers

**Causal AI & Machine Learning:**
- Neural causal models (integration with Flux.jl/Lux.jl)
- Causal representation learning (disentangled representations)
- Causal reinforcement learning (counterfactual policy evaluation)
- Large-scale causal inference (billions of variables, distributed computing)

**Quantum Causal Models:**
- Quantum Bayesian networks
- Causal indefiniteness (no fixed causal order)
- Quantum interventions and counterfactuals

**Formal Verification:**
- Proof export to Coq/Lean (causal reasoning proofs)
- Certified causal inference (verified correctness guarantees)
- Integration with Axiom.jl for theorem proving

**Domain Expansions:**
- Genomic causality (gene regulatory networks, GWAS)
- Climate modeling (attribution of extreme events)
- Economic causality (policy impact analysis)
- Social network dynamics (influence propagation)

### Ecosystem Integration

- **Turing.jl:** Probabilistic programming interface for Bayesian causal models
- **DifferentialEquations.jl:** Continuous-time causal dynamics
- **Graphs.jl:** Advanced graph algorithms for causal structure
- **MLJ.jl:** Causal feature selection and causal prediction

### Ambitious Features

- **Causal foundation models** - Pre-trained causal reasoning on knowledge graphs
- **Natural language causal extraction** - Parse causal claims from text
- **Interactive causal sandbox** - Visual programming for causal modeling (Makie.jl + web UI)
- **Causal explanation engine** - Generate human-readable justifications for inferences

---

## Future Horizons (v2.0+)

### Synthetic Twin Universes (STU)
- [ ] **High-Fidelity Counterfactual Simulation**: Create causal-consistent "Alternative History" simulators using `Agents.jl` to test the long-term impact of policy interventions before deployment.
- [ ] **Stochastic Causal Worlds**: Support for branching causal realities where probabilities themselves are subject to interventional shifts.

### Causal Ethics & Moral Attribution
- [ ] **Moral Responsibility Scoring**: Integrate with `Axiology.jl` to formally attribute "Blame" or "Credit" based on causal necessity and sufficiency in multi-agent systems.
- [ ] **Equitable Causal Chains**: Verify that causal paths from protected attributes to outcomes do not violate specific fairness invariants (linking to `Axiology.jl` fairness metrics).

### Neuro-Symbolic Causal Discovery
- [ ] **Visual Causal Discovery**: Use deep learning (Flux.jl/Lux.jl) to extract symbolic causal DAGs directly from raw video or sensor streams.
- [ ] **Latent Causal Search**: Discover "hidden" causal variables in high-dimensional latent spaces of Generative AI models.

### Causal Legal & Forensic Oracles
- [ ] **Automated "But-For" Briefs**: Generate human-readable forensic reports that meet legal standards for "Causation in Fact" and "Proximate Cause".
- [ ] **Blockchain Causal Audits**: Store causal intervention logs on an immutable ledger for verifiable post-incident forensic analysis.

---

## Migration Path

**v1.0 → v1.1:** Backward compatible (performance improvements only)
**v1.1 → v1.2:** Mostly compatible (new features, minor API additions)
**v1.2 → v1.3+:** Breaking changes possible (research features may require API redesign)

## Community Goals

- **10 citations** in academic papers by v1.2
- **100 GitHub stars** by v1.2
- **JuliaCon talk** submission for v1.2 release
- **Collaboration** with causal inference research groups (MIT, UCL, CMU)
