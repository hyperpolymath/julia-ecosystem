# Solver Support

SMTLib.jl auto-detects installed SMT solvers and provides a unified interface.

## Supported Solvers

### Z3 (Recommended)

**Developer:** Microsoft Research
**Website:** https://github.com/Z3Prover/z3

**Strengths:**
- Most comprehensive theory support
- Excellent documentation
- Active development
- Good performance across all logics

**Installation:**
```bash
# macOS
brew install z3

# Ubuntu/Debian
apt install z3

# Arch Linux
pacman -S z3

# From source
git clone https://github.com/Z3Prover/z3
cd z3 && python scripts/mk_make.py && cd build && make
```

### CVC5

**Developer:** Stanford, University of Iowa, others
**Website:** https://cvc5.github.io/

**Strengths:**
- Strong theory combinations
- Good for arrays and datatypes
- Formal verification focus

**Installation:**
```bash
# macOS
brew install cvc5

# Ubuntu/Debian
apt install cvc5

# From release
wget https://github.com/cvc5/cvc5/releases/latest/download/cvc5-Linux
chmod +x cvc5-Linux
sudo mv cvc5-Linux /usr/local/bin/cvc5
```

### Yices 2

**Developer:** SRI International
**Website:** https://yices.csl.sri.com/

**Strengths:**
- Very fast for linear arithmetic
- Low memory footprint
- Good for embedded/resource-constrained use

**Installation:**
```bash
# macOS
brew install yices

# Ubuntu/Debian
apt install yices2

# From source
wget https://yices.csl.sri.com/releases/2.6.4/yices-2.6.4-x86_64-pc-linux-gnu.tar.gz
tar xzf yices-2.6.4-x86_64-pc-linux-gnu.tar.gz
sudo cp yices-2.6.4/bin/yices-smt2 /usr/local/bin/
```

### MathSAT

**Developer:** FBK and University of Trento
**Website:** https://mathsat.fbk.eu/

**Strengths:**
- Optimization (MaxSMT)
- Interpolation
- UNSAT core generation

**Installation:**
```bash
# Download from website (requires registration for academic use)
wget https://mathsat.fbk.eu/download.php?file=mathsat-5.6.10-linux-x86_64.tar.gz
tar xzf mathsat-5.6.10-linux-x86_64.tar.gz
sudo cp mathsat-5.6.10-linux-x86_64/bin/mathsat /usr/local/bin/
```

## Solver Detection

SMTLib.jl searches for solvers in your `PATH`:

```julia
# List all available solvers
solvers = available_solvers()
for solver in solvers
    println("$(solver.kind): $(solver.path) (version $(solver.version))")
end

# Find a specific solver
z3 = find_solver(:z3)
if isnothing(z3)
    error("Z3 not found. Please install it.")
end
```

## Choosing a Solver

```julia
# Use a specific solver
ctx = SMTContext(solver=find_solver(:z3), logic=:QF_LIA)

# Or let SMTLib.jl choose automatically (prefers Z3)
ctx = SMTContext(logic=:QF_LIA)
```

## Solver-Specific Features

### Z3 Extensions

Z3 supports some extensions beyond SMT-LIB2:

```julia
# Set Z3-specific options
ctx = SMTContext(logic=:QF_LIA)
ctx.solver_options[:timeout] = 5000  # milliseconds
ctx.solver_options[:random_seed] = 42
```

### CVC5 Options

```julia
ctx = SMTContext(solver=find_solver(:cvc5), logic=:QF_LIA)
ctx.solver_options[:finite_model_find] = true
```

## Solver Comparison

| Feature | Z3 | CVC5 | Yices | MathSAT |
|---------|----|----|-------|---------|
| Linear arithmetic | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Nonlinear arithmetic | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ |
| Bitvectors | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| Arrays | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| Datatypes | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ |
| Quantifiers | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ |
| Performance | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Documentation | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |

## Troubleshooting

### Solver Not Found

```julia
# Check your PATH
println(ENV["PATH"])

# Manually specify solver path
solver = SMTSolver(:z3, "/opt/homebrew/bin/z3", "4.12.2")
ctx = SMTContext(solver=solver, logic=:QF_LIA)
```

### Timeout Issues

```julia
# Increase timeout
ctx = SMTContext(logic=:QF_NRA, timeout=30000)  # 30 seconds

# Or use a faster solver for your logic
ctx = SMTContext(solver=find_solver(:yices), logic=:QF_LIA)
```

### Memory Issues

```julia
# Use Yices for lower memory footprint
ctx = SMTContext(solver=find_solver(:yices), logic=:QF_LIA)

# Or limit solver memory (Z3)
ctx.solver_options[:max_memory] = 4096  # MB
```

## Contributing Solver Support

To add support for a new solver, implement:

1. Detection logic in `find_solver()`
2. SMT-LIB2 generation (usually standard)
3. Result parsing
4. Add to CI tests

See `src/SMTLib.jl` for details.
