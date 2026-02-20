# API Reference

## Types

```@docs
SMTLib.SMTSolver
SMTLib.SMTResult
SMTLib.SMTContext
```

## Solver Discovery

```@docs
SMTLib.available_solvers
SMTLib.find_solver
```

## Context Management

```@docs
SMTLib.SMTContext
SMTLib.declare
SMTLib.assert!
SMTLib.check_sat
SMTLib.get_model
SMTLib.reset!
```

## Incremental Solving

```@docs
Base.push!(::SMTLib.SMTContext)
Base.pop!(::SMTLib.SMTContext)
```

## Conversion

```@docs
SMTLib.to_smtlib
SMTLib.from_smtlib
```

## Macros

```@docs
SMTLib.@smt
```

## Constants

```@docs
SMTLib.LOGICS
```
