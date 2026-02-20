# SPDX-License-Identifier: PMPL-1.0-or-later
# Copyright (c) 2026 Jonathan D.A. Jewell <jonathan.jewell@open.ac.uk>

"""
Core value type system for ethical and economic values in ML.
"""

# Abstract base type
"""
    Value

Abstract base type for all value types in Axiology.

Values represent ethical or economic objectives that ML systems should optimize or satisfy.
"""
abstract type Value end

# Fairness metrics enum
"""
    FairnessMetric

An enumeration of supported fairness metrics. These metrics are used to quantify
and assess different aspects of fairness in machine learning models, typically
to identify and mitigate biases against protected groups.

Variants:
- `demographic_parity_metric`: Ensures that a positive outcome is equally likely
                               across different protected groups.
- `equalized_odds_metric`: Requires that false positive rates and false negative
                           rates are equal across protected groups.
- `equal_opportunity_metric`: A weaker form of equalized odds, requiring only that
                              true positive rates are equal across protected groups.
- `disparate_impact_metric`: Measures whether the selection rate for a protected
                             group is substantially less than for a non-protected group.
- `individual_fairness_metric`: Requires similar treatment for similar individuals.
"""
@enum FairnessMetric begin
    demographic_parity_metric
    equalized_odds_metric
    equal_opportunity_metric
    disparate_impact_metric
    individual_fairness_metric
end

# Welfare metrics enum
"""
    WelfareMetric

An enumeration of supported social welfare functions. These functions are used
to aggregate individual utilities or outcomes into a single measure of collective
well-being, guiding optimization towards socially desirable states.

Variants:
- `utilitarian_metric`: Aims to maximize the sum of individual utilities across
                        all individuals in a group.
- `rawlsian_metric`: Focuses on maximizing the utility of the worst-off individual
                     or group.
- `egalitarian_metric`: Strives to reduce disparities and promote equality
                        among individuals.
"""
@enum WelfareMetric begin
    utilitarian_metric
    rawlsian_metric
    egalitarian_metric
end

# Efficiency metrics enum
"""
    EfficiencyMetric

An enumeration of supported efficiency metrics. These metrics are used to
assess how effectively resources are utilized or how quickly a system
achieves its objectives, encompassing both economic and computational
aspects.

Variants:
- `pareto_metric`: Related to Pareto efficiency, where resources are
                   allocated such that no one can be made better off
                   without making at least one individual worse off.
- `kaldor_hicks_metric`: A measure of economic efficiency where an outcome
                         is considered more efficient if those who benefit
                         could hypothetically compensate those who lose.
- `computation_time_metric`: Measures the computational resources or time
                             required for a model to perform its task,
                             a critical aspect of practical efficiency.
"""
@enum EfficiencyMetric begin
    pareto_metric
    kaldor_hicks_metric
    computation_time_metric
end

"""
    Fairness <: Value

Represents fairness as a core value in machine learning models. Fairness in ML
aims to ensure that model predictions or outcomes do not unfairly discriminate
against certain demographic groups, or that individuals are treated equitably.
This struct allows for the explicit definition of a fairness objective with a
specific metric, protected attributes, and an acceptable threshold for disparity.

# Fields
- `metric::Symbol`: The specific fairness metric to be used. Valid metrics are
                    defined in the `FairnessMetric` enum (e.g., `:demographic_parity`,
                    `:equalized_odds`, `:equal_opportunity`, `:disparate_impact`,
                    `:individual_fairness`).
- `protected_attributes::Vector{Symbol}`: A list of demographic or sensitive attributes
                                        (e.g., `[:gender, :race]`) that should not
                                        unduly influence model predictions or lead to
                                        disparate impact. The specific interpretation
                                        depends on the chosen `metric`.
- `threshold::Float64`: The maximum acceptable level of disparity or bias for the
                        chosen `metric`. For example, a threshold of `0.05` for
                        demographic parity might mean that the difference in positive
                        prediction rates between groups should not exceed 5%.
                        Defaults to `0.05`.
- `weight::Float64`: A non-negative weight used in multi-objective optimization to
                     balance fairness against other competing values (e.g., profit,
                     efficiency). Defaults to `1.0`.

# Example

```julia
# Define a fairness objective based on demographic parity for gender and race
fairness_objective = Fairness(
    metric = :demographic_parity,
    protected_attributes = [:gender, :race],
    threshold = 0.05,
    weight = 0.4 # Assign a weight in a multi-objective context
)
```
"""
struct Fairness <: Value
    metric::Symbol
    protected_attributes::Vector{Symbol}
    threshold::Float64
    weight::Float64

    """
        Fairness(; metric::Symbol = :demographic_parity,
                   protected_attributes::Vector{Symbol} = Symbol[],
                   threshold::Float64 = 0.05,
                   weight::Float64 = 1.0)

    Constructs a `Fairness` value.

    Arguments:
    - `metric`: The fairness metric (see `FairnessMetric` for valid options).
                Defaults to `:demographic_parity`.
    - `protected_attributes`: A `Vector` of `Symbol`s for attributes to protect against bias.
                              Defaults to an empty `Vector`.
    - `threshold`: The maximum acceptable disparity, a `Float64` between `0.0` and `1.0`.
                   Defaults to `0.05`.
    - `weight`: The non-negative `Float64` weight for multi-objective optimization.
                Defaults to `1.0`.

    Throws:
    - `AssertionError` if `metric` is invalid, `threshold` is out of `[0,1]` range,
                      or `weight` is negative.
    """
    function Fairness(; metric::Symbol = :demographic_parity,
                        protected_attributes::Vector{Symbol} = Symbol[],
                        threshold::Float64 = 0.05,
                        weight::Float64 = 1.0)
        @assert metric in [:demographic_parity, :equalized_odds, :equal_opportunity,
                          :disparate_impact, :individual_fairness] "Invalid fairness metric: `$(metric)`. Must be one of $(instances(FairnessMetric))."
        @assert threshold >= 0.0 && threshold <= 1.0 "Threshold must be in [0,1]."
        @assert weight >= 0.0 "Weight must be non-negative."
        new(metric, protected_attributes, threshold, weight)
    end
end

"""
    Welfare <: Value

Represents social welfare functions as a core value for optimizing machine
learning outcomes. Social welfare functions provide a framework for evaluating
the overall well-being or desirability of different states or distributions
of resources in a society influenced by ML systems. This struct allows defining
a specific welfare objective based on established metrics.

# Fields
- `metric::Symbol`: The specific social welfare metric to be used. Valid metrics
                    are defined in the `WelfareMetric` enum (e.g., `:utilitarian`,
                    `:rawlsian`, `:egalitarian`).
- `weight::Float64`: A non-negative weight used in multi-objective optimization to
                     balance welfare against other competing values (e.g., fairness,
                     profit). Defaults to `1.0`.

# Example

```julia
# Define a welfare objective based on utilitarian principles
welfare_objective = Welfare(metric = :utilitarian, weight = 0.5)
```
"""
struct Welfare <: Value
    metric::Symbol
    weight::Float64

    """
        Welfare(; metric::Symbol = :utilitarian, weight::Float64 = 1.0)

    Constructs a `Welfare` value.

    Arguments:
    - `metric`: The welfare metric (see `WelfareMetric` for valid options).
                Defaults to `:utilitarian`.
    - `weight`: The non-negative `Float64` weight for multi-objective optimization.
                Defaults to `1.0`.

    Throws:
    - `AssertionError` if `metric` is invalid or `weight` is negative.
    """
    function Welfare(; metric::Symbol = :utilitarian, weight::Float64 = 1.0)
        @assert metric in [:utilitarian, :rawlsian, :egalitarian] "Invalid welfare metric: `$(metric)`. Must be one of $(instances(WelfareMetric))."
        @assert weight >= 0.0 "Weight must be non-negative."
        new(metric, weight)
    end
end

"""
    Profit <: Value

Represents economic profit optimization as a value for machine learning models.
In many business applications, maximizing profit is a primary objective.
However, Axiology.jl allows `Profit` to be considered alongside other ethical
or social values, recognizing that profit maximization often operates within
a broader set of constraints.

# Fields
- `target::Float64`: The desired or aspirational profit value that the ML
                     system aims to achieve or exceed. This can serve as
                     an optimization goal or a threshold for satisfaction.
                     Defaults to `0.0`.
- `constraints::Vector{Value}`: A list of other `Value` objectives (e.g., `Fairness`,
                               `Safety`) that must be satisfied while optimizing
                               for profit. This allows defining bounded rationality
                               or ethical profit-seeking. Defaults to an empty `Vector`.
- `weight::Float64`: A non-negative weight used in multi-objective optimization
                     to balance profit against other competing values. Defaults to `1.0`.

# Example

```julia
# Define a profit objective with a target and a fairness constraint
profit_objective = Profit(
    target = 1_000_000.0, # Aim for 1M profit
    constraints = [Fairness(metric = :demographic_parity, threshold = 0.1)],
    weight = 0.6
)
```
"""
struct Profit <: Value
    target::Float64
    constraints::Vector{Value}
    weight::Float64

    """
        Profit(; target::Float64 = 0.0,
                 constraints::Vector{Value} = Value[],
                 weight::Float64 = 1.0)

    Constructs a `Profit` value.

    Arguments:
    - `target`: The `Float64` target profit value. Defaults to `0.0`.
    - `constraints`: A `Vector` of `Value` objects representing additional
                     constraints to be considered. Defaults to an empty `Vector`.
    - `weight`: The non-negative `Float64` weight for multi-objective optimization.
                Defaults to `1.0`.

    Throws:
    - `AssertionError` if `weight` is negative.
    """
    function Profit(; target::Float64 = 0.0,
                      constraints::Vector{Value} = Value[],
                      weight::Float64 = 1.0)
        @assert weight >= 0.0 "Weight must be non-negative."
        new(target, constraints, weight)
    end
end

"""
    Efficiency <: Value

Represents efficiency metrics as a core value in machine learning systems.
Efficiency can encompass various aspects, including computational resource
utilization, speed, economic efficiency (e.g., resource allocation), or
allocative efficiency. This struct allows defining an efficiency objective
based on specific metrics and target values.

# Fields
- `metric::Symbol`: The specific efficiency metric to be used. Valid metrics
                    are defined in the `EfficiencyMetric` enum (e.g., `:pareto`,
                    `:kaldor_hicks`, `:computation_time`).
- `target::Float64`: The desired or aspirational efficiency value. The interpretation
                     of this target depends on the chosen `metric`. For example,
                     for `:computation_time`, it might represent a maximum acceptable
                     execution time in seconds. Defaults to `1.0`.
- `weight::Float64`: A non-negative weight used in multi-objective optimization to
                     balance efficiency against other competing values (e.g., fairness,
                     safety). Defaults to `1.0`.

# Example

```julia
# Define a computational efficiency objective
computational_efficiency = Efficiency(
    metric = :computation_time,
    target = 0.5, # Target 0.5 seconds for a task
    weight = 0.2
)
```
"""
struct Efficiency <: Value
    metric::Symbol
    target::Float64
    weight::Float64

    """
        Efficiency(; metric::Symbol = :pareto,
                     target::Float64 = 1.0,
                     weight::Float64 = 1.0)

    Constructs an `Efficiency` value.

    Arguments:
    - `metric`: The efficiency metric (see `EfficiencyMetric` for valid options).
                Defaults to `:pareto`.
    - `target`: The `Float64` target efficiency value. Defaults to `1.0`.
    - `weight`: The non-negative `Float64` weight for multi-objective optimization.
                Defaults to `1.0`.

    Throws:
    - `AssertionError` if `metric` is invalid or `weight` is negative.
    """
    function Efficiency(; metric::Symbol = :pareto,
                         target::Float64 = 1.0,
                         weight::Float64 = 1.0)
        @assert metric in [:pareto, :kaldor_hicks, :computation_time] "Invalid efficiency metric: `$(metric)`. Must be one of $(instances(EfficiencyMetric))."
        @assert weight >= 0.0 "Weight must be non-negative."
        new(metric, target, weight)
    end
end

"""
    Safety <: Value

Represents safety invariants and constraints for machine learning models.
Safety is a critical ethical value, ensuring that ML systems operate reliably,
robustly, and without causing undue harm to individuals or society. This
struct allows defining explicit safety properties that models must adhere to.

# Fields
- `invariant::String`: A logical formula or descriptive statement defining the
                       safety property that the ML model must maintain. This could
                       be a formal specification (e.g., "prediction(X) not in harmful_range")
                       or a high-level goal (e.g., "No harmful recommendations").
- `critical::Bool`: A flag indicating whether this is a critical safety constraint.
                    Critical constraints typically imply that any violation should
                    trigger immediate intervention or halt system operation.
                    Defaults to `true`.
- `weight::Float64`: A non-negative weight used in multi-objective optimization to
                     balance safety against other competing values. While safety
                     is often a hard constraint, `weight` can be used when safety
                     is expressed as a soft objective or when balancing
                     different safety aspects. Defaults to `1.0`.

# Example

```julia
# Define a critical safety invariant for a medical diagnosis model
patient_safety = Safety(
    invariant = "Model must not misdiagnose life-threatening conditions with > 1% false negative rate.",
    critical = true,
    weight = 100.0 # High weight due to criticality
)
```
"""
struct Safety <: Value
    invariant::String
    critical::Bool
    weight::Float64

    """
        Safety(; invariant::String = "",
                 critical::Bool = true,
                 weight::Float64 = 1.0)

    Constructs a `Safety` value.

    Arguments:
    - `invariant`: A `String` defining the safety property. Cannot be empty.
                   Defaults to `""`.
    - `critical`: A `Bool` indicating if this is a critical safety constraint.
                  Defaults to `true`.
    - `weight`: The non-negative `Float64` weight for multi-objective optimization.
                Defaults to `1.0`.

    Throws:
    - `AssertionError` if `invariant` is empty or `weight` is negative.
    """
    function Safety(; invariant::String = "",
                     critical::Bool = true,
                     weight::Float64 = 1.0)
        @assert !isempty(invariant) "Safety invariant cannot be empty."
        @assert weight >= 0.0 "Weight must be non-negative."
        new(invariant, critical, weight)
    end
end

# Display methods
"""
    Base.show(io::IO, f::Fairness)

Prints a concise string representation of a `Fairness` object to the given
I/O stream, typically for console output.
"""
Base.show(io::IO, f::Fairness) = print(io, "Fairness(:$(f.metric), threshold=$(f.threshold))")

"""
    Base.show(io::IO, w::Welfare)

Prints a concise string representation of a `Welfare` object to the given
I/O stream.
"""
Base.show(io::IO, w::Welfare) = print(io, "Welfare(:$(w.metric))")

"""
    Base.show(io::IO, p::Profit)

Prints a concise string representation of a `Profit` object to the given
I/O stream.
"""
Base.show(io::IO, p::Profit) = print(io, "Profit(target=$(p.target))")

"""
    Base.show(io::IO, e::Efficiency)

Prints a concise string representation of an `Efficiency` object to the given
I/O stream.
"""
Base.show(io::IO, e::Efficiency) = print(io, "Efficiency(:$(e.metric), target=$(e.target))")

"""
    Base.show(io::IO, s::Safety)

Prints a concise string representation of a `Safety` object to the given
I/O stream.
"""
Base.show(io::IO, s::Safety) = print(io, "Safety(\"$(s.invariant)\")")
