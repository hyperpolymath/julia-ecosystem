# SPDX-License-Identifier: PMPL-1.0-or-later
# Copyright (c) 2026 Jonathan D.A. Jewell <jonathan.jewell@open.ac.uk>

"""
Pedagogical examples of zero-probability paradoxes.

These functions demonstrate counterintuitive properties of zero-probability events
for teaching and exploration.
"""

"""
    continuum_paradox(dist::Distribution, examples::Int=5) -> Dict

Demonstrate the continuum paradox: the unit interval (or any continuous distribution)
can be decomposed as a union of disjoint zero-probability points, yet P(Ω) = 1, not 0.

# Returns
A dictionary with:
- `:points` - Sample points (each has P = 0)
- `:individual_probs` - All zeros
- `:union_prob` - 1.0 (the whole space)
- `:explanation` - Why this isn't a contradiction

# Examples

```julia
result = continuum_paradox(Normal(0, 1), 5)
@assert all(result[:individual_probs] .== 0.0)
@assert result[:union_prob] == 1.0
println(result[:explanation])
```
"""
function continuum_paradox(dist::Distribution, examples::Int=5)
    # Sample some points from the distribution
    points = rand(dist, examples)

    # Each point has zero probability
    individual_probs = [probability(ContinuousZeroProbEvent(dist, p)) for p in points]

    # But the whole space has probability 1
    union_prob = 1.0

    explanation = """
    The Continuum Paradox:

    Each individual point has P(X = x) = 0.
    Yet the entire space (union of ALL such points) has P(Ω) = 1.

    Resolution:
    - Probability is σ-additive (countably additive), not fully additive
    - You cannot sum uncountably many numbers (even zeros) and expect 0
    - The sum of uncountably many zeros is not defined
    - Measure theory resolves this: Lebesgue measure of a point is 0,
      but Lebesgue measure of [a,b] is b-a

    Key Insight:
    Zero-probability ≠ impossibility in continuous spaces.
    """

    return Dict(
        :points => points,
        :individual_probs => individual_probs,
        :union_prob => union_prob,
        :explanation => explanation
    )
end

"""
    borel_kolmogorov_paradox() -> Dict

Demonstrate the Borel-Kolmogorov paradox: conditioning on a zero-probability
event can lead to ambiguous results depending on how you approach the limit.

The classic example: Given (X, Y) ~ Uniform(unit circle), what is the conditional
distribution of X given Y = 0?

# Returns
A dictionary explaining the paradox and showing different limits give different answers.

# Examples

```julia
result = borel_kolmogorov_paradox()
println(result[:explanation])
println("Approach 1 (horizontal): ", result[:approach_1])
println("Approach 2 (radial): ", result[:approach_2])
```
"""
function borel_kolmogorov_paradox()
    explanation = """
    Borel-Kolmogorov Paradox:

    Setup: (X, Y) uniformly distributed on the unit circle.
    Question: What is P(X | Y = 0)?

    The problem: Y = 0 is a zero-probability event!
    Conditioning on P = 0 events is ambiguous.

    Approach 1 (Horizontal limit):
    Condition on Y ∈ [-ε, ε] and take ε → 0.
    Result: X is uniform on the intersection (two points on circle).

    Approach 2 (Radial limit):
    Condition on angle θ ∈ [-ε, ε] and take ε → 0.
    Result: X is concentrated at (1, 0) and (-1, 0) differently.

    Different limits → different conditional distributions!

    Resolution:
    You must specify HOW you condition (which σ-algebra).
    There's no unique "right" conditional probability on zero-probability events.

    Key Insight:
    P(A | B) when P(B) = 0 requires extra structure beyond probability alone.
    """

    return Dict(
        :explanation => explanation,
        :approach_1 => "Horizontal limit: X ~ depends on limiting process",
        :approach_2 => "Radial limit: X ~ depends on limiting process",
        :resolution => "Must specify the limiting σ-algebra"
    )
end

"""
    rational_points_paradox(interval=(0.0, 1.0), samples::Int=1000) -> Dict

Demonstrate that rational numbers in [0,1] have zero probability yet are infinite.

# Examples

```julia
result = rational_points_paradox((0.0, 1.0), 1000)
println(result[:explanation])
@assert result[:prob_rational] == 0.0
@assert result[:count_rationals] == Inf
```
"""
function rational_points_paradox(interval=(0.0, 1.0), samples::Int=1000)
    a, b = interval

    # Sample from uniform distribution
    dist = Uniform(a, b)
    samples_drawn = rand(dist, samples)

    # Check how many are rational (in practice, none due to Float64 representation)
    # But mathematically, rationals are dense in reals
    count_rational = 0  # In practice, floating point numbers aren't truly rational

    explanation = """
    Rational Points Paradox:

    Facts:
    1. Rational numbers ℚ are countably infinite
    2. Rational numbers are dense in ℝ (between any two reals, there's a rational)
    3. Yet P(X ∈ ℚ) = 0 for X ~ Uniform[0,1]

    Why?
    - Probability depends on measure, not cardinality
    - Countable sets have Lebesgue measure zero
    - ℚ is countable, so μ(ℚ ∩ [0,1]) = 0

    Intuition:
    - If you pick a random real in [0,1], you'll "almost surely" get an irrational
    - The rationals are "too sparse" to have positive probability
    - Yet they're everywhere (dense)!

    Key Insight:
    Dense ≠ positive measure. Cardinality ≠ probability.
    """

    return Dict(
        :samples => samples_drawn,
        :count_sampled_rationals => count_rational,
        :prob_rational => 0.0,  # Theoretical probability
        :count_rationals => Inf,  # But infinitely many of them!
        :are_rationals_dense => true,
        :explanation => explanation
    )
end

"""
    uncountable_union_paradox(dist::Distribution, num_points::Int=10) -> Dict

Show that the union of uncountably many zero-probability events can have probability 1.

# Examples

```julia
result = uncountable_union_paradox(Normal(0, 1), 10)
println(result[:explanation])
```
"""
function uncountable_union_paradox(dist::Distribution, num_points::Int=10)
    # Sample some points
    points = rand(dist, num_points)

    # Each has zero probability
    point_probs = zeros(num_points)

    explanation = """
    Uncountable Union Paradox:

    Consider the sample space Ω of a continuous distribution.
    - Ω = ⋃{x} for all x in the support (uncountable union)
    - Each individual point {x} has P({x}) = 0
    - Yet P(Ω) = 1

    Why doesn't 0 + 0 + 0 + ... = 0?

    Resolution:
    - You cannot sum uncountably many numbers
    - Probability is only σ-additive (countably additive)
    - For uncountable unions, you need measure theory, not arithmetic

    Countable additivity:
    If A₁, A₂, ... are disjoint and countable:
    P(⋃ Aᵢ) = Σ P(Aᵢ)

    Uncountable unions:
    This rule does NOT extend to uncountable unions!

    Key Insight:
    The sum of uncountably many zeros is NOT zero.
    Countable vs. uncountable is a fundamental divide.
    """

    return Dict(
        :sample_points => points,
        :individual_probs => point_probs,
        :union_prob => 1.0,
        :countable_additivity => true,
        :uncountable_additivity => false,
        :explanation => explanation
    )
end

"""
    almost_sure_vs_sure() -> String

Explain the crucial distinction between "almost sure" and "sure" events.

# Examples

```julia
println(almost_sure_vs_sure())
```
"""
function almost_sure_vs_sure()
    return """
    Almost Surely vs. Surely: The Critical Distinction

    SURE EVENT:
    - Holds for ALL sample points, without exception
    - P(E) = 1 AND E contains the entire sample space
    - Example: "A real number is either rational or irrational" (sure)

    ALMOST SURE (a.s.) EVENT:
    - Holds except possibly on a zero-probability set
    - P(E) = 1 BUT there exist exceptions (just with P = 0)
    - Example: "A random real in [0,1] is irrational" (almost sure, not sure)

    Key Examples:

    1. X ~ Uniform[0,1]
       - "X ∈ [0,1]" is SURE (no exceptions)
       - "X is irrational" is ALMOST SURE (rationals are exceptions, but P(ℚ) = 0)

    2. Continuous random walks
       - "Walk visits every point" is NOT sure (won't hit specific points)
       - "Walk gets arbitrarily close to every point" is ALMOST SURE

    3. Strong Law of Large Numbers
       - Sample mean converges to expectation ALMOST SURELY
       - Not SURELY (pathological sequences exist, but have P = 0)

    Why It Matters:

    In practice, "almost surely" and "surely" are often treated the same
    (zero-probability exceptions don't occur in finite samples).

    But theoretically, the distinction is crucial:
    - Formal proofs require careful handling
    - Conditioning on zero-probability events is problematic
    - Foundations of probability theory depend on this

    Rule of Thumb:
    If you can say "P = 1", you have "almost surely".
    If you can say "no exceptions possible", you have "surely".
    """
end

# ============================================================================
# Extended Paradoxes and Demonstrations
# ============================================================================

"""
    construct_cantor_set(iterations::Int) -> Vector{Tuple{Float64, Float64}}

Construct the Cantor set after N iterations of middle-third removal.

The Cantor set is constructed by starting with the interval [0, 1] and repeatedly
removing the open middle third of each remaining interval:

- Iteration 0: [0, 1]
- Iteration 1: [0, 1/3] union [2/3, 1]
- Iteration 2: [0, 1/9] union [2/9, 1/3] union [2/3, 7/9] union [8/9, 1]
- ...

The limiting Cantor set C has remarkable properties:
- It is uncountable (same cardinality as [0, 1])
- It has Lebesgue measure zero (total length = 0)
- It is a perfect set (closed, every point is a limit point)
- It is nowhere dense (contains no intervals)
- It is self-similar with Hausdorff dimension log(2)/log(3) ≈ 0.6309
- It is totally disconnected

# Arguments
- `iterations::Int`: Number of middle-third removal steps to perform (must be >= 0)

# Returns
A `Vector{Tuple{Float64, Float64}}` of intervals `[(a1, b1), (a2, b2), ...]`
representing the remaining intervals after `iterations` steps.
After n iterations, there are 2^n intervals, each of length (1/3)^n.

# Examples

```julia
# After 0 iterations: just [0, 1]
intervals = construct_cantor_set(0)
@assert length(intervals) == 1
@assert intervals[1] == (0.0, 1.0)

# After 1 iteration: [0, 1/3] and [2/3, 1]
intervals = construct_cantor_set(1)
@assert length(intervals) == 2

# After 3 iterations: 8 intervals
intervals = construct_cantor_set(3)
@assert length(intervals) == 8

# Total length decreases as (2/3)^n
total_length = sum(b - a for (a, b) in intervals)
@assert total_length ≈ (2/3)^3
```
"""
function construct_cantor_set(iterations::Int)
    @assert iterations >= 0 "Number of iterations must be non-negative"

    # Start with the unit interval
    intervals = [(0.0, 1.0)]

    for _ in 1:iterations
        new_intervals = Tuple{Float64, Float64}[]
        for (a, b) in intervals
            third = (b - a) / 3.0
            # Keep the left third [a, a + third] and right third [a + 2*third, b]
            push!(new_intervals, (a, a + third))
            push!(new_intervals, (a + 2.0 * third, b))
        end
        intervals = new_intervals
    end

    return intervals
end

"""
    banach_tarski_paradox() -> Dict

Explain the Banach-Tarski paradox: a solid ball can be decomposed into finitely many
pieces and reassembled into two solid balls of the same size.

This paradox demonstrates the bizarre consequences of the Axiom of Choice in
measure theory. It shows that not all sets of points can be assigned a meaningful
"volume" (Lebesgue measure), and that geometric intuition breaks down when
non-measurable sets are involved.

# Returns
A `Dict` with the following keys:
- `:explanation` - Pedagogical description of the paradox
- `:num_pieces` - Minimum number of pieces needed (5 for 3D, using free group on 2 generators)
- `:key_ingredients` - List of mathematical prerequisites
- `:implications` - What the paradox tells us about measure theory
- `:resolution` - Why this doesn't violate physics

# Examples

```julia
result = banach_tarski_paradox()
println(result[:explanation])
@assert result[:num_pieces] == 5
```
"""
function banach_tarski_paradox()
    explanation = """
    Banach-Tarski Paradox (1924):

    Statement:
    A solid ball in 3D space can be decomposed into a finite number of
    non-overlapping pieces, which can then be reassembled (using only
    rotations and translations) into TWO solid balls, each the same
    size as the original.

    Construction (sketch):
    1. Use the free group F_2 on two generators (two rotations) to partition
       the sphere into orbits
    2. Apply the Axiom of Choice to select one point from each orbit
    3. This creates 5 non-measurable pieces of the ball
    4. Rotate and translate these pieces to form two complete balls

    Why 5 pieces?
    The free group F_2 = <a, b> can be decomposed as:
    F_2 = {e} union W(a) union W(a^-1) union W(b) union W(b^-1)
    where W(x) = words starting with x.
    Key identity: W(a) union a*W(a^-1) = F_2 (paradoxical decomposition)

    Why it works:
    - The pieces are NON-MEASURABLE (they have no well-defined volume)
    - Lebesgue measure cannot be extended to ALL subsets of R^3
    - The Axiom of Choice allows constructing sets with no definable volume

    Why it doesn't violate physics:
    - Physical matter is made of atoms (discrete, countable)
    - The pieces are infinitely complex and physically unrealisable
    - It's a statement about mathematical sets, not physical objects
    """

    return Dict(
        :explanation => explanation,
        :num_pieces => 5,
        :key_ingredients => [
            "Axiom of Choice",
            "Free group on 2 generators (SO(3) contains F_2)",
            "Non-measurable sets",
            "Paradoxical decomposition of groups"
        ],
        :implications => [
            "Not all subsets of R^3 are Lebesgue measurable",
            "Finitely additive, isometry-invariant measures extending Lebesgue measure don't exist on all subsets of R^3",
            "The Axiom of Choice has counterintuitive consequences"
        ],
        :resolution => "The pieces are non-measurable sets that cannot be physically constructed."
    )
end

"""
    vitali_set_paradox() -> Dict

Explain the Vitali set paradox: non-measurable subsets of the real line.

The Vitali set construction shows that, assuming the Axiom of Choice, there
exist subsets of [0, 1] that cannot be assigned a Lebesgue measure in a
consistent way. This is the first historical example of a non-measurable set
(1905) and motivates the entire theory of measurable sets and sigma-algebras.

# Returns
A `Dict` with the following keys:
- `:explanation` - Pedagogical description of the construction and paradox
- `:construction_steps` - Step-by-step construction of the Vitali set
- `:contradiction` - Why measuring the Vitali set leads to a contradiction
- `:resolution` - How modern measure theory resolves this

# Examples

```julia
result = vitali_set_paradox()
println(result[:explanation])
```
"""
function vitali_set_paradox()
    explanation = """
    Vitali Set Paradox (1905):

    Question: Can every subset of [0, 1] be assigned a "length" (Lebesgue measure)?
    Answer: NO! (Assuming the Axiom of Choice)

    The Vitali set V is a subset of [0, 1] that has no consistent Lebesgue measure.

    Construction:
    1. Define equivalence: x ~ y iff x - y is rational
    2. This partitions [0, 1] into uncountably many equivalence classes
    3. Use the Axiom of Choice to pick exactly one representative from each class
    4. Call this set V (the Vitali set)

    Why V is non-measurable:
    Let {r_n} be an enumeration of rationals in [-1, 1].
    Define V_n = V + r_n (mod 1) = shift V by r_n.

    Key facts:
    - The V_n are pairwise disjoint (by construction)
    - Their union is all of [0, 1] (every x is equivalent to some v in V)
    - By translation invariance: mu(V_n) = mu(V) for all n

    Contradiction:
    - If mu(V) = 0, then mu([0,1]) = sum of mu(V_n) = 0. But mu([0,1]) = 1!
    - If mu(V) > 0, then mu([0,1]) = sum of mu(V_n) = infinity. But mu([0,1]) = 1!
    - Therefore mu(V) cannot be defined consistently.

    Key Insight:
    Not all subsets of R are Lebesgue measurable. This is why we need sigma-algebras
    to define which sets we can meaningfully measure.
    """

    return Dict(
        :explanation => explanation,
        :construction_steps => [
            "Define equivalence relation: x ~ y iff x - y is rational",
            "Partition [0, 1] into equivalence classes (each class is dense in [0,1])",
            "Apply Axiom of Choice: select one representative from each class",
            "The collection of representatives is the Vitali set V"
        ],
        :contradiction => "If mu(V) = 0, then [0,1] has measure 0. If mu(V) > 0, then [0,1] has infinite measure. Neither is correct.",
        :resolution => "Restrict attention to Lebesgue-measurable sets (a sigma-algebra). The Vitali set is excluded."
    )
end

"""
    gabriels_horn_paradox() -> Dict

Demonstrate Gabriel's Horn (Torricelli's Trumpet): a shape with finite volume
but infinite surface area.

Gabriel's Horn is the solid of revolution obtained by rotating the curve y = 1/x
(for x >= 1) around the x-axis. It demonstrates that:
- The volume integral converges: V = pi (finite)
- The surface area integral diverges: A = infinity

This connects to zero-probability thinking: the "probability" of uniformly
painting the horn (covering all surface area) is in some sense zero, yet the
horn can be "filled" with a finite amount of paint (volume).

# Returns
A `Dict` with the following keys:
- `:explanation` - Pedagogical description of the paradox
- `:volume` - The exact volume (pi)
- `:surface_area` - The surface area (Inf)
- `:volume_formula` - The integral formula for volume
- `:surface_formula` - The integral formula for surface area
- `:paint_paradox` - The classic "paint the inside" thought experiment

# Examples

```julia
result = gabriels_horn_paradox()
@assert result[:volume] ≈ pi
@assert result[:surface_area] == Inf
println(result[:explanation])
```
"""
function gabriels_horn_paradox()
    explanation = """
    Gabriel's Horn (Torricelli's Trumpet):

    Construction:
    Rotate the curve y = 1/x for x >= 1 around the x-axis.
    This creates an infinitely long trumpet/horn shape.

    Volume (disk method):
    V = pi * integral_1^inf (1/x)^2 dx
      = pi * integral_1^inf 1/x^2 dx
      = pi * [-1/x]_1^inf
      = pi * (0 - (-1))
      = pi
    FINITE! The horn holds exactly pi cubic units.

    Surface Area (surface of revolution):
    A = 2*pi * integral_1^inf (1/x) * sqrt(1 + 1/x^4) dx
      >= 2*pi * integral_1^inf (1/x) dx  (since sqrt(1 + 1/x^4) > 1)
      = 2*pi * [ln(x)]_1^inf
      = INFINITY
    INFINITE! The horn's surface area is unbounded.

    The Paint Paradox:
    - You can FILL the horn with pi cubic units of paint (finite)
    - But you can NEVER PAINT the outside (infinite surface area)
    - Yet if you fill it... the inside is painted!?

    Resolution:
    The paradox arises from conflating 3D volume with 2D surface area.
    Paint has thickness; mathematical surfaces have none.
    A finite volume of paint with infinitesimal thickness can cover
    infinite area (the thickness decreases as 1/x^2, fast enough for
    volume to converge, but the area still diverges).

    Connection to Zero-Probability:
    The horn illustrates how measure depends on dimension:
    - 3D Lebesgue measure (volume): finite
    - 2D Hausdorff measure (surface area): infinite
    Same object, different measures, radically different answers.
    """

    return Dict(
        :explanation => explanation,
        :volume => Float64(pi),  # Exact: pi
        :surface_area => Inf,
        :volume_formula => "V = pi * integral_1^inf (1/x^2) dx = pi",
        :surface_formula => "A = 2*pi * integral_1^inf (1/x) * sqrt(1 + 1/x^4) dx = inf",
        :paint_paradox => "You can fill the horn with finite paint, which covers the infinite interior surface -- but you cannot paint the exterior."
    )
end

"""
    bertrand_paradox() -> Dict

Demonstrate Bertrand's Paradox: different methods of choosing a "random" chord
of a circle give different probabilities for the same event.

Bertrand's paradox (1889) asks: what is the probability that a random chord of
a unit circle is longer than the side of an inscribed equilateral triangle
(i.e., longer than sqrt(3))?

Three equally plausible methods give three different answers, showing that
"random" is not well-defined without specifying the probability distribution.

# Returns
A `Dict` with the following keys:
- `:explanation` - Pedagogical description of the paradox
- `:method_1` - Random endpoint method (P = 1/3)
- `:method_2` - Random midpoint (radial) method (P = 1/4)
- `:method_3` - Random midpoint (area) method (P = 1/4) with correction
- `:simulated_probs` - Monte Carlo estimates for each method
- `:resolution` - How the paradox is resolved

# Examples

```julia
result = bertrand_paradox()
@assert result[:method_1][:probability] ≈ 1/3
@assert result[:method_2][:probability] ≈ 1/2
@assert result[:method_3][:probability] ≈ 1/4
```
"""
function bertrand_paradox()
    n_samples = 10000

    # Method 1: Random Endpoints
    # Choose two random points on the circle (uniform angle) and compute chord length.
    # P(chord > sqrt(3)) = 1/3
    count_1 = 0
    for _ in 1:n_samples
        theta1 = 2.0 * pi * rand()
        theta2 = 2.0 * pi * rand()
        # Chord length for unit circle: 2 * sin(|theta2 - theta1| / 2)
        chord_length = 2.0 * abs(sin((theta2 - theta1) / 2.0))
        if chord_length > sqrt(3.0)
            count_1 += 1
        end
    end
    prob_1 = count_1 / n_samples

    # Method 2: Random Radius (Midpoint on radius)
    # Choose a random radius and a random point on it (uniform distance from centre).
    # The chord perpendicular to the radius at that point has length > sqrt(3)
    # iff the midpoint distance from centre < 1/2.
    # P(chord > sqrt(3)) = 1/2
    count_2 = 0
    for _ in 1:n_samples
        # Distance of midpoint from centre, uniform on [0, 1]
        d = rand()
        # Chord length = 2 * sqrt(1 - d^2)
        chord_length = 2.0 * sqrt(1.0 - d^2)
        if chord_length > sqrt(3.0)
            count_2 += 1
        end
    end
    prob_2 = count_2 / n_samples

    # Method 3: Random Midpoint (uniform in disc)
    # Choose a random point uniformly inside the circle as the chord's midpoint.
    # The chord through this midpoint perpendicular to the radius has length > sqrt(3)
    # iff the midpoint is within distance 1/2 of centre.
    # Area of inner circle / area of unit circle = (1/2)^2 = 1/4.
    # P(chord > sqrt(3)) = 1/4
    count_3 = 0
    for _ in 1:n_samples
        # Random point in unit disc (rejection sampling)
        x, y = 2.0 * rand() - 1.0, 2.0 * rand() - 1.0
        while x^2 + y^2 > 1.0
            x, y = 2.0 * rand() - 1.0, 2.0 * rand() - 1.0
        end
        d = sqrt(x^2 + y^2)
        if d < 0.5  # Within inner circle of radius 1/2
            count_3 += 1
        end
    end
    prob_3 = count_3 / n_samples

    explanation = """
    Bertrand's Paradox (1889):

    Question: What is the probability that a random chord of a unit circle
    is longer than sqrt(3) (the side of an inscribed equilateral triangle)?

    Method 1 - Random Endpoints:
    Choose two random points uniformly on the circumference.
    The chord connecting them is longer than sqrt(3) with probability 1/3.

    Method 2 - Random Radius:
    Choose a random radius and a random point on it (uniform distance).
    The perpendicular chord through that point exceeds sqrt(3) with probability 1/2.

    Method 3 - Random Midpoint:
    Choose a random point uniformly inside the circle as the chord's midpoint.
    The chord exceeds sqrt(3) with probability 1/4.

    Three methods, three answers: 1/3, 1/2, 1/4!

    Resolution:
    "Random chord" is ambiguous. Each method defines a different probability
    measure on the space of chords. The paradox is resolved by recognising that
    you must specify WHICH probability distribution you mean.

    Connection to Zero-Probability:
    Each individual chord has zero probability (it's a single element in an
    uncountable space of chords). The paradox shows that even specifying
    "choose uniformly at random" is insufficient -- you need to say
    WHICH parameterisation is uniform.
    """

    return Dict(
        :explanation => explanation,
        :method_1 => Dict(:name => "Random Endpoints", :probability => 1.0/3.0, :simulated => prob_1),
        :method_2 => Dict(:name => "Random Radius", :probability => 1.0/2.0, :simulated => prob_2),
        :method_3 => Dict(:name => "Random Midpoint", :probability => 1.0/4.0, :simulated => prob_3),
        :simulated_probs => Dict(:method_1 => prob_1, :method_2 => prob_2, :method_3 => prob_3),
        :resolution => "The phrase 'random chord' is ambiguous; each method implies a different measure on the chord space."
    )
end

"""
    buffon_needle_problem(length::Real=1.0, spacing::Real=2.0;
                          n_samples::Int=100000) -> Dict

Demonstrate Buffon's Needle Problem: estimating pi by dropping needles on parallel lines.

In 1777, Georges-Louis Leclerc, Comte de Buffon, posed this question: If a needle
of length L is dropped at random onto a floor ruled with parallel lines spaced D
apart (where L <= D), what is the probability that the needle crosses a line?

The answer is P = 2L / (pi * D), which provides a Monte Carlo method for estimating pi.

# Arguments
- `length::Real=1.0`: Length of the needle (must be <= spacing)
- `spacing::Real=2.0`: Distance between parallel lines
- `n_samples::Int=100000`: Number of needle drops in the simulation

# Returns
A `Dict` with the following keys:
- `:explanation` - Pedagogical description of the problem
- `:theoretical_probability` - The exact probability 2L/(pi*D)
- `:simulated_probability` - Monte Carlo estimate of crossing probability
- `:pi_estimate` - Estimate of pi from the simulation: pi ≈ 2L/(D*P_simulated)
- `:needle_length` - The needle length used
- `:line_spacing` - The line spacing used
- `:n_samples` - Number of samples used

# Mathematical Background

A needle dropped at random has:
- Centre distance from nearest line: y ~ Uniform(0, D/2)
- Angle with lines: theta ~ Uniform(0, pi)

The needle crosses a line iff y <= (L/2) * sin(theta).

P(cross) = integral_0^pi integral_0^{(L/2)*sin(theta)} (2/D) dy (1/pi) dtheta
          = (2L) / (pi * D)

# Examples

```julia
result = buffon_needle_problem(1.0, 2.0, n_samples=100000)
@assert abs(result[:pi_estimate] - pi) < 0.2  # Rough estimate
println("Pi estimate: ", result[:pi_estimate])
```
"""
function buffon_needle_problem(length::Real=1.0, spacing::Real=2.0;
                                n_samples::Int=100000)
    @assert length > 0 "Needle length must be positive"
    @assert spacing > 0 "Line spacing must be positive"
    @assert length <= spacing "Needle length must not exceed line spacing (short needle case)"

    L = Float64(length)
    D = Float64(spacing)

    # Simulate needle drops
    crossings = 0
    for _ in 1:n_samples
        # Distance of needle centre from nearest line: Uniform(0, D/2)
        y = rand() * D / 2.0
        # Angle of needle with lines: Uniform(0, pi)
        theta = rand() * pi
        # Needle crosses if y <= (L/2) * sin(theta)
        if y <= (L / 2.0) * sin(theta)
            crossings += 1
        end
    end

    simulated_prob = crossings / n_samples
    theoretical_prob = 2.0 * L / (pi * D)

    # Estimate pi from the simulation
    pi_estimate = if simulated_prob > 0
        2.0 * L / (D * simulated_prob)
    else
        Inf  # No crossings (very unlikely for large n_samples)
    end

    explanation = """
    Buffon's Needle Problem (1777):

    Setup:
    - Parallel lines on a floor, spaced D = $D apart
    - A needle of length L = $L is dropped at random
    - Question: What is P(needle crosses a line)?

    Analysis:
    The needle is characterised by:
    - y: distance of its centre from the nearest line (Uniform on [0, D/2])
    - theta: angle with the lines (Uniform on [0, pi])

    The needle crosses a line iff y <= (L/2) * sin(theta).

    P(cross) = (2/(D*pi)) * integral_0^pi (L/2)*sin(theta) dtheta
             = (2L)/(D*pi) * [-cos(theta)]_0^pi
             = (2L)/(D*pi) * 2
             = 2L / (pi * D)

    For L=$L, D=$D: P(cross) = $(round(theoretical_prob, digits=6))

    Estimating Pi:
    Since P = 2L/(pi*D), we can rearrange: pi = 2L/(D*P).
    By estimating P via Monte Carlo (drop many needles), we estimate pi!

    Simulation result ($n_samples drops):
    - Simulated P(cross) = $(round(simulated_prob, digits=6))
    - Pi estimate = $(round(pi_estimate, digits=6))
    - True pi     = $(round(Float64(pi), digits=6))

    Connection to Zero-Probability:
    Each individual needle position (y, theta) has zero probability in the
    continuous sample space. Yet the event "needle crosses a line" has a
    well-defined positive probability obtained by integrating over the
    uncountably many zero-probability positions that satisfy the crossing condition.
    """

    return Dict(
        :explanation => explanation,
        :theoretical_probability => theoretical_prob,
        :simulated_probability => simulated_prob,
        :pi_estimate => pi_estimate,
        :needle_length => L,
        :line_spacing => D,
        :n_samples => n_samples
    )
end
