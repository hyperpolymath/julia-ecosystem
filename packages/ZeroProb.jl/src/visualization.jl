# SPDX-License-Identifier: PMPL-1.0-or-later
# Copyright (c) 2026 Jonathan D.A. Jewell <jonathan.jewell@open.ac.uk>

"""
Visualization tools for zero-probability events.

Makes abstract concepts tangible through interactive plots.
Requires Plots.jl to be loadable; functions degrade gracefully if unavailable.
"""

const _PLOTS_LOADED = Ref(false)

function _ensure_plots()
    if !_PLOTS_LOADED[]
        try
            @eval using Plots
            _PLOTS_LOADED[] = true
        catch e
            error(
                "Plots.jl is required for visualization functions but could not be loaded. " *
                "Install it with `using Pkg; Pkg.add(\"Plots\")` and ensure its dependencies are working. " *
                "Original error: $e"
            )
        end
    end
    nothing
end

"""
    plot_zero_probability(event::ContinuousZeroProbEvent; kwargs...)

Visualize why a point has zero probability but non-zero relevance.

Shows:
- The probability density function (PDF)
- The specific zero-probability point
- The PDF value at that point (relevance measure)

# Examples

```julia
using Plots
event = ContinuousZeroProbEvent(Normal(0, 1), 0.0, :density)
plot_zero_probability(event)
```
"""
function plot_zero_probability(event::ContinuousZeroProbEvent{T};
                               xlims=nothing, title="") where T
    _ensure_plots()
    dist = event.distribution
    point = event.point

    # Determine plot range
    if xlims === nothing
        if isa(dist, Normal)
            μ, σ = params(dist)
            xlims = (μ - 4σ, μ + 4σ)
        else
            xlims = (point - 5, point + 5)
        end
    end

    # Generate x values
    xs = range(xlims[1], xlims[2], length=500)
    ys = pdf.(dist, xs)

    # Plot PDF
    p = plot(xs, ys, label="PDF", linewidth=2, legend=:topright)
    xlabel!("x")
    ylabel!("Density")

    # Highlight the zero-probability point
    point_density = pdf(dist, point)
    scatter!([point], [point_density], markersize=8, color=:red,
             label="Zero-prob point (x=$point)")

    # Add vertical line
    vline!([point], linestyle=:dash, color=:red, alpha=0.5, label="")

    # Add annotation
    annotate!(point, point_density * 1.1,
              text("P(X=$point) = 0\nDensity = $(round(point_density, digits=4))",
                   :center, 8))

    # Title
    if title == ""
        title = "Zero-Probability Event: P(X=$point) = 0 but PDF=$( round(point_density, digits=4))"
    end
    plot!(title=title)

    return p
end

"""
    plot_continuum_paradox(dist::Distribution; num_points::Int=10)

Visualize the continuum paradox: uncountably many zero-probability points
sum to probability 1.

# Examples

```julia
plot_continuum_paradox(Normal(0, 1), num_points=20)
```
"""
function plot_continuum_paradox(dist::Distribution; num_points::Int=10)
    _ensure_plots()
    # Sample points
    points = rand(dist, num_points)
    sort!(points)

    # Each has zero probability
    probs = zeros(num_points)

    # But their union (the whole space) has probability 1
    xs = range(minimum(points) - 1, maximum(points) + 1, length=500)
    cdf_vals = cdf.(dist, xs)

    p = plot(xs, cdf_vals, label="CDF (total probability)", linewidth=2,
             legend=:topleft)
    xlabel!("x")
    ylabel!("Cumulative Probability")

    # Show sample points
    scatter!(points, cdf.(dist, points), markersize=6, color=:red,
             label="Sample points (each P=0)")

    # Add annotations
    annotate!(minimum(xs), 0.9,
              text("Each point: P=0\nUnion: P=1", :left, 8))

    title!("Continuum Paradox: Uncountably Many Zeros Sum to One")

    return p
end

"""
    plot_density_vs_probability(event::ContinuousZeroProbEvent; ε_max::Float64=1.0)

Compare probability vs. density for a zero-probability event.

Shows how P(|X - x| < ε) grows as ε increases, while P(X = x) = 0.

# Examples

```julia
event = ContinuousZeroProbEvent(Normal(0, 1), 0.0, :density)
plot_density_vs_probability(event, ε_max=2.0)
```
"""
function plot_density_vs_probability(event::ContinuousZeroProbEvent{T};
                                     ε_max::Float64=1.0) where T
    _ensure_plots()
    εs = range(0.001, ε_max, length=100)
    probs = [epsilon_neighborhood(event, ε) for ε in εs]

    p = plot(εs, probs, label="P(|X - $(event.point)| < ε)", linewidth=2)
    xlabel!("ε (neighborhood size)")
    ylabel!("Probability")

    # Add horizontal line at density value
    density = density_ratio(event)
    hline!([density], linestyle=:dash, color=:red,
           label="PDF at point (relevance)")

    title!("Probability vs. Density for Zero-Probability Event")

    return p
end

"""
    plot_epsilon_neighborhood(event::ContinuousZeroProbEvent; ε::Float64=0.1)

Visualize the ε-neighborhood around a zero-probability point.

# Examples

```julia
event = ContinuousZeroProbEvent(Normal(0, 1), 0.0, :epsilon)
plot_epsilon_neighborhood(event, ε=0.5)
```
"""
function plot_epsilon_neighborhood(event::ContinuousZeroProbEvent{T};
                                   ε::Float64=0.1) where T
    _ensure_plots()
    dist = event.distribution
    point = event.point

    # Determine plot range
    if isa(dist, Normal)
        μ, σ = params(dist)
        xlims = (μ - 4σ, μ + 4σ)
    else
        xlims = (point - 5, point + 5)
    end

    xs = range(xlims[1], xlims[2], length=500)
    ys = pdf.(dist, xs)

    # Plot PDF
    p = plot(xs, ys, label="PDF", linewidth=2, legend=:topright)
    xlabel!("x")
    ylabel!("Density")

    # Highlight neighborhood
    neighborhood_xs = filter(x -> abs(x - point) < ε, xs)
    neighborhood_ys = pdf.(dist, neighborhood_xs)

    plot!(neighborhood_xs, neighborhood_ys, fillrange=0, fillalpha=0.3,
          color=:blue, label="ε-neighborhood")

    # Add vertical lines at boundaries
    vline!([point - ε, point + ε], linestyle=:dash, color=:blue,
           label="")
    vline!([point], linestyle=:dash, color=:red, label="Zero-prob point")

    # Compute neighborhood probability
    neighborhood_prob = epsilon_neighborhood(event, ε)

    title!("ε-Neighborhood: P(|X-$point| < $ε) = $(round(neighborhood_prob, digits=4))")

    return p
end

"""
    plot_black_swan_impact(event::BlackSwanEvent; samples::Int=10000)

Visualize the distribution of outcomes and the black swan threshold.

# Examples

```julia
crash = MarketCrashEvent(severity = :catastrophic)
plot_black_swan_impact(crash, samples=10000)
```
"""
function plot_black_swan_impact(event::BlackSwanEvent; samples::Int=10000)
    _ensure_plots()
    dist = event.distribution
    threshold = event.threshold

    # Generate samples
    xs_sample = rand(dist, samples)

    # Plot histogram
    p = histogram(xs_sample, bins=50, alpha=0.6, label="Sampled outcomes",
                  normalize=:probability)
    xlabel!("Outcome")
    ylabel!("Probability")

    # Add threshold line
    vline!([threshold], linewidth=3, color=:red, linestyle=:dash,
           label="Black swan threshold")

    # Count how many exceed threshold
    beyond_threshold = sum(xs_sample .<= threshold)
    pct = beyond_threshold / samples * 100

    # Annotate
    annotate!(threshold * 0.9, maximum(ylims()) * 0.9,
              text("$(beyond_threshold)/$samples = $(round(pct, digits=2))%\nexceed threshold",
                   :right, 8))

    title!("Black Swan Event: $(round(probability(event) * 100, digits=4))% probability")

    return p
end
