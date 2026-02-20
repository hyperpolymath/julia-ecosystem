# SPDX-License-Identifier: PMPL-1.0-or-later

"""
Plot recipes for Cliodynamics.jl outputs.

Loaded automatically when `using RecipesBase` or `using Plots` is called
alongside Cliodynamics. Provides convenient plotting for:

- Political Stress Indicator (PSI) results
- Elite Overproduction Index (EOI) results
- Secular cycle analysis (trend + cycle decomposition)
- Cycle phase detection (phase-colored timeline)
- Conflict intensity timelines
"""
module CliodynamicsPlotsExt

using RecipesBase
using Cliodynamics
import DataFrames: DataFrame, hasproperty

# ============================================================================
# PSI Plot Recipe
# ============================================================================

"""
Plot Political Stress Indicator results with component breakdown.

Usage: `plot(psi_result, :psi)`
"""
@recipe function f(df::DataFrame, ::Val{:psi})
    hasproperty(df, :psi) || error("DataFrame must contain :psi column (from political_stress_indicator)")

    title --> "Political Stress Indicator"
    xlabel --> "Year"
    ylabel --> "Stress Index"
    legend --> :topleft
    linewidth --> 2

    @series begin
        label := "PSI (composite)"
        linecolor := :red
        linewidth := 3
        df.year, df.psi
    end

    if hasproperty(df, :mmp)
        @series begin
            label := "Mass Mobilization (MMP)"
            linestyle := :dash
            linecolor := :blue
            df.year, df.mmp
        end
    end

    if hasproperty(df, :emp)
        @series begin
            label := "Elite Mobilization (EMP)"
            linestyle := :dash
            linecolor := :orange
            df.year, df.emp
        end
    end

    if hasproperty(df, :sfd)
        @series begin
            label := "State Fiscal Distress (SFD)"
            linestyle := :dash
            linecolor := :green
            df.year, df.sfd
        end
    end
end

# ============================================================================
# EOI Plot Recipe
# ============================================================================

"""
Plot Elite Overproduction Index results.

Usage: `plot(eoi_result, :eoi)`
"""
@recipe function f(df::DataFrame, ::Val{:eoi})
    hasproperty(df, :eoi) || error("DataFrame must contain :eoi column (from elite_overproduction_index)")

    title --> "Elite Overproduction Index"
    xlabel --> "Year"
    ylabel --> "EOI"
    legend --> :topleft
    linewidth --> 2
    linecolor --> :purple

    # Zero line for baseline
    @series begin
        label := "Baseline"
        linestyle := :dot
        linecolor := :gray
        df.year, zeros(length(df.year))
    end

    @series begin
        label := "EOI"
        fillalpha := 0.3
        fillrange := 0
        fillcolor := :purple
        df.year, df.eoi
    end
end

# ============================================================================
# Secular Cycle Plot Recipe
# ============================================================================

"""
Plot secular cycle analysis results (trend-cycle decomposition).

Usage: `plot(analysis_result, :secular_cycle)`
where `analysis_result` is the NamedTuple from `secular_cycle_analysis`.
"""
@recipe function f(result::NamedTuple, ::Val{:secular_cycle})
    haskey(result, :trend) || error("NamedTuple must contain :trend (from secular_cycle_analysis)")

    layout --> (2, 1)
    title --> ["Trend Component" "Cycle Component"]
    xlabel --> "Time"
    linewidth --> 2

    n = length(result.trend)

    @series begin
        subplot := 1
        label := "Trend"
        linecolor := :blue
        1:n, result.trend
    end

    @series begin
        subplot := 2
        label := "Cycle (period ≈ $(result.period))"
        linecolor := :red
        fillalpha := 0.2
        fillrange := 0
        fillcolor := :red
        1:n, result.cycle
    end
end

# ============================================================================
# Phase Detection Plot Recipe
# ============================================================================

"""
Plot secular cycle phases as colored regions.

Usage: `plot(phases_df, :phases)`
"""
@recipe function f(df::DataFrame, ::Val{:phases})
    hasproperty(df, :phase) || error("DataFrame must contain :phase column (from detect_cycle_phases)")

    title --> "Secular Cycle Phases"
    xlabel --> "Year"
    ylabel --> "Phase"
    legend --> :topright
    seriestype --> :scatter
    markersize --> 3

    phase_colors = Dict(
        Cliodynamics.Expansion => :green,
        Cliodynamics.Stagflation => :yellow,
        Cliodynamics.Crisis => :red,
        Cliodynamics.Depression => :blue
    )

    phase_nums = Dict(
        Cliodynamics.Expansion => 1,
        Cliodynamics.Stagflation => 2,
        Cliodynamics.Crisis => 3,
        Cliodynamics.Depression => 4
    )

    for phase in [Cliodynamics.Expansion, Cliodynamics.Stagflation,
                  Cliodynamics.Crisis, Cliodynamics.Depression]
        mask = df.phase .== phase
        any(mask) || continue
        @series begin
            label := string(phase)
            markercolor := phase_colors[phase]
            df.year[mask], [phase_nums[phase] for _ in 1:count(mask)]
        end
    end
end

# ============================================================================
# Conflict Intensity Plot Recipe
# ============================================================================

"""
Plot conflict intensity timeline.

Usage: `plot(intensity_df, :conflict)`
"""
@recipe function f(df::DataFrame, ::Val{:conflict})
    hasproperty(df, :intensity) || error("DataFrame must contain :intensity column (from conflict_intensity)")

    title --> "Conflict Intensity"
    xlabel --> "Year"
    ylabel --> "Intensity"
    linewidth --> 2
    linecolor --> :darkred
    fillalpha --> 0.4
    fillrange --> 0
    fillcolor --> :red
    label --> "Conflict Intensity"

    df.year, df.intensity
end

end # module CliodynamicsPlotsExt
