# SPDX-License-Identifier: PMPL-1.0-or-later
# SPDX-FileCopyrightText: 2026 Jonathan D.A. Jewell

"""
    Cliometrics

A Julia library for quantitative economic history analysis.

Cliometrics applies economic theory and quantitative methods to the study of
historical economic phenomena. This package provides tools for analyzing
historical datasets, performing econometric analysis on historical data, and
modeling long-term economic trends.

# Features
- Historical data analysis and visualization
- Time series analysis for historical economic data
- Growth accounting and decomposition
- Institutional analysis quantification
- Counterfactual historical modeling
- Historical GDP estimation and comparison

# Example
```julia
using Cliometrics

# Analyze historical GDP growth
data = load_historical_gdp("maddison_project.csv")
growth_rates = calculate_growth_rates(data)
decompose_growth(growth_rates, factors=[:capital, :labor, :tfp])
```
"""
module Cliometrics

using Statistics
using StatsBase
using DataFrames
using CSV
using Dates

export
    # Data loading and preparation
    load_historical_data,
    clean_historical_series,
    interpolate_missing_years,

    # Growth analysis
    calculate_growth_rates,
    decompose_growth,
    solow_residual,

    # Institutional analysis
    quantify_institutions,
    institutional_quality_index,

    # Comparative analysis
    compare_historical_trajectories,
    convergence_analysis,

    # Counterfactuals
    counterfactual_scenario,
    estimate_treatment_effect

"""
    load_historical_data(filepath::String; kwargs...) -> DataFrame

Load historical economic data from CSV file with proper date parsing.

# Arguments
- `filepath::String`: Path to CSV file containing historical data
- `date_format::String`: Format string for date parsing (default: "yyyy")
- `start_year::Int`: Optional filter for starting year
- `end_year::Int`: Optional filter for ending year

# Returns
- `DataFrame`: Parsed historical data with standardized column names

# Example
```julia
data = load_historical_data("historical_gdp.csv", start_year=1800, end_year=2000)
```
"""
function load_historical_data(filepath::String;
                             date_format::String="yyyy",
                             start_year::Union{Int,Nothing}=nothing,
                             end_year::Union{Int,Nothing}=nothing)
    df = CSV.read(filepath, DataFrame)

    # Ensure year column exists
    if !("year" in names(df)) && !("Year" in names(df))
        error("Data must contain a 'year' or 'Year' column")
    end

    # Standardize column name
    if "Year" in names(df)
        rename!(df, :Year => :year)
    end

    # Filter by year range if specified
    if !isnothing(start_year)
        filter!(row -> row.year >= start_year, df)
    end
    if !isnothing(end_year)
        filter!(row -> row.year <= end_year, df)
    end

    return df
end

"""
    calculate_growth_rates(data::DataFrame, variable::Symbol; method=:geometric) -> Vector{Float64}

Calculate growth rates for a historical time series.

# Arguments
- `data::DataFrame`: Historical data containing the variable
- `variable::Symbol`: Column name of the variable to analyze
- `method::Symbol`: Growth rate calculation method (`:geometric` or `:arithmetic`)

# Returns
- `Vector{Float64}`: Growth rates as decimal values (e.g., 0.03 for 3% growth)

# Example
```julia
growth = calculate_growth_rates(gdp_data, :real_gdp, method=:geometric)
```
"""
function calculate_growth_rates(data::DataFrame, variable::Symbol; method::Symbol=:geometric)
    values = data[!, variable]

    if method == :geometric
        return [log(values[i] / values[i-1]) for i in 2:length(values)]
    elseif method == :arithmetic
        return [(values[i] - values[i-1]) / values[i-1] for i in 2:length(values)]
    else
        error("Unknown method: $method. Use :geometric or :arithmetic")
    end
end

"""
    solow_residual(output::Vector, capital::Vector, labor::Vector; alpha=0.3) -> Vector{Float64}

Calculate the Solow residual (Total Factor Productivity) using growth accounting.

# Arguments
- `output::Vector`: Real output (GDP) series
- `capital::Vector`: Capital stock series
- `labor::Vector`: Labor input series
- `alpha::Float64`: Capital share of output (default: 0.3)

# Returns
- `Vector{Float64}`: TFP growth rates

# Example
```julia
tfp = solow_residual(gdp, capital_stock, labor_force, alpha=0.35)
```
"""
function solow_residual(output::Vector, capital::Vector, labor::Vector; alpha::Float64=0.3)
    # Calculate growth rates
    g_Y = [log(output[i] / output[i-1]) for i in 2:length(output)]
    g_K = [log(capital[i] / capital[i-1]) for i in 2:length(capital)]
    g_L = [log(labor[i] / labor[i-1]) for i in 2:length(labor)]

    # Solow residual: g_A = g_Y - alpha*g_K - (1-alpha)*g_L
    return g_Y .- alpha .* g_K .- (1 - alpha) .* g_L
end

"""
    decompose_growth(data::DataFrame; output=:gdp, capital=:capital, labor=:labor, alpha=0.3)

Decompose economic growth into contributions from capital, labor, and TFP.

# Arguments
- `data::DataFrame`: Historical data with output, capital, and labor series
- `output::Symbol`: Column name for output variable
- `capital::Symbol`: Column name for capital stock
- `labor::Symbol`: Column name for labor input
- `alpha::Float64`: Capital share of output

# Returns
- `DataFrame`: Decomposition with columns for each growth component

# Example
```julia
decomposition = decompose_growth(historical_data, alpha=0.35)
```
"""
function decompose_growth(data::DataFrame;
                         output::Symbol=:gdp,
                         capital::Symbol=:capital,
                         labor::Symbol=:labor,
                         alpha::Float64=0.3)
    Y = data[!, output]
    K = data[!, capital]
    L = data[!, labor]

    g_Y = calculate_growth_rates(DataFrame(; x=Y), :x)
    g_K = calculate_growth_rates(DataFrame(; x=K), :x)
    g_L = calculate_growth_rates(DataFrame(; x=L), :x)
    tfp = solow_residual(Y, K, L, alpha=alpha)

    # Contributions
    capital_contrib = alpha .* g_K
    labor_contrib = (1 - alpha) .* g_L
    tfp_contrib = tfp

    return DataFrame(
        year = data.year[2:end],
        output_growth = g_Y,
        capital_contribution = capital_contrib,
        labor_contribution = labor_contrib,
        tfp_contribution = tfp_contrib
    )
end

"""
    convergence_analysis(data::DataFrame, initial_var::Symbol, growth_var::Symbol) -> NamedTuple

Test for convergence in economic growth across regions or countries.

# Arguments
- `data::DataFrame`: Cross-sectional data with initial levels and growth rates
- `initial_var::Symbol`: Column name for initial income/GDP level
- `growth_var::Symbol`: Column name for subsequent growth rate

# Returns
- `NamedTuple`: Regression results including beta coefficient and R²

# Example
```julia
result = convergence_analysis(country_data, :gdp_1950, :growth_1950_2000)
```
"""
function convergence_analysis(data::DataFrame, initial_var::Symbol, growth_var::Symbol)
    x = log.(data[!, initial_var])
    y = data[!, growth_var]

    # Simple OLS: growth = alpha - beta*log(initial)
    n = length(x)
    x_mean = mean(x)
    y_mean = mean(y)

    beta = sum((x .- x_mean) .* (y .- y_mean)) / sum((x .- x_mean).^2)
    alpha = y_mean - beta * x_mean

    # Calculate R²
    y_pred = alpha .+ beta .* x
    ss_tot = sum((y .- y_mean).^2)
    ss_res = sum((y .- y_pred).^2)
    r_squared = 1 - (ss_res / ss_tot)

    # Beta should be negative for convergence
    is_converging = beta < 0

    return (
        alpha = alpha,
        beta = beta,
        r_squared = r_squared,
        converging = is_converging,
        half_life = is_converging ? -log(2) / beta : Inf
    )
end

"""
    institutional_quality_index(data::DataFrame, indicators::Vector{Symbol}; weights=nothing) -> Vector{Float64}

Create a composite institutional quality index from multiple indicators.

# Arguments
- `data::DataFrame`: Data containing institutional indicators
- `indicators::Vector{Symbol}`: Column names of indicators to combine
- `weights::Union{Vector{Float64},Nothing}`: Optional weights (default: equal weights)

# Returns
- `Vector{Float64}`: Composite institutional quality scores

# Example
```julia
quality = institutional_quality_index(
    institutions_data,
    [:rule_of_law, :property_rights, :contract_enforcement]
)
```
"""
function institutional_quality_index(data::DataFrame,
                                    indicators::Vector{Symbol};
                                    weights::Union{Vector{Float64},Nothing}=nothing)
    # Use equal weights if not specified
    if isnothing(weights)
        weights = ones(length(indicators)) ./ length(indicators)
    end

    # Normalize each indicator to [0, 1]
    normalized = DataFrame()
    for ind in indicators
        values = data[!, ind]
        min_val, max_val = extrema(values)
        normalized[!, ind] = (values .- min_val) ./ (max_val - min_val)
    end

    # Calculate weighted sum
    index = zeros(nrow(data))
    for (i, ind) in enumerate(indicators)
        index .+= weights[i] .* normalized[!, ind]
    end

    return index
end

"""
    quantify_institutions(data::DataFrame, entity::Symbol, indicators::Vector{Symbol};
                         period::Union{Tuple{Int,Int},Nothing}=nothing) -> DataFrame

Measure how institutional indicators change over time for given entities.

# Arguments
- `data::DataFrame`: Panel data with `:year`, entity column, and indicator columns
- `entity::Symbol`: Column name identifying entities (e.g., `:country`, `:region`)
- `indicators::Vector{Symbol}`: List of institutional indicator columns to analyze
- `period::Union{Tuple{Int,Int},Nothing}`: Optional (start_year, end_year) to filter data

# Returns
- `DataFrame`: Entity-level statistics including average change rate, volatility, and direction

# Example
```julia
df = DataFrame(
    year = repeat(2000:2004, 2),
    country = repeat(["A", "B"], inner=5),
    rule_of_law = [0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 0.78, 0.76, 0.74, 0.72]
)
result = quantify_institutions(df, :country, [:rule_of_law])
# Returns statistics on institutional change for each country
```
"""
function quantify_institutions(data::DataFrame, entity::Symbol, indicators::Vector{Symbol};
                              period::Union{Tuple{Int,Int},Nothing}=nothing)
    # Convert entity symbol to string for DataFrame operations
    entity_str = String(entity)

    if !("year" in names(data))
        error("DataFrame must have a :year column")
    end
    if !(entity_str in names(data))
        error("Entity column :$entity not found in DataFrame")
    end
    for ind in indicators
        if !(String(ind) in names(data))
            error("Indicator :$ind not found in DataFrame")
        end
    end

    # Filter by period if specified
    filtered_data = data
    if !isnothing(period)
        start_year, end_year = period
        filtered_data = data[(data.year .>= start_year) .& (data.year .<= end_year), :]
    end

    # Get unique entities
    entities = unique(filtered_data[!, entity_str])

    # Initialize result vectors
    entity_vec = similar(entities)
    avg_change_rate_vec = Float64[]
    volatility_vec = Float64[]
    direction_vec = String[]

    for (idx, ent) in enumerate(entities)
        # Filter data for this entity
        entity_data = filtered_data[filtered_data[!, entity_str] .== ent, :]

        # Sort by year
        sort!(entity_data, :year)

        # Calculate year-over-year changes for all indicators
        all_changes = Float64[]

        for ind in indicators
            values = entity_data[!, String(ind)]
            # Calculate differences between consecutive years
            for i in 2:length(values)
                if !ismissing(values[i]) && !ismissing(values[i-1])
                    change = values[i] - values[i-1]
                    push!(all_changes, change)
                end
            end
        end

        # Calculate statistics
        if length(all_changes) > 0
            avg_change = mean(all_changes)
            volatility = std(all_changes)

            # Determine direction
            direction = if avg_change > 0.001
                "improving"
            elseif avg_change < -0.001
                "deteriorating"
            else
                "stable"
            end

            entity_vec[idx] = ent
            push!(avg_change_rate_vec, avg_change)
            push!(volatility_vec, volatility)
            push!(direction_vec, direction)
        else
            # No valid changes found
            entity_vec[idx] = ent
            push!(avg_change_rate_vec, 0.0)
            push!(volatility_vec, 0.0)
            push!(direction_vec, "unknown")
        end
    end

    # Create result DataFrame
    result = DataFrame(
        Symbol(entity_str) => entity_vec,
        :avg_change_rate => avg_change_rate_vec,
        :volatility => volatility_vec,
        :direction => direction_vec
    )

    return result
end

"""
    clean_historical_series(data::Vector; method=:linear) -> Vector{Float64}

Clean and interpolate missing or anomalous values in historical time series.

# Arguments
- `data::Vector`: Time series data (may contain missing values)
- `method::Symbol`: Interpolation method (`:linear` or `:forward_fill`)

# Returns
- `Vector{Float64}`: Cleaned series with interpolated values

# Example
```julia
clean_data = clean_historical_series(messy_gdp_series, method=:linear)
```
"""
function clean_historical_series(data::Vector; method::Symbol=:linear)
    # Convert to Union{Float64, Missing} to preserve missing values
    cleaned = Vector{Union{Float64,Missing}}(data)

    if method == :linear
        # Simple linear interpolation for missing values
        for i in 2:length(cleaned)-1
            if ismissing(cleaned[i]) || isnan(cleaned[i])
                # Find surrounding valid values
                prev_idx = findlast(x -> !ismissing(x) && !isnan(x), cleaned[1:i-1])
                next_idx = findfirst(x -> !ismissing(x) && !isnan(x), cleaned[i+1:end])

                if !isnothing(prev_idx) && !isnothing(next_idx)
                    next_idx += i  # Adjust for offset
                    # Linear interpolation
                    cleaned[i] = cleaned[prev_idx] +
                        (cleaned[next_idx] - cleaned[prev_idx]) *
                        (i - prev_idx) / (next_idx - prev_idx)
                end
            end
        end
    elseif method == :forward_fill
        # Forward fill missing values
        last_valid = cleaned[1]
        for i in 2:length(cleaned)
            if ismissing(cleaned[i]) || isnan(cleaned[i])
                cleaned[i] = last_valid
            else
                last_valid = cleaned[i]
            end
        end
    else
        error("Unknown method: $method. Use :linear or :forward_fill")
    end

    # Convert any remaining missing values to NaN, then return as Float64 vector
    result = Vector{Float64}(undef, length(cleaned))
    for i in 1:length(cleaned)
        if ismissing(cleaned[i])
            result[i] = NaN
        else
            result[i] = Float64(cleaned[i])
        end
    end

    return result
end

"""
    interpolate_missing_years(data::DataFrame, variable::Symbol; method::Symbol=:linear) -> DataFrame

Fill in missing years in a historical time series with interpolated values.

# Arguments
- `data::DataFrame`: Data with a `:year` column and variable column
- `variable::Symbol`: Column name of the variable to interpolate
- `method::Symbol`: Interpolation method (`:linear` only currently supported)

# Returns
- `DataFrame`: Expanded DataFrame with all years filled and interpolated values

# Example
```julia
df = DataFrame(year=[2000, 2002, 2005], gdp=[100.0, 110.0, 130.0])
result = interpolate_missing_years(df, :gdp)
# Returns DataFrame with years 2000:2005 and linearly interpolated values
```
"""
function interpolate_missing_years(data::DataFrame, variable::Symbol; method::Symbol=:linear)
    if !("year" in names(data))
        error("DataFrame must have a :year column")
    end
    if !(String(variable) in names(data))
        error("Variable :$variable not found in DataFrame")
    end

    # Get year range
    year_col = data.year
    min_year = minimum(year_col)
    max_year = maximum(year_col)
    all_years = collect(min_year:max_year)

    # Create result DataFrame with all years
    result = DataFrame(year = all_years)

    # Add the variable column with interpolation
    if method == :linear
        # Create mapping of existing years to values
        year_to_value = Dict(zip(data.year, data[!, variable]))

        # Interpolate values for all years
        values = Vector{Float64}(undef, length(all_years))
        for (i, year) in enumerate(all_years)
            if haskey(year_to_value, year)
                # Year exists in original data
                values[i] = year_to_value[year]
            else
                # Find surrounding years for interpolation
                prev_year = nothing
                next_year = nothing

                for y in reverse(min_year:(year-1))
                    if haskey(year_to_value, y)
                        prev_year = y
                        break
                    end
                end

                for y in (year+1):max_year
                    if haskey(year_to_value, y)
                        next_year = y
                        break
                    end
                end

                # Linear interpolation
                if !isnothing(prev_year) && !isnothing(next_year)
                    prev_val = year_to_value[prev_year]
                    next_val = year_to_value[next_year]
                    year_diff = next_year - prev_year
                    weight = (year - prev_year) / year_diff
                    values[i] = prev_val + weight * (next_val - prev_val)
                elseif !isnothing(prev_year)
                    # Only previous value available, use it
                    values[i] = year_to_value[prev_year]
                elseif !isnothing(next_year)
                    # Only next value available, use it
                    values[i] = year_to_value[next_year]
                else
                    # No surrounding values (shouldn't happen)
                    values[i] = NaN
                end
            end
        end

        result[!, variable] = values
    else
        error("Unknown interpolation method: $method. Only :linear is supported.")
    end

    # Copy over any other columns from original data (matching by year)
    for col in names(data)
        if col != "year" && col != String(variable)
            year_to_col = Dict(zip(data.year, data[!, col]))
            result[!, col] = [get(year_to_col, y, missing) for y in all_years]
        end
    end

    return result
end

"""
    compare_historical_trajectories(data::DataFrame, regions::Vector{String};
                                   variable=:gdp_per_capita) -> DataFrame

Compare economic trajectories across multiple regions or countries.

# Arguments
- `data::DataFrame`: Historical data with region identifiers
- `regions::Vector{String}`: List of regions to compare
- `variable::Symbol`: Variable to compare

# Returns
- `DataFrame`: Comparative statistics and growth paths

# Example
```julia
comparison = compare_historical_trajectories(
    world_data,
    ["Western Europe", "East Asia", "Latin America"],
    variable=:gdp_per_capita
)
```
"""
function compare_historical_trajectories(data::DataFrame,
                                        regions::Vector{String};
                                        variable::Symbol=:gdp_per_capita)
    results = DataFrame(
        region = String[],
        initial_level = Float64[],
        final_level = Float64[],
        avg_growth = Float64[],
        std_growth = Float64[],
        cumulative_growth = Float64[]
    )

    for region in regions
        region_data = filter(row -> row.region == region, data)

        # Calculate summary statistics
        values = region_data[!, variable]
        growth = calculate_growth_rates(region_data, variable)

        push!(results, (
            region = region,
            initial_level = first(values),
            final_level = last(values),
            avg_growth = mean(growth),
            std_growth = std(growth),
            cumulative_growth = last(values) / first(values) - 1
        ))
    end

    return results
end

"""
    counterfactual_scenario(data::DataFrame, variable::Symbol, break_year::Int;
                           adjustment::Float64=1.0, method::Symbol=:multiplicative) -> DataFrame

Create a counterfactual time series by modifying a variable at a specific point in time.

# Arguments
- `data::DataFrame`: Historical data with `:year` column and variable column
- `variable::Symbol`: Variable to create counterfactual for
- `break_year::Int`: Year when counterfactual diverges from actual
- `adjustment::Float64`: Adjustment factor (default 1.0 = no change)
- `method::Symbol`: How to apply adjustment (`:multiplicative` or `:additive`)

# Returns
- `DataFrame`: Data with both `actual` and `counterfactual` columns

# Example
```julia
df = DataFrame(year=2000:2004, gdp=[100.0, 110.0, 121.0, 133.1, 146.41])
result = counterfactual_scenario(df, :gdp, 2002, adjustment=0.9, method=:multiplicative)
# Shows what would have happened if GDP was 10% lower starting in 2002
```
"""
function counterfactual_scenario(data::DataFrame, variable::Symbol, break_year::Int;
                                adjustment::Float64=1.0, method::Symbol=:multiplicative)
    if !("year" in names(data))
        error("DataFrame must have a :year column")
    end
    if !(String(variable) in names(data))
        error("Variable :$variable not found in DataFrame")
    end

    # Sort by year
    sorted_data = sort(data, :year)

    # Extract years and values
    years = sorted_data.year
    actual_values = sorted_data[!, String(variable)]

    # Initialize counterfactual values (start as copy of actual)
    counterfactual_values = copy(actual_values)

    # Find index of break year
    break_idx = findfirst(y -> y == break_year, years)

    if isnothing(break_idx)
        error("Break year $break_year not found in data")
    end

    # Apply adjustment from break year onward
    if method == :multiplicative
        # At break year, multiply by adjustment
        counterfactual_values[break_idx] = actual_values[break_idx] * adjustment

        # For subsequent years, calculate growth rates from actual
        # and apply them to counterfactual base
        for i in (break_idx+1):length(years)
            if i > 1
                # Calculate growth rate from actual data
                growth_rate = actual_values[i] / actual_values[i-1]
                # Apply that growth rate to counterfactual base
                counterfactual_values[i] = counterfactual_values[i-1] * growth_rate
            end
        end
    elseif method == :additive
        # At break year, add adjustment
        counterfactual_values[break_idx] = actual_values[break_idx] + adjustment

        # For subsequent years, calculate absolute changes from actual
        # and apply them to counterfactual base
        for i in (break_idx+1):length(years)
            if i > 1
                # Calculate absolute change from actual data
                change = actual_values[i] - actual_values[i-1]
                # Apply that change to counterfactual base
                counterfactual_values[i] = counterfactual_values[i-1] + change
            end
        end
    else
        error("Unknown method: $method. Use :multiplicative or :additive")
    end

    # Create result DataFrame
    result = DataFrame(
        year = years,
        actual = actual_values,
        counterfactual = counterfactual_values
    )

    # Copy over any other columns from original data
    for col in names(sorted_data)
        if col != "year" && col != String(variable)
            result[!, col] = sorted_data[!, col]
        end
    end

    return result
end

"""
    estimate_treatment_effect(data::DataFrame, variable::Symbol,
                              group::Symbol, treatment_year::Int) -> NamedTuple

Estimate treatment effect using difference-in-differences (DiD) methodology.

# Arguments
- `data::DataFrame`: Panel data with `:year`, group indicator, and outcome variable
- `variable::Symbol`: Outcome variable to analyze
- `group::Symbol`: Boolean column indicating treated (true) vs control (false) group
- `treatment_year::Int`: Year when treatment begins

# Returns
- `NamedTuple`: DiD statistics including treatment effect, pre/post differences, and t-statistic

# Example
```julia
df = DataFrame(
    year = repeat(1990:1999, 2),
    treated = repeat([true, false], inner=10),
    gdp = vcat([100, 102, 104, 106, 108, 115, 120, 125, 130, 135],
               [100, 102, 104, 106, 108, 110, 112, 114, 116, 118]) .* 1.0
)
result = estimate_treatment_effect(df, :gdp, :treated, 1995)
# Returns DiD estimate of treatment effect
```
"""
function estimate_treatment_effect(data::DataFrame, variable::Symbol,
                                  group::Symbol, treatment_year::Int)
    if !("year" in names(data))
        error("DataFrame must have a :year column")
    end
    if !(String(variable) in names(data))
        error("Variable :$variable not found in DataFrame")
    end
    if !(String(group) in names(data))
        error("Group column :$group not found in DataFrame")
    end

    # Split data into pre and post treatment periods
    pre_data = data[data.year .< treatment_year, :]
    post_data = data[data.year .>= treatment_year, :]

    # Split by treatment group
    treated_pre = pre_data[pre_data[!, String(group)] .== true, String(variable)]
    treated_post = post_data[post_data[!, String(group)] .== true, String(variable)]

    control_pre = pre_data[pre_data[!, String(group)] .== false, String(variable)]
    control_post = post_data[post_data[!, String(group)] .== false, String(variable)]

    # Calculate means for each group-period combination
    mean_treated_pre = mean(treated_pre)
    mean_treated_post = mean(treated_post)
    mean_control_pre = mean(control_pre)
    mean_control_post = mean(control_post)

    # Calculate differences
    treated_diff = mean_treated_post - mean_treated_pre
    control_diff = mean_control_post - mean_control_pre

    # Difference-in-differences estimator
    treatment_effect = treated_diff - control_diff

    # Calculate standard errors for simple t-statistic
    # (This is a simplified version; full DiD would use regression with clustered SEs)
    var_treated_pre = var(treated_pre)
    var_treated_post = var(treated_post)
    var_control_pre = var(control_pre)
    var_control_post = var(control_post)

    n_treated_pre = length(treated_pre)
    n_treated_post = length(treated_post)
    n_control_pre = length(control_pre)
    n_control_post = length(control_post)

    # Standard error of DiD estimator (simplified formula)
    se_did = sqrt(
        var_treated_post / n_treated_post +
        var_treated_pre / n_treated_pre +
        var_control_post / n_control_post +
        var_control_pre / n_control_pre
    )

    # t-statistic
    t_stat = treatment_effect / se_did

    return (
        treatment_effect = treatment_effect,
        pre_treatment_diff = mean_treated_pre - mean_control_pre,
        post_treatment_diff = mean_treated_post - mean_control_post,
        treated_change = treated_diff,
        control_change = control_diff,
        se = se_did,
        t_statistic = t_stat,
        mean_treated_pre = mean_treated_pre,
        mean_treated_post = mean_treated_post,
        mean_control_pre = mean_control_pre,
        mean_control_post = mean_control_post
    )
end

end # module Cliometrics
