# SPDX-License-Identifier: PMPL-1.0-or-later
# Growth Decomposition Example for Cliometrics.jl
# Author: Jonathan D.A. Jewell <jonathan.jewell@open.ac.uk>

using Cliometrics
using DataFrames
using Statistics

println("=" ^ 60)
println("Cliometrics.jl - Growth Decomposition Example")
println("=" ^ 60)

# Create synthetic historical economic data (1950-2000)
years = 1950:2000
n = length(years)

# Simulate realistic economic growth
gdp = 1000.0 * (1.03 .^ (0:n-1))  # 3% annual growth
capital = 3000.0 * (1.035 .^ (0:n-1))  # 3.5% annual capital growth
labor = 100.0 * (1.015 .^ (0:n-1))  # 1.5% annual labor growth

data = DataFrame(
    year = years,
    gdp = gdp,
    capital = capital,
    labor = labor
)

println("\n1. Load Historical Data")
println("-" ^ 60)
println("Loaded data for $(nrow(data)) years ($(first(data.year))-$(last(data.year)))")
println("Initial GDP: \$$(round(data.gdp[1], digits=2))")
println("Final GDP: \$$(round(data.gdp[end], digits=2))")

# Calculate growth rates
println("\n2. Calculate Growth Rates")
println("-" ^ 60)
growth_rates = calculate_growth_rates(data, :gdp, method=:geometric)
avg_growth = mean(growth_rates)
println("Average annual growth rate: $(round(avg_growth * 100, digits=2))%")

# Decompose growth into sources
println("\n3. Growth Decomposition (Solow Framework)")
println("-" ^ 60)
decomp = decompose_growth(data, alpha=0.35)  # Capital share α = 0.35

# Show first 5 years of decomposition
println("\nFirst 5 years:")
println(first(decomp, 5))

# Calculate average contributions
avg_capital_contrib = mean(decomp.capital_contribution)
avg_labor_contrib = mean(decomp.labor_contribution)
avg_tfp_contrib = mean(decomp.tfp_contribution)

println("\nAverage contributions to growth:")
println("  Capital:  $(round(avg_capital_contrib * 100, digits=2))%")
println("  Labor:    $(round(avg_labor_contrib * 100, digits=2))%")
println("  TFP:      $(round(avg_tfp_contrib * 100, digits=2))%")
println("  Total:    $(round((avg_capital_contrib + avg_labor_contrib + avg_tfp_contrib) * 100, digits=2))%")

# Calculate Solow residual (TFP growth)
println("\n4. Solow Residual (TFP Growth)")
println("-" ^ 60)
tfp_growth = solow_residual(data.gdp, data.capital, data.labor, alpha=0.35)
avg_tfp = mean(tfp_growth)
println("Average TFP growth: $(round(avg_tfp * 100, digits=2))%")
println("TFP accounts for $(round(avg_tfp / avg_growth * 100, digits=1))% of output growth")

# Convergence analysis (simulate 2 countries)
println("\n5. Convergence Analysis")
println("-" ^ 60)

convergence_data = DataFrame(
    country = ["Country A", "Country B", "Country C", "Country D"],
    initial_gdp = [1000.0, 2000.0, 4000.0, 8000.0],
    growth_rate = [0.05, 0.04, 0.03, 0.02]  # Poorer countries grow faster
)

conv_result = convergence_analysis(convergence_data, :initial_gdp, :growth_rate)

println("Beta coefficient: $(round(conv_result.beta, digits=4))")
println("R-squared: $(round(conv_result.r_squared, digits=3))")
println("Convergence detected: $(conv_result.converging)")
if conv_result.converging
    println("Half-life: $(round(conv_result.half_life, digits=1)) years")
end

# Institutional quality analysis
println("\n6. Institutional Quality Index")
println("-" ^ 60)

inst_data = DataFrame(
    country = ["Germany", "Brazil", "Nigeria"],
    rule_of_law = [0.9, 0.5, 0.3],
    property_rights = [0.85, 0.55, 0.35],
    corruption_control = [0.8, 0.4, 0.25]
)

indicators = [:rule_of_law, :property_rights, :corruption_control]
inst_index = institutional_quality_index(inst_data, indicators)

println("\nInstitutional Quality Index (0-1):")
for (i, country) in enumerate(inst_data.country)
    println("  $(country): $(round(inst_index[i], digits=3))")
end

# Counterfactual scenario
println("\n7. Counterfactual Analysis")
println("-" ^ 60)

# What if GDP growth was 10% lower starting in 1975?
counterfactual_data = DataFrame(year=1970:1980, gdp=[1000, 1030, 1061, 1093, 1126, 1159, 1194, 1229, 1265, 1303, 1342] .* 1.0)
cf_result = counterfactual_scenario(counterfactual_data, :gdp, 1975, adjustment=0.9, method=:multiplicative)

println("\nCounterfactual: 10% GDP reduction in 1975")
println("Actual 1980 GDP: \$$(round(cf_result.actual[end], digits=0))")
println("Counterfactual 1980 GDP: \$$(round(cf_result.counterfactual[end], digits=0))")
println("Difference: \$$(round(cf_result.actual[end] - cf_result.counterfactual[end], digits=0))")

println("\n" * "=" ^ 60)
println("Example complete! All core Cliometrics.jl features demonstrated.")
println("=" ^ 60)
