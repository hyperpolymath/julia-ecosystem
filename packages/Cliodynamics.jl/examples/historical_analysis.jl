# SPDX-License-Identifier: PMPL-1.0-or-later
# Historical Analysis Examples for Cliodynamics.jl
#
# Demonstrates Seshat data integration, model fitting, parameter estimation,
# phase detection, and conflict intensity analysis.
#
# Run with: julia --project=. examples/historical_analysis.jl

using Cliodynamics
using DataFrames
using Statistics

println("=== Cliodynamics.jl Historical Analysis Examples ===\n")

# ── Example 1: Load and Analyze Seshat Data ──────────────────────────────────
println("1. Seshat Global History Databank Integration")
println("-" ^ 55)

seshat_path = joinpath(@__DIR__, "..", "data", "seshat_sample.csv")
raw = load_seshat_csv(seshat_path)
println("  Loaded $(nrow(raw)) records from Seshat sample data")
println("  Polities: ", join(unique(raw.polity), ", "))
println()

# Prepare Roman Republic/Empire data
roman_data = prepare_seshat_data(raw, polity="RomPrinworlds")
println("  Roman Principate (RomPrinworlds):")
println("    Years: ", Int(minimum(roman_data.year)), " to ", Int(maximum(roman_data.year)))
println("    Population range: ", Int(minimum(roman_data.population)), " - ", Int(maximum(roman_data.population)))
println("    Elite ratio: ", round(roman_data.elite_ratio[1], digits=4), " → ", round(roman_data.elite_ratio[end], digits=4))
println()

# ── Example 2: Elite Overproduction in Roman Empire ──────────────────────────
println("2. Elite Overproduction in Roman Empire")
println("-" ^ 55)

# Combine all Roman periods for long-term analysis
roman_all = prepare_seshat_data(raw)
roman_all = filter(row -> occursin("Rom", string(row.polity)), roman_all)
sort!(roman_all, :year)

eoi = elite_overproduction_index(roman_all)
println("  Roman elite overproduction index (500 BCE - 450 CE):")
for i in 1:nrow(eoi)
    marker = eoi.eoi[i] > 0.5 ? " ← OVERPRODUCTION" : ""
    println("    Year $(Int(eoi.year[i])): EOI = $(round(eoi.eoi[i], digits=3))$marker")
end
println()

# ── Example 3: Model Fitting to Population Data ─────────────────────────────
println("3. Model Fitting (Malthusian) to English Population")
println("-" ^ 55)

# Extract English data
english = prepare_seshat_data(raw)
english = filter(row -> occursin("Eng", string(row.polity)), english)
sort!(english, :year)

# Fit Malthusian model to English population growth
eng_years = Float64.(english.year)
eng_pop = Float64.(english.population)

fit = fit_malthusian(eng_years, eng_pop, r_init=0.005, K_init=2_000_000.0)
println("  Fitted Malthusian parameters:")
println("    Growth rate r = ", round(fit.params.r, digits=6))
println("    Carrying capacity K = ", round(fit.params.K, digits=0))
println("    Converged: ", fit.converged)
println("    Loss: ", round(fit.loss, digits=2))
println()

# Compare fit vs observed at key dates
println("  Fit vs Observed:")
fit_sol = malthusian_model(fit.params, tspan=(eng_years[1], eng_years[end]))
for i in [1, div(length(eng_years), 2), length(eng_years)]
    yr = eng_years[i]
    obs = eng_pop[i]
    pred = fit_sol(yr)[1]
    err = round(abs(pred - obs) / obs * 100, digits=1)
    println("    Year $(Int(yr)): observed=$(Int(obs)), predicted=$(round(pred, digits=0)), error=$(err)%")
end
println()

# ── Example 4: Parameter Estimation with Bootstrap CI ────────────────────────
println("4. Parameter Estimation with Confidence Intervals")
println("-" ^ 55)

# Estimate exponential growth parameters with uncertainty
function exp_model(p, t)
    A, r = p
    return A .* exp.(r .* (t .- t[1]))
end

# Use early English growth phase (500-1200 CE)
early_eng = filter(row -> row.year >= 500 && row.year <= 1200, english)
early_years = Float64.(early_eng.year)
early_pop = Float64.(early_eng.population)

result = estimate_parameters(exp_model, early_pop, early_years, [50000.0, 0.002],
                              n_bootstrap=200)

println("  Exponential growth fit (500-1200 CE):")
println("    A₀ = $(round(result.params[1], digits=0))  95% CI: [$(round(result.ci_lower[1], digits=0)), $(round(result.ci_upper[1], digits=0))]")
println("    r  = $(round(result.params[2], digits=5))  95% CI: [$(round(result.ci_lower[2], digits=5)), $(round(result.ci_upper[2], digits=5))]")
println("    Converged: ", result.converged)
println()

# ── Example 5: Cycle Phase Detection ─────────────────────────────────────────
println("5. Secular Cycle Phase Detection")
println("-" ^ 55)

# Create synthetic data representing a 300-year secular cycle
n_years = 301
years_cycle = 1500:1800

# Simulate Turchin-Nefedov secular cycle phases
phase_signal(low, high, n) = vcat(
    range(low, high, length=div(n, 4)),           # expansion
    range(high, high*0.9, length=div(n, 4)),       # stagflation
    range(high*0.9, high*1.1, length=div(n, 4)),   # crisis
    range(high*1.1, low*1.2, length=n - 3*div(n, 4))  # depression
)

phase_data = DataFrame(
    year = collect(years_cycle),
    population_pressure = clamp.(phase_signal(0.2, 0.85, n_years) .+ 0.03 .* randn(n_years), 0.0, 1.0),
    elite_overproduction = clamp.(phase_signal(0.1, 0.6, n_years) .+ 0.03 .* randn(n_years), 0.0, 1.0),
    instability = clamp.(phase_signal(0.15, 0.75, n_years) .+ 0.03 .* randn(n_years), 0.0, 1.0)
)

phases = detect_cycle_phases(phase_data)

phase_counts = Dict{String, Int}()
for p in phases.phase
    name = string(p)
    phase_counts[name] = get(phase_counts, name, 0) + 1
end

println("  Phase distribution across 1500-1800:")
for (name, count) in sort(collect(phase_counts), by=x->x[2], rev=true)
    bar = repeat("█", div(count, 3))
    println("    $name: $count years  $bar")
end
println()

# ── Example 6: Instability Events and Conflict Intensity ─────────────────────
println("6. Instability Events and Conflict Analysis")
println("-" ^ 55)

# Create synthetic indicator data with crisis periods
indicator_data = DataFrame(
    year = 1700:1900,
    indicator = vcat(
        0.3 .+ 0.1 .* randn(48),    # 1700-1747: stable
        0.8 .+ 0.1 .* randn(5),      # 1748-1752: crisis
        0.3 .+ 0.1 .* randn(37),     # 1753-1789: stable
        0.85 .+ 0.15 .* randn(10),   # 1790-1799: revolution
        0.4 .+ 0.1 .* randn(48),     # 1800-1847: moderate
        0.9 .+ 0.1 .* randn(3),      # 1848-1850: revolution
        0.35 .+ 0.1 .* randn(50)     # 1851-1900: stable
    )
)

events = instability_events(indicator_data, 0.7)

println("  Instability events detected: ", length(events))
for e in events
    println("    $(e.year): intensity=$(round(e.intensity, digits=2)), type=$(e.type)")
end
println()

# Compute conflict intensity from events
if !isempty(events)
    intensity = conflict_intensity(events, window=10)
    peak_year = intensity.year[argmax(intensity.intensity)]
    println("  Conflict intensity:")
    println("    Peak year: ", peak_year)
    println("    Peak intensity: ", round(maximum(intensity.intensity), digits=2))
    println("    Years with intensity > 0.5: ", count(intensity.intensity .> 0.5))
end
println()

# ── Example 7: Collective Action and State Formation ─────────────────────────
println("7. Collective Action in State Formation")
println("-" ^ 55)

println("  Group size → Probability of successful collective action:")
println("  (benefit=10000, cost=5)")
for n in [5, 10, 50, 100, 500, 1000, 5000]
    prob = collective_action_problem(n, 10000.0, 5.0)
    bar = repeat("█", round(Int, prob * 30))
    println("    n=$(lpad(n, 5)): P=$(round(prob, digits=3))  $bar")
end
println()

println("=== Historical Analysis Complete ===")
println("\nFor plotting (requires Plots.jl):")
println("  using Plots")
println("  plot(psi_result, Val(:psi))       # PSI breakdown")
println("  plot(eoi_result, Val(:eoi))       # Elite overproduction")
println("  plot(phases, Val(:phases))        # Phase timeline")
println("  plot(intensity, Val(:conflict))   # Conflict intensity")
