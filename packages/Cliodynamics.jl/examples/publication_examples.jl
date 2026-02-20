# SPDX-License-Identifier: PMPL-1.0-or-later
# Publication-Quality Analysis Examples for Cliodynamics.jl
#
# Demonstrates research-grade analyses replicating key results from the
# cliodynamics literature. Each example maps to published findings.
#
# Run with: julia --project=. examples/publication_examples.jl

using Cliodynamics
using DataFrames
using Statistics

println("=" ^ 70)
println("  Cliodynamics.jl — Publication-Quality Analysis Examples")
println("  Replicating key results from the cliodynamics literature")
println("=" ^ 70)
println()

# ══════════════════════════════════════════════════════════════════════════════
# Example 1: Roman Secular Cycles (Turchin & Nefedov 2009)
# ══════════════════════════════════════════════════════════════════════════════
println("━" ^ 70)
println("  1. Roman Secular Cycles (cf. Turchin & Nefedov 2009, Ch. 4)")
println("━" ^ 70)

# Load Seshat sample data for Roman polities
seshat_path = joinpath(@__DIR__, "..", "data", "seshat_sample.csv")
raw = load_seshat_csv(seshat_path)
roman = prepare_seshat_data(raw)
roman = filter(row -> occursin("Rom", string(row.polity)), roman)
sort!(roman, :year)

println("\n  Roman data: $(nrow(roman)) observations, $(Int(minimum(roman.year))) to $(Int(maximum(roman.year))) CE")

# Compute elite overproduction across the Republic-Empire transition
eoi = elite_overproduction_index(roman)
println("\n  Elite Overproduction Index across Roman history:")
for i in 1:nrow(eoi)
    yr = Int(eoi.year[i])
    val = round(eoi.eoi[i], digits=4)
    bar = repeat("█", max(0, round(Int, (val + 0.5) * 20)))
    marker = val > 0.5 ? " ← CRISIS ZONE" : ""
    println("    $(lpad(yr, 5)) CE: EOI = $(lpad(string(val), 8))  $bar$marker")
end

# Fit Malthusian model to Roman population data
println("\n  Malthusian fit to Roman population:")
rom_years = Float64.(roman.year)
rom_pop = Float64.(roman.population)
rom_fit = fit_malthusian(rom_years, rom_pop, r_init=0.005, K_init=maximum(rom_pop) * 2)
println("    Growth rate r = $(round(rom_fit.params.r, digits=6))")
println("    Carrying capacity K = $(round(rom_fit.params.K, digits=0))")
println("    Converged: $(rom_fit.converged), Loss: $(round(rom_fit.loss, digits=2))")

# ══════════════════════════════════════════════════════════════════════════════
# Example 2: Ages of Discord — US Analysis (Turchin 2016)
# ══════════════════════════════════════════════════════════════════════════════
println("\n")
println("━" ^ 70)
println("  2. Ages of Discord — US-Inspired Analysis (cf. Turchin 2016)")
println("━" ^ 70)

# Synthetic data inspired by US 1780-2020 structural-demographic indicators
# Real wages peaked ~1970, elite overproduction accelerated post-1980
us_years = 1780:10:2020
n_us = length(us_years)

# Simulate US-like indicators
real_wages = vcat(
    range(80.0, 120.0, length=div(n_us, 3)),       # 1780-1860: rising
    range(120.0, 150.0, length=div(n_us, 3)),       # 1860-1940: continuing rise
    range(150.0, 100.0, length=n_us - 2*div(n_us, 3))  # 1940-2020: stagnation/decline
)

elite_ratio = vcat(
    range(0.005, 0.008, length=div(n_us, 3)),       # 1780-1860: gilded age elites
    range(0.008, 0.006, length=div(n_us, 3)),       # 1860-1940: progressive era
    range(0.006, 0.025, length=n_us - 2*div(n_us, 3))  # 1940-2020: explosion
)

state_revenue = vcat(
    range(500.0, 800.0, length=div(n_us, 3)),       # 1780-1860: growing
    range(800.0, 1200.0, length=div(n_us, 3)),       # 1860-1940: strong
    range(1200.0, 600.0, length=n_us - 2*div(n_us, 3))  # 1940-2020: fiscal strain
)

us_data = DataFrame(
    year = collect(us_years),
    real_wages = real_wages,
    elite_ratio = elite_ratio,
    state_revenue = state_revenue
)

psi = political_stress_indicator(us_data)
println("\n  Political Stress Indicator (US-inspired, 1780-2020):")
println("  " * "-" ^ 66)
println("  $(lpad("Year", 6)) | $(lpad("PSI", 8)) | $(lpad("MMP", 8)) | $(lpad("EMP", 8)) | $(lpad("SFD", 8)) |")
println("  " * "-" ^ 66)
for i in 1:nrow(psi)
    yr = Int(psi.year[i])
    println("  $(lpad(yr, 6)) | $(lpad(round(psi.psi[i], digits=3), 8)) | $(lpad(round(psi.mmp[i], digits=3), 8)) | $(lpad(round(psi.emp[i], digits=3), 8)) | $(lpad(round(psi.sfd[i], digits=3), 8)) |")
end
println("  " * "-" ^ 66)

# Find peak stress decade
peak_idx = argmax(psi.psi)
println("\n  Peak PSI: $(round(psi.psi[peak_idx], digits=3)) at $(Int(psi.year[peak_idx]))")
println("  Instability probability at peak: $(round(instability_probability(psi.psi[peak_idx]), digits=3))")

# ══════════════════════════════════════════════════════════════════════════════
# Example 3: Multi-Region Instability Diffusion
# ══════════════════════════════════════════════════════════════════════════════
println("\n")
println("━" ^ 70)
println("  3. Spatial Instability Diffusion (Multi-Region Model)")
println("━" ^ 70)

# Model 1848 European revolutions spreading from France
regions = [
    (name=:France,   psi0=0.85, growth_rate=0.08),
    (name=:Germany,  psi0=0.3,  growth_rate=0.05),
    (name=:Austria,  psi0=0.4,  growth_rate=0.04),
    (name=:Italy,    psi0=0.5,  growth_rate=0.06),
    (name=:Britain,  psi0=0.2,  growth_rate=0.02)
]

# Adjacency: France-Germany, France-Italy, Germany-Austria, France-Britain (weak)
adjacency = [0.0 1.0 0.0 1.0 0.3;
              1.0 0.0 1.0 0.0 0.0;
              0.0 1.0 0.0 0.5 0.0;
              1.0 0.0 0.5 0.0 0.0;
              0.3 0.0 0.0 0.0 0.0]

result = spatial_instability_diffusion(regions, adjacency,
                                        diffusion_rate=0.15, tspan=(0.0, 30.0))

println("\n  1848 Revolutions — Instability diffusion model:")
println("  Initial PSI → Final PSI (after 30 years):")
for i in 1:length(result.regions)
    name = rpad(result.regions[i], 10)
    initial = round(result.psi[1, i], digits=3)
    final_val = round(result.psi[end, i], digits=3)
    delta = round(final_val - initial, digits=3)
    arrow = delta > 0.1 ? " ↑↑" : delta > 0.0 ? " ↑" : " →"
    println("    $name: $initial → $final_val  (Δ=$(lpad(string(delta), 6)))$arrow")
end

# ══════════════════════════════════════════════════════════════════════════════
# Example 4: Territorial Competition (Turchin's Meta-Ethnic Frontier Theory)
# ══════════════════════════════════════════════════════════════════════════════
println("\n")
println("━" ^ 70)
println("  4. Territorial Competition & Frontier Formation")
println("━" ^ 70)

# Model competing states with different military capacities
states = [
    (name=:Rome,     territory0=100.0, military=2.0, growth_rate=0.02),
    (name=:Carthage, territory0=80.0,  military=1.5, growth_rate=0.015),
    (name=:Macedon,  territory0=60.0,  military=1.8, growth_rate=0.01),
    (name=:Ptolemaic, territory0=50.0, military=1.0, growth_rate=0.005)
]

terr = territorial_competition_model(states, tspan=(0.0, 200.0))

println("\n  Mediterranean territorial competition (200 years):")
println("  " * "-" ^ 50)
for i in 1:length(terr.states)
    name = rpad(terr.states[i], 10)
    initial = round(terr.territory[1, i], digits=1)
    final_val = round(terr.territory[end, i], digits=1)
    pct = round((final_val / initial - 1) * 100, digits=1)
    println("    $name: $(lpad(string(initial), 6)) → $(lpad(string(final_val), 8))  ($(pct > 0 ? "+" : "")$(pct)%)")
end

# Meta-ethnic frontier formation
println("\n  Frontier formation index:")
cultural_dist = [0.0 0.9 0.4 0.5;
                  0.9 0.0 0.7 0.6;
                  0.4 0.7 0.0 0.3;
                  0.5 0.6 0.3 0.0]
pops = [1500000.0, 800000.0, 600000.0, 400000.0]
terrs = [100.0, 80.0, 60.0, 50.0]
ffi = frontier_formation_index(cultural_dist, pops, terrs)

state_names = ["Rome", "Carthage", "Macedon", "Ptolemaic"]
for i in 1:4
    bar = repeat("█", round(Int, ffi[i] * 25))
    println("    $(rpad(state_names[i], 10)): FFI = $(round(ffi[i], digits=3))  $bar")
end

# ══════════════════════════════════════════════════════════════════════════════
# Example 5: Parameter Estimation with Bootstrap CI
# ══════════════════════════════════════════════════════════════════════════════
println("\n")
println("━" ^ 70)
println("  5. Parameter Estimation with Confidence Intervals")
println("━" ^ 70)

# Fit exponential growth model to English population data
english = prepare_seshat_data(raw)
english = filter(row -> occursin("Eng", string(row.polity)), english)
sort!(english, :year)

eng_years = Float64.(english.year)
eng_pop = Float64.(english.population)

# Exponential model: P(t) = A * exp(r * (t - t0))
exp_model(p, t) = p[1] .* exp.(p[2] .* (t .- t[1]))

est = estimate_parameters(exp_model, eng_pop, eng_years, [eng_pop[1], 0.002],
                           n_bootstrap=500)

println("\n  English population growth (exponential model):")
println("    A₀ = $(round(est.params[1], digits=0))")
println("         95% CI: [$(round(est.ci_lower[1], digits=0)), $(round(est.ci_upper[1], digits=0))]")
println("    r  = $(round(est.params[2], digits=6))")
println("         95% CI: [$(round(est.ci_lower[2], digits=6)), $(round(est.ci_upper[2], digits=6))]")
println("    Converged: $(est.converged)")
println("    Doubling time: $(round(log(2) / max(est.params[2], 1e-10), digits=0)) years")

# ══════════════════════════════════════════════════════════════════════════════
# Example 6: Full Secular Cycle with Phase Detection
# ══════════════════════════════════════════════════════════════════════════════
println("\n")
println("━" ^ 70)
println("  6. Synthetic Secular Cycle with Phase Detection")
println("━" ^ 70)

# Generate a 300-year cycle matching Turchin-Nefedov pattern
n = 301
years_cycle = 1500:1800

# Four phases across 301 years
phase_gen(low, mid1, mid2, high) = vcat(
    range(low, mid1, length=div(n,4)),
    range(mid1, high, length=div(n,4)),
    range(high, mid2, length=div(n,4)),
    range(mid2, low, length=n - 3*div(n,4))
)

cycle_data = DataFrame(
    year = collect(years_cycle),
    population_pressure = clamp.(phase_gen(0.2, 0.5, 0.9, 0.85) .+ 0.02 .* randn(n), 0.0, 1.0),
    elite_overproduction = clamp.(phase_gen(0.1, 0.3, 0.45, 0.6) .+ 0.02 .* randn(n), 0.0, 1.0),
    instability = clamp.(phase_gen(0.15, 0.3, 0.55, 0.75) .+ 0.02 .* randn(n), 0.0, 1.0)
)

phases = detect_cycle_phases(cycle_data)

# Count and display phases
phase_summary = Dict{String, Vector{Int}}()
for i in 1:nrow(phases)
    p = string(phases.phase[i])
    if !haskey(phase_summary, p)
        phase_summary[p] = Int[]
    end
    push!(phase_summary[p], Int(phases.year[i]))
end

println("\n  Secular cycle phases (1500-1800):")
for (name, yrs) in sort(collect(phase_summary), by=x->minimum(x[2]))
    bar = repeat("█", div(length(yrs), 4))
    println("    $(rpad(name, 14)): $(minimum(yrs))-$(maximum(yrs))  ($(length(yrs)) years)  $bar")
end

# Cycle analysis on population pressure time series
analysis = secular_cycle_analysis(Float64.(cycle_data.population_pressure), window=30)
println("\n  Cycle analysis:")
println("    Detected period: $(analysis.period) years")
println("    Amplitude: $(round(analysis.amplitude, digits=3))")

# ══════════════════════════════════════════════════════════════════════════════
# Example 7: Collective Action & State Formation
# ══════════════════════════════════════════════════════════════════════════════
println("\n")
println("━" ^ 70)
println("  7. Collective Action & State Formation (cf. Turchin 2003, Ch. 3)")
println("━" ^ 70)

println("\n  Olsonian collective action problem:")
println("  How group size affects coordination success")
println("  (benefit=10000, cost per capita=5)")
println()
println("  $(rpad("Group Size", 12)) $(rpad("P(success)", 12)) Visual")
println("  " * "-" ^ 50)
for n_group in [5, 10, 25, 50, 100, 250, 500, 1000, 5000, 10000]
    prob = collective_action_problem(n_group, 10000.0, 5.0)
    bar = repeat("█", round(Int, prob * 30))
    empty = repeat("░", 30 - round(Int, prob * 30))
    println("  $(rpad(n_group, 12)) $(rpad(round(prob, digits=4), 12)) $bar$empty")
end

println("\n  Key insight: Small groups coordinate effectively;")
println("  large groups face free-rider problems (Olson 1965)")

# ══════════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════════
println("\n")
println("=" ^ 70)
println("  Analysis Complete")
println("=" ^ 70)
println()
println("  Models demonstrated:")
println("    1. Seshat data integration + Roman elite overproduction")
println("    2. US-inspired political stress indicator (PSI)")
println("    3. Spatial instability diffusion (1848 revolutions)")
println("    4. Territorial competition + frontier formation")
println("    5. Bootstrap parameter estimation with CI")
println("    6. Secular cycle phase detection")
println("    7. Collective action and group size effects")
println()
println("  For Bayesian analysis, load Turing.jl:")
println("    using Turing")
println("    result = bayesian_malthusian(years, population)")
println()
println("  For visualization, load Plots.jl:")
println("    using Plots")
println("    plot(psi_result, Val(:psi))  # and other recipes")
