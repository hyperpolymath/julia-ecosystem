# SPDX-License-Identifier: PMPL-1.0-or-later
# Basic Usage Examples for Cliodynamics.jl
#
# Run with: julia --project=. examples/basic_usage.jl

using Cliodynamics
using DataFrames
using Statistics

println("=== Cliodynamics.jl Basic Usage Examples ===\n")

# ── Example 1: Malthusian Population Model ────────────────────────────────────
println("1. Malthusian Population Dynamics")
println("-" ^ 40)

params_malthus = MalthusianParams(r=0.02, K=1000.0, N0=100.0)
sol = malthusian_model(params_malthus, tspan=(0.0, 200.0))

println("  Initial population: ", params_malthus.N0)
println("  Carrying capacity:  ", params_malthus.K)
println("  Growth rate:         ", params_malthus.r)
println("  Population at t=100: ", round(sol(100.0)[1], digits=1))
println("  Population at t=200: ", round(sol(200.0)[1], digits=1))
println()

# ── Example 2: Demographic-Structural Theory Model ───────────────────────────
println("2. Demographic-Structural Theory Model")
println("-" ^ 40)

params_dst = DemographicStructuralParams(
    r=0.015, K=1000.0, w=2.0, δ=0.03, ε=0.001,
    N0=500.0, E0=10.0, S0=100.0
)
sol_dst = demographic_structural_model(params_dst, tspan=(0.0, 300.0))

println("  Initial state: N₀=", params_dst.N0, ", E₀=", params_dst.E0, ", S₀=", params_dst.S0)
println("  At t=150:")
state_150 = sol_dst(150.0)
println("    Population:     ", round(state_150[1], digits=1))
println("    Elites:         ", round(state_150[2], digits=1))
println("    State capacity: ", round(state_150[3], digits=1))
println("  At t=300:")
state_300 = sol_dst(300.0)
println("    Population:     ", round(state_300[1], digits=1))
println("    Elites:         ", round(state_300[2], digits=1))
println("    State capacity: ", round(state_300[3], digits=1))
println()

# ── Example 3: Elite Overproduction Index ─────────────────────────────────────
println("3. Elite Overproduction Index")
println("-" ^ 40)

data_eoi = DataFrame(
    year = 1800:1900,
    population = collect(100_000:1000:200_000),
    elites = [1000 + 10*i + 5*i^1.5 for i in 0:100]
)

eoi = elite_overproduction_index(data_eoi)

println("  Time period: 1800-1900")
println("  Initial EOI: ", round(eoi.eoi[1], digits=3))
println("  Final EOI:   ", round(eoi.eoi[end], digits=3))
println("  Max EOI:     ", round(maximum(eoi.eoi), digits=3))
println("  (Positive values = elite overproduction relative to baseline)")
println()

# ── Example 4: Political Stress Indicator ─────────────────────────────────────
println("4. Political Stress Indicator (PSI)")
println("-" ^ 40)

data_psi = DataFrame(
    year = 1800:1900,
    real_wages = 100.0 .- collect(0:100).^1.2 ./ 10,
    elite_ratio = 0.01 .+ collect(0:100) ./ 5000,
    state_revenue = 1000.0 .- collect(0:100).^1.5 ./ 5
)

psi_result = political_stress_indicator(data_psi)

println("  Time period: 1800-1900")
println("  PSI at 1800: ", round(psi_result.psi[1], digits=3))
println("  PSI at 1850: ", round(psi_result.psi[51], digits=3))
println("  PSI at 1900: ", round(psi_result.psi[end], digits=3))
println("  Max PSI:     ", round(maximum(psi_result.psi), digits=3))
println("  Components at 1900:")
println("    MMP (mass mobilization):   ", round(psi_result.mmp[end], digits=3))
println("    EMP (elite mobilization):  ", round(psi_result.emp[end], digits=3))
println("    SFD (state fiscal distress):", round(psi_result.sfd[end], digits=3))
println()

# ── Example 5: Secular Cycle Analysis ─────────────────────────────────────────
println("5. Secular Cycle Analysis")
println("-" ^ 40)

t = 1:300
cycle_data = 100.0 .+ 50.0 .* sin.(2π .* t ./ 100) .+ 2 .* randn(300)

analysis = secular_cycle_analysis(Float64.(cycle_data), window=30)

println("  Data length:     ", length(cycle_data), " time steps")
println("  Detected period: ", analysis.period, " (expected ≈100)")
println("  Cycle amplitude: ", round(analysis.amplitude, digits=2))
println()

# ── Example 6: State Capacity Model ──────────────────────────────────────────
println("6. State Capacity Model")
println("-" ^ 40)

params_state = StateCapacityParams(τ=0.15, α=1.0, β=0.8, γ=0.5)
cap_normal = state_capacity_model(params_state, 1000.0, 50.0)
cap_overpop = state_capacity_model(params_state, 2000.0, 50.0)
cap_elite_glut = state_capacity_model(params_state, 1000.0, 100.0)

println("  Normal (pop=1000, elites=50):   ", round(cap_normal, digits=1))
println("  High pop (pop=2000, elites=50): ", round(cap_overpop, digits=1))
println("  Elite glut (pop=1000, elites=100): ", round(cap_elite_glut, digits=1))
println("  → More population increases capacity")
println("  → Excessive elites decrease capacity")
println()

# ── Example 7: Instability Probability ────────────────────────────────────────
println("7. Instability Probability")
println("-" ^ 40)

for psi_val in [0.1, 0.3, 0.5, 0.7, 0.9]
    prob = instability_probability(psi_val)
    bar = repeat("█", round(Int, prob * 30))
    println("  PSI=$psi_val → P(instability)=$(round(prob, digits=3))  $bar")
end
println()

println("=== Examples Complete ===")
println("\nFor historical analysis examples, see historical_analysis.jl")
println("For plotting, load Plots.jl: using Plots; plot(psi_result, Val(:psi))")
