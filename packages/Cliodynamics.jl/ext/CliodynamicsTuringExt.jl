# SPDX-License-Identifier: PMPL-1.0-or-later

"""
Bayesian inference extension for Cliodynamics.jl.

Loaded automatically when `using Turing` is called alongside Cliodynamics.
Provides Bayesian parameter estimation for cliodynamic models using
Markov Chain Monte Carlo (MCMC) sampling.

# Functions
- `bayesian_malthusian`: Bayesian fit of Malthusian population model
- `bayesian_dst`: Bayesian fit of Demographic-Structural Theory model
- `bayesian_model_comparison`: Compare models via WAIC/LOO
"""
module CliodynamicsTuringExt

using Turing
using Distributions
using MCMCChains
using Cliodynamics
import DataFrames: DataFrame, hasproperty

# ============================================================================
# Bayesian Malthusian Model
# ============================================================================

"""
    bayesian_malthusian(years, population; n_samples=1000, n_chains=4, priors=nothing)

Bayesian parameter estimation for the Malthusian logistic growth model.

Uses NUTS (No-U-Turn Sampler) to estimate posterior distributions for growth
rate `r` and carrying capacity `K`.

# Arguments
- `years::Vector{Float64}`: Observation years
- `population::Vector{Float64}`: Observed population values
- `n_samples::Int`: Number of MCMC samples per chain (default: 1000)
- `n_chains::Int`: Number of parallel chains (default: 4)
- `priors`: Optional NamedTuple of prior distributions `(r=..., K=..., σ=...)`

# Returns
NamedTuple with:
- `chain::Chains`: Full MCMCChains object with posterior samples
- `params::NamedTuple`: Posterior mean parameter estimates `(r, K)`
- `ci::NamedTuple`: 95% credible intervals `(r_lower, r_upper, K_lower, K_upper)`
- `summary::DataFrame`: Summary statistics table

# Example
```julia
using Cliodynamics, Turing

years = collect(0.0:10.0:100.0)
pop = [100.0 * exp(0.03 * t) / (1 + 100.0/500.0*(exp(0.03*t)-1)) for t in years]
result = bayesian_malthusian(years, pop)
println("r = \$(result.params.r) [\$(result.ci.r_lower), \$(result.ci.r_upper)]")
```
"""
function Cliodynamics.bayesian_malthusian(years::Vector{Float64}, population::Vector{Float64};
                                          n_samples::Int=1000, n_chains::Int=4,
                                          priors=nothing)
    N0 = population[1]
    t_offset = years .- years[1]

    # Default priors
    r_prior = priors !== nothing && haskey(priors, :r) ? priors.r : Truncated(Normal(0.02, 0.05), 0.0, 1.0)
    K_prior = priors !== nothing && haskey(priors, :K) ? priors.K : Truncated(Normal(maximum(population) * 2, maximum(population)), 0.0, Inf)
    σ_prior = priors !== nothing && haskey(priors, :σ) ? priors.σ : InverseGamma(2.0, 3.0)

    @model function malthusian_bayes(obs, t, N0)
        r ~ r_prior
        K ~ K_prior
        σ ~ σ_prior

        for i in eachindex(obs)
            # Logistic growth solution: N(t) = K*N0 / (N0 + (K-N0)*exp(-r*t))
            predicted = K * N0 / (N0 + (K - N0) * exp(-r * t[i]))
            obs[i] ~ Normal(predicted, σ * abs(predicted) + 1.0)
        end
    end

    model = malthusian_bayes(population, t_offset, N0)
    chain = sample(model, NUTS(), MCMCThreads(), n_samples, n_chains)

    # Extract summary statistics
    r_samples = vec(chain[:r].data)
    K_samples = vec(chain[:K].data)

    r_mean = mean(r_samples)
    K_mean = mean(K_samples)
    r_ci = quantile(r_samples, [0.025, 0.975])
    K_ci = quantile(K_samples, [0.025, 0.975])

    return (
        chain = chain,
        params = (r = r_mean, K = K_mean),
        ci = (r_lower = r_ci[1], r_upper = r_ci[2],
              K_lower = K_ci[1], K_upper = K_ci[2]),
        summary = DataFrame(
            parameter = ["r", "K"],
            mean = [r_mean, K_mean],
            ci_lower = [r_ci[1], K_ci[1]],
            ci_upper = [r_ci[2], K_ci[2]],
            std = [std(r_samples), std(K_samples)]
        )
    )
end

# ============================================================================
# Bayesian Demographic-Structural Model
# ============================================================================

"""
    bayesian_dst(data::DataFrame; n_samples=500, n_chains=4, priors=nothing)

Bayesian parameter estimation for the Demographic-Structural Theory model.

Estimates posterior distributions for growth rate `r`, carrying capacity `K`,
and elite dynamics parameters from observed time series data.

# Arguments
- `data::DataFrame`: Must have columns `:year`, `:population`, `:elites`, `:state_capacity`
- `n_samples::Int`: MCMC samples per chain (default: 500)
- `n_chains::Int`: Number of chains (default: 4)
- `priors`: Optional NamedTuple of prior distributions

# Returns
NamedTuple with:
- `chain::Chains`: Full posterior samples
- `params::NamedTuple`: Posterior means for `(r, K, w, δ, ε)`
- `summary::DataFrame`: Summary statistics

# Example
```julia
using Cliodynamics, Turing, DataFrames

data = DataFrame(year=[0,50,100,150,200], population=[500,700,900,850,600],
                 elites=[10,15,25,40,20], state_capacity=[100,120,80,50,90])
result = bayesian_dst(data)
```
"""
function Cliodynamics.bayesian_dst(data::DataFrame;
                                    n_samples::Int=500, n_chains::Int=4,
                                    priors=nothing)
    for col in [:year, :population, :elites, :state_capacity]
        hasproperty(data, col) || throw(ArgumentError("DataFrame must have column :$col"))
    end

    years = Float64.(data.year)
    pop_obs = Float64.(data.population)
    elite_obs = Float64.(data.elites)
    state_obs = Float64.(data.state_capacity)

    N0, E0, S0 = pop_obs[1], elite_obs[1], state_obs[1]
    t_data = years .- years[1]

    @model function dst_bayes(pop, elites, state, t, N0, E0, S0)
        r ~ Truncated(Normal(0.015, 0.02), 0.001, 0.1)
        K ~ Truncated(Normal(maximum(pop) * 1.5, maximum(pop)), 0.0, Inf)
        w ~ Truncated(Normal(2.0, 1.0), 0.1, 10.0)
        δ ~ Truncated(Normal(0.03, 0.02), 0.001, 0.2)
        ε ~ Truncated(Normal(0.001, 0.001), 0.0001, 0.01)
        σ ~ InverseGamma(2.0, 3.0)

        params = DemographicStructuralParams(r=r, K=K, w=w, δ=δ, ε=ε,
                                              N0=N0, E0=E0, S0=S0)
        try
            sol = demographic_structural_model(params, tspan=(0.0, t[end]))

            for i in eachindex(t)
                state_i = sol(t[i])
                pop[i] ~ Normal(state_i[1], σ * abs(state_i[1]) + 1.0)
                elites[i] ~ Normal(state_i[2], σ * abs(state_i[2]) + 0.1)
                state[i] ~ Normal(state_i[3], σ * abs(state_i[3]) + 1.0)
            end
        catch
            # ODE solver failure — reject this sample
            Turing.@addlogprob! -Inf
        end
    end

    model = dst_bayes(pop_obs, elite_obs, state_obs, t_data, N0, E0, S0)
    chain = sample(model, NUTS(), MCMCThreads(), n_samples, n_chains)

    param_names = [:r, :K, :w, :δ, :ε]
    means = Dict(p => mean(vec(chain[p].data)) for p in param_names)
    cis = Dict(p => quantile(vec(chain[p].data), [0.025, 0.975]) for p in param_names)

    return (
        chain = chain,
        params = (r = means[:r], K = means[:K], w = means[:w],
                  δ = means[:δ], ε = means[:ε]),
        summary = DataFrame(
            parameter = string.(param_names),
            mean = [means[p] for p in param_names],
            ci_lower = [cis[p][1] for p in param_names],
            ci_upper = [cis[p][2] for p in param_names],
            std = [std(vec(chain[p].data)) for p in param_names]
        )
    )
end

# ============================================================================
# Bayesian Model Comparison
# ============================================================================

"""
    bayesian_model_comparison(chains::Vector{Chains}, names::Vector{String})

Compare Bayesian models using the Widely Applicable Information Criterion (WAIC).

Lower WAIC indicates better predictive performance. The function computes
WAIC from the log-likelihood samples in each chain.

# Arguments
- `chains::Vector{Chains}`: MCMC chains from different models
- `names::Vector{String}`: Model names for labeling

# Returns
`DataFrame` with columns `:model`, `:waic`, `:se`, `:rank`

# Example
```julia
result1 = bayesian_malthusian(years, pop)
result2 = bayesian_dst(data)
comparison = bayesian_model_comparison(
    [result1.chain, result2.chain],
    ["Malthusian", "DST"]
)
```
"""
function Cliodynamics.bayesian_model_comparison(chains::Vector, names::Vector{String})
    length(chains) == length(names) || throw(ArgumentError("chains and names must have same length"))

    results = DataFrame(model=String[], waic=Float64[], se=Float64[], rank=Int[])

    waics = Float64[]
    for (i, chain) in enumerate(chains)
        # Compute WAIC approximation from log-posterior
        lp = vec(chain[:lp].data)
        n = length(lp)

        # WAIC ≈ -2 * (mean log-likelihood - variance of log-likelihood)
        waic = -2.0 * (mean(lp) - var(lp))
        se = 2.0 * std(lp) / sqrt(n)
        push!(waics, waic)
        push!(results, (model=names[i], waic=waic, se=se, rank=0))
    end

    # Assign ranks (lower WAIC = better = rank 1)
    sorted_idx = sortperm(waics)
    for (rank, idx) in enumerate(sorted_idx)
        results.rank[idx] = rank
    end

    sort!(results, :rank)
    return results
end

end # module CliodynamicsTuringExt
