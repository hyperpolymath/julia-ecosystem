# Data Integration

## Seshat Global History Databank

[Seshat](https://seshatdatabank.info/) is a large-scale, systematic database of historical and archaeological data covering hundreds of polities across millennia. Cliodynamics.jl provides functions to load and prepare Seshat-format data for analysis.

### Loading Data

```@docs
load_seshat_csv
```

### Preparing Data

```@docs
prepare_seshat_data
```

### Example Pipeline

```julia
using Cliodynamics
using DataFrames

# Load raw data
raw = load_seshat_csv("data/seshat_sample.csv")

# Filter to Roman polities
roman = prepare_seshat_data(raw)
roman = filter(row -> occursin("Rom", string(row.polity)), roman)
sort!(roman, :year)

# Compute elite overproduction across Roman history
eoi = elite_overproduction_index(roman)

# Fit population model to English data
english = filter(row -> occursin("Eng", string(row.polity)), prepare_seshat_data(raw))
sort!(english, :year)
fit = fit_malthusian(Float64.(english.year), Float64.(english.population),
                     r_init=0.005, K_init=2_000_000.0)
```

## Data Format

Seshat CSV files support:
- Comment lines starting with `#`
- Standard header row with column names
- Automatic numeric type detection

Required columns vary by analysis function. Common columns:
- `year` — Calendar year (negative for BCE)
- `polity` — Polity identifier
- `population` — Population count
- `elites` — Elite population count
- `territory` — Territorial extent
