# ALM System — Actuarial Glossary

Hand-curated reference for concepts used in AI agent system prompts.
Covers terms not already explained in detail within DECISIONS.md.

---

## Run Types

| Run type | Description |
|---|---|
| `LIABILITY_ONLY` | Projects policyholder liability cash flows only. No asset model. Produces BEL per timestep. |
| `DETERMINISTIC` | Projects both liabilities and assets under a single deterministic economic scenario. Produces BEL, asset values, and fund surplus per timestep. |
| `STOCHASTIC` | Runs the full projection N times, once per ESG scenario. Produces BEL per scenario per timestep, and aggregates into TVOG. |

---

## Key Output Metrics

### BEL — Best Estimate Liability
The present value of all future net policyholder cash outflows, discounted at the risk-free rate.

- Calculated at each timestep using a **backward pass**: once all future cash flows are known from the forward projection, BEL at time t is the sum of all future net outgos discounted back to t.
- A higher BEL means more money is needed to meet future obligations.
- BEL decreases over time as policies lapse, die, or mature (the book runs off).
- For stochastic runs, BEL varies by scenario — higher in adverse scenarios (low rates, low lapses).

### TVOG — Time Value of Options and Guarantees
The additional liability cost arising from the asymmetric nature of policyholder options (e.g. surrender options, guaranteed bonuses).

- Calculated as: `TVOG = Mean(Stochastic BEL) − Deterministic BEL`
- A positive TVOG means the stochastic scenarios on average produce a higher liability than the deterministic base case — policyholders benefit more from guarantees in adverse conditions.
- TVOG is only meaningful for STOCHASTIC runs.
- TVOG is zero or near-zero for pure protection products with no embedded options.

### Net Outgo
Cash out minus cash in for one timestep:
`net_outgo = claims + expenses − premiums + surrenders`

A positive net outgo means the fund paid out more than it received that month.

### Duration / Projection Term
The number of monthly timesteps in the projection. A 10-year run has 120 timesteps.

---

## Liability Decrements

Policies leave the in-force book through three decrements:

| Decrement | Description |
|---|---|
| **Mortality** | Policyholder dies. Death benefit (sum assured) paid. |
| **Lapse** | Policyholder stops paying premiums and abandons the policy. No surrender value paid (pure lapse). |
| **Surrender** | Policyholder voluntarily exits and receives the surrender value (a fraction of the reserve). |
| **Maturity** | Policy reaches its term and the maturity benefit is paid. Not a decrement in the strict sense — it is the scheduled end of the policy. |

---

## Assumption Tables

The `ConventionalAssumptions` object holds all per-policy assumption rates:

| Table | Key | Meaning |
|---|---|---|
| `mortality_rates` | attained age (int) | Probability of death per year at each age, q_x |
| `lapse_rates` | policy duration (int, years) | Probability of lapse per year at each duration |
| `surrender_value_factors` | policy duration (int, years) | Surrender value as a fraction of the policy reserve |
| `expense_pct_premium` | scalar | Expense loading as a percentage of annual premium |
| `expense_per_policy` | scalar | Fixed expense per policy per year |
| `bonus_rate_yr` | scalar | Annual bonus rate added to sum assured (participating / PAR products only) |

---

## Run Configuration — Key Fields

| Field | Meaning |
|---|---|
| `run_name` | Human-readable label for the run |
| `run_type` | LIABILITY_ONLY, DETERMINISTIC, or STOCHASTIC |
| `valuation_date` | The date at which the projection starts (t = 0) |
| `projection_term_years` | How many years to project forward |
| `n_scenarios` | Number of ESG scenarios (STOCHASTIC only) |
| `input_sources.model_points` | File path to the CSV/Excel model points table |
| `input_sources.assumption_tables.tables_root_dir` | Folder containing assumption CSV files |
| `input_sources.fund_config_path` | YAML file defining the asset portfolio and investment strategy |

---

## ESG Scenarios

ESG (Economic Scenario Generator) scenarios provide stochastic paths for:
- Risk-free interest rates (one rate per timestep per scenario)
- Equity returns (one return per timestep per scenario)

Each scenario is an independent simulation. For N scenarios there are N independent
interest rate paths and N independent equity return paths.

In adverse scenarios: lower rates → higher BEL (future cash flows discounted less).
In benign scenarios: higher rates → lower BEL.

---

## Status Lifecycle

| Status | Meaning |
|---|---|
| `PENDING` | Run accepted by API; not yet picked up by a worker |
| `RUNNING` | Worker has started the projection |
| `COMPLETED` | Projection finished successfully; results available |
| `FAILED` | Projection ended with an error; see `error_message` |

---

## Batch Runs

A batch is a collection of runs submitted together, typically used to:
- Compare different assumption sets side by side
- Run sensitivity tests (vary one assumption, hold others constant)
- Submit multiple scenarios for regulatory reporting

Each run in a batch is independent. The batch status is COMPLETED only when all member runs are COMPLETED.
