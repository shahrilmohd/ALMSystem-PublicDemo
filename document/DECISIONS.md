# DECISIONS.md — Financial and Actuarial Modelling Decisions

> **Version:** Public Demo (redacted from production)
> **Last Updated:** April 2026

> **Purpose:** Records *why* key modelling choices were made, not just *what* was built.
>
> **Note:** This is the public-facing decisions document. Sections covering BPA mortality
> assumptions, MA calibration methodology, IFRS 17 CSM mechanics, SCR correlation parameters,
> and bonus strategy calibration are omitted to protect proprietary implementations.
> The decisions presented here are representative of the modelling philosophy and
> standards applied throughout the production system.
>
> Further detail is available to qualified parties under NDA.
> Contact [mohd.shahrils@yahoo.com](mailto:mohd.shahrils@yahoo.com)

---

## Table of Contents

1. [Bond Accounting Basis](#1-bond-accounting-basis)
2. [Effective Interest Rate Method](#2-effective-interest-rate-method)
3. [ResultStore Indexing](#3-resultstore-indexing)
4. [Seriatim vs Group Model Points](#4-seriatim-vs-group-model-points)
5. [Opening Balance Sheet: Asset-Liability Mismatch](#5-opening-balance-sheet-asset-liability-mismatch)
6. [DeterministicRun Two-Pass BEL Design](#6-deterministicrun-two-pass-bel-design)
7. [ESG Scenario Data Structure](#7-esg-scenario-data-structure)
8. [StochasticRun Design](#8-stochasticrun-design)
9. [IFRS 17 Engine Architecture](#9-ifrs-17-engine-architecture)
10. [Storage Layer Design](#10-storage-layer-design)
11. [AI Layer Design](#11-ai-layer-design)
12. [IFRS 17 Computation Scope](#12-ifrs-17-computation-scope)

---

## 1. Bond Accounting Basis

### Decision
Each bond carries its own `accounting_basis` field (AC, FVTPL, or FVOCI).
This is a bond-level property, not a model-level switch.

### Why
A realistic insurer balance sheet holds bonds under multiple accounting bases simultaneously.
Held-to-maturity and annuity-matching portfolios are typically AC.
Trading portfolios and shorter-duration holdings are typically FVTPL.
A single model-level toggle would be incorrect — it would force the entire portfolio onto
one basis, which does not reflect reality and would produce wrong capital numbers.

### Implications
All downstream calculations — P&L, balance sheet values, spread stress, rebalancing —
must branch on `accounting_basis` at the individual bond level. Results are aggregated
only after per-bond calculations are complete. Aggregating early and splitting later
is prohibited.

### What changes between bases

| Item | AC | FVTPL | FVOCI |
|---|---|---|---|
| Balance sheet asset value | Amortised book value | Market value | Market value |
| Investment income in P&L | EIR income | Coupon + MV movement | EIR income |
| Unrealised G/L | Disclosed only, not in P&L | Hits P&L immediately | Goes to OCI reserve |
| Surplus volatility | Low — smoothed | High — market-driven | Low — smoothed |
| Matching Adjustment eligibility | Yes | No | No |

### Immutability rule
Accounting basis is designated at bond inception and cannot change during projection.
Under IFRS 9, reclassification is only permitted in rare circumstances and triggers
a full portfolio review. The model does not support mid-projection reclassification.

---

## 2. Effective Interest Rate Method

### Decision
EIR amortisation applies to AC bonds only. EIR is calculated once at purchase from the
bond's cash flows and locked for the life of the bond.

### Why
The EIR method is required under IFRS 9 for amortised cost instruments. It spreads the
purchase discount or premium evenly over the bond's life using the internal rate of return
of cash flows at the purchase price. This produces a stable, predictable income stream
and ensures book value converges to par at maturity regardless of market movements.

### Mechanics

```
EIR = IRR of [purchase_price, coupon_1, coupon_2, ..., coupon_n + par]

Each period:
  new_book_value = old_book_value + (old_book_value × EIR) - coupon_paid
```

### What EIR is NOT
EIR is not the current yield. It is not recalculated when market conditions change.
It is not the coupon rate. It is the yield-to-maturity at the original purchase date,
applied consistently for the entire projection.

---

## 3. ResultStore Indexing

### Decision
`ResultStore` uses a multi-level index of `(run_id, scenario_id, cohort_id, timestep)`.
`cohort_id` is optional and defaults to `"all"` for non-BPA runs.

### Why
A flat result table keyed only by `(run_id, timestep)` cannot distinguish multiple
scenarios in a stochastic run, nor multiple cohorts in a BPA run. A hierarchical index
allows a single store to hold results from all run types without schema changes.
Cohort granularity is added as an optional dimension so conventional runs bear no overhead.

---

## 4. Seriatim vs Group Model Points

### Decision
All liability models receive a unified DataFrame regardless of whether model points are
seriatim (one row per policy) or group (one row per cell). The distinction is resolved
by the data loader before the engine sees the data.

### Why
Mixing seriatim and group awareness into the liability engine would create two code paths
for the same calculation. The engine should not need to know the source granularity —
only the data shape matters. Group model points use a `weight` column (sum of lives);
seriatim model points set `weight = 1` per row. The liability model treats both identically.

### Stochastic constraint
Stochastic runs prohibit seriatim model points. With thousands of scenarios, a
seriatim run would be computationally infeasible. The data loader enforces this constraint;
`StochasticRun` validates at startup.

---

## 5. Opening Balance Sheet: Asset-Liability Mismatch

### Decision
If the initial asset portfolio value does not equal the opening BEL, the difference is
held as a cash position. A validation threshold warns if the mismatch exceeds 5% of BEL.
The model does not automatically rebalance to eliminate the mismatch.

### Why
In practice, an ALM model is initialised from actual balance sheet data which will
never perfectly balance to the BEL due to rounding, data lag, and market movements.
Forcing a match would mask real economic mismatches and distort the projection.
Cash is the correct residual asset — it is liquid, easily modelled, and observable.
The 5% threshold is a data quality gate, not an auto-correction.

---

## 6. DeterministicRun Two-Pass BEL Design

### Decision
`DeterministicRun` executes two passes. Pass 1 projects liabilities only to compute
the BEL. Pass 2 uses the BEL as input to set the discount curve, then runs the full
asset + liability projection.

### Why
The MA-adjusted discount curve depends on the MA benefit, which depends on the
asset portfolio. But the asset rebalancing strategy depends on the liability duration,
which requires a BEL. This circular dependency is resolved by the two-pass approach:
Pass 1 gives a BEL estimate; Pass 2 uses a properly calibrated discount curve.
This is consistent with how BPA MA calculations are performed in practice.

---

## 7. ESG Scenario Data Structure

### Decision
ESG scenarios are stored as a `ScenarioStore` containing a list of `EsgScenario`
dataclasses. Each scenario has columns: `t` (timestep), `risk_free_rate`,
`equity_return`, `cpi_annual_rate`, `rpi_annual_rate`. Scenarios are loaded from
Parquet files and validated on load.

### Why
Parquet gives efficient columnar storage and fast random access across many scenarios.
The schema is fixed and validated at load time so the engine never encounters
malformed scenario data mid-run. CPI/RPI columns are optional and default to zero,
allowing the same schema to serve both conventional and BPA runs without divergence.

---

## 8. StochasticRun Design

### Decision
`StochasticRun` iterates over N scenarios. Each scenario runs the same projection
loop as `DeterministicRun` but with scenario-specific risk factors from `ScenarioStore`.
TVOG is computed post-run as the difference between the stochastic mean BEL and
the deterministic BEL.

### Why
TVOG under the real-world measure requires a distribution of BELs across scenarios.
Running each scenario independently (rather than batching) simplifies debugging and
ensures that single-scenario failures are isolated. The vectorised path (`use_vectorised`)
batches model point steps within a single scenario using JAX `vmap` — it does not
batch across scenarios, which preserves scenario isolation.

---

## 9. IFRS 17 Engine Architecture

### Decision
The IFRS 17 engine (`GmmEngine`) is a standalone component, injected into `BPARun`
and called at each accounting period boundary. It does not participate in the monthly
time loop directly. IFRS 17 state (CSM, RA, LRC) is persisted to `Ifrs17StateStore`
at each period end to support roll-forward accounting across multiple valuation dates.

### Why
IFRS 17 reporting is on an annual or quarterly accounting basis, while the liability
projection runs monthly. Conflating the two loops would force monthly IFRS 17 updates
and significantly increase computational cost with no actuarial benefit. Separating them
allows the time loop to run at the required monthly granularity while IFRS 17
aggregation occurs only at period boundaries.

Persistent state is necessary because IFRS 17 movements (CSM unlocking, experience
variance, assumption changes) are cumulative. A stateless GMM engine would need to
re-project from inception at every valuation date, which is impractical for live runs.

### Components
- `GmmEngine` — orchestrates the GMM step: CSM unlock, experience variance, RA release
- `CsmTracker` — accumulates CSM balance and tracks locked-in assumptions
- `LossComponentTracker` — monitors onerous contracts and loss reversal
- `CostOfCapitalRA` — Risk Adjustment under the cost-of-capital method
- `CoverageUnitProvider` — allocates CSM release by coverage unit

---

## 10. Storage Layer Design

### Decision
All database access goes through repository classes only. No code outside the
repositories calls SQLAlchemy directly. The repository interface is defined in terms
of domain objects (run configs, result dataframes) not ORM models.

### Why
The repository pattern decouples the engine and API from the database implementation.
This allows the database backend to be swapped (SQLite → PostgreSQL) without
touching the engine or API code. It also makes testing easier: tests inject in-memory
repositories without needing a real database connection.

---

## 11. AI Layer Design

### Decision
The AI layer uses a multi-agent architecture with a central orchestrator routing
queries to specialist agents. LLM provider is configurable (Anthropic Claude,
OpenAI-compatible endpoint, or in-house model). Agents have access to a tool
framework that provides read-only access to run configs, results, assumption tables,
and code module interfaces.

### Why
A single monolithic LLM prompt handling all actuarial questions (IFRS 17, SII, BPA,
data quality, regulatory research) would perform poorly — context dilution reduces
accuracy on specialist topics. Routing to specialist agents with narrow, deep context
produces better answers. The configurable provider design ensures the system is not
locked to a single LLM vendor as the market evolves.

Read-only tool access is enforced as a safeguard: the AI assistant can explain and
analyse but cannot modify production data.

---

## 12. IFRS 17 Computation Scope

### Decision
IFRS 17 GMM calculations run on the deterministic scenario only. Stochastic IFRS 17
(i.e. running GMM across all ESG scenarios) is not implemented.

### Why
IFRS 17 under the GMM is a deterministic standard. The standard requires locked-in
assumptions at inception and current assumptions at each reporting date — both of which
are single deterministic sets. Running GMM across stochastic scenarios would produce
a distribution of CSM outcomes that has no IFRS 17 interpretation.

Where stochastic inputs are needed (e.g. for the RA under a stochastic approach),
these are handled separately outside the GMM loop.

---

*Detailed decisions covering BPA mortality basis, MA calibration methodology,
IFRS 17 CSM mechanics, SCR stress calibration, and bonus strategy parameters
are available to qualified parties under NDA.
Contact [mohd.shahrils@yahoo.com](mailto:mohd.shahrils@yahoo.com)*
