# DECISIONS.md — Financial and Actuarial Modelling Decisions

> **Purpose:** Records *why* key modelling choices were made, not just *what* was built.
> Model Developers and all team members should read this before making changes to the engine.
>
> **How to use this file:**
> - When adding a new feature (e.g., IFRS17 metrics), add a new section here explaining
>   the modelling approach and key decisions before writing any code.
> - When changing an existing approach, update the relevant section and note the reason.
> - Never create a DECISIONS_v2.md — this file is the single source of financial truth.
>   Use Git history to see how decisions evolved.
>
> **Companion document:** `document/ALM_Architecture.md` describes *what* exists and *how*
> it is structured. This document explains *why*.

---

## Table of Contents

1. [Bond Accounting Basis](#1-bond-accounting-basis)
2. [Effective Interest Rate Method](#2-effective-interest-rate-method)
3. [Calibration Spread](#3-calibration-spread)
4. [Credit Spread vs Calibration Spread](#4-credit-spread-vs-calibration-spread)
5. [Credit Spread Stress Testing](#5-credit-spread-stress-testing)
6. [Default Allowance](#6-default-allowance)
7. [Rebalancing Constraints on AC Bonds](#7-rebalancing-constraints-on-ac-bonds)
8. [ResultStore Indexing](#8-resultstore-indexing)
9. [Seriatim vs Group Model Points](#9-seriatim-vs-group-model-points)
10. [Opening Balance Sheet: Asset-Liability Mismatch](#10-opening-balance-sheet-asset-liability-mismatch)
11. [Fund Step Ordering: Rebalance Before step_time](#11-fund-step-ordering-rebalance-before-step_time)
12. [DeterministicRun Two-Pass BEL Design](#12-deterministicrun-two-pass-bel-design)
13. [ESG Scenario Data Structure and ScenarioStore](#13-esg-scenario-data-structure-and-scenariostore)
14. [StochasticRun Design](#14-stochasticrun-design)
15. [BaseLiability Multi-Decrement Interface](#15-baseliability-multi-decrement-interface)
16. [ESG Schema Inflation Extension](#16-esg-schema-inflation-extension)
17. [ResultStore BPA Cohort Granularity](#17-resultstore-bpa-cohort-granularity)
18. [BPA Product Scope and Phase Placement](#18-bpa-product-scope-and-phase-placement)
19. [BPA Mortality Basis](#19-bpa-mortality-basis)
20. [LPI Option Treatment](#20-lpi-option-treatment)
21. [BPA BEL Discounting: Pre-MA and Post-MA](#21-bpa-bel-discounting-pre-ma-and-post-ma)
22. [Matching Adjustment Eligibility Definition](#22-matching-adjustment-eligibility-definition)
23. [Fundamental Spread Source and Computation](#23-fundamental-spread-source-and-computation)
24. [IFRS 17 Measurement Model for BPA](#24-ifrs-17-measurement-model-for-bpa)
25. [Deferred Member Retirement and Decrement Assumptions](#25-deferred-member-retirement-and-decrement-assumptions)
26. [Dependant Proportion and Joint-Life Assumptions](#26-dependant-proportion-and-joint-life-assumptions)
27. [Hybrid Timestep Design for BPA](#27-hybrid-timestep-design-for-bpa)
19. [BPA Mortality Basis](#19-bpa-mortality-basis) *(not in public demo)*
20. [LPI Option Treatment](#20-lpi-option-treatment) *(not in public demo)*
21. [BPA BEL Discounting: Pre-MA and Post-MA](#21-bpa-bel-discounting-pre-ma-and-post-ma) *(not in public demo)*
22. [Matching Adjustment Eligibility Definition](#22-matching-adjustment-eligibility-definition) *(not in public demo)*
23. [Fundamental Spread Source and Computation](#23-fundamental-spread-source-and-computation) *(not in public demo)*
24. [IFRS 17 Measurement Model for BPA](#24-ifrs-17-measurement-model-for-bpa) *(not in public demo)*
25. [Deferred Member Retirement and Decrement Assumptions](#25-deferred-member-retirement-and-decrement-assumptions) *(not in public demo)*
26. [Dependant Proportion and Joint-Life Assumptions](#26-dependant-proportion-and-joint-life-assumptions) *(not in public demo)*
27. [Hybrid Timestep Design for BPA](#27-hybrid-timestep-design-for-bpa) *(not in public demo)*
28. [IFRS 17 Engine Architecture](#28-ifrs-17-engine-architecture)
29. [Storage Layer Design](#29-storage-layer-design)
30. [AI Layer Design (Step 15)](#30-ai-layer-design-step-15)
31. [Asset Data Loader (Step 15a)](#31-asset-data-loader-step-15a)
32. [Stochastic Vectorisation Design](#32-stochastic-vectorisation-design)
33. [IFRS 17 Persistent State and Roll-Forward Accounting](#33-ifrs-17-persistent-state-and-roll-forward-accounting)
34. [IFRS 17 Assumption Attribution — Locked-In vs Current](#34-ifrs-17-assumption-attribution--locked-in-vs-current)
35. [Coverage Unit Definition for BPA](#35-coverage-unit-definition-for-bpa) *(not in public demo)*
36. [IFRS 17 Computation Scope — Deterministic Scenario Only](#36-ifrs-17-computation-scope--deterministic-scenario-only)
37. [IFRS 17 State and Movement Storage Tables](#37-ifrs-17-state-and-movement-storage-tables)
38. [IFRS 17 Contract Boundary — BPA](#38-ifrs-17-contract-boundary--bpa) *(not in public demo)*
39. [Reinsurance — Longevity Swap Design Hook](#39-reinsurance--longevity-swap-design-hook) *(not in public demo)*
40. [IFRS 17 Transition](#40-ifrs-17-transition) *(not in public demo)*
41. [Experience Variance — Past Service Attribution](#41-experience-variance--past-service-attribution) *(not in public demo)*
42. [Expense Loading — Product-Level Attributable Expenses](#42-expense-loading--product-level-attributable-expenses) *(not in public demo)*
43. [Multi-Cohort Aggregation](#43-multi-cohort-aggregation) *(not in public demo)*
44. [BPA Deal Registry and Model Point Identification](#44-bpa-deal-registry-and-model-point-identification) *(not in public demo)*
45. [BPA Deal Registry — Implementation](#45-bpa-deal-registry--implementation) *(not in public demo)*
46. [BPA Investment Strategy — Three-Strategy Design](#46-bpa-investment-strategy--three-strategy-design) *(not in public demo)*

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

### EIR for existing bonds vs new purchases

**Existing bonds (loaded from asset data file):**
The EIR is read directly from the input file. It was locked at the original purchase date
and must not be recomputed. The `eir` constructor parameter accepts this value directly.

**New bonds purchased during projection (e.g. reinvestment of maturing proceeds):**
The EIR is computed at the purchase timestep using the current `RiskFreeRateCurve`.
The process is:
1. Derive `initial_book_value` by discounting all future cash flows at `(rf_rate + calibration_spread)`.
   This is the fair purchase price implied by the current yield environment.
2. Solve the flat YTM from that purchase price via `scipy.brentq` (IFRS 9 constant-rate requirement).
3. Lock that flat rate as the EIR for the bond's lifetime.

This two-step approach satisfies both requirements simultaneously:
- The purchase price is curve-consistent (respects the current interest rate environment)
- The EIR is a constant flat rate (as required by IFRS 9 amortised cost accounting)

### Monthly coupon compounding note
When coupons are paid monthly (as in this model), a par bond (purchase price = face value)
will have an EIR slightly above the stated coupon rate. This is mathematically correct:
monthly coupon payments arrive earlier than annual payments, making the effective yield
marginally higher. The difference for a 5% bond over 36 months is approximately 0.12%.

### Numerical anchor
Bond purchased at 95, par 100, 5% annual coupon, 3 years. EIR ≈ 6.9%.
Year 1: BV moves from 95.00 → 96.55. Year 2: 96.55 → 98.22. Year 3: 98.22 → 100.00.
Total EIR income over 3 years = 20.00 = 3 coupons (15) + discount unwind (5). Correct.
(DECISIONS.md anchor uses annual coupon payments for simplicity. Monthly model results
match to within ~£0.15 per £100 par; total income anchor is timing-invariant.)

---

## 3. Calibration Spread

### Decision
At the valuation date, solve a single parallel spread shift to the risk-free discount curve
that equates discounted bond cash flows to the market value in the input data.
This spread is locked for the entire projection.

### Why
The risk-free discount curve alone will not reprice bonds to their market values because
market prices embed credit risk, liquidity premia, and other factors beyond the risk-free rate.
The calibration spread is the model's way of capturing "what additional yield does the market
require for this specific bond." Locking it at the valuation date preserves consistency —
we do not assume the market's view of credit risk changes unless we explicitly run a stress.

### Mechanics
```
Solve s such that: Σ [CF_t / (1 + r_t + s)^t] = Market Value at valuation date
where r_t = risk-free rate at tenor t
```

### What the calibration spread captures
The calibration spread implicitly contains:
- Expected credit loss (probability of default × loss given default)
- Liquidity premium
- Any residual pricing factors not in the risk-free curve

### What it does NOT do
The calibration spread is not a term-structure-aware credit model. It is a single parallel
shift — all tenors receive the same addition. This is a deliberate simplification for Phase 1.
A term-structure spread model is a potential Phase 2 enhancement.

---

## 4. Credit Spread vs Calibration Spread

### Decision
These are treated as distinct concepts in the model. They are not interchangeable.

### Why the distinction matters

**Credit spread** (market observable):
- The yield differential between a corporate bond and an equivalent-maturity government bond
- Quoted in markets, changes daily
- Reflects aggregate market view of default and liquidity risk for an issuer/rating

**Calibration spread** (model construct):
- The parallel shift this model needs to match market value given this model's discount curve
- Computed internally, not quoted externally
- Will differ from quoted credit spread because:
  - Our risk-free curve may not match the exact government benchmark in the quoted spread
  - Calibration spread is a single parallel shift; quoted credit spread is term-structure-aware
  - Liquidity premia are absorbed into calibration spread but may not be in the quoted spread

### In practice
They will be numerically close for investment-grade bonds with standard maturities.
The difference matters most for longer-dated bonds, non-standard structures, or
bonds trading at significant discounts.

---

## 5. Credit Spread Stress Testing

### Decision
Credit spread stresses must simultaneously reprice assets AND adjust liability discount rates.
They must not be modelled as a simple addition to the discount rate in isolation.

### Why
When credit spreads widen:
1. Bond market values fall (asset side)
2. The Matching Adjustment or Volatility Adjustment increases (liability side)
3. The liability discount rate increases → liability present value falls
4. Net own funds impact depends on the relative magnitude of both movements

For an AC-matched annuity portfolio, the MA offset is designed to make the net impact
near-zero for genuine spread widening (not default events). Modelling only the asset
repricing without the liability offset produces wildly overstated capital impacts.

### Implementation rules
- Spread widening and tightening shocks must be asymmetric (widening > tightening in magnitude)
- Spreads cannot go below zero — tightening is bounded
- Duration of the bond determines sensitivity: `ΔMV ≈ -duration × Δspread × MV`
- AC bonds: MV changes but book value does not — only disclosed unrealised G/L is affected
- FVTPL bonds: MV change = immediate P&L and own funds impact
- Default allowance must be re-estimated consistently with the stress (see Section 6)
- MA is only available for AC-designated portfolios — FVTPL bonds do not qualify

### What NOT to do
Do not apply a spread shock only to the asset discount rate without:
- Adjusting the MA/VA on the liability side
- Re-estimating the default allowance on future cash flows
- Checking for asymmetry between up and down shocks

---

## 6. Default Allowance

### Decision
Default allowance is derived from the calibration spread using a Loss Given Default (LGD)
assumption. It represents the expected reduction in future cash flows due to issuer default.

### Why
The calibration spread implicitly compensates the investor for expected credit losses.
When projecting future asset cash flows (particularly in annuity matching models where
asset cash flows directly back liability payments), the raw contractual cash flows must
be haircut to reflect the probability that the issuer defaults and the full payment is
not received. Failing to apply this haircut overstates the asset backing and
understates the net liability position.

### Mechanics (Phase 1 simplified approach)
```
default_allowance_t = projected_CF_t × PD_t × LGD
where:
  PD_t = implied default probability from calibration spread and duration
  LGD = Loss Given Default assumption (e.g., 40% for senior unsecured)
```

### Consistency requirement
Under a credit spread stress, the calibration spread changes, which changes the implied
PD, which changes the default allowance. These must be updated consistently — a stressed
spread must produce a stressed default allowance, not the base allowance.

---

## 7. Rebalancing Constraints on AC Bonds

### Decision
AC-designated bonds must not be sold for routine SAA rebalancing.
`InvestmentStrategy` enforces this constraint unless an explicit override is set in run config.

### Why
Under IFRS 9, selling AC bonds before maturity:
1. Crystallises a gain/loss that was not previously recognised in P&L —
   creating a discontinuity in the income stream
2. May trigger a "tainting" review of the entire AC portfolio designation —
   if an entity sells AC bonds too frequently, IFRS 9 requires reclassification
   of the remaining AC portfolio to FVTPL, which has significant P&L volatility implications
3. Has different tax treatment from FVTPL sales in most jurisdictions

### Permitted AC bond sales
- Bond reaches maturity (scheduled redemption — not a sale)
- Bond issuer defaults (forced realisation)
- Explicit `force_sell_ac` override in run config (for asset disposal scenarios)

### Consequence for rebalancing logic
When the SAA target weight for bonds changes and the model would normally sell bonds
to rebalance, the strategy must:
1. Check accounting basis of each bond being considered for sale
2. Exclude AC bonds from the sell order
3. Rebalance using only FVTPL/FVOCI bonds and new purchases
4. Log a warning if the AC constraint prevents full rebalancing to target weights

---

## 8. ResultStore Indexing

### Decision
Results are indexed by `(run_id, scenario_id, timestep, asset_id, accounting_basis)`.
`asset_id` and `accounting_basis` are nullable for liability-only runs.

### Why
The accounting basis split must be preserved in stored results because:
- Regulatory reporting requires separate disclosure of AC and FVTPL/FVOCI portfolios
- Spread stress attribution requires knowing which bonds are on which basis
- OCI reserve reconciliation requires FVOCI bonds to be identifiable separately
- Future IFRS17 integration will require matching asset basis to insurance contract groupings

Aggregating AC and FVTPL results before storage would permanently destroy this information
and make audit and attribution impossible.

---

## 9. Seriatim vs Group Model Points

### Decision
Liability models receive a DataFrame and never know whether it came from seriatim policies
or group model points. This is a data loader concern, not a model concern.

Stochastic runs are prohibited from using seriatim data — group model points only.

### Why
Seriatim runs over 1,000 ESG scenarios with 50,000 policies would be computationally
infeasible for a small actuarial team's system. Group model points aggregate policies with
similar characteristics into representative points, reducing the liability calculation
cost by orders of magnitude while preserving sufficient accuracy for stochastic distributions.

The prohibition is enforced in `stochastic_run.py`, not left to user discipline.

---

## 10. Opening Balance Sheet: Asset-Liability Mismatch

### Decision
The opening surplus (total asset market value minus BEL at valuation date) is initialised
into a cash account and projected forward as a first-class fund component. Assets and
liabilities are never scaled or modified to force a match.

### Why
Asset values come from a pricing system or custodian. Liability values are calculated
by the actuarial model. They are produced independently and will almost never agree
exactly. The difference is the fund's opening surplus — real capital that earns returns,
absorbs losses, and should be visible in all output reports.

Zeroing or hiding this difference is the most common ALM implementation mistake.
It makes the model appear balanced while concealing the true capitalisation of the fund,
which distorts projected solvency ratios, bonus crediting capacity, and stress test results.

Scaling assets would destroy EIRs and calibration spreads computed from original market
values. Scaling liabilities would change the actuarial calculation to fit the data.
Both are wrong.

### The cash account is not a rounding bin
The cash account is an interest-bearing, first-class asset. It earns the risk-free short
rate each period. It is reported explicitly in all balance sheet outputs. A persistently
growing cash balance signals the fund is generating surplus faster than it is crediting
bonuses. A persistently declining or negative balance signals the fund is consuming capital
— both are meaningful actuarial outputs.

### Validation threshold
A mismatch exceeding 20% of BEL triggers an error before the projection starts.
This threshold is configurable in run config (`max_opening_mismatch_ratio`).
A large mismatch almost always indicates a data problem: different valuation dates,
wrong discount curve, currency mismatch, or non-fund assets included in error.

### Liability basis for surplus
Opening surplus is defined against BEL only. Risk margin, statutory reserve, and
IFRS17 measures are calculated alongside but do not affect the opening cash balance
or the projection engine.

---

## 11. Fund Step Ordering: Rebalance Before step_time

### Decision
Within `Fund.step_time()`, the per-period sequence is:

1. Collect liability cashflows and decrements.
2. Collect asset income (coupons, dividends, maturity proceeds) via `AssetModel.collect_cashflows()`.
3. Update cash balance: `cash += asset_income; cash -= net_liability_outflow`.
4. Execute forced sells (if cash < 0): call `asset.rebalance()` on FVTPL/FVOCI assets first.
5. Execute SAA rebalancing (if drift exceeds tolerance): call `asset.rebalance()` on affected assets.
6. Call `asset_model.step_time()` — advance book values for all remaining assets.
7. Call `asset_model.aggregate_pnl()` — read the period P&L.

Rebalancing (steps 4–5) happens **before** `step_time()` (step 6).

### Why
`Bond.rebalance()` accumulates `_realised_gl_this_period`. `Bond.step_time()` incorporates
that accumulated value into `_last_pnl` and then resets it to zero. If `step_time()` were
called before `rebalance()`, the realised G/L from sales made in this period would not be
captured in the current period's P&L — it would be included in the next period or lost.

The architecture specification (ALM_Architecture.md Section 6.2) lists book-value advancement
as Step 4 and rebalancing as Step 5. The implementation deviates by swapping these to
preserve the realised GL timing invariant. The financial result is identical: rebalancing
occurs intra-period using current market values; book values at period-end reflect both
the EIR amortisation and the rebalanced position.

### Implications
- `Fund.step_time()` always calls `rebalance()` before `step_time()` on assets.
- Tests must verify that realised GL appears in the **same** period as the trade.
- Do not insert a `step_time()` call between `rebalance()` and `aggregate_pnl()`.

---

## 12. DeterministicRun Two-Pass BEL Design

### Decision
`DeterministicRun.execute()` uses the same two-pass approach as `LiabilityOnlyRun.execute()`:

- **Pass 1 (forward):** At each month `t`, call `Fund.step_time()`. Collect and store the
  `FundTimestepResult` (cashflows, decrements, asset state). Advance model points to `t+1`.
- **Pass 2 (backward):** Compute BEL at each `t` as the sum of discounted future `net_outgo`
  values from Pass 1. Uses plain Python float arithmetic — no pandas overhead.

### Why
BEL at month `t` requires knowing all future liability cashflows. The cleanest way to obtain
these without re-running the full Fund loop at every timestep (O(N²)) is to collect all
cashflows in a single forward pass, then compute BEL values in a backward summation (O(N)).
For a 120-month projection this is 7,260 float multiplications — microseconds, not minutes.

The asset model is run only once (in Pass 1) — there is no nested asset projection.
Asset state is not rewound: the model is stateful (bonds advance each period).
This matches the physical reality of a live portfolio being managed over time.

### Strategy injection
`DeterministicRun` receives a fully constructed `InvestmentStrategy` as a constructor
parameter. It does not build the strategy from `FundConfig` internally. This is consistent
with CLAUDE.md Rule 5 ("Strategies are always injected, never hardcoded") and makes
`DeterministicRun` testable without a real `FundConfig`.

### Rate curve for assets
The `AssetScenarioPoint` at each timestep is constructed by `DeterministicRun` using:
- `rate_curve`: taken from `ConventionalAssumptions.rate_curve` (same curve used for liability
  BEL discounting — one consistent risk-free curve per deterministic scenario).
- `equity_total_return_yr`: injected as a constructor parameter (`equity_return_yr: float`).

For stochastic runs (Step 9), each scenario will supply its own path of rate curves and
equity returns via `ScenarioStore`. The deterministic design is forward-compatible with this.

---

## 13. ESG Scenario Data Structure and ScenarioStore

### Decision
Each ESG scenario is a **full time-series path**: a sequence of T `AssetScenarioPoint`
objects, one per projection month, each carrying a complete `RiskFreeRateCurve` and an
annual equity total return for that month. Scenarios are stored in `ScenarioStore`, indexed
by integer `scenario_id`.

### Why a full time-series path (not a flat rate per scenario)
A real ESG generator produces a rate curve that evolves differently each month for each
scenario. A flat rate per scenario would collapse a 120-point path into a single number,
discarding the term structure dynamics that drive both asset repricing and liability
discounting. TVOG is sensitive precisely to this variation — using flat rates would produce
a TVOG of approximately zero for rate scenarios and would defeat the purpose of the
stochastic run.

### CSV input format
ESG scenario files are CSV with the following mandatory columns:

| Column | Type | Description |
|---|---|---|
| `scenario_id` | int | Unique scenario identifier. Does not need to be contiguous or start at 1. |
| `timestep` | int | Month index (0 = valuation month). One row per scenario per month. |
| `r_{n}m` | float | Spot rate at maturity `n` months (e.g., `r_1m`, `r_12m`, `r_60m`, `r_360m`). At least one required. |
| `equity_return_yr` | float | Annual total equity return for this month (decimal, e.g. 0.07 = 7%). |

Column names `r_{n}m` encode the maturity: `r_1m` = 1-month spot rate, `r_120m` = 10-year
spot rate. `ScenarioLoader` parses these dynamically, so any set of maturity knot points is
supported without code changes. The resulting `RiskFreeRateCurve` uses these
(maturity, spot rate) pairs as its knot points.

### Scenario selection
`ScenarioStore` is indexed by `scenario_id`, not by position. This allows any subset of
scenarios to be retrieved without loading all preceding scenarios. Use cases:

- **Full stochastic run:** `StochasticRun(scenario_ids=None)` → runs all scenarios in the store
- **Tail analysis:** `StochasticRun(scenario_ids=[650])` → runs only scenario 650
- **Spot check:** `StochasticRun(scenario_ids=[1, 500, 999])` → runs three specific scenarios

`num_scenarios` in `StochasticConfig` controls how many rows are loaded from the CSV file.
`scenario_ids` in `StochasticRun` controls which of the loaded scenarios are actually run.
These are independent: you can load 1,000 scenarios and run only 3 of them.

---

## 14. StochasticRun Design

`StochasticRun` repeats the two-pass Fund projection loop N times — once per ESG scenario.
The asset model is deep-copied before each scenario so every scenario starts from the
identical valuation-date portfolio state. Results are stored in `ResultStore` with
`scenario_id` as the distinguishing key. TVOG is derived as the mean stochastic BEL minus
the deterministic BEL across all scenarios.

> **Not included in public demo.**
> Implementation detail of scenario isolation, group MP enforcement, BEL discounting
> approach, progress reporting, and parallelism architecture.

---

## 15. BaseLiability Multi-Decrement Interface

### Decision
BPA liabilities extend `BaseLiability` via a new intermediate abstract class
`MultiDecrementLiability`, inserted between `BaseLiability` and all BPA-specific
liability classes. This class defines the interface for multi-decrement table logic
that BPA requires but conventional and UL products do not.

`BaseLiability` itself is not modified. No existing subclasses are affected.

### Why a separate intermediate class
Conventional and UL products operate on a single primary decrement (lapse or surrender)
with mortality handled implicitly. BPA liabilities require simultaneous projection of
multiple competing decrements at each timestep:

- Mortality (in-payment pensioners and deferred members)
- Dependant pension trigger (death of in-payment pensioner)
- Deferred retirement (conversion of deferred member to in-payment)
- Transfer value (deferred member exits before retirement)
- Commutation (partial lump sum at retirement, reducing in-payment pension)

Forcing these into `BaseLiability` would pollute the base class with BPA-specific
logic and break the clean hierarchy used by conventional and UL products. A separate
intermediate class isolates multi-decrement logic without touching existing code.

### Interface contract
`MultiDecrementLiability` exposes one additional abstract method:

```python
@abstractmethod
def get_decrements(self, t: int, model_points: pd.DataFrame) -> pd.DataFrame:
    """
    Return a DataFrame of decrement probabilities for each model point at timestep t.
    Columns must include at minimum: ['mp_id', 'q_death', 'q_retire', 'q_transfer',
                                      'q_commute'].
    All columns are floats in [0, 1]. Columns irrelevant to a subclass return 0.0.
    The sum of all decrements for a model point must not exceed 1.0 at any timestep.
    """
```

`BaseLiability.project(t, model_points)` is unchanged. The decrement DataFrame
is consumed internally by each `MultiDecrementLiability` subclass in its own
`project()` implementation.

### Class hierarchy after this decision

```
BaseLiability (abstract)
├── ConventionalLiability          ← unchanged
├── ULLiability                    ← unchanged
└── MultiDecrementLiability (abstract, new)   ← intermediate class
    ├── InPaymentLiability         ← BPA pensioners
    ├── DeferredLiability          ← BPA deferred members
    ├── DependantLiability         ← BPA contingent dependants
    └── EnhancedLiability          ← BPA impaired lives
```

### What NOT to do
- Do not add `get_decrements()` to `BaseLiability`. Conventional products must not
  be required to implement a method they have no use for.
- Do not create a parallel liability hierarchy outside `BaseLiability`. All liability
  types must be projectable by the same run mode orchestrators.
- Do not merge in-payment and deferred members into a single class with a flag.
  Their cashflow projection logic diverges substantially (see §25), and a single
  class with conditional branches is harder to test and validate independently.

---

## 16. ESG Schema Inflation Extension

The ESG CSV schema includes two optional inflation columns: `cpi_annual_rate` and
`rpi_annual_rate`. Both are floats (decimal). They are added early to avoid a schema
migration once BPA runs require them. UK DB pension benefits are often split between
RPI and CPI indexation, so both columns are necessary.

> **Not included in public demo.**
> Correlation requirements between inflation and interest rate paths, RPI/CPI distinction,
> and the updated CSV schema specification.

---

## 17. ResultStore BPA Cohort Granularity

### Decision
An optional `cohort_id: str | None` field is added to the `ResultStore` index.
The index becomes `(run_id, scenario_id, timestep, asset_id, accounting_basis, cohort_id)`.

`cohort_id` defaults to `None` for all existing non-BPA runs. No existing result
queries, tests, or output schemas are affected.

BPA runs populate `cohort_id` using `make_cohort_id(deal_id, population_type)` from
`engine/liability/bpa/registry.py` (DECISIONS.md §44, §45):

```python
cohort_id = f"{deal_id}_{population_type}"
# e.g. "AcmePension_2024Q3_pensioner"
#      "AcmePension_2024Q3_deferred"
#      "AcmePension_2024Q3_dependant"
#      "AcmePension_2024Q3_enhanced"
```

Where `deal_id` comes from `BPADealRegistry` and `population_type` is one of
`"pensioner"`, `"deferred"`, `"dependant"`, `"enhanced"`.

This design encodes both the deal and the liability class in a single stable string.
For a portfolio of multiple deals, each deal produces four `cohort_id` values —
one per population type — all distinct and independently roll-forwarded.

Fund-level aggregates are stored with `cohort_id = None`, consistent with existing
non-BPA behaviour.

### Why cohort-level granularity is necessary for BPA
Three regulatory requirements drive this:

**MA eligibility testing** — The cash-flow matching test (see §22) must be run
against the total liability cashflow schedule. But when MA eligibility is partially
met, the model must identify which liability cohorts are backed by eligible assets
and which fall into the non-MA portfolio. This requires cashflows attributable to
each cohort to be separable in the output.

**IFRS 17 GMM coverage units** — Under IFRS 17, the CSM is released over the
coverage period in proportion to coverage units. For BPA, coverage units differ
by cohort type (see §24). The CSM release schedule cannot be computed without
cohort-level liability cashflow data.

**Attribution and validation** — For a consultancy tool, client validation runs
will require comparison of pensioner BEL, deferred BEL, and dependant BEL
separately against the client's own model. Fund-level aggregates are insufficient
for this purpose.

### What NOT to do
- Do not create a separate `BPAResultStore` class. A parallel store creates two
  result retrieval interfaces, doubles the query surface, and makes cross-product
  reporting (e.g., BPA fund + conventional fund in same entity) unnecessarily complex.
- Do not store cohort results in a separate table. The `cohort_id` index dimension
  is sufficient for query-time filtering.

---

## 18. BPA Product Scope and Phase Placement

### Decision
BPA modelling is placed in Phase 3. Phase 1 (Steps 1–10) and Phase 2 (Steps 11–13)
are not extended to include BPA liability or MA logic, with three bridging exceptions
recorded in §15, §16, and §17.

Phase 3 build order for BPA:

| Phase 3 Step | What | Depends on |
|---|---|---|
| 14 | `engine/liability/bpa/` — all liability subclasses | §15 (MultiDecrementLiability) |
| 15 | `engine/matching_adjustment/` — eligibility, FS, MA benefit | §16 (ESG inflation), §22, §23 |
| 16 | MA-adjusted BEL discount curve | §21, Step 15 above |
| 17 | IFRS 17 GMM for BPA — CSM, coverage units, loss component | §24, Step 14 above |
| 18 | BPA SCR — spread stress and interest stress with MA offset | §5, Step 15 above |
| 19 | BPA cohort reporting in ResultStore | §17, all above steps |

Phase 4 (post-BPA):

| Phase 4 Step | What |
|---|---|
| 20 | Matching investment strategy — liability cashflow matching, duration matching, MA optimisation |
| 21 | AI layer |

### Liability types in scope
All four BPA liability types are in scope for Phase 3:
- In-payment pensioners
- Deferred members (with retirement, transfer value, and commutation decrements)
- Contingent dependants (joint-life survival)
- Enhanced / impaired lives (individual mortality ratings)

Excluding any type would produce a BPA tool that cannot model a real UK DB scheme in
full buyout, which is the end-state insurers are working toward.

### Why not Phase 1 or Phase 2
The existing engine was designed around unit-linked and conventional products with
dynamic asset strategies. BPA introduces a qualitatively different liability structure
(multi-decrement, inflation-linked, joint-life) and a new asset-liability coupling
(MA discount rate depends on asset composition). Building BPA before the core engine
is validated and the API/frontend layer is stable would introduce instability at the
most critical architectural layer.

The three bridging decisions (§15, §16, §17) are the minimum changes required now
to prevent Phase 3 from requiring schema migrations or base class surgery on
production code.

### Investment strategy deferral
The Phase 4 matching investment strategy (cashflow matching, duration matching, MA
portfolio optimisation) is intentionally deferred. Phase 3 BPA runs will use a
static buy-and-hold strategy — the MA portfolio is loaded from asset data and held
to maturity without active rebalancing. This is the correct modelling choice for a
closed MA portfolio and avoids implementing optimisation logic before the underlying
BPA liability and MA modules are validated.

See §46 for the full three-strategy design and the buy-and-hold injection pattern
used in `BPARun`.

---

## 19. BPA Mortality Basis

> **Not included in public demo.**
> S3 base tables, CMI improvement model, and age-rating for enhanced/impaired lives.

---

## 20. LPI Option Treatment

> **Not included in public demo.**
> Limited Price Indexation option valuation and LPI cap/floor assumptions.

---

## 21. BPA BEL Discounting: Pre-MA and Post-MA

> **Not included in public demo.**
> Pre-MA and post-MA adjusted discount curves and the two-pass BEL calibration.

---

## 22. Matching Adjustment Eligibility Definition

> **Not included in public demo.**
> Four-condition eligibility framework and highly-predictable cap (PRA PS10/24).

---

## 23. Fundamental Spread Source and Computation

> **Not included in public demo.**
> PRA fundamental spread table lookup, governance metadata, and FS computation.

---

## 24. IFRS 17 Measurement Model for BPA

> **Not included in public demo.**
> GMM measurement model applied to BPA cohorts, dual BEL wiring, and RA attribution.

---

## 25. Deferred Member Retirement and Decrement Assumptions

> **Not included in public demo.**
> Four-decrement model (death, ill-health retirement, NRA retirement, TV) and
> two-phase BEL forward projection for deferred members.

---

## 26. Dependant Proportion and Joint-Life Assumptions

> **Not included in public demo.**
> Contingent dependant pension modelling, weight convention, and convolution BEL.

---

## 27. Hybrid Timestep Design for BPA

> **Not included in public demo.**
> Monthly near-term / annual long-term timestep design and ProjectionCalendar integration.

---

## 28. IFRS 17 Engine Architecture

### Decision
Build `engine/ifrs17/` as a **standalone module** at the start of Phase 3 (Step 17).
All IFRS 17 GMM mechanics are product-agnostic and live exclusively in this module.
Products provide three inputs via injection: BEL values, a `CoverageUnitProvider`,
and the locked-in discount rate. `GmmEngine` handles everything else uniformly.

### Why standalone, not inside `engine/liability/bpa/`

The GMM mechanics — CSM accretion, coverage unit release, LRC/LIC assembly, loss
component tracking — are identical regardless of whether the underlying liability is
a BPA annuity, a conventional endowment, or a term assurance contract. Only three
things differ per product:

| Product-specific | Product-agnostic (in `engine/ifrs17/`) |
|---|---|
| BEL calculation | CSM accretion at locked-in rate |
| Coverage unit definition | CSM release by coverage unit fraction |
| Locked-in discount rate at inception | Loss component tracking |
| Contract group identification | LRC = BEL + RA + CSM assembly |
| Risk Adjustment basis | LIC = BEL_past + RA_past |

Building GMM inside `engine/liability/bpa/` would require duplication when
conventional products need IFRS 17 reporting. Duplicated mechanics create
maintenance divergence risk: a correction to CSM accretion logic applied in
one place but not the other produces silently inconsistent results across products.
A standalone module eliminates this risk entirely.

### Module structure

**See also §33 (persistent state), §34 (assumption attribution and dual BEL), §35
(coverage unit definition), §36 (stochastic scope), §37 (IFRS 17 storage tables)
for supporting decisions that complete this architecture.**

```
engine/ifrs17/
├── __init__.py
├── gmm.py              — GmmEngine: orchestrates all components per contract group
├── csm.py              — CSM accretion at locked-in rate; release by coverage units
├── loss_component.py   — Loss component tracking for onerous contract groups
├── risk_adjustment.py  — Cost of Capital method (6% CoC on longevity/lapse SCR)
├── coverage_units.py   — CoverageUnitProvider protocol (injected per product)
├── state.py            — Ifrs17State dataclass; holds cross-period rolling balances
└── assumptions.py      — LockedInAssumptions, CurrentAssumptions, AssumptionProvider

storage/
└── ifrs17_state_repository.py   — Ifrs17StateStore: load/save by cohort_id + valuation_date
```

### `GmmEngine` interface

`GmmEngine` receives opening IFRS 17 state (loaded from prior period by the run mode
orchestrator via `Ifrs17StateStore`) and produces a `GmmStepResult` with closing state.
The dual BEL requirement (§34) means `step()` accepts both current and locked-in BEL
values. See §33 for why opening state must be loaded, not recomputed.

```python
class GmmEngine:
    def __init__(
        self,
        contract_groups:          list[str],
        locked_in_rates:          dict[str, float],          # {cohort_id: post-MA rate at inception}
        coverage_unit_providers:  dict[str, CoverageUnitProvider],
        opening_states:           dict[str, Ifrs17State],    # loaded from Ifrs17StateStore
    ) -> None: ...

    def step(
        self,
        cohort_id:                  str,
        t:                          int,
        bel_current:                float,   # BEL discounted at current-period rate (balance sheet)
        bel_locked:                 float,   # BEL discounted at locked-in rate (finance income split)
        risk_adjustment:            float,
        remaining_coverage_units:   float,
        fcf_change_non_financial:   float = 0.0,  # FCF change from mortality/inflation (→ adjusts CSM)
        fcf_change_financial:       float = 0.0,  # FCF change from discount rate (→ OCI)
    ) -> GmmStepResult: ...
```

`GmmStepResult` carries:

| Field | Description |
|---|---|
| `csm_opening` | CSM at start of period (from opening state) |
| `csm_accretion` | Interest accretion at locked-in rate |
| `csm_adjustment_non_financial` | FCF change for future service — non-financial assumptions |
| `csm_adjustment_financial` | FCF change for future service — financial assumptions (→ OCI if elected) |
| `csm_release` | Released to P&L by coverage unit fraction |
| `csm_closing` | CSM at end of period (saved to Ifrs17StateStore) |
| `loss_component_opening` | Opening loss component balance |
| `loss_component_release` | Amount released as onerous contracts generate cashflows |
| `loss_component_closing` | Closing loss component balance |
| `insurance_finance_pl` | BEL unwinding at locked-in rate (P&L) |
| `insurance_finance_oci` | BEL change due to current vs locked-in rate difference (OCI) |
| `lrc` | Liability for Remaining Coverage = BEL_current + RA + CSM |
| `lic` | Liability for Incurred Claims = BEL_past + RA_past |
| `p_and_l_csm_release` | P&L from CSM release |
| `p_and_l_insurance_finance` | P&L from insurance finance income/expense |

### Contract group granularity and `cohort_id`

The `cohort_id` field added to `ResultStore` in §17 is the IFRS 17 contract group
identifier. No additional `ResultStore` changes are needed.

For BPA, `cohort_id` maps directly to deal cohort:
- `"pensioner"`, `"deferred"`, `"dependant"`, `"enhanced"` — or finer granularity
  at the deal tranche level (e.g., `"pensioner_2024Q1"`) if separate locked-in rates
  are required per tranche.

For conventional products, `cohort_id` will be a combination of product code,
inception year, and profitability class at initial recognition
(e.g., `"ENDOW_2023_PROFITABLE"`). Contracts with different inception dates must
not be grouped together — each forms its own contract group with its own locked-in
rate and CSM balance (IFRS 17 para 28).

### Locked-in discount rate

The locked-in rate is internal state of `GmmEngine`, frozen at construction:

```python
locked_in_rates: dict[str, float]   # set once, immutable after __init__
```

This rate is used for CSM accretion only. BEL measurement uses the current-period
risk-free rate (as it does today). The distinction matters: using the current rate
for CSM accretion is an IFRS 17 misstatement (para 44).

For BPA, the locked-in rate is the **post-MA discount rate** at the transaction date
(§21). If a new tranche is bought in during Phase 3 validation testing, it forms a
new contract group with a new locked-in rate — it does not inherit the rate of an
earlier tranche.

The locked-in rate is an input to `GmmEngine`, not computed by it. The run mode
orchestrator (initially `BPARun`, later extended to conventional runs) is responsible
for supplying the correct rate at contract group inception.

### Risk Adjustment

The Risk Adjustment (RA) is computed using the Cost of Capital method at 6% CoC,
consistent with the Solvency II Risk Margin methodology (§24). Dual use of the same
CoC calculation for both Solvency II Risk Margin and IFRS 17 RA is a deliberate
efficiency choice — the underlying longevity SCR projection is identical.

`risk_adjustment.py` exposes a `CostOfCapitalRA` class that takes a sequence of
future SCR values and discount factors and returns the RA as a scalar. The SCR
projection feeding this class is product-specific and injected by the run mode.

### Phase placement: Phase 3 Step 17, not Phase 2

IFRS 17 is not an immediate client reporting need. Phase 2 (API, worker, storage,
desktop, AI core) is focused on making the engine accessible. Adding IFRS 17 result
schemas to Phase 2 storage and API before the GMM module exists would produce
placeholder columns with no values.

Phase 3 Step 17 placement ensures:
1. The storage and API schema (Phase 2) is stable before IFRS 17 output fields are
   added to it.
2. GMM mechanics are validated against simple conventional product test cases
   (two-period contracts with known cashflows) before BPA complexity is layered on.
3. BPA Step 20 (`BPA IFRS 17 integration`) builds on a validated `engine/ifrs17/`
   module rather than implementing GMM from scratch inside the BPA package.

### BPA IFRS 17 integration step (Phase 3 Step 20)

Step 20 is not a reimplementation of GMM. It is exclusively:
- Defining `BPACoverageUnitProvider` for each cohort type (§24)
- Wiring BPA BEL values and RA into `GmmEngine.step()`
- Extending `ResultStore` output with IFRS 17 fields per cohort
- Validation tests: reproduce known CSM release schedule from a simple BPA
  contract group with fixed mortality and fixed cashflows

### What NOT to do
- Do not build GMM logic inside `engine/liability/bpa/`. A BPA-only IFRS 17
  implementation cannot be reused for conventional products without duplication.
- Do not store the locked-in discount rate in `ResultStore`. It is a static
  assumption per contract group, not a per-timestep result.
- Do not use the current-period risk-free rate for CSM accretion. The locked-in
  rate is set at inception and never changes (IFRS 17 para 44).
- Do not aggregate contract groups with different inception dates. Each deal tranche
  is a separate contract group with its own locked-in rate and CSM balance.

---

## 29. Storage Layer Design

### Decision

Persist runs and results in a SQLAlchemy ORM backed by SQLite (development) with a
single connection-string change to switch to PostgreSQL (production).
All database access is routed exclusively through two repository classes:
`RunRepository` and `ResultRepository`. Nothing outside `storage/` imports
SQLAlchemy directly.

Results are stored at the same granularity as `ResultStore.as_dataframe()`:
one row per `(run_id, scenario_id, timestep, cohort_id)`. All 28 numeric columns
from `TimestepResult` are stored, with asset fields nullable for LIABILITY_ONLY runs.
Large result sets are written using SQLAlchemy bulk inserts (batch size 500 rows).

### Schema

#### `run_records` table (`storage/models/run_record.py`)

| Column | Type | Notes |
|---|---|---|
| `run_id` | VARCHAR PK | UUID string generated by the run mode |
| `run_type` | VARCHAR | `"LIABILITY_ONLY"`, `"DETERMINISTIC"`, `"STOCHASTIC"` |
| `status` | VARCHAR | `"PENDING"`, `"RUNNING"`, `"COMPLETED"`, `"FAILED"` |
| `created_at` | DATETIME | Set when the record is first inserted |
| `started_at` | DATETIME NULL | Set when execution begins |
| `completed_at` | DATETIME NULL | Set on COMPLETED or FAILED |
| `duration_seconds` | FLOAT NULL | Wall-clock time for the full run |
| `error_message` | TEXT NULL | First-line error string on FAILED runs |
| `config_json` | TEXT | Full `RunConfig` serialised as JSON (Pydantic `.model_dump_json()`) |
| `n_scenarios` | INTEGER NULL | Number of ESG scenarios run (STOCHASTIC only) |
| `n_timesteps` | INTEGER NULL | Number of monthly projection steps |

#### `result_records` table (`storage/models/result_record.py`)

One row per `(run_id, scenario_id, timestep, cohort_id)`. This matches
`ResultStore.as_dataframe()` exactly; every column in `RESULT_COLUMNS` is a
column here (plus a surrogate primary key).

| Column | Type | Notes |
|---|---|---|
| `id` | INTEGER PK autoincrement | Surrogate key |
| `run_id` | VARCHAR FK → run_records | Indexed |
| `scenario_id` | INTEGER | Indexed (with run_id) |
| `timestep` | INTEGER | |
| `cohort_id` | VARCHAR NULL | Phase 3 BPA; NULL for all Phase 1–2 runs |
| `bel` | FLOAT | |
| `reserve` | FLOAT | |
| `premiums` | FLOAT | |
| `death_claims` | FLOAT | |
| `surrender_payments` | FLOAT | |
| `maturity_payments` | FLOAT | |
| `expenses` | FLOAT | |
| `net_outgo` | FLOAT | |
| `in_force_start` | FLOAT | |
| `deaths` | FLOAT | |
| `lapses` | FLOAT | |
| `maturities` | FLOAT | |
| `in_force_end` | FLOAT | |
| `total_market_value` | FLOAT NULL | NULL for LIABILITY_ONLY runs |
| `total_book_value` | FLOAT NULL | NULL for LIABILITY_ONLY runs |
| `cash_balance` | FLOAT NULL | NULL for LIABILITY_ONLY runs |
| `eir_income` | FLOAT NULL | NULL for LIABILITY_ONLY runs |
| `coupon_income` | FLOAT NULL | NULL for LIABILITY_ONLY runs |
| `dividend_income` | FLOAT NULL | NULL for LIABILITY_ONLY runs |
| `unrealised_gl` | FLOAT NULL | NULL for LIABILITY_ONLY runs |
| `realised_gl` | FLOAT NULL | NULL for LIABILITY_ONLY runs |
| `oci_reserve` | FLOAT NULL | NULL for LIABILITY_ONLY runs |
| `mv_ac` | FLOAT NULL | NULL for LIABILITY_ONLY runs |
| `mv_fvtpl` | FLOAT NULL | NULL for LIABILITY_ONLY runs |
| `mv_fvoci` | FLOAT NULL | NULL for LIABILITY_ONLY runs |

Composite unique constraint on `(run_id, scenario_id, timestep, cohort_id)` mirrors
the `ResultStore` duplicate-key rule.

### Repository interfaces

#### `RunRepository`

```python
class RunRepository:
    def __init__(self, session: Session) -> None: ...

    def save(self, record: RunRecord) -> None:
        """Insert or update a RunRecord. Idempotent on run_id."""

    def get(self, run_id: str) -> RunRecord:
        """Raises KeyError if not found."""

    def list_all(self) -> list[RunRecord]:
        """All runs ordered by created_at descending."""

    def update_status(
        self,
        run_id: str,
        status: str,
        *,
        started_at: datetime | None = None,
        completed_at: datetime | None = None,
        duration_seconds: float | None = None,
        error_message: str | None = None,
    ) -> None:
        """Partial update — only non-None fields are written."""
```

#### `ResultRepository`

```python
class ResultRepository:
    def __init__(self, session: Session) -> None: ...

    def save_all(self, run_id: str, store: ResultStore) -> None:
        """Bulk-insert all results from a ResultStore. Batch size 500."""

    def get_dataframe(self, run_id: str) -> pd.DataFrame:
        """All results for a run as a DataFrame matching RESULT_COLUMNS schema."""

    def get_scenario(
        self, run_id: str, scenario_id: int, cohort_id: str | None = None
    ) -> pd.DataFrame:
        """Results for one scenario, sorted by timestep."""
```

### Why one row per (scenario_id, timestep, cohort_id)

**Alternative rejected: per-asset granularity**
Storing one row per `(scenario_id, timestep, asset_id, accounting_basis)` would
give full asset-level attribution in the database but would increase the row count
by a factor of 10–50× (depending on portfolio size). For a 1,000-scenario ×
120-month run with 20 bonds plus 5 equities, this means ~3 million rows per run.
This level of granularity is not needed by the API or desktop UI — asset-level
detail is available from the `ResultStore.as_dataframe()` if needed within a run.
The database stores run summaries for retrieval and comparison; per-asset breakdown
is a within-run diagnostic.

**Why flat rows rather than JSON blobs**
Storing the full result as a JSON blob per timestep would simplify the schema but
prevent any SQL-level filtering or aggregation (e.g., "return BEL for scenario 5").
Flat columns preserve query flexibility at the cost of a wider table — an
acceptable trade given the modest row counts (120,000 rows for a 1,000-scenario run).

### Bulk insert strategy

SQLAlchemy's `session.bulk_insert_mappings()` is used for result rows.
This bypasses the ORM object construction overhead and sends one parameterised
INSERT per batch of 500 rows. For 120,000 rows this means 240 round-trips instead
of 120,000. This is sufficient for SQLite; PostgreSQL's `COPY FROM` would be faster
but is SQLite-incompatible and premature until production load is measured.

### SQLite → PostgreSQL migration

The only change required to switch from SQLite to PostgreSQL is the connection string
in `storage/db.py`:

```python
# SQLite (development)
engine = create_engine("sqlite:///alm.db")

# PostgreSQL (production)
engine = create_engine("postgresql://user:pass@host/dbname")
```

All ORM models, repositories, and migrations work identically on both engines.
No code outside `storage/db.py` references the database URL.

### What NOT to do

- Do not call SQLAlchemy directly from `api/`, `worker/`, or `engine/`. All database
  access goes through `RunRepository` or `ResultRepository`.
- Do not store the raw `ResultStore` object. Serialise to flat rows via
  `ResultStore.as_dataframe()` before writing.
- Do not add IFRS 17 GMM columns (`csm`, `ra`, `lrc`, etc.) to `result_records` now.
  These fields will be added in Phase 3 Step 17 once `engine/ifrs17/` is built and
  validated. Adding placeholder NULL columns now would pollute the schema with
  purpose-less columns for the entire Phase 2 lifecycle.
- Do not implement a separate `BPAResultRepository`. The `cohort_id` column in
  `result_records` is sufficient for BPA granularity (see §17).

---

## §30 — AI Layer Design (Step 15)

### Context

Step 15 adds a conversational AI assistant that can help actuaries understand ALM results,
modify run configurations, and launch new runs — all through a natural language interface
embedded in the desktop application.  The AI layer is scoped to **conventional segregated
fund products only** for Phase 2.  BPA and IFRS 17 specialist agents are deferred to Phase 3.

---

### Model Selection and Provider Flexibility

**Decision:** Model selection is fully configurable via an `AILayerConfig` dataclass.  The
system ships with Anthropic / Claude as the default but can be pointed at any OpenAI-compatible
endpoint, including in-house or self-hosted models.

`AILayerConfig` fields:

| Field | Type | Default | Notes |
|---|---|---|---|
| `provider` | `str` | `"anthropic"` | `"anthropic"` or `"openai_compatible"` |
| `model` | `str` | `"claude-opus-4-6"` | Model ID as accepted by the provider |
| `api_key_env_var` | `str` | `"ANTHROPIC_API_KEY"` | Name of the env var holding the key |
| `base_url` | `str \| None` | `None` | Required for `"openai_compatible"` providers |
| `api_version` | `str \| None` | `None` | Used by Azure OpenAI endpoints |

**Why this design:**
- The `openai` Python SDK accepts a custom `base_url`, so any OpenAI-compatible in-house
  endpoint (self-hosted Llama, Azure OpenAI, etc.) works without a custom HTTP client.
- The Anthropic SDK is used directly for Claude models because its native tool-use contract
  differs slightly from the OpenAI function-calling contract.
- Keeping the config in one dataclass means the desktop app can expose a single settings
  dialog where all fields are entered once and stored locally (never committed to the repo).

---

### Multi-Agent Architecture

**Decision:** The AI layer is designed as a **team of specialist agents** rather than a
single monolithic assistant.  Each agent has a focused role, its own system prompt, and its
own set of allowed tools.  Agents do not share state — the orchestrator routes the user's
request to the appropriate agent and optionally passes the response to a reviewer.

#### Agent Roster (Phase 2 — conventional products)

| Agent | Role | Tools | Phase |
|---|---|---|---|
| `RunAnalystAgent` | Explains BEL, TVOG, cash flow results for a specific run | `get_run_results`, `get_run_config` | 2 |
| `ModellingAgent` | Explains how the engine calculates BEL/TVOG — algorithms, code, design rationale | `get_run_config` (optional) | 2 |
| `ConfigAdvisorAgent` | Suggests assumption changes; proposes a revised `RunConfig` | `get_run_config`, `submit_run` | 2 |
| `ReviewerAgent` | Cross-checks a proposed config change for consistency and flags actuarial risks | `get_run_config` (read-only) | 2 |

#### Agent Roster (Phase 3 — specialist domains, to be built in Step 23)

| Agent | Role | Tools | Phase |
|---|---|---|---|
| `IFRS17Agent` | Explains CSM, RA, LRC/LIC movements; interprets GMM results | `get_run_results`, `get_run_config` | 3 |
| `SolvencyIIAgent` | Explains SCR spread stress, interest stress, MA offset; flags capital adequacy concerns | `get_run_results`, `get_run_config` | 3 |
| `BPAAgent` | Explains matching adjustment, eligibility, fundamental spread; interprets BPA BEL | `get_run_results`, `get_run_config` | 3 |

**Why specialist agents over one general agent:**
- A single large system prompt covering all domains (conventional, BPA, IFRS 17, SII) becomes
  unwieldy and degrades accuracy.  A focused system prompt with a narrow tool set produces
  more reliable, domain-specific responses.
- Specialist agents can be loaded with different models.  For example, a lighter/cheaper model
  may be sufficient for `RunAnalystAgent` (data retrieval + narration), while `SolvencyIIAgent`
  may need a stronger model for regulatory reasoning.
- Each agent can be developed and tested independently as the corresponding engine module is
  built.  `IFRS17Agent` should not exist before `engine/ifrs17/` is validated (Phase 3).

#### Reviewer Pattern

The `ReviewerAgent` is always invoked after `ConfigAdvisorAgent` proposes a config change.
The orchestrator:
1. Calls `ConfigAdvisorAgent` → receives proposed `RunConfig` JSON.
2. Passes the proposed config to `ReviewerAgent` with the question: _"Does this change
   introduce any actuarial inconsistencies or risk factors?"_
3. Presents both the proposed config **and** the reviewer's commentary to the actuary.
4. The actuary approves, rejects, or requests a revision before `submit_run` is called.

This keeps a human in the loop for all mutating actions while still providing an automated
consistency check.

#### Orchestrator

`ai_layer/agent.py` contains `ALMOrchestrator` which:
- Classifies the user's message via a lightweight LLM call (see Routing decision below).
- Routes to the correct specialist agent.
- Automatically invokes `ReviewerAgent` when `ConfigAdvisorAgent` proposes a config change.
- Manages conversation history (`self._history`) across turns in-memory.
- Returns `OrchestratorResponse(reply, agent_used, pending_submit, reviewer_verdict, tool_calls)`.

---

### Dual-SDK Design

**Decision:** `BaseAgent` uses two SDK clients conditionally based on `AILayerConfig.provider`:

- `"anthropic"` → `anthropic.Anthropic` client + Anthropic messages API (native tool-use format)
- `"openai_compatible"` → `openai.OpenAI(base_url=...)` client + OpenAI chat completions API

**Why two SDKs:**
The Anthropic and OpenAI tool-use contracts differ structurally (different message formats for
tool results, different schema field names).  Using the native SDK for each provider avoids
a translation layer that could introduce subtle bugs.  The `openai` SDK's `base_url` parameter
means any OpenAI-compatible endpoint — Azure, local Ollama, in-house models — works without
a custom HTTP client.

Tool schemas are written once in Anthropic format.  `BaseAgent._anthropic_to_openai_tool()`
converts them automatically when the OpenAI path is active.  Subclasses never branch on
provider — they implement `system_prompt`, `tool_schemas`, and `_execute_tool` only.

---

### Routing — LLM Classifier Call

**Decision:** `ALMOrchestrator` routes user messages via a **single, cheap LLM call** rather
than keyword matching.

The router call uses `max_tokens=10`, `temperature=0`, no tools, and no conversation history.
The model returns a single word: `analyst`, `advisor`, `modelling`, or `unknown`.

**Why LLM routing over keywords:**
- Natural language intent is ambiguous.  "Can you check this for me?" could be analyst or
  advisor depending on context.  A language model infers intent reliably; a keyword list does not.
- Maintainability: no keyword list to update when new phrasing patterns emerge.
- Cost is negligible — a 10-token response on a small prompt is the cheapest possible call.

**Phase 3 short-circuit:** Before the classifier call, a simple `str.lower()` check tests for
IFRS 17, SCR, BPA, CSM, and related terms.  These always return "not available yet" without
an LLM call, because the stub agents raise `NotImplementedError` unconditionally.

On any router failure (API error, unexpected response), the orchestrator defaults to `"analyst"`
so the user always receives a response.

**The `analyst` vs `modelling` distinction** is a key routing rule:
- `"analyst"` — user wants to understand a specific run's output numbers (BEL, TVOG, cash flows)
- `"modelling"` — user wants to understand how the engine computes those numbers (algorithms, code)
  Example: *"Why is BEL high for run ABC?"* → `analyst`; *"How is BEL calculated?"* → `modelling`

---

### Per-Agent Model Tiering

**Decision:** Different agents use different model tiers.  Not every AI call requires the most
capable (and most rate-limited) model.  `AILayerConfig` carries an `agent_models` dict that
maps agent type keys to model IDs.  `BaseAgent` reads its tier via a class-level `_AGENT_TYPE`
string and calls `config.model_for(_AGENT_TYPE)` — the orchestrator needs no per-agent wiring.

Default tier assignment:

| Agent key | Default model | Rationale |
|---|---|---|
| `router` | `claude-haiku-4-5-20251001` | Returns one word — no actuarial reasoning needed |
| `reviewer` | `claude-sonnet-4-6` | Config sanity-check — moderate complexity |
| `analyst` | primary model (e.g. Opus) | Deep actuarial interpretation of BEL/TVOG |
| `advisor` | primary model | Config modification requires precise schema understanding |
| `modelling` | primary model | Engine code explanation requires deep reasoning |

The "primary model" is `AILayerConfig.model`, set by the GUI.  Any tier can be overridden by
passing a custom `agent_models` dict to `AILayerConfig`.

**Why not use Opus everywhere:**
`claude-opus-4-6` has a 30,000 input token/minute rate limit on standard plans.  The large
knowledge-base system prompts (DECISIONS.md + architecture sections + schema) consume most of
that budget on a single request.  Routing classification and config review do not need the
same depth of reasoning as actuarial result interpretation, so lighter models are appropriate.

---

### Prompt Caching (Anthropic path)

**Decision:** On the Anthropic provider path, `BaseAgent._call_anthropic()` wraps the system
prompt and tool definitions in `cache_control: {"type": "ephemeral"}` content blocks.

```python
system=[{"type": "text", "text": self.system_prompt, "cache_control": {"type": "ephemeral"}}]
tools=[..., {**last_tool, "cache_control": {"type": "ephemeral"}}]  # breakpoint after last tool
```

**Why prompt caching:**
Agent system prompts are large — the RunAnalystAgent injects DECISIONS.md financial sections,
ALM_Architecture.md result fields, and the RunConfig schema (~15–25k tokens).  Without caching,
every turn within a session re-sends the full system prompt to the API, consuming rate-limit
budget on static content.

With caching, the first call pays the cache-write cost (125% of normal input token cost).
Subsequent turns within the 5-minute cache window pay only 10% of normal input token cost for
the cached portion, and that portion does not count against the input token rate limit.

The OpenAI-compatible path is unchanged — `cache_control` is an Anthropic-specific extension.

---

### Session Management

**Decision:** Conversation history is stored in an **in-memory dict** in `api/routers/ai.py`
keyed by `session_id` (UUID).  No database, no Redis, no persistence across server restarts.

**Why in-memory:**
The AI Assistant is embedded in a single-user desktop application.  At most a handful of
sessions exist simultaneously.  The actuary expects each app launch to start fresh.
In-memory storage adds zero infrastructure overhead and zero latency.

If the system is later exposed to multiple concurrent users (web frontend), sessions should
be moved to Redis.  The `session_id` protocol already anticipates this — the client owns the
ID and passes it back each turn, so the storage backend is swappable without API changes.

---

### Tool Design Philosophy

**Decision:** Three tools are available in Phase 2.  All are wrappers over existing HTTP
endpoints — the AI layer has no direct database access.

| Tool | HTTP call | Mutating? |
|---|---|---|
| `get_run_results` | `GET /results/{run_id}?format=csv` + `GET /results/{run_id}/summary` | No |
| `get_run_config` | `GET /runs/{run_id}` → extract `config_json` | No |
| `submit_run` | `POST /runs` | Yes — intercepted; requires explicit actuary approval |

`submit_run` is intercepted by `BaseAgent` before `_execute_tool` is called.  The proposed
config is stored in `AgentResponse.pending_submit` and returned to the orchestrator.  The
orchestrator passes it through `ReviewerAgent`, then the desktop UI shows a confirmation
dialog.  The tool is only executed (via `ALMApiClient.submit_run()`) after the actuary
clicks Approve.  `OrchestratorResponse.pending_submit` is `None` if the reviewer rejected
the config — the actuary must request a different change.

Phase 3 will add read-only tools for IFRS 17 and SII specialist agents as those engine
modules are validated.

---

### Safety Constraints — Human in the Loop

**Decision:** The AI layer **never automatically submits a run** without displaying the
proposed `RunConfig` to the actuary for approval.  The `submit_run` tool is available to the
model, but `BaseAgent` intercepts all `submit_run` tool calls and the desktop UI presents a
confirmation dialog (showing config JSON + reviewer verdict) before executing.

No code-write, file-delete, or database-mutation tools are exposed in Phase 2 or Phase 3.

---

### API Key Management and Configuration Split

**Decision:** API keys are stored **only** in a `.env` file on the server and loaded via
`python-dotenv` (`load_dotenv()` called once in `api/main.py`).  They are never sent over
the wire in the HTTP request body.

All other AI settings — `provider`, `model`, `base_url`, `deployment_mode` — are **user
preferences** controlled via the desktop GUI settings panel and sent in the `POST /ai/chat`
request body.  They are not secrets and do not belong in `.env`.

**Why this split:**
- API keys are secrets.  Placing them in a GUI field risks accidental logging or
  transmission.  `.env` files are excluded from version control and scoped to the server process.
- Model selection and deployment mode are per-user preferences that the actuary may change
  between sessions (e.g. switching from development to production mode, or testing a new model).
  Forcing these into `.env` would require a server restart on every change.

The `AIChatRequest` schema carries `provider`, `model`, `base_url`, `deployment_mode`.
The server reads `ANTHROPIC_API_KEY` (or `LLM_API_KEY` for `openai_compatible`) from the
environment.  GUI settings are only applied when creating a **new** session; subsequent turns
reuse the session's existing settings.

---

### Knowledge Base Strategy — Context Injection, not RAG

**Decision:** Agent system prompts are assembled from project documentation injected directly
into the context window on every request.  Retrieval-Augmented Generation (RAG) is explicitly
**not used** in Phase 2.

#### What "context injection" means

At agent instantiation, four sources are loaded and concatenated into the system prompt:

1. `model_docs.md` — a hand-curated actuarial glossary covering BEL, TVOG, SCR, run types,
   and other concepts that the agent may need to explain.  Written once; updated when new
   concepts are introduced.

2. `decisions_loader.py` — programmatically reads `document/DECISIONS.md` and extracts the
   sections relevant to the requesting agent (e.g. financial modelling decisions for
   `RunAnalystAgent`; modelling constraint decisions for `ConfigAdvisorAgent`).

3. `architecture_loader.py` — programmatically reads `document/ALM_Architecture.md` and
   extracts the module map, result field definitions, and run type descriptions.

4. `schema_export.py` — calls `RunConfig.model_json_schema()` at runtime and formats it as
   readable text.  Because it is generated from the live Python class, the AI context stays
   in sync with code changes automatically without any manual update step.

#### Why not RAG

RAG (Retrieval-Augmented Generation) involves chunking documents, embedding each chunk with
a vector model, storing embeddings in a vector database, and at query time retrieving the
chunks most similar to the user's question.

RAG is the right choice when the document corpus is too large to fit in the model's context
window.  It is the wrong choice here for three reasons:

1. **Document set is small.**  `DECISIONS.md` + `ALM_Architecture.md` + `model_docs.md` +
   the RunConfig schema combined are approximately 30,000–50,000 tokens.  Claude Opus 4.6
   has a 200,000-token context window.  The documents fit with room to spare.

2. **Reliability.**  RAG introduces a retrieval failure mode: if the vector search returns
   the wrong chunks, the agent is given incorrect or incomplete context.  Direct injection
   guarantees the agent always sees the complete, authoritative documents.  For an actuarial
   system where incorrect results carry professional liability risk, this reliability
   advantage outweighs the marginal cost of a larger prompt.

3. **Infrastructure simplicity.**  RAG requires an embedding model, a vector store (Chroma,
   FAISS, Pinecone, etc.), and a retrieval pipeline.  For a small actuarial team with a
   bounded document set, this infrastructure adds complexity with no measurable benefit.

#### When to revisit this decision

Switch to RAG in Phase 3 or later if any of the following occurs:
- The combined documentation exceeds ~150,000 tokens (leaving insufficient room for tool
  results and conversation history in the context window).
- New specialist domains (BPA, IFRS 17, SII) each add large document volumes that make the
  full corpus unwieldy to inject for every request.
- A need for semantic search across historical run results or audit trails arises.

---

### ModellingAgent — Separate Agent for Engine Mechanics

**Decision:** A dedicated `ModellingAgent` handles questions about *how the model works
internally* — algorithms, calculation sequences, class/method behaviour — rather than bundling
this into `RunAnalystAgent`.

**Why not bundle into `RunAnalystAgent`:**
- `RunAnalystAgent` is focused on *what the results mean* for a specific run.  Mixing
  in algorithm explanations would require its system prompt to carry the full engine source
  code on every call — even when the user is asking a simple "why is TVOG high?" question.
- Separation keeps each agent's context lean.  `RunAnalystAgent` injects results context;
  `ModellingAgent` injects engine source code.  Neither gets the other's overhead.
- The router can cheaply distinguish the two intents (`"analyst"` vs `"modelling"`), so
  routing cost is negligible.

**Why engine source code injection instead of hand-written documentation:**

The engine source files are the single source of truth for how calculations work.  Hand-written
summaries of algorithms drift from the code as the engine evolves; source code never lies.

At `ModellingAgent` instantiation, `code_loader.py` reads ~15 selected engine `.py` files
and formats them as fenced code blocks in the system prompt.  Combined token budget is
~15,000–20,000 tokens — well within the 200K context window.

A short `model_mechanics.md` primer (~1,000 tokens) provides the conceptual *why* that source
code alone does not convey: the two-pass BEL design rationale, the coordinator pattern intent,
the deep-copy isolation rule for stochastic scenarios.  This pairs with the code to give the
agent both the *what* and the *why*.

**Files involved:**
- `ai_layer/agents/modelling_agent.py` — agent class; injects code + primer + full DECISIONS.md + schema
- `ai_layer/knowledge_base/code_loader.py` — reads engine `.py` files; graceful missing-file handling
- `ai_layer/knowledge_base/model_mechanics.md` — conceptual primer (~1,000 tokens)

**What the agent must NOT do:**
- Propose config changes (that is `ConfigAdvisorAgent`).
- Interpret a specific run's output numbers (that is `RunAnalystAgent`).
- Use `submit_run` or `get_run_results`.

---

### What NOT to do

- Do not give any agent direct SQLAlchemy or filesystem access.  All data is mediated via HTTP.
- Do not build `IFRS17Agent`, `SolvencyIIAgent`, or `BPAAgent` before the corresponding
  engine modules are validated in Phase 3.
- Do not persist conversation history server-side in Phase 2.  The desktop client owns it.
- Do not expose `code_modifier.py` or any file-write tool before Phase 3 and only with a
  human-approval gate.
- Do not hardcode model names or API keys anywhere outside `AILayerConfig`.
- Do not implement RAG in Phase 2 — the document corpus fits in the context window and
  direct injection is more reliable.  Revisit when the corpus exceeds ~150,000 tokens.

---

### Privacy and Confidentiality (PNC) — Data Governance for the AI Layer

**Decision:** The default provider for **production** deployments is `openai_compatible`
pointing to an on-premise or private-network model endpoint.  The Anthropic external API
(`provider = "anthropic"`) is permitted for **development and testing only**, and only with
sanitised, non-production data.

#### Why this matters

When an external AI API is used, everything in the context window — system prompt, user
message, tool results — is transmitted over the internet to a third-party service.  For
this ALM system the context window may contain:

| Content | PNC Classification | Risk |
|---|---|---|
| `model_docs.md` (generic glossary) | Low — generic actuarial concepts | Acceptable for external API |
| `DECISIONS.md` injected sections | High — firm's proprietary modelling methodology and assumption rationale | Must not leave network perimeter |
| `ALM_Architecture.md` injected sections | High — proprietary system design and internal module structure | Must not leave network perimeter |
| `RunConfig` schema | Medium — proprietary parameter structure | Acceptable if anonymised |
| `get_run_config` output | High — client/product-specific assumption sets | Must not leave network perimeter |
| `get_run_results` output | Very High — projection results derived from policyholder liability data | Must not leave network perimeter |

Sending High or Very High content to an external API likely conflicts with:
- GDPR / UK Data Protection Act 2018 obligations on policyholder-derived data
- Client confidentiality agreements
- Internal data governance and information security policies

This applies even when the external provider has a Data Processing Agreement (DPA)
prohibiting training on customer data — the data still leaves the network perimeter.

#### Approved deployment patterns

Four patterns are supported, ordered from lowest to highest PNC protection:

**Pattern 1 — External API + context injection (development / testing only)**

`provider = "anthropic"` (or any external API) with full document injection.
All documents and tool results leave the network.
- Permitted only with synthetic, non-production data
- Must set `deployment_mode = "development"` in `AILayerConfig`; a warning is logged on
  every request
- Do not use real run IDs, real assumption sets, or real policyholder-derived projections

**Pattern 2 — Private RAG + external LLM (intermediate — documents protected)**

On-premise embedding model and on-premise vector store (e.g. Chroma or FAISS running
locally).  Only the retrieved chunks — not the full documents — are injected into the
context sent to the external LLM.

- `DECISIONS.md`, `ALM_Architecture.md`, and `model_docs.md` are embedded and stored
  entirely on-premises.  The full documents never leave the network.
- Only the 2–4 most semantically relevant paragraphs are retrieved per request and sent
  to the external LLM — materially less exposure than full document injection.
- **Tool results (`get_run_results`, `get_run_config`) still go to the external LLM.**
  This pattern does not protect real run data.  Use only with test/synthetic run IDs, or
  upgrade to Pattern 3 or 4 when real run data is needed.
- Appropriate when: an on-premise LLM is not yet available but is planned; or when the
  document corpus has grown large enough (>150K tokens) that RAG is preferable anyway.

**Pattern 3 — Private RAG + on-premise LLM (recommended for production with large corpus)**

Both the RAG infrastructure (embedding + vector store) and the LLM run on-premises.
Nothing leaves the network.  Use when the document corpus exceeds ~150K tokens and full
context injection is no longer practical.

**Pattern 4 — Context injection + on-premise LLM (recommended for production, Phase 2)**

`provider = "openai_compatible"` with `base_url` pointing to a model within the firm's
network perimeter.  Full document injection is used (no RAG overhead).  Nothing leaves the
network.  This is the recommended production pattern for Phase 2 because the document
corpus fits in the context window and RAG infrastructure is not yet warranted.

#### Pattern selection summary

| Pattern | Docs leave network? | Tool results leave network? | When to use |
|---|---|---|---|
| 1 — External API + injection | Yes (full) | Yes | Dev/test with synthetic data only |
| 2 — Private RAG + external LLM | No (chunks only) | Yes | Intermediate; no real run data |
| 3 — Private RAG + on-premise LLM | No | No | Production; large corpus (Phase 3+) |
| 4 — Injection + on-premise LLM | No | No | Production; Phase 2 (recommended) |

#### Configuration enforcement

`AILayerConfig` must include a `deployment_mode` field:
- `"development"` — external APIs permitted; warning logged on every request
- `"production"` — `provider = "anthropic"` raises `RuntimeError` at agent instantiation;
  only `"openai_compatible"` with a non-public `base_url` is allowed

The desktop AI tab must display the active `deployment_mode` visibly so the actuary always
knows whether data may leave the network.

#### What NOT to do (PNC)

- Do not use `provider = "anthropic"` (or any external API) with real production run data.
- Do not inject the full `DECISIONS.md` or `ALM_Architecture.md` when using an external API.
- Do not log context window contents — they may contain proprietary assumptions or results.
- Do not store conversation history to disk without encryption.
- Do not use Pattern 2 (private RAG + external LLM) with real run IDs — tool results still
  leave the network even when documents are protected by the RAG layer.

---

---

## 31. Asset Data Loader (Step 15a)

`AssetDataLoader` loads bond portfolios from CSV into a validated `AssetModel` ready for
injection into a run mode. It enforces required columns (`asset_id`, `face_value`,
`annual_coupon_rate`, `maturity_month`, `accounting_basis`, `initial_book_value`) with
optional `eir` and `calibration_spread`. The engine never reads files directly.

> **Not included in public demo.**
> Column validation rules, EIR derivation logic, and loader interface design detail.

---

## 32. Stochastic Vectorisation Design

The stochastic run is designed to support scenario-level vectorisation as a Phase 3
enhancement: all N scenarios are computed simultaneously as batched NumPy/JAX array
operations, eliminating the Python loop over scenarios. The IFRS 17 GMM inner step
(`engine/ifrs17/_gmm_jit.py`) is the first production use of JAX JIT compilation,
introduced in Step 18a as the lowest-risk entry point before touching the stochastic
layer. A `use_vectorised: bool = False` flag in `StochasticConfig` gates the batched
path; both paths must produce identical TVOG output before the flag can be enabled.

> **Not included in public demo.**
> Array shape design, JAX `vmap`/`lax.scan` approach, migration sequence, and
> interface change specifications.

---

## 33. IFRS 17 Persistent State and Roll-Forward Accounting

### Decision
IFRS 17 reporting requires **stateful roll-forward accounting**, not a fresh calculation
each reporting cycle. The closing CSM balance from Q1 is the opening balance for Q2.
The closing loss component from one period determines whether onerous relief is available
in the next. Without persisting this state, the tool can compute inception-date metrics
but cannot produce the quarterly movement tables required for financial statement
presentation.

Two components are added to support this:

**`Ifrs17State` (in `engine/ifrs17/state.py`)** — a frozen dataclass holding all
rolling IFRS 17 balances for a single contract group at a single valuation date:

```python
@dataclass(frozen=True)
class Ifrs17State:
    cohort_id:                 str
    valuation_date:            date
    csm_balance:               float
    loss_component:            float      # 0.0 for non-onerous groups
    remaining_coverage_units:  float
    total_coverage_units:      float      # fixed at inception; used for release fraction
    locked_in_rate:            float      # post-MA rate at inception (§21); never changes
    inception_date:            date
```

**`Ifrs17StateStore` (in `storage/ifrs17_state_repository.py`)** — loads and saves
`Ifrs17State` objects by cohort_id and valuation_date. Backed by the `ifrs17_state`
database table (§37):

```python
class Ifrs17StateStore:
    def load(self, cohort_id: str, valuation_date: date) -> Ifrs17State | None:
        """Returns None on first run (inception) — GmmEngine initialises from FCF."""

    def save(self, state: Ifrs17State) -> None:
        """Upsert: insert if not exists, overwrite if rerun."""
```

### Run mode integration
The run mode orchestrator (initially `BPARun`) is responsible for:
1. Calling `Ifrs17StateStore.load()` for each contract group before `GmmEngine` is
   constructed. On first run (state = None), `GmmEngine` initialises CSM from FCF at
   inception: `CSM_0 = max(0, -FCF_0)`.
2. After all timesteps complete, calling `Ifrs17StateStore.save()` with the closing
   state from the final `GmmStepResult` for each cohort.

The run mode must not reconstruct `GmmEngine` mid-run with different opening states.
State loading happens once at run start; state saving happens once at run end.

### Why this is a conceptual shift
Every other module in the engine is stateless between runs — a fresh `DeterministicRun`
reproduces the same results given the same inputs. `GmmEngine` is different: two runs
with different opening CSM balances (e.g., after a retrospective assumption change)
produce different results even with identical current-period inputs. This is correct
IFRS 17 behaviour and must be documented prominently for anyone maintaining the engine.

### What NOT to do
- Do not recompute CSM from inception on every run. This ignores retrospective
  adjustments and produces CSM values inconsistent with prior-period financial statements.
- Do not store `Ifrs17State` inside `ResultStore`. It is a cross-period rolling balance,
  not a within-run per-timestep result. The two storage layers serve different purposes.
- Do not omit `total_coverage_units` from the state. The CSM release fraction
  (`remaining / total_remaining_at_start_of_period`) requires knowing how many units
  have already been consumed. Recomputing this from scratch each period is incorrect
  after any assumption change.

---

## 34. IFRS 17 Assumption Attribution — Locked-In vs Current

### Decision
IFRS 17 GMM requires distinguishing between changes arising from **financial assumptions**
(discount rate movements) and **non-financial assumptions** (mortality, inflation, lapses).
These follow different accounting paths:

| Source of change | Accounting treatment |
|---|---|
| Non-financial assumption change — future service | Adjust CSM (para 44) |
| Non-financial assumption change — past/current service | P&L immediately |
| Financial assumption change — future service | OCI (if OCI option elected, para 88b) |
| Financial assumption change — past/current service | P&L |

Without this split, IFRS 17 P&L and OCI figures are miscategorised.

### Two assumption objects

**`LockedInAssumptions` (in `engine/ifrs17/assumptions.py`)** — frozen at contract group
inception; used only for CSM accretion and the locked-in BEL calculation:

```python
@dataclass(frozen=True)
class LockedInAssumptions:
    cohort_id:       str
    inception_date:  date
    locked_in_rate:  float   # post-MA parallel shift rate at inception (§21)
```

The locked-in rate is stored as a single parallel shift (not a full yield curve) for
consistency with the MA benefit design (§21), which is itself a single spread added to
the EIOPA risk-free curve. Upgrading to a full locked-in curve is a Phase 4 enhancement
if greater precision is required; for BPA, a single parallel shift is materially
adequate given the flat nature of the MA benefit.

**`CurrentAssumptions` (in `engine/ifrs17/assumptions.py`)** — current-period inputs
supplied by the run mode at each timestep:

```python
@dataclass(frozen=True)
class CurrentAssumptions:
    t:                int
    current_rate:     float     # current-period discount rate
    mortality_table:  str       # identifier — used to detect table change between periods
    inflation_index:  float     # current CPI/RPI rate from ESG scenario
```

**`AssumptionProvider` (protocol)** — injected into the run mode orchestrator:

```python
class AssumptionProvider(Protocol):
    def get_locked_in(self, cohort_id: str) -> LockedInAssumptions: ...
    def get_current(self, t: int) -> CurrentAssumptions: ...
```

### Dual BEL requirement

`GmmEngine.step()` receives two BEL values (see §28 updated interface):

- **`bel_current`** — BEL discounted at the current-period discount rate. This is the
  balance sheet value and the base for LRC/LIC assembly.
- **`bel_locked`** — BEL discounted at the locked-in rate. Used to compute insurance
  finance income/expense at the locked-in rate, isolating the OCI component.

The **insurance finance P&L** is: `bel_locked(t) - bel_locked(t-1)` (unwinding at
locked-in rate). The **OCI component** is: `(bel_current(t) - bel_locked(t)) -
(bel_current(t-1) - bel_locked(t-1))` (change in the current-vs-locked gap).

The liability model computes both BEL values from the same cashflow projection, using
two discount curves. The computational cost is negligible; the compliance cost of
omitting it is material.

### FCF change attribution inputs

`fcf_change_non_financial` and `fcf_change_financial` are supplied by the run mode
orchestrator by comparing current-period BEL with prior-period BEL under controlled
conditions:

```
fcf_change_non_financial = BEL(current mortality/inflation, locked-in rate)
                         - BEL(prior mortality/inflation, locked-in rate)

fcf_change_financial     = BEL(current mortality/inflation, current rate)
                         - BEL(current mortality/inflation, locked-in rate)
                         = bel_current - bel_locked
```

The second identity means `fcf_change_financial` need not be passed separately — it
is always `bel_current - bel_locked`. `GmmEngine` computes it internally. Only
`fcf_change_non_financial` requires an additional BEL computation by the run mode.

### What NOT to do
- Do not use a single `fcf_change` field and assume all changes are non-financial.
  This overstates CSM adjustments and understates OCI.
- Do not apply the locked-in rate to BEL measurement for the balance sheet. The
  balance sheet BEL (`bel_current`) uses the current-period rate. Only CSM accretion
  and the finance income split use the locked-in rate.
- Do not store `LockedInAssumptions` in `ResultStore`. They are static per contract
  group; loading them from `Ifrs17State` is sufficient.

---

## 36. IFRS 17 Computation Scope — Deterministic Scenario Only

IFRS 17 GMM is computed on the deterministic scenario only. TVOG and TVLPI from the
stochastic run are added to the deterministic BEL to form `bel_current` passed to
`GmmEngine`. The stochastic engine does not produce per-scenario IFRS 17 output — IFRS 17
is an accounting standard requiring a single best-estimate measurement, not a distribution.

> **Not included in public demo.**
> Detailed computation sequence, quarterly reporting roll-forward approach, and
> design constraints between `GmmEngine` and `StochasticRun`.

---

## 37. IFRS 17 State and Movement Storage Tables

Two database tables support IFRS 17 reporting: `ifrs17_state` (one row per
`(cohort_id, valuation_date)` — the closing CSM, loss component, coverage units, and
locked-in rate for each reporting period) and `ifrs17_movements` (one row per
`(cohort_id, valuation_date, period_index)` — the full movement table for auditor review).
Both are separate from `result_records` and keyed by cohort and valuation date, not run ID,
so state persists across quarterly reporting cycles.

> **Not included in public demo.**
> Full column specifications, upsert behaviour, and repository interface detail.

---

## 35. Coverage Unit Definition for BPA

> **Not included in public demo.**
> Coverage unit definition per BPA cohort type and CSM release methodology.

---

## 38. IFRS 17 Contract Boundary — BPA *(Tier 1)*

> **Not included in public demo.**
> Contract boundary determination for BPA under IFRS 17 AASB 17 / IFRS 17 paragraphs.

---

## 39. Reinsurance — Longevity Swap Design Hook *(Tier 3)*

> **Not included in public demo.**
> Longevity swap design hook and reinsurance treaty modelling approach.

---

## 40. IFRS 17 Transition *(Tier 3)*

> **Not included in public demo.**
> Modified retrospective and fair value transition approaches.

---

## 41. Experience Variance — Past Service Attribution *(Tier 1)*

> **Not included in public demo.**
> Experience variance attribution methodology for past service adjustments.

---

## 42. Expense Loading — Product-Level Attributable Expenses *(Tier 2)*

> **Not included in public demo.**
> Per-cohort expense attribution and loading methodology.

---

## 43. Multi-Cohort Aggregation *(Tier 2)*

> **Not included in public demo.**
> Aggregation rules across BPA cohorts for IFRS 17 reporting.

---

## 44. BPA Deal Registry and Model Point Identification

> **Not included in public demo.**
> Deal registry design, cohort_id derivation, and model point identification scheme.

---

## 45. BPA Deal Registry — Implementation

> **Not included in public demo.**
> Implementation details of the deal registry and its integration with the run mode.

---

## 46. BPA Investment Strategy — Three-Strategy Design

> **Not included in public demo.**
> Buy-and-hold, liability-matching, and active strategy design for BPA portfolios.

---

*End of DECISIONS.md*
