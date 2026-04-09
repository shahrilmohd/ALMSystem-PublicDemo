# ALM Model — Full System Architecture

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Directory Structure](#2-directory-structure)
3. [Layer Descriptions](#3-layer-descriptions)
   - 3.7 [AI Layer (Phase 2 — Step 15)](#37-ai-layer-phase-2--step-15)
   - 3.9 [IFRS 17 Engine (Phase 3 — Step 17)](#39-ifrs-17-engine-phase-3--step-17)
4. [Module Responsibilities](#4-module-responsibilities)
5. [Run Mode Logic](#5-run-mode-logic)
6. [Data Flow](#6-data-flow)
7. [Class Hierarchy](#7-class-hierarchy)
8. [Bond Accounting Basis Framework](#8-bond-accounting-basis-framework)
9. [Opening Balance Sheet Initialisation](#9-opening-balance-sheet-initialisation)
10. [Technology Stack](#10-technology-stack)
11. [Build Order](#11-build-order)
12. [Design Principles](#12-design-principles)

---

## Documentation Map

The `document/` folder contains the full technical reference for this project.
All documents are kept in plain Markdown so they can be read alongside code in
any editor. This file (`ALM_Architecture.md`) is the entry point — read it first,
then follow the links below for deeper detail.

| Document | Audience | Purpose |
|---|---|---|
| **[ALM_Architecture.md](ALM_Architecture.md)** | Everyone | System-wide architecture: directory layout, module interfaces, data flow, class hierarchy, build order. Start here. |
| **[DECISIONS.md](DECISIONS.md)** | Developers, actuaries | Records *why* key modelling choices were made (IFRS 9 basis, EIR computation, BEL design, rebalancing constraints, etc.). Must be read before changing the engine. Add a new section here before writing any new model code. |
| **[testing_guide.md](testing_guide.md)** | Developers | Testing conventions, pytest features used, test directory structure, helper patterns, and a formal run log with a bug history for every step. |
| **[calculation_engine_guide.md](calculation_engine_guide.md)** | Actuaries, quants | Explains the vectorised two-pass calculation mechanism, how it compares to traditional serial actuarial platforms (Prophet, MoSes), and why it was chosen for this engine. |
| **[developer_guide.md](developer_guide.md)** | Developers | Recurring development tasks: adding new output fields, extending the result schema, understanding result granularity, and working with the build order. |
| **[EIOPA-BoS-24-533 - RFR Technical Documentation.pdf](EIOPA-BoS-24-533%20-%20RFR%20Technical%20Documentation.pdf)** | Actuaries, regulators | EIOPA's official technical documentation for the Risk-Free Rate (RFR) methodology. Reference source for the Smith-Wilson extrapolation implemented in `engine/curves/rate_curve.py`. |
| **[validation/rate_curve_validation.md](validation/rate_curve_validation.md)** | Developers, actuaries | Numerical validation of `RiskFreeRateCurve`: hand-calculated discount factors, interpolation checks, and Smith-Wilson convergence evidence. Values reproduced exactly by the unit tests in `tests/unit/curves/`. |
| **[validation/rate_curve_comparison.png](validation/rate_curve_comparison.png)** | Actuaries | Visual comparison of flat-forward vs Smith-Wilson extrapolation beyond the last observable maturity. Produced by `validation/plot_rate_curve.py`. |

**When to update which document:**

- Adding a new module or changing an interface → update this file (`ALM_Architecture.md`)
- Making a financial or actuarial modelling decision → add a section to `DECISIONS.md` *before* writing code
- Adding or changing tests → update the directory structure table and test run log in `testing_guide.md`
- Changing the calculation mechanism or projection loop → update `calculation_engine_guide.md`
- Adding a new output field or result column → follow the steps in `developer_guide.md`

---

## 1. System Overview

The ALM model is a segregated fund-based Asset and Liability Model (ALM) built to:

- Project asset and liability cash flows at **fund level**
- Support **seriatim** (per-policy) liability runs for deterministic/liability-only modes
- Support **group model point** liability runs for stochastic modes (1,000–3,000 MPs)
- Calculate **Time Value of Options and Guarantees (TVOG)** from stochastic results
- Serve a **team of actuaries** via desktop app (Phase 1) and web app (Phase 2)
- Integrate an **AI assistant layer** (Phase 2) that can advise, explain, and modify the model

### High-Level System Diagram

```
┌─────────────────────────────────────────────────────────┐
│                     FRONTEND                            │
│         Desktop (PyQt6)  ──►  Web (React, Phase 2)      │
└──────────────────────┬──────────────────────────────────┘
                       │ HTTP
┌──────────────────────▼──────────────────────────────────┐
│                      API (FastAPI)                       │
│     /runs    /config    /results    /ai (Phase 2)        │
└──────────────────────┬──────────────────────────────────┘
                       │ Job dispatch
┌──────────────────────▼──────────────────────────────────┐
│               WORKER (RQ + Redis)                        │
│         Async execution + progress reporting             │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│                  ENGINE (Pure Python ALM)                │
│                                                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐              │
│  │  Config  │  │Run Modes │  │ Results  │              │
│  └──────────┘  └──────────┘  └──────────┘              │
│                                                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐              │
│  │   Core   │  │  Asset   │  │Liability │              │
│  │Top/Fund/ │  │Bond/Eq/  │  │Conv/UL/  │              │
│  │ Company  │  │  Deriv   │  │ Annuity  │              │
│  └──────────┘  └──────────┘  └──────────┘              │
│                                                         │
│  ┌──────────┐  ┌──────────┐                            │
│  │ Strategy │  │Scenarios │                            │
│  │Inv/Bonus │  │ESG/Store │                            │
│  └──────────┘  └──────────┘                            │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│              DATA + STORAGE                              │
│   Loaders / Validators / SQLite (→ PostgreSQL)           │
└─────────────────────────────────────────────────────────┘
```

---

## 2. Directory Structure

```
alm_system/
│
├── frontend/                              # Purpose: user interaction only — zero business logic; no engine imports
│   ├── desktop/                           # Purpose: PyQt6 native desktop app for the actuarial team
│   │   ├── app.py                         # Responsibility: QApplication entry point and event loop
│   │   ├── windows/
│   │   │   ├── main_window.py             # Responsibility: main window shell — navigation, menu bar, window management
│   │   │   ├── run_config_window.py       # Responsibility: form to build, load, edit and submit a RunConfig
│   │   │   └── results_window.py          # Responsibility: results viewer — tables, summary statistics, charts
│   │   ├── components/
│   │   │   ├── progress_panel.py          # Responsibility: live progress bar; polls GET /runs/{run_id} every 2 s while RUNNING
│   │   │   ├── file_picker.py             # Responsibility: file browser widget for selecting liability/asset CSV inputs
│   │   │   └── run_type_selector.py       # Responsibility: radio-button selector for LIABILITY_ONLY / DETERMINISTIC / STOCHASTIC
│   │   └── api_client.py                  # Responsibility: HTTP client wrapping requests — all API calls go through here; no direct engine imports
│   └── web/                               # Purpose: web frontend placeholder (Phase 3+)
│       └── README.md
│
├── api/                                   # Purpose: single HTTP interface between all frontends and the engine; no business logic
│   ├── main.py                            # Responsibility: FastAPI app init, router mounting, CORS configuration
│   ├── routers/
│   │   ├── runs.py                        # Responsibility: POST /runs (submit), GET /runs (list), GET /runs/{run_id} (status)
│   │   ├── config.py                      # Responsibility: POST /config/validate, GET /config/template
│   │   ├── results.py                     # Responsibility: GET /results/{run_id} (full results), GET /results/{run_id}/summary
│   │   └── ai.py                          # Responsibility: AI assistant endpoints (Phase 2 — Step 15)
│   ├── schemas/
│   │   ├── run_schema.py                  # Responsibility: Pydantic request/response models for run submission and status
│   │   ├── config_schema.py               # Responsibility: Pydantic schema for config submission and validation responses
│   │   └── result_schema.py               # Responsibility: Pydantic schema for result responses (JSON and CSV modes)
│   └── dependencies.py                    # Responsibility: FastAPI dependency injection — DB session (get_db), queue connection
│
├── worker/                                # Purpose: async execution of long-running model runs via RQ + Redis
│   ├── job_queue.py                       # Responsibility: Redis connection and RQ queue setup
│   ├── tasks.py                           # Responsibility: run_alm_job(run_id) — deserialises config, runs engine, persists results, updates status
│   └── progress.py                        # Responsibility: progress reporting via Redis pub/sub or RQ job metadata
│
├── engine/                                # Purpose: pure Python ALM model — zero imports from frontend/, api/, or worker/
│   │
│   ├── config/                            # Purpose: run parameter definitions and validation
│   │   ├── run_config.py                  # Responsibility: RunConfig — master Pydantic config object for a full projection run
│   │   ├── fund_config.py                 # Responsibility: FundConfig — SAA target weights and crediting group definitions
│   │   ├── projection_config.py           # Responsibility: ProjectionConfig — projection term, timestep, currency, valuation date
│   │   └── config_loader.py               # Responsibility: loads and deserialises RunConfig from YAML/JSON file
│   │
│   ├── core/                              # Purpose: projection loop and fund-level asset/liability coordination
│   │   ├── top.py                         # Responsibility: owns the time step loop — calls Fund at each period; emits progress callbacks
│   │   ├── company.py                     # Responsibility: aggregates all fund results to company level after each scenario
│   │   ├── fund.py                        # Responsibility: coordinates AssetModel ↔ BaseLiability per fund per timestep (see Section 6.2)
│   │   └── projection_calendar.py         # Responsibility: ProjectionCalendar + ProjectionPeriod — hybrid timestep (monthly/annual) for BPA runs (DECISIONS.md §27)
│   │
│   ├── asset/                             # Purpose: asset projection — fully basis-aware (AC / FVTPL / FVOCI)
│   │   ├── base_asset.py                  # Responsibility: BaseAsset ABC — valuation, cashflows, P&L decomposition, step_time, rebalance interface
│   │   ├── bond.py                        # Responsibility: fixed-income bond — EIR amortisation (AC), MV reset (FVTPL), OCI reserve (FVOCI), calibration spread
│   │   ├── equity.py                      # Responsibility: equity — FVTPL only; applies total return minus dividend yield to market value
│   │   ├── derivative.py                  # Responsibility: derivative placeholder (Phase 3)
│   │   └── asset_model.py                 # Responsibility: portfolio container — aggregates cashflows, P&L and market value by accounting basis
│   │
│   ├── liability/                         # Purpose: liability cash flow projection by product type
│   │   ├── base_liability.py              # Responsibility: BaseLiability ABC — decrement interface and cashflow contract for all liability types
│   │   ├── multi_decrement.py             # Responsibility: MultiDecrementLiability ABC — multi-decrement extension; bridging step 10a for BPA (Phase 3)
│   │   ├── conventional.py                # Responsibility: conventional with-profits / endowment liability — death, lapse, maturity decrements
│   │   ├── unit_linked.py                 # Responsibility: unit-linked liability — fund value projection with policyholder account
│   │   ├── annuity.py                     # Responsibility: annuity liability — in-payment cashflows with mortality decrement
│   │   └── liability_model.py             # Responsibility: LiabilityModel container — holds and dispatches to all BaseLiability instances
│   │
│   ├── strategy/                          # Purpose: investment and bonus strategies — always injected into Fund, never hardcoded
│   │   ├── base_strategy.py               # Responsibility: BaseStrategy ABC
│   │   ├── investment_strategy.py         # Responsibility: SAA rebalancing — AC bonds excluded from routine sell orders (DECISIONS.md §7); conventional funds only
│   │   ├── buy_and_hold_strategy.py       # Responsibility: no-op rebalancing for BPA MA portfolio held to maturity (DECISIONS.md §46); Step 20
│   │   └── bonus_strategy.py              # Responsibility: bonus crediting rate calculation and application to policyholder reserves
│   │
│   ├── scenarios/                         # Purpose: ESG scenario data management for stochastic runs
│   │   ├── scenario_engine.py             # Responsibility: ScenarioLoader — constructs ScenarioStore from CSV file or flat-rate analytical generator
│   │   └── scenario_store.py              # Responsibility: ScenarioStore — holds all EsgScenario objects; O(1) retrieval by scenario_id
│   │
│   ├── run_modes/                         # Purpose: run orchestrators — wire all components together; own no calculations themselves
│   │   ├── base_run.py                    # Responsibility: BaseRun ABC — validate / setup / execute / teardown contract
│   │   ├── liability_only_run.py          # Responsibility: LiabilityOnlyRun — liability cashflows and BEL only; no asset layer
│   │   ├── deterministic_run.py           # Responsibility: DeterministicRun — two-pass BEL design with full asset/liability projection loop
│   │   └── stochastic_run.py              # Responsibility: StochasticRun — N ESG scenarios; group model points only; produces TVOG distribution
│   │
│   ├── results/                           # Purpose: result collection, aggregation and post-processing
│   │   ├── result_store.py                # Responsibility: ResultStore — collects TimestepResult per (run_id, scenario_id, timestep, cohort_id)
│   │   ├── aggregator.py                  # Responsibility: fund → company rollup; AC and FVTPL portfolios aggregated separately before netting
│   │   └── tvog_calculator.py             # Responsibility: TVOG — E[max(guarantee_cost, 0)] discounted to valuation date across all scenarios
│   │
│   ├── ifrs17/                            # Purpose: standalone IFRS 17 GMM engine — product-agnostic; all GMM mechanics live here (DECISIONS.md §28)
│   │   ├── __init__.py
│   │   ├── gmm.py                         # Responsibility: GmmEngine — orchestrates CSM, loss component, finance income per contract group per period
│   │   ├── _gmm_jit.py                    # Responsibility: _gmm_step_inner — JAX JIT-compiled pure arithmetic core; NumPy fallback when JAX unavailable (DECISIONS.md §32)
│   │   ├── csm.py                         # Responsibility: CsmTracker — accretion at locked-in rate; release by coverage unit fraction
│   │   ├── loss_component.py              # Responsibility: LossComponentTracker — onerous excess recognition and release
│   │   ├── risk_adjustment.py             # Responsibility: CostOfCapitalRA — 6% CoC on longevity/lapse SCR projection
│   │   ├── coverage_units.py              # Responsibility: CoverageUnitProvider protocol — injected per product; BPA impl in Step 21
│   │   ├── state.py                       # Responsibility: Ifrs17State frozen dataclass — cross-period rolling balances per contract group
│   │   └── assumptions.py                 # Responsibility: LockedInAssumptions, CurrentAssumptions, AssumptionProvider protocol
│   │
│   └── bpa/                               # Purpose: BPA-specific liability classes and deal identification (DECISIONS.md §18, §44, §45)
│       ├── __init__.py
│       ├── registry.py                    # Responsibility: BPADealMetadata, BPADealRegistry (from_csv), make_cohort_id — deal lookup keyed by deal_id
│       ├── mortality.py                   # Responsibility: MortalityBasis, q_x(), survival_probs() — S3 tables + CMI improvement projection
│       ├── assumptions.py                 # Responsibility: BPAAssumptions, RetirementRates, DependantAssumptions frozen dataclasses
│       ├── in_payment.py                  # Responsibility: InPaymentLiability — single-decrement (death) annuity projection
│       ├── enhanced.py                    # Responsibility: EnhancedLiability — age-rated mortality; subclasses InPaymentLiability
│       ├── dependant.py                   # Responsibility: DependantLiability — convolution BEL; contingent on member death
│       └── deferred.py                    # Responsibility: DeferredLiability — 4-decrement (death, TV, ill-health, retirement) with phase transition
│
├── data/                                  # Purpose: raw file ingestion — translates inputs into validated engine-ready structures
│   ├── loaders/
│   │   ├── base_loader.py                 # Responsibility: BaseLoader ABC
│   │   ├── liability_data_loader.py       # Responsibility: returns unified DataFrame regardless of seriatim vs group MP source
│   │   ├── asset_data_loader.py           # Responsibility: loads asset portfolio CSV — enforces column contract and accounting basis rules (see Section 4)
│   │   ├── scenario_loader.py             # Responsibility: loads ESG scenario CSV into ScenarioStore via ScenarioEngine
│   │   └── bpa_data_loader.py             # Responsibility: loads BPA MP CSV files (4 pop. types) + mortality basis CSVs; accepts optional BPADealRegistry (DECISIONS.md §44)
│   └── validators/
│       ├── liability_validator.py         # Responsibility: validates liability MP DataFrame — columns, data types, value ranges
│       ├── asset_validator.py             # Responsibility: enforces asset column contract; rejects files with missing columns or invalid basis values
│       ├── scenario_validator.py          # Responsibility: validates ESG scenario CSV — required columns, no duplicate (scenario_id, timestep) pairs
│       └── bpa_validator.py              # Responsibility: validates BPA MP DataFrames — deal_id, tv_eligible, tranche_id completeness, registry cross-reference (DECISIONS.md §44)
│
├── storage/                               # Purpose: DB persistence — all SQLAlchemy access goes through repositories; nothing else imports SQLAlchemy
│   ├── models/
│   │   ├── run_record.py                  # Responsibility: RunRecord ORM model — run_id, run_type, status, config_json, timing, error_message
│   │   ├── result_record.py               # Responsibility: ResultRecord ORM model — one row per (run_id, scenario_id, timestep, cohort_id); 28 numeric columns
│   │   └── ifrs17_record.py               # Responsibility: Ifrs17StateRecord + Ifrs17MovementRecord ORM models — cross-period IFRS 17 accounting tables (DECISIONS.md §37)
│   ├── run_repository.py                  # Responsibility: CRUD for RunRecord — save, get, list_all, update_status
│   ├── result_repository.py               # Responsibility: bulk insert and query for ResultRecord — save_batch, get_dataframe
│   ├── ifrs17_state_repository.py         # Responsibility: Ifrs17StateStore — load/save Ifrs17State by (cohort_id, valuation_date); upsert movements (DECISIONS.md §33, §37)
│   └── db.py                              # Responsibility: SQLAlchemy engine + Session factory — swap SQLite → PostgreSQL by changing connection string here only
│
├── ai_layer/                              # Purpose: Multi-agent AI assistant (Phase 2 — Step 15); specialist agents added in Phase 3
│   ├── config.py                          # Responsibility: AILayerConfig dataclass — provider, model, api_key_env_var, base_url, api_version
│   ├── agent.py                           # Responsibility: ALMOrchestrator — routes requests to specialist agents; manages session history
│   ├── agents/
│   │   ├── base_agent.py                  # Responsibility: abstract agent with shared tool-loop logic and provider switching (Anthropic / OpenAI-compatible)
│   │   ├── run_analyst.py                 # Responsibility: RunAnalystAgent — explains BEL/TVOG/cash flows for a specific run (Phase 2)
│   │   ├── config_advisor.py              # Responsibility: ConfigAdvisorAgent — reads RunConfig, proposes revised assumptions, calls submit_run (Phase 2)
│   │   ├── reviewer.py                    # Responsibility: ReviewerAgent — cross-checks proposed RunConfig for actuarial consistency; always called after ConfigAdvisorAgent (Phase 2)
│   │   ├── ifrs17_specialist.py           # Responsibility: IFRS17Agent — CSM/RA/LRC/LIC explanation; deferred to Phase 3 Step 23
│   │   ├── solvency2_specialist.py        # Responsibility: SolvencyIIAgent — SCR stress, MA offset, capital adequacy; deferred to Phase 3 Step 23
│   │   └── bpa_specialist.py              # Responsibility: BPAAgent — matching adjustment, eligibility, fundamental spread; deferred to Phase 3 Step 23
│   ├── tools/
│   │   ├── get_run_results.py             # Responsibility: calls GET /results/{run_id}?format=csv + GET /results/{run_id}/summary; read-only
│   │   ├── get_run_config.py              # Responsibility: calls GET /runs/{run_id} and returns config_json as structured dict; read-only
│   │   └── submit_run.py                  # Responsibility: calls POST /runs with a RunConfig JSON string; requires human approval in UI before execution
│   ├── knowledge_base/
│   │   ├── model_docs.md                  # Responsibility: hand-curated actuarial glossary (BEL, TVOG, SCR, run types) — concepts not already covered in DECISIONS.md
│   │   ├── schema_export.py               # Responsibility: programmatically exports live RunConfig JSON schema as formatted text — stays in sync with code automatically
│   │   ├── decisions_loader.py            # Responsibility: loads and filters DECISIONS.md sections relevant to the requesting agent (financial rationale, modelling constraints)
│   │   └── architecture_loader.py         # Responsibility: loads selected sections of ALM_Architecture.md (module map, result fields, run type definitions)
│   └── sandbox/
│       └── test_runner.py                 # Responsibility: runs pytest against any AI-proposed code change before human approval (Phase 3)
│
├── tests/                                 # Purpose: test suite — unit and integration tests written alongside code
│   ├── unit/
│   │   ├── asset/
│   │   │   ├── test_bond.py               # AC and FVTPL paths tested independently (Step 7) — passing AC does not imply FVTPL correctness
│   │   │   ├── test_equity.py             # Total return / dividend yield split (Step 7)
│   │   │   └── test_asset_model.py        # Portfolio aggregation, MV by basis (Step 7)
│   │   ├── strategy/
│   │   │   └── test_investment_strategy.py # AC sell constraint, rebalancing logic (Step 7)
│   │   ├── core/
│   │   │   ├── test_fund.py               # Cash mechanics, forced sells, rebalancing step order (Step 8)
│   │   │   └── test_projection_calendar.py # ProjectionCalendar period counts, dt values, discount factors, constructor validation (Step 17)
│   │   ├── run_modes/
│   │   │   ├── test_deterministic_run.py  # validate/setup/execute/teardown/progress (Step 8)
│   │   │   ├── test_stochastic_run.py     # validate/setup/execute/teardown/progress (Step 9)
│   │   │   └── test_liability_only_run.py
│   │   ├── scenarios/
│   │   │   ├── test_scenario_store.py     # EsgScenario, ScenarioStore construction/retrieval (Step 9)
│   │   │   └── test_scenario_engine.py    # ScenarioLoader.from_csv and .flat (Step 9)
│   │   ├── results/
│   │   │   ├── test_result_store.py       # Index correctness, as_dataframe, cohort_id dimension (Step 10c)
│   │   │   └── test_tvog_calculator.py    # TVOG numerical anchor (Step 10)
│   │   ├── ifrs17/
│   │   │   ├── __init__.py
│   │   │   ├── test_ifrs17_state.py       # Ifrs17State construction, validation, immutability (Step 17)
│   │   │   ├── test_ifrs17_assumptions.py # LockedInAssumptions, CurrentAssumptions, AssumptionProvider (Step 17)
│   │   │   ├── test_ifrs17_coverage_units.py  # CoverageUnitProvider protocol contract (Step 17)
│   │   │   ├── test_ifrs17_csm.py         # CsmTracker accretion, release, onerous excess (Step 17)
│   │   │   ├── test_ifrs17_loss_component.py  # LossComponentTracker addition and release (Step 17)
│   │   │   ├── test_ifrs17_risk_adjustment.py # CostOfCapitalRA numerical anchors (Step 17)
│   │   │   ├── test_ifrs17_gmm.py         # GmmEngine end-to-end: CSM release schedule, LRC assembly, P&L (Step 17)
│   │   │   └── test_ifrs17_gmm_jit.py     # JAX availability, float64 precision, _gmm_step_inner numerical regression, GmmEngine JIT wiring (Step 18a)
│   │   ├── liability/
│   │   │   └── bpa/
│   │   │       ├── conftest.py            # Shared flat MortalityBasis fixture (synthetic S3/CMI tables for deterministic tests)
│   │   │       ├── test_mortality.py      # q_x formula, CMI improvement, LTR sensitivity, survival_probs (Step 18)
│   │   │       ├── test_in_payment.py     # BEL numerical anchor, cashflow per period, LPI cap/floor (Step 18)
│   │   │       ├── test_enhanced.py       # Age-rating: BEL with rating_years=5 matches InPayment at age+5 (Step 18)
│   │   │       ├── test_dependant.py      # Convolution BEL, zero-weight guard (Step 18)
│   │   │       └── test_deferred.py       # TV cashflow, 4-decrement sum ≤ 1.0, retirement conversion BEL (Step 18)
│   │   └── data/
│   │       ├── test_bpa_validator.py      # deal_id, tv_eligible, tranche_id, registry cross-reference (Step 18 — updated §44)
│   │       └── test_bpa_data_loader.py    # MP loading, mortality basis loading, column map, float coercion (Step 18)
│   │   ├── storage/
│   │   │   ├── conftest.py                # In-memory SQLite DB fixture shared across storage tests
│   │   │   ├── test_run_repository.py     # save, get, list_all, update_status (Step 11)
│   │   │   ├── test_result_repository.py  # save_batch, get_dataframe, bulk insert (Step 11)
│   │   │   └── test_ifrs17_state_repository.py  # Ifrs17StateStore: load, save, upsert, movements roundtrip (Step 17)
│   │   ├── liability/
│   │   │   ├── test_base_liability.py
│   │   │   ├── test_conventional.py
│   │   │   └── test_multi_decrement.py    # MultiDecrementLiability ABC contract (Step 10a)
│   │   ├── config/
│   │   │   ├── conftest.py
│   │   │   ├── test_run_config.py
│   │   │   ├── test_projection_config.py
│   │   │   └── test_fund_config.py
│   │   └── curves/
│   │       └── test_rate_curve.py
│   ├── integration/
│   │   ├── test_liability_only_run.py     # End-to-end: load data → run → check BEL
│   │   ├── test_deterministic_run.py      # Scenarios A (FVTPL) and B (AC amortisation) (Step 8)
│   │   └── test_fund_interaction.py
│   └── fixtures/
│       ├── sample_group_mps.csv
│       ├── sample_assets.csv              # Must include accounting_basis column
│       └── sample_scenarios.csv
│
├── utils/                                 # Purpose: shared helpers — no layer-specific logic; if a utility imports from engine/ it belongs in engine/
│   ├── logging_config.py                  # Responsibility: structured logging configuration
│   ├── date_utils.py                      # Responsibility: date arithmetic helpers
│   └── math_utils.py                      # Responsibility: numerical utilities (e.g. IRR solver wrappers)
│
├── config_files/                          # Purpose: example and template configuration files for actuarial team use
│   ├── run_config_template.yaml
│   ├── fund_config_template.yaml
│   └── projection_config_template.yaml
│
├── document/                              # Purpose: technical documentation — read before changing any code
│   ├── ALM_Architecture.md               # Responsibility: system-wide architecture — modules, interfaces, data flow, build order (this file)
│   └── DECISIONS.md                      # Responsibility: financial and actuarial modelling decisions — why, not what
│
├── main.py                                # CLI entry point
└── README.md
```

---

## 3. Layer Descriptions

### 3.1 Frontend

**Purpose:** User interaction only. Contains zero business logic.

The desktop app (PyQt6) communicates exclusively with the FastAPI layer via HTTP on `localhost`.
It never imports from `engine/` directly.

### 3.2 API

**Purpose:** Single interface between all frontends and the execution engine. Built with FastAPI.
Does not execute the model — hands off to the worker layer.

### 3.3 Worker

**Purpose:** Execute long-running model runs asynchronously via RQ + Redis.
Critical for runs of 30 minutes to 4 hours.

### 3.4 Engine

**Purpose:** The ALM model itself. Pure Python, no web/UI dependencies.

| Sub-layer | Responsibility |
|---|---|
| `config/` | Defines and validates all input parameters |
| `core/` | Owns the projection loop and fund-level coordination |
| `asset/` | Projects asset cash flows and market values — basis-aware |
| `liability/` | Projects liability cash flows per policy type |
| `strategy/` | Controls investment rebalancing and bonus crediting |
| `scenarios/` | Manages ESG scenario data for stochastic runs |
| `run_modes/` | Orchestrates which components run in which order |
| `results/` | Collects, aggregates, and exports all outputs |

### 3.5 Data

**Purpose:** Translate raw input files into validated structures the engine can consume.

The `asset_data_loader.py` must enforce the column contract defined in Section 4.
The `liability_data_loader.py` returns a unified DataFrame regardless of seriatim vs group MP source.

### 3.6 Storage

SQLAlchemy ORM against SQLite locally. Swap to PostgreSQL by changing connection string in `db.py` only.

### 3.7 AI Layer (Phase 2 — Step 15)

Multi-agent AI assistant built on a provider-agnostic foundation (`AILayerConfig`).  Supports
Anthropic Claude (default), any OpenAI-compatible endpoint, or in-house self-hosted models —
configured via `provider`, `model`, `api_key_env_var`, and `base_url`.

#### Agent roles (Phase 2 — conventional products)

| Agent class | Role | Mutating? |
|---|---|---|
| `ALMOrchestrator` | Routes user message to correct specialist agent; owns session history | No |
| `RunAnalystAgent` | Explains BEL, TVOG, cash flow results for a given run | No |
| `ModellingAgent` | Explains how the engine calculates results — algorithms, code logic, design rationale | No |
| `ConfigAdvisorAgent` | Suggests assumption changes; proposes a revised RunConfig JSON | Via `submit_run` (human-approved) |
| `ReviewerAgent` | Cross-checks proposed config for actuarial consistency; returns JSON verdict | No |

#### Agent roles (Phase 3 — specialist domains, Step 23)

| Agent class | Role |
|---|---|
| `IFRS17Agent` | CSM, RA, LRC/LIC movements; GMM result interpretation |
| `SolvencyIIAgent` | SCR spread/interest stress, MA offset, capital adequacy |
| `BPAAgent` | Matching adjustment eligibility, fundamental spread, BPA BEL explanation |


The AI layer supports three deployment modes: Anthropic API (default for development), any
OpenAI-compatible endpoint, or in-house self-hosted models. A lightweight LLM classifier
routes each user query to the appropriate specialist agent. A reviewer pattern cross-checks
any proposed config change before the actuary approves submission. The tool surface is
intentionally narrow (read-only results, read-only config, submit-with-approval) — the AI
never mutates the model state autonomously.

Knowledge base content (actuarial glossary, relevant DECISIONS.md sections, and the live
RunConfig schema) is injected directly into each agent's system prompt. This is not RAG.

### 3.8 Tests

Write tests at the same time as code. Most critical tests are numerical:
given known inputs, cash flows / BEL / TVOG must reproduce known correct values.

**Bond tests must cover both AC and FVTPL paths independently.**
A bond that passes AC tests is not assumed to pass FVTPL tests — they are separate code paths.

### 3.9 IFRS 17 Engine (Phase 3 — Steps 17 & 18a)

**Purpose:** Standalone, product-agnostic IFRS 17 General Measurement Model (GMM) engine.
Products inject BEL values, a `CoverageUnitProvider`, and the locked-in discount rate.
`GmmEngine` handles all GMM mechanics uniformly. See DECISIONS.md §28 for the full design rationale.

#### GmmStepResult fields

| Field | Description |
|---|---|
| `csm_opening` | CSM at start of period |
| `csm_accretion` | Interest at locked-in rate × year_fraction |
| `csm_adjustment_non_financial` | FCF change from mortality/inflation (adjusts CSM) |
| `csm_release` | Released to P&L by coverage unit fraction |
| `csm_closing` | CSM at end of period (always ≥ 0) |
| `loss_component_opening` | Opening loss component balance |
| `loss_component_addition` | Onerous excess transferred from CSM |
| `loss_component_release` | Released to P&L as cashflows run off |
| `loss_component_closing` | Closing loss component balance |
| `insurance_finance_pl` | BEL unwinding at locked-in rate — P&L |
| `insurance_finance_oci` | Change in current-vs-locked gap — OCI |
| `lrc` | Liability for Remaining Coverage = bel_current + RA + csm_closing |
| `lic` | Liability for Incurred Claims (past-service BEL + RA) |
| `p_and_l_csm_release` | P&L from CSM release |
| `p_and_l_loss_component` | Net P&L from loss component (release − addition) |
| `p_and_l_insurance_finance` | = insurance_finance_pl |
| `bel_current`, `bel_locked`, `risk_adjustment` | Input echoes for traceability |


---

## 4. Module Responsibilities

> This section lists each module and its single-sentence responsibility.
> Full method signatures and interfaces are available in the source code.

| Module | Responsibility |
|---|---|
| `engine/config/run_config.py` | Master Pydantic config object for a full projection run |
| `engine/core/top.py` | Owns the time step loop; calls Fund at each period |
| `engine/core/fund.py` | Coordinates AssetModel ↔ BaseLiability per fund per timestep |
| `engine/core/projection_calendar.py` | Hybrid timestep calendar (monthly/annual) for BPA runs |
| `engine/asset/base_asset.py` | ABC for all assets — valuation, P&L decomposition, step_time |
| `engine/asset/bond.py` | Fixed-income bond — EIR (AC), MV reset (FVTPL), OCI (FVOCI) |
| `engine/asset/equity.py` | Equity — FVTPL only; total return applied to market value |
| `engine/asset/asset_model.py` | Portfolio container — aggregates by accounting basis |
| `engine/liability/base_liability.py` | ABC for all liabilities — decrement and cashflow contract |
| `engine/liability/multi_decrement.py` | Multi-decrement extension for BPA liability classes |
| `engine/liability/conventional.py` | Conventional product — death, lapse, maturity decrements |
| `engine/liability/bpa/` | BPA liability classes (in-payment, deferred, dependant, enhanced) |
| `engine/matching_adjustment/` | MA eligibility, FS lookup, MA benefit computation |
| `engine/ifrs17/gmm.py` | IFRS 17 GMM engine — CSM, loss component, finance income |
| `engine/ifrs17/_gmm_jit.py` | JAX JIT-compiled GMM inner step; NumPy fallback |
| `engine/strategy/investment_strategy.py` | SAA rebalancing — AC bond constraint enforced here |
| `engine/scenarios/scenario_store.py` | Holds all ESG scenarios; O(1) retrieval by scenario_id |
| `engine/results/result_store.py` | Collects TimestepResult per (run_id, scenario_id, timestep) |
| `engine/results/tvog_calculator.py` | TVOG = E[stochastic BEL] − deterministic BEL |
| `engine/curves/rate_curve.py` | Smith-Wilson RFR curve (EIOPA methodology) |


## 5. Run Mode Logic

```
User selects run type
        │
        ├──► Liability Only Run
        │       ├── Load: seriatim policies OR group MPs
        │       ├── Run: Conventional / Unit-Linked / Annuity models
        │       ├── Aggregate at liability model level
        │       └── Output: BEL, reserves, liability CFs (no assets)
        │
        ├──► Deterministic Run
        │       ├── Load: seriatim policies, full asset portfolio, single scenario
        │       ├── Run: Full Fund loop (liability + asset + strategy)
        │       ├── Aggregate: Fund → Company
        │       └── Output: Full P&L, asset/liability values by accounting basis, bonus rates
        │
        └──► Stochastic Run
                ├── Load: GROUP MODEL POINTS ONLY (seriatim not permitted)
                ├── Load: N ESG scenarios
                ├── For each scenario: run full Fund loop
                ├── Aggregate: Fund → Company per scenario
                ├── Collect stochastic result distribution
                └── Output: TVOG, stochastic P&L distribution
```

---

## 6. Data Flow

### 6.1 System-Level Flow

```
Input Files (CSV/Excel)
        │
        ▼
data/loaders/                    ← Reads raw files into DataFrames
        │
        ▼
data/validators/                 ← Checks completeness, validity, accounting basis rules,
        │                           required columns, date ranges, rating scale membership
        ▼
engine/config/run_config.py      ← Validated Pydantic config object injected into run mode
        │
        ▼
engine/run_modes/{mode}_run.py   ← Constructs and wires all components; owns no calculations
        │
        ├──► engine/scenarios/   ← ESG scenario data loaded into ScenarioStore (stochastic only)
        │
        ▼
engine/core/top.py               ← Outer loop: iterates over time steps (and scenarios)
        │
        ▼
engine/core/fund.py              ← Inner loop: coordinates asset ↔ liability at each time step
        │                           See Section 6.2 for detailed calculation sequence
        ▼
engine/results/result_store.py   ← Collects per-bond, per-basis outputs at each time step
        │
        ▼
engine/results/aggregator.py     ← Fund → Company rollup
        │                           AC and FVTPL bonds aggregated separately before netting
        ▼
storage/result_repository.py     ← Persists to DB
        │
        ▼
api/routers/results.py           ← Served to frontend
```

---

### 6.2 Per-Time-Step Calculation Sequence (inside Fund)

Each timestep executes six ordered stages inside `engine/core/fund.py`:

1. **Liability cash flows** — decrements applied; gross outflows calculated per product
2. **Asset income collection** — coupons, redemptions, dividends collected into cash
3. **Pay liability outflows** — net outflow deducted from cash; shortfall flagged
4. **Portfolio valuation** — reprice MV (risk-free + calibration spread); book value walked via EIR; unrealised G/L computed by basis
5. **Rebalancing** — `InvestmentStrategy` trades to SAA targets; AC bonds excluded from sale candidates
6. **Advance time** — `step_time()` called on each asset; liability in-force population updated

---

## 7. Class Hierarchy

```
BaseAsset (ABC)
    ├── Bond                  (accounting_basis: AC | FVTPL | FVOCI)
    ├── Equity
    └── Derivative

AssetModel
    └── holds list of BaseAsset instances

BaseLiability (ABC)
    ├── Conventional          (mode: seriatim | group_mp)
    ├── UnitLinked            (mode: seriatim | group_mp)
    └── Annuity               (mode: seriatim | group_mp)

LiabilityModel
    └── holds instances of BaseLiability subclasses

BaseStrategy (ABC)
    ├── InvestmentStrategy    (SAA rebalancing — conventional funds; AC bond constraint enforced here)
    ├── BuyAndHoldStrategy    (no-op rebalancing — BPA MA portfolio; Step 20)
    └── BonusStrategy

BaseRun (ABC)
    ├── LiabilityOnlyRun
    ├── DeterministicRun
    └── StochasticRun

BaseLoader (ABC)
    ├── LiabilityDataLoader
    ├── AssetDataLoader       (enforces column contract + accounting basis validation)
    └── ScenarioLoader

--- IFRS 17 Engine (Phase 3 — Step 17) ---

GmmEngine                     (product-agnostic; orchestrates CSM + loss component + finance income)
    └── holds CsmTracker and LossComponentTracker per contract group

CsmTracker                    (accretion at locked-in rate; release by coverage unit fraction)
LossComponentTracker          (onerous excess addition; proportional release)
CostOfCapitalRA               (6% CoC × SCR projection; scalar RA output)

CoverageUnitProvider (Protocol)
    └── BPACoverageUnitProvider (Step 21 — engine/liability/bpa/)

Ifrs17State (frozen dataclass)  (cross-period rolling balances; immutable; persisted via Ifrs17StateStore)
LockedInAssumptions (frozen dataclass)
CurrentAssumptions (frozen dataclass)
AssumptionProvider (Protocol)

GmmStepResult (frozen dataclass)  (all GMM movements for one contract group one period)

--- Projection Calendar (Phase 3 — Step 17) ---

ProjectionCalendar            (generates ordered sequence of ProjectionPeriod objects)
    └── ProjectionPeriod (frozen dataclass)  (period_index, year_fraction, is_monthly, time_in_years)

--- IFRS 17 Storage (Phase 3 — Step 17) ---

Ifrs17StateStore              (load/save Ifrs17State; save/get GmmStepResult movements)
    └── uses Ifrs17StateRecord ORM (ifrs17_state table)
    └── uses Ifrs17MovementRecord ORM (ifrs17_movements table)
```

---

## 8. Bond Accounting Basis Framework

This section is the authoritative reference for how accounting basis affects every downstream
calculation. All bond-related code must be consistent with this section.

### 8.1 Three Accounting Bases

| Basis | Balance Sheet Asset Value | P&L Impact | OCI Reserve |
|---|---|---|---|
| AC (Amortised Cost) | Closing book value (EIR amortised) | EIR income only | None |
| FVTPL (Fair Value through P&L) | Market value | Coupon + MV movement | None |
| FVOCI (Fair Value through OCI) | Market value | EIR income only | Cumulative MV movements |

### 8.2 Effective Interest Rate (EIR) Method — AC Only

The EIR is the internal rate of return of the bond's cash flows at purchase price.
It is locked at purchase and never recalculated.

Each period under AC:
```
new_book_value = old_book_value + (old_book_value × EIR) - coupon_paid
```

The book value converges to par at maturity. If the bond was purchased at a discount,
book value walks upward. If purchased at a premium, it walks downward.

**Numerical example (purchase at 95, 5% coupon on par 100, EIR ≈ 6.9%, 3 years):**

| Year | Opening BV | EIR Income | Coupon | Closing BV | Market Value | Unrealised G/L |
|---|---|---|---|---|---|---|
| 1 | 95.00 | 6.55 | 5.00 | 96.55 | 97.00 | +0.45 (disclosed, not in P&L) |
| 2 | 96.55 | 6.67 | 5.00 | 98.22 | 99.00 | +0.78 (disclosed, not in P&L) |
| 3 | 98.22 | 6.78 | 5.00 | 100.00 | 100.00 | 0 |

### 8.3 FVTPL — MV Movement Through P&L

Book value is reset to market value every period. MV movement hits P&L immediately.
"Unrealised" means the bond has not been sold — it does not mean the gain is unrecognised.

**Same bond under FVTPL (MV path: 97, 99, 100):**

| Year | Opening BV | Coupon | MV Movement | Closing BV | Total P&L |
|---|---|---|---|---|---|
| 1 | 95.00 | 5.00 | +2.00 | 97.00 | 7.00 |
| 2 | 97.00 | 5.00 | +2.00 | 99.00 | 7.00 |
| 3 | 99.00 | 5.00 | +1.00 | 100.00 | 6.00 |

Total P&L = 20 = 3 × coupon (15) + discount gain (5). Consistent with economic return.

### 8.4 Calibration Spread

The calibration spread is a parallel shift to the risk-free discount curve, solved at the
valuation date so that discounted bond cash flows equal the market value in the input data:

```
Solve s such that: Σ [CF_t / (1 + r_t + s)^t] = Market Value at valuation date
```

Once solved, the calibration spread is **locked** for the entire projection.
It is used for:
1. Projecting market value at future time steps
2. Calculating the default allowance on future cash flows

The calibration spread is distinct from the quoted credit spread (corporate minus government yield).
They will be numerically close but differ because the calibration spread is model-specific
(dependent on the exact discount curve used) and is a single parallel shift rather than
a term-structure-aware spread.

### 8.5 Credit Spread Stress

When running a credit spread stress scenario (e.g., +/- 100bps):

**Asset side:**
- Reprice all bonds using stressed spread: `s_stressed = s_calibration + shock`
- Duration determines sensitivity: longer duration = larger MV impact
- AC bonds: book value unchanged; only market value (and disclosed unrealised G/L) changes
- FVTPL bonds: book value changes = immediate P&L impact

**Liability side:**
- Matching Adjustment (MA) or Volatility Adjustment (VA) changes with credit spreads
- Liability discount rate changes → liability value changes
- MA is only available for AC-designated bond portfolios (IFRS 9 / Solvency II constraint)
- Net own funds impact = change in assets − change in liabilities (must be computed together)

**Default allowance:**
- Spread widening implies higher default probability
- Default allowance on future cash flows must be re-estimated consistently with the stress
- Do not shock the discount rate without also adjusting expected cash flows

**Asymmetry:**
- Spread widening shock is typically larger than tightening shock
- Spreads are bounded near zero — symmetric treatment is incorrect

### 8.6 Accounting Basis and Rebalancing Constraints

AC bonds must not be sold for routine SAA rebalancing because:
- Sale crystallises a P&L gain/loss previously not recognised
- Under IFRS 9, frequent sales of AC bonds can trigger reclassification of the entire portfolio
- Tax implications differ from FVTPL sales

`InvestmentStrategy` must respect this constraint. AC bonds can only be sold when:
- The bond matures
- An explicit forced-sale override is set in run config
- The bond defaults

---

## 9. Opening Balance Sheet Initialisation

This section defines how the model handles the inevitable mismatch between total asset
values (read from input) and total liability values (calculated by the liability model)
at the valuation date. This logic executes once, inside the run mode orchestrator,
before the first time step.

### 9.1 Why a Mismatch Always Exists

Asset values come from the asset input file — market prices at the valuation date sourced
from a pricing system or custodian. Liability values are calculated by the liability model
from policy data, assumptions, and a discount curve. These are produced independently,
at potentially different times, by different teams and systems.

The difference between them is not an error. It is the **opening surplus** of the fund —
the capital buffer sitting above the liability value. For a solvent insurer this will be
positive. It is a real economic quantity that must be modelled, not zeroed out.

### 9.2 Approach: Surplus Initialised into Cash Account

The model uses a combination of explicit surplus tracking and a cash account as the
balancing item. This preserves the integrity of all individual bond data (EIRs,
calibration spreads, book values) while ensuring the balance sheet balances from day one.

```
At valuation date initialisation (run mode orchestrator):

  total_asset_MV      = Σ market_value of all assets in input file
  total_liability_BEL = calculated BEL from liability model

  opening_surplus     = total_asset_MV - total_liability_BEL

  cash_balance        = opening_surplus   ← surplus parked in cash account
```

The cash account is a first-class component of the fund. It is not a rounding bin.
It earns interest each period at the risk-free short rate from the scenario.
It absorbs net liability outflows when coupon and dividend income is insufficient.
It grows when asset income exceeds liability outflows.

### 9.3 Balance Sheet at Valuation Date

```
ASSETS                              LIABILITIES & SURPLUS
─────────────────────────────────   ─────────────────────────────────
Bond portfolio (AC)      £Xm        BEL (calculated)         £Lm
Bond portfolio (FVTPL)   £Ym
Equities                 £Zm        SURPLUS
Cash (opening surplus)   £Sm        Opening surplus           £Sm
                                    (= total assets - BEL)
─────────────────────────────────   ─────────────────────────────────
Total assets             £(X+Y+Z+S)m  Total L + surplus      £(L+S)m
```

The balance sheet balances by construction. The surplus is shown explicitly on the
liability side as a separate line, not netted into any other figure.

### 9.4 Validation: Mismatch Threshold

Before initialising the cash account, the run mode orchestrator must validate that the
mismatch is within an acceptable range. A very large mismatch almost always indicates
a data problem, not a genuine surplus.

Common causes of a large mismatch to investigate before proceeding:
- Asset data is not at the same valuation date as the liability run
- Liability model is using a different discount curve than intended
- Currency mismatch between asset and liability inputs
- Assets include non-fund assets (e.g., shareholder assets loaded in error)
- Liability BEL excludes a product line that the assets back

### 9.5 Liability Basis Used for Surplus Calculation

The opening surplus is always calculated against the **Best Estimate Liability (BEL)**.

Other liability measures are calculated alongside BEL but do not affect the opening
cash balance:

| Liability Measure | Used for Surplus? | Purpose |
|---|---|---|
| BEL | **Yes** — defines opening surplus | Projection baseline |
| Risk Margin | No | Added to BEL for Solvency II SCR |
| Statutory Reserve | No | Regulatory reporting only |
| IFRS17 (future) | No | Separate reporting basis |

The risk margin is reported as a separate output. It does not enter the projection
engine as a cash flow or asset requirement in Phase 1.

### 9.6 Cash Account Behaviour Through the Projection

The cash account evolves at each time step as part of the Step 6 calculation
(see Section 6.2). Its closing balance each period becomes the opening balance next period.

```
cash_balance(t) = cash_balance(t-1)
                + coupon_income(t)
                + dividend_income(t)
                + redemption_proceeds(t)      ← bonds maturing
                + sale_proceeds(t)            ← from rebalancing
                - liability_outflows(t)       ← claims, surrenders, expenses
                - purchase_costs(t)           ← from rebalancing
                + cash_interest(t)            ← risk-free short rate × cash_balance(t-1)
```

If `cash_balance(t)` goes negative at any point, the model triggers the forced asset
sale logic in Step 5f (Section 6.2) before allowing the time step to complete.

A persistently negative cash balance across multiple periods is a model output signal
that the fund is consuming its capital buffer — this is meaningful actuarial information,
not an error to be suppressed.

### 9.7 What NOT to Do

These approaches are explicitly prohibited in this model:

- **Do not scale asset values to match liabilities.** This destroys book values, EIRs,
  and calibration spreads that were computed from the original market values.
- **Do not scale liabilities to match assets.** This changes the actuarial calculation
  to fit the data, which is the wrong direction.
- **Do not zero out the mismatch silently.** The opening surplus is real economic
  information. Hiding it produces a model that appears balanced but is not.
- **Do not aggregate the surplus into an existing asset category.** It must be held
  in the explicit cash account so it is visible in results and earns the correct return.

---

## 10. Technology Stack

| Layer | Technology | Notes |
|---|---|---|
| Desktop UI | PyQt6 | Native, handles long runs cleanly |
| Web UI (Phase 2) | React | Same FastAPI backend |
| API | FastAPI | Async, Pydantic-native, auto-docs |
| Job Queue | RQ + Redis | Right-sized for team |
| Database | SQLAlchemy + SQLite | Swap to PostgreSQL for web |
| Numerical | NumPy + Pandas | Core computation layer |
| Config Validation | Pydantic v2 | Throughout: config, schemas, contracts |
| Testing | pytest | Unit + integration from day one |
| AI Layer | Anthropic/GPT/Llama/Gwen API | Phase 2 agent |
| AI Code Editing | libcst | Structured AST edits, not string replace |
| Version Control | Git + GitPython | AI agent commits via API |

---

## 11. Build Order

The model was built in four phases:

| Phase | Scope | Status |
|---|---|---|
| 1 | Core engine — config, liability models, result store, asset layer, fund loop, stochastic run, TVOG | Complete |
| 2 | API, worker, storage, desktop UI, AI assistant core (conventional products) | Complete |
| 3 | IFRS 17 GMM engine, BPA liability classes, Matching Adjustment, BPA run mode, BPA IFRS 17 integration, SCR, specialist AI agents | In progress |
| 4 | BPA portfolio construction optimiser (cashflow/duration matching, MA optimisation) | Planned |

Full modelling decisions for each step are in [DECISIONS.md](DECISIONS.md).

---

## 12. Design Principles

**1. The engine knows nothing about the UI.**
`engine/` has zero imports from `frontend/`, `api/`, or `worker/`. Enforced, not aspirational.

**2. Models receive inputs; they do not fetch them.**
No model class reads from files or databases. Data is loaded, validated, and injected by the run mode orchestrator.

**3. The time loop lives only in `top.py`.**
No liability or asset model advances time internally.

**4. Results are never stored inside models.**
All outputs go to `ResultStore`. Models are stateless between time steps.

**5. Strategies are always injected, never hardcoded.**
`Fund` receives an `InvestmentStrategy` at construction. Swapping strategy logic requires no changes to `Fund`.

**6. Seriatim is a data concern, not a model concern.**
Liability models receive a DataFrame. Whether it came from seriatim policies or group MPs is the loader's concern.

**7. The database is behind a repository.**
No router or engine module calls SQLAlchemy directly.

**8. Accounting basis is a bond-level property, not a model-level switch.**
Each bond carries its own `accounting_basis` field. All downstream calculations — P&L, balance sheet,
rebalancing, spread stress — branch on this field at the individual bond level.
There is no single model-level AC/FVTPL toggle. Aggregation happens after per-bond calculations, never before.

**9. The opening surplus is a real economic quantity, not a rounding error.**
The difference between total asset market value and BEL at the valuation date is initialised
into the cash account and projected forward. It is never zeroed, scaled, or silently absorbed.
Asset and liability inputs are never modified to force a match. See Section 9.

**10. The AI layer modifies strategy and config first, engine logic never (initially).**
`code_modifier.py` is scoped to `engine/strategy/` and `config_files/` only in Phase 2.
