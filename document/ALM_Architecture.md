# PythonALM — System Architecture

> **Version:** Public Demo (redacted from production v2.7)
> **Last Updated:** April 2026
> **Architecture Pattern:** Package-Based OOP with Plugin AI Layer

> **Note:** This is the public-facing architecture document. Certain module interfaces,
> algorithm details, and calibration specifications are omitted to protect proprietary
> implementations. The structure, layer design, and design principles are fully representative
> of the production system.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Directory Structure](#2-directory-structure)
3. [Layer Descriptions](#3-layer-descriptions)
4. [Run Mode Logic](#4-run-mode-logic)
5. [Data Flow](#5-data-flow)
6. [Technology Stack](#6-technology-stack)
7. [Build Order](#7-build-order)
8. [Design Principles](#8-design-principles)

---

## 1. System Overview

PythonALM is a segregated fund-based Asset and Liability Model (ALM) for life insurance actuarial teams. It supports:

- **Conventional with-profits products** — deterministic and stochastic projections, TVOG, bonus crediting
- **Bulk Purchase Annuity (BPA) portfolios** — in-payment, deferred, dependant, and enhanced annuities with full mortality decrements
- **IFRS 17 Gross Margin Model** — CSM, Risk Adjustment, LRC/LIC, loss component
- **Solvency II Standard Formula** — full BSCR aggregation with correlation matrix, Risk Margin
- **Matching Adjustment** — eligibility screening, fundamental spread, MA-adjusted discount curve

The system is structured in four layers: Calculation Engine → Storage → API → Frontend/AI. The engine has zero imports from any layer above it.

---

## 2. Directory Structure

```
alm_system/
│
├── engine/                      # Core calculation engine (zero UI/API imports)
│   ├── config/                  # Run configuration schema (Pydantic v2)
│   ├── liability/               # Liability models
│   │   ├── bpa/                 # BPA liability classes (in-payment, deferred, dependant)
│   │   └── ...                  # Conventional, multi-decrement base
│   ├── asset/                   # Asset models (bonds, equities)
│   ├── scenarios/               # ESG scenario store and loader
│   ├── ifrs17/                  # IFRS 17 GMM engine
│   ├── scr/                     # Solvency II SCR engines
│   ├── matching_adjustment/     # MA eligibility and fundamental spread
│   ├── portfolio_optimiser/     # LP-based MA-optimal asset selection
│   ├── strategy/                # Investment and bonus strategies
│   ├── core/                    # Fund coordinator, projection calendar
│   ├── run_modes/               # Run orchestrators
│   ├── results/                 # ResultStore and TVOG calculator
│   └── curves/                  # Rate curves
│
├── api/                         # FastAPI REST layer
│   ├── routers/                 # Endpoint handlers
│   └── schemas/                 # Pydantic request/response models
│
├── worker/                      # RQ background job processor
│
├── storage/                     # SQLAlchemy persistence layer
│   ├── models/                  # ORM models
│   └── ...                      # Repositories (run, result, batch, IFRS 17 state)
│
├── ai_layer/                    # Multi-agent AI system
│   ├── agents/                  # Specialist agents
│   ├── tools/                   # AI tool framework
│   └── knowledge_base/          # Architecture and code context loaders
│
├── frontend/
│   └── desktop/                 # PyQt6 desktop application
│
├── data/                        # Data loaders and validators
│   ├── loaders/                 # Liability, asset, BPA loaders
│   ├── validators/              # Model point and assumption validators
│   ├── mortality/               # Mortality table handlers
│   └── tools/                   # Utility tools (MP compression, basis comparison)
│
├── tests/                       # Test suite (2,000+ tests)
│   ├── unit/                    # Unit tests mirroring engine structure
│   ├── integration/             # Full end-to-end run tests
│   └── fixtures/                # Test data and sample model points
│
├── config_files/                # Sample run configuration files
├── outputs/                     # Run outputs (Parquet, Excel)
├── document/                    # Documentation
├── main.py                      # CLI entry point
├── Dockerfile                   # Container definition
└── docker-compose.yml           # Full-stack orchestration
```

---

## 3. Layer Descriptions

### 3.1 Calculation Engine (`engine/`)

The engine is a self-contained Python package with **zero imports** from `api/`, `frontend/`, or `worker/`. All data is injected by the run mode orchestrator; the engine never reads files or databases directly.

**Key modules:**

| Module | Responsibility |
|---|---|
| `engine/config/` | Pydantic v2 run configuration schema. Validates all inputs before a run starts. |
| `engine/liability/` | Abstract `BaseLiability` with multi-decrement interface. Concrete implementations: conventional with-profits, BPA in-payment, deferred, dependant, enhanced. |
| `engine/asset/` | `BaseAsset` with `step_time()` interface. Bond model handles AC/FVTPL/FVOCI accounting per IFRS 9. Equity model. `AssetModel` portfolio container. |
| `engine/scenarios/` | `ScenarioStore` holds ESG paths (risk-free rates, equity returns, inflation). `ScenarioLoader` reads from Parquet. |
| `engine/ifrs17/` | `GmmEngine` — CSM accumulation, RA projection, LRC/LIC, loss component, coverage unit allocation. JAX JIT inner step with NumPy fallback. |
| `engine/scr/` | Stress engines for spread, interest, longevity, lapse, expense, currency, counterparty. `BSCRAggregator` with SII correlation matrix. `RiskMarginCalculator`. |
| `engine/matching_adjustment/` | MA eligibility rules, fundamental spread lookup, MA benefit per asset. |
| `engine/portfolio_optimiser/` | LP solver for MA-optimal BPA asset selection at deal pricing. |
| `engine/strategy/` | `InvestmentStrategy` ABC with buy-and-hold and rebalancing implementations. `BonusStrategy` for with-profits PAR products. |
| `engine/core/` | `Fund` coordinates assets, liabilities, and strategies. `ProjectionCalendar` defines the hybrid monthly/annual timestep grid. |
| `engine/run_modes/` | `DeterministicRun`, `StochasticRun`, `BPARun`, `LiabilityOnlyRun`. Each orchestrates the time loop and writes to `ResultStore`. |
| `engine/results/` | `ResultStore` (Pandas-backed result collector), `TvogCalculator`, cohort pivot. |

### 3.2 API Layer (`api/`)

FastAPI application exposing:
- `POST /runs` — submit a run config; returns `run_id`
- `GET /runs/{run_id}` — poll run status and progress
- `GET /results/{run_id}` — retrieve result data
- `POST /batches` — submit multiple run configs in one call
- `GET /batches/{id}` — batch status
- `GET /workers` — worker status (read-only)
- `POST /ai/query` — AI assistant query

Pydantic v2 schemas enforce request/response contracts. Dependency injection wires repositories into route handlers.

### 3.3 Worker Layer (`worker/`)

RQ (Redis Queue) background job processor. Picks up run jobs enqueued by the API, executes the appropriate run mode, and writes progress back via the API. Multiple workers can run in parallel; the desktop UI controls spawn/kill via worker count selector.

### 3.4 Storage Layer (`storage/`)

Repository pattern over SQLAlchemy. Nothing outside the repository files calls SQLAlchemy directly.

| Repository | Stores |
|---|---|
| `RunRepository` | Run records, status, config snapshot |
| `ResultRepository` | Time-series results per run |
| `BatchRepository` | Batch submission records |
| `Ifrs17StateStore` | IFRS 17 opening balances and movement tables |

SQLite in development; PostgreSQL in production.

### 3.5 AI Layer (`ai_layer/`)

Multi-agent system powered by a configurable LLM provider (Anthropic Claude, OpenAI-compatible, or in-house model). Specialist agents cover IFRS 17, Solvency II, BPA, architecture, data review, and regulatory research. A tool framework gives agents read access to run configs, results, assumption tables, and code interfaces. A dual-config orchestrator separates primary modelling queries from regulatory research.

### 3.6 Desktop UI (`frontend/desktop/`)

PyQt6 application. Key windows:
- **Run Config Window** — full run configuration form (run type, input sources, projection settings, output)
- **Results Window** — tabular and chart result viewer
- **Workers Window** — live worker count and status
- **ESG Scenarios Window** — scenario path viewer
- **AI Assistant Window** — in-app AI chat with context binding to current run

---

## 4. Run Mode Logic

### Time Loop

The projection time loop lives **only** in `engine/core/top.py`. No liability or asset model advances time internally. Each time step:

1. `Fund.rebalance()` — strategy-driven asset rebalancing
2. `Fund.step_time()` — asset coupon/dividend, liability decrement, premium/claim cash flows
3. `ResultStore.store()` — write timestep result

### Run Modes

| Run Mode | Description |
|---|---|
| `LiabilityOnlyRun` | Projects liabilities only — no assets. Used for BEL validation. |
| `DeterministicRun` | Single deterministic scenario. Two-pass design: first pass computes BEL for discount curve; second pass runs full projection. |
| `StochasticRun` | N Monte Carlo scenarios. Supports serial and JAX-vectorised paths. TVOG computed from spread of scenario BELs. |
| `BPARun` | Full BPA projection: MA calibration pre-pass, BEL projection, IFRS 17 GMM, SCR calculations. Per-cohort result storage. |

### Vectorised Stochastic Path

`StochasticConfig.use_vectorised = True` activates JAX `vmap` over model points. Pure step functions are stateless and JAX-traceable. `BaseLiability.batch_step()` provides a default loop fallback; concrete implementations override with `vmap`. Regression tests confirm numerical equivalence between serial and vectorised paths.

---

## 5. Data Flow

```
Input Files (YAML / CSV / Parquet)
        │
        ▼
  Data Loaders + Validators
        │
        ▼
  Run Mode Orchestrator  ──────────────────────────────┐
        │                                              │
        ▼                                              ▼
  engine/core/Fund                             ResultStore
        │
   ┌────┴────────────────┐
   ▼                     ▼
AssetModel          BaseLiability
(bonds, equities)   (conventional / BPA)
        │
        ▼ (BPARun only)
  GmmEngine  ──►  Ifrs17StateStore
  SCRCalculator
  BSCRAggregator
```

---

## 6. Technology Stack

| Layer | Technology |
|---|---|
| Language | Python 3.12 |
| Package manager | `uv` |
| Numerical | NumPy, Pandas, JAX (`jit`, `vmap`, float64 x64 mode) |
| Config validation | Pydantic v2 |
| Desktop UI | PyQt6 |
| API | FastAPI |
| Job queue | RQ + Redis |
| Database | SQLAlchemy + SQLite (dev) / PostgreSQL (prod) |
| AI | Anthropic API · OpenAI-compatible · in-house LLM |
| Testing | pytest |
| Containers | Docker + docker-compose |

---

## 7. Build Order

The system was built in four phases:

| Phase | Steps | Scope |
|---|---|---|
| **Phase 1** | 1–10 | Core engine: config, liabilities, assets, scenarios, stochastic runs, TVOG |
| **Phase 2** | 11–16 | API, worker, storage, desktop UI, AI core, end-to-end validation |
| **Phase 3** | 17–27a | IFRS 17 GMM, BPA engine, MA, SCR, portfolio optimiser, bonus strategy |
| **Phase 4** | Ongoing | Extended SCR modules, BPA stochastic vectorisation, AI agent expansion |

Each phase was validated with a full test suite before the next phase began.

---

## 8. Design Principles

1. **`engine/` has zero imports from `frontend/`, `api/`, or `worker/`** — enforced, not aspirational.
2. **Models do not read files or databases.** Data is loaded and injected by the run mode orchestrator.
3. **The time loop lives only in `engine/core/top.py`.** No model advances time internally.
4. **Results are never stored inside models.** All outputs go to `ResultStore`. Models are stateless between time steps.
5. **Strategies are always injected, never hardcoded.** `Fund` receives `InvestmentStrategy` at construction.
6. **Seriatim vs group model points is a data concern.** All liability models receive a unified DataFrame — they never know the source.
7. **Database access goes through repositories only.** Nothing calls SQLAlchemy directly except the repository files.
8. **Stochastic runs use group model points only** — seriatim is prohibited in `StochasticRun`.
9. **Accounting basis is a bond-level property, not a model-level switch.** Each bond carries its own `accounting_basis` field (AC, FVTPL, or FVOCI). All calculations branch on this field at the individual bond level. Results are aggregated only after per-bond calculations are complete.

---

*Further technical detail — including module interfaces, class hierarchies, and calibration specifications — is available to qualified parties under NDA. Contact [mohd.shahrils@yahoo.com](mailto:mohd.shahrils@yahoo.com).*
