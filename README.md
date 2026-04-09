# ALM System — Public Demo

> **This is an educational/prototype subset.**
> It is intended to illustrate architecture, design decisions, and modelling patterns —
> not to be run as a production system.
>
> The following are intentionally excluded from this demo:
> BPA mortality, matching adjustment computation, and associated actuarial assumptions
> (stubbed with `NotImplementedError`).
>
> **The production-grade modules are available for a walk-through.**
> If you are an actuary, engineer, or insurer interested in seeing the full model,
> reach out via the contact details below.
>
> See `document/DECISIONS.md` for a full index of what exists but is not included here.

---

## What is this?

A production-grade **Asset and Liability Model (ALM)** built for a small actuarial team,
covering the full stack from cash flow projection to IFRS 17 reporting, Solvency II Matching
Adjustment, and an AI assistant layer.

**Products in scope (full model):**
- Segregated fund conventional products (with-profits, unit-linked, endowments)
- Bulk Purchase Annuity (BPA) — in-payment pensioners, deferred members, dependants, enhanced lives

**Accounting frameworks:**
- IFRS 9: bonds at AC, FVTPL, or FVOCI — per bond, never a portfolio-level switch
- IFRS 17 GMM: CSM, RA, LRC/LIC, loss component, coverage unit release
- Solvency II: Matching Adjustment with PRA PS10/24 fundamental spreads

---

## Architecture overview

```
Frontend (PyQt6 desktop)
        │ HTTP
API (FastAPI: /runs, /config, /results, /batches, /ai)
        │ job dispatch
Worker (RQ + Redis)
        │
Engine (pure Python ALM — zero imports from above layers)
├── config/          Pydantic run config validation
├── core/            Projection loop and fund coordinator
├── asset/           Bond (AC/FVTPL/FVOCI), Equity, AssetModel
├── liability/       Conventional products + BPA classes
├── ifrs17/          GMM engine: CSM, RA, LRC/LIC, loss component, JAX JIT
├── matching_adjustment/  Eligibility, fundamental spread, MA benefit
├── curves/          Smith-Wilson RFR extrapolation (EIOPA methodology)
├── scenarios/       ESG scenario store (1,000–3,000 paths)
├── run_modes/       Deterministic, Stochastic, LiabilityOnly, BPARun
└── results/         ResultStore, TVOG calculator
        │
Storage (SQLAlchemy: SQLite dev → PostgreSQL prod)
        │
AI Layer (Anthropic API — specialist agents per domain)
```

**Hard separation rules (enforced, not aspirational):**
- `engine/` has zero imports from `api/`, `frontend/`, or `worker/`
- The time loop lives only in `engine/core/top.py` — no model advances time internally
- Results never stored inside models — all output goes to `ResultStore`
- DB access only through repository classes

---

## Projection engine design

The engine supports three run modes that reflect how actuaries actually work:

**LiabilityOnly and Deterministic runs** use a traditional serial projection — one timestep at
a time, iterating over the in-force model point population. This matches the mental model
actuaries have from platforms like Prophet or MoSes, making the logic easy to audit.
For large seriatim portfolios the deterministic run supports a two-pass BEL design where
the liability-only pass produces the BEL used as the outgo driver for the asset pass.

**Stochastic runs** operate over 1,000–3,000 ESG scenarios, each running the full
fund loop. Group model points are required (seriatim is explicitly prohibited). The result
is a distribution of BEL outcomes from which TVOG = E[stochastic BEL] − deterministic BEL
is computed. An optional vectorised batch mode is available for the inner scenario loop —
`use_vectorised: bool = False` in `StochasticConfig` — enabling significant speedup when
model points are structured as pure array operations.

The **IFRS 17 GMM** inner step (CSM accretion, RA release, loss component) is JIT-compiled
via `@jax.jit` with `jax_enable_x64=True`. A NumPy fallback activates automatically when
JAX is not available, so the engine runs without a GPU.

---

## Key modelling decisions

Full rationale for every decision is in [document/DECISIONS.md](document/DECISIONS.md).
Sections marked *"not in public demo"* cover BPA-specific actuarial assumptions.

| Decision | Summary |
|---|---|
| Bond accounting basis | Per-bond property (AC / FVTPL / FVOCI) — never a portfolio-level switch. Aggregate only after per-bond calculations. |
| EIR method | Locked-in at purchase; solved via `scipy.brentq` against the risk-free curve. Monthly coupon compounding handled correctly. |
| Calibration spread | Parallel shift to RFR solved at valuation date so discounted CFs equal market value; locked for the entire projection. |
| Two-pass BEL | Liability-only pass gives BEL; asset pass uses BEL as cash outgo. No circular dependency. |
| TVOG | Mean(stochastic BEL) − deterministic BEL across all ESG scenarios. |
| IFRS 17 GMM | Standalone product-agnostic module. CSM accretion at locked-in rate; non-financial assumption changes adjust CSM. JAX JIT on inner step. |
| Matching Adjustment | Four-condition eligibility, highly-predictable cap (35% per PS10/24), FS from PRA tables; PV cashflow-weighted MA benefit. |
| AI layer | Claude-based orchestrator routes to specialist agents (analyst, advisor, modelling). Deployment-mode gate controls whether data leaves the network. |

---

## Technology stack

| Layer | Technology |
|---|---|
| Package manager | `uv` |
| Python | 3.12 |
| Numerical | NumPy, Pandas, JAX (JIT on IFRS 17 GMM inner step) |
| Config validation | Pydantic v2 |
| Desktop UI | PyQt6 |
| API | FastAPI |
| Job queue | RQ + Redis |
| Database | SQLAlchemy + SQLite (dev) → PostgreSQL (prod) |
| Testing | pytest — 1,173 tests passing |
| AI layer | Anthropic API (Claude), specialist agents per domain |

---

## Running the tests

```bash
uv sync
uv run pytest
```

All 1,173 tests pass on this branch. Tests covering BPA computation internals and
matching adjustment computation are excluded from this demo.

---

## What is stubbed in this demo

The following modules show their full class interfaces and docstrings but raise
`NotImplementedError` for all computation methods:

| Module | What is stubbed |
|---|---|
| `engine/liability/bpa/in_payment.py` | `project_cashflows`, `get_bel`, `get_decrements` |
| `engine/liability/bpa/deferred.py` | Four-decrement period calculations and two-phase BEL |
| `engine/liability/bpa/dependant.py` | Convolution BEL over trigger periods |
| `engine/liability/bpa/enhanced.py` | Age-rating override (inherits InPaymentLiability) |
| `engine/liability/bpa/mortality.py` | `q_x`, `improvement_factor`, `survival_probs_variable_dt` |
| `engine/matching_adjustment/eligibility.py` | Four-condition eligibility + HP cap trimming |
| `engine/matching_adjustment/fundamental_spread.py` | `get_fs` lookup |
| `engine/matching_adjustment/ma_calculator.py` | `MACalculator.compute` |

The class signatures, docstrings, model point schemas, and DECISIONS.md section headers
are all visible — you can see exactly what the full model does.

---

## Documentation

| Document | Purpose |
|---|---|
| [document/ALM_Architecture.md](document/ALM_Architecture.md) | System architecture: directory layout, module table, class hierarchy, data flow, design principles |
| [document/DECISIONS.md](document/DECISIONS.md) | Why key modelling choices were made (46 sections; BPA-specific sections marked *not in public demo*) |

Additional internal documents (available in the full model):
- **Calculation engine guide** — vectorised two-pass design vs traditional serial platforms (Prophet, MoSes)
- **Developer guide** — adding output fields, extending the result schema, working with the build order
- **Testing guide** — conventions, directory structure, step-by-step run log with 1,173 tests

---

## Contact

Interested in the full model or a commercial engagement?
Connect via email: mohd.shahrils@yahoo.com, or open a GitHub Discussion.
