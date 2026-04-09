"""
tools — HTTP wrapper functions exposed to AI agents as callable tools.

Each function maps to one or more ALM API endpoints.
No function has direct database or filesystem access.

Phase 2 tools (conventional products):
  get_run_results  — GET /results/{run_id} (read-only)
  get_run_config   — GET /runs/{run_id}    (read-only)
  submit_run       — POST /runs            (mutating; requires human approval in UI)
"""
