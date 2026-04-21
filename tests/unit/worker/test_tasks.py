"""
Unit tests for worker/tasks.py — run_alm_job().

Strategy
--------
run_alm_job() orchestrates the full stack: DB reads/writes, engine execution,
and result persistence.  We do not run the real engine in unit tests — that
belongs in integration tests.  Instead we patch the three private helpers
(_run_liability_only, _run_deterministic, _run_stochastic) to return a
pre-built minimal ResultStore, and verify that run_alm_job() correctly:

  - marks the run RUNNING before execution
  - calls the correct helper for each run_type
  - persists results after the helper returns
  - marks the run COMPLETED with a duration
  - marks the run FAILED and stores the error message when the helper raises
  - re-raises the exception after marking FAILED

Pre-engine mocking
------------------
run_alm_job() parses RunConfig, loads FundConfig, and loads model points
*before* calling the helper.  All three steps are patched out so the tests
never touch real files or the Pydantic validator — only the orchestration
logic is under test.

DB isolation
------------
Each test gets a fresh in-memory SQLite DB (via db_session fixture).
run_alm_job() constructs its own engine from the ALM_DB_URL env var, so we
patch that URL to point at the same in-memory DB used by the fixture.
StaticPool ensures both connections share the same in-memory database.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy.pool import StaticPool

from engine.config.run_config import RunType
from engine.results.result_store import ResultStore
from storage.run_repository import RunRepository
from tests.unit.worker.conftest import make_pending_run


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_empty_store(run_id: str) -> ResultStore:
    """Minimal ResultStore with no rows — sufficient for save_all()."""
    return ResultStore(run_id=run_id)


def _make_mock_run_config(run_type: RunType, num_scenarios: int = 50) -> MagicMock:
    """
    Minimal mock RunConfig satisfying every attribute run_alm_job() accesses.

    We never want to validate a real RunConfig in unit tests — the test data
    is intentionally minimal.  This mock provides exactly the attributes the
    orchestrator reads without triggering Pydantic validation.
    """
    cfg = MagicMock()
    cfg.run_type = run_type
    cfg.projection.projection_term_years = 1
    cfg.stochastic.num_scenarios = num_scenarios
    return cfg


def _patch_engine_for_session(db_session, monkeypatch):
    """
    Redirect run_alm_job's internal DB engine to the test session's engine.

    run_alm_job() calls get_engine(ALM_DB_URL) internally.  We patch
    get_engine to return an engine that shares the same StaticPool as the
    test session, so all DB writes in the job are visible to the test.
    """
    from sqlalchemy.orm import sessionmaker
    test_engine = db_session.get_bind()

    monkeypatch.setattr("worker.tasks.get_engine", lambda *a, **kw: test_engine)
    monkeypatch.setattr(
        "worker.tasks.get_session_factory",
        lambda engine: sessionmaker(bind=engine, expire_on_commit=False),
    )


@contextmanager
def _patch_pre_engine_steps(run_type: RunType, num_scenarios: int = 50):
    """
    Patch the three steps that precede the engine helper in run_alm_job():

        1. RunConfig.model_validate_json  — skip Pydantic validation
        2. FundConfig.from_yaml           — skip file I/O
        3. LiabilityDataLoader            — skip CSV loading

    Yields the mock RunConfig so individual tests can inspect or adjust it.
    """
    mock_config = _make_mock_run_config(run_type, num_scenarios)
    mock_fund   = MagicMock()
    mock_loader = MagicMock()
    mock_loader.return_value.load.return_value = MagicMock()  # mock DataFrame

    with patch("worker.tasks.RunConfig.model_validate", return_value=mock_config), \
         patch("worker.tasks.FundConfig.from_yaml", return_value=mock_fund), \
         patch("data.loaders.liability_data_loader.LiabilityDataLoader", mock_loader):
        yield mock_config


# ---------------------------------------------------------------------------
# Tests — successful runs
# ---------------------------------------------------------------------------

class TestRunAlmJobSuccess:
    def test_liability_only_marks_running_then_completed(
        self, db_session, monkeypatch
    ):
        _patch_engine_for_session(db_session, monkeypatch)
        make_pending_run(db_session, run_id="r1", run_type="LIABILITY_ONLY")

        store = _make_empty_store("r1")
        with _patch_pre_engine_steps(RunType.LIABILITY_ONLY), \
             patch("worker.tasks._run_liability_only", return_value=store), \
             patch("worker.tasks.ResultRepository.save_all"):
            from worker.tasks import run_alm_job
            run_alm_job("r1")

        db_session.expire_all()
        updated = RunRepository(db_session).get("r1")
        assert updated.status == "COMPLETED"
        assert updated.started_at is not None
        assert updated.completed_at is not None
        assert updated.duration_seconds >= 0.0
        assert updated.error_message is None

    def test_deterministic_calls_correct_helper(self, db_session, monkeypatch):
        _patch_engine_for_session(db_session, monkeypatch)
        make_pending_run(
            db_session,
            run_id="r2",
            run_type="DETERMINISTIC",
            config_json='{"run_type": "deterministic"}',
        )

        store = _make_empty_store("r2")
        with _patch_pre_engine_steps(RunType.DETERMINISTIC), \
             patch("worker.tasks._run_deterministic", return_value=store) as mock_det, \
             patch("worker.tasks._run_liability_only") as mock_liab, \
             patch("worker.tasks.ResultRepository.save_all"):
            from worker.tasks import run_alm_job
            run_alm_job("r2")

        mock_det.assert_called_once()
        mock_liab.assert_not_called()

    def test_stochastic_stores_n_scenarios_on_record(self, db_session, monkeypatch):
        _patch_engine_for_session(db_session, monkeypatch)
        config = '{"run_type": "stochastic", "stochastic": {"num_scenarios": 50}}'
        make_pending_run(db_session, run_id="r3", run_type="STOCHASTIC", config_json=config)

        store = _make_empty_store("r3")
        with _patch_pre_engine_steps(RunType.STOCHASTIC, num_scenarios=50), \
             patch("worker.tasks._run_stochastic", return_value=store), \
             patch("worker.tasks.ResultRepository.save_all"):
            from worker.tasks import run_alm_job
            run_alm_job("r3")

        db_session.expire_all()
        updated = RunRepository(db_session).get("r3")
        assert updated.n_scenarios == 50

    def test_results_are_persisted_after_run(self, db_session, monkeypatch):
        _patch_engine_for_session(db_session, monkeypatch)
        make_pending_run(db_session, run_id="r4")

        store = _make_empty_store("r4")
        save_mock = MagicMock()
        with _patch_pre_engine_steps(RunType.LIABILITY_ONLY), \
             patch("worker.tasks._run_liability_only", return_value=store), \
             patch("worker.tasks.ResultRepository.save_all", save_mock):
            from worker.tasks import run_alm_job
            run_alm_job("r4")

        save_mock.assert_called_once_with("r4", store)


# ---------------------------------------------------------------------------
# Tests — failure handling
# ---------------------------------------------------------------------------

class TestRunAlmJobFailure:
    def test_marks_failed_when_helper_raises(self, db_session, monkeypatch):
        _patch_engine_for_session(db_session, monkeypatch)
        make_pending_run(db_session, run_id="r5")

        with _patch_pre_engine_steps(RunType.LIABILITY_ONLY), \
             patch(
                "worker.tasks._run_liability_only",
                side_effect=RuntimeError("engine exploded"),
             ):
            from worker.tasks import run_alm_job
            with pytest.raises(RuntimeError, match="engine exploded"):
                run_alm_job("r5")

        db_session.expire_all()
        updated = RunRepository(db_session).get("r5")
        assert updated.status == "FAILED"
        assert "engine exploded" in updated.error_message
        assert updated.completed_at is not None

    def test_exception_is_reraised_after_marking_failed(self, db_session, monkeypatch):
        _patch_engine_for_session(db_session, monkeypatch)
        make_pending_run(db_session, run_id="r6")

        with _patch_pre_engine_steps(RunType.LIABILITY_ONLY), \
             patch(
                "worker.tasks._run_liability_only",
                side_effect=ValueError("bad input"),
             ):
            from worker.tasks import run_alm_job
            with pytest.raises(ValueError):
                run_alm_job("r6")

    def test_unknown_run_type_marks_failed(self, db_session, monkeypatch):
        _patch_engine_for_session(db_session, monkeypatch)
        make_pending_run(
            db_session,
            run_id="r7",
            run_type="UNKNOWN",
            config_json='{"run_type": "unknown_type"}',
        )

        # Use a mock config with an unrecognised run_type string (not a real RunType enum)
        mock_config = MagicMock()
        mock_config.run_type = "UNKNOWN_VALUE"
        mock_config.projection.projection_term_years = 1
        mock_fund = MagicMock()
        mock_loader = MagicMock()
        mock_loader.return_value.load.return_value = MagicMock()

        with patch("worker.tasks.RunConfig.model_validate_json", return_value=mock_config), \
             patch("worker.tasks.FundConfig.from_yaml", return_value=mock_fund), \
             patch("data.loaders.liability_data_loader.LiabilityDataLoader", mock_loader):
            from worker.tasks import run_alm_job
            with pytest.raises(Exception):
                run_alm_job("r7")

        db_session.expire_all()
        assert RunRepository(db_session).get("r7").status == "FAILED"
