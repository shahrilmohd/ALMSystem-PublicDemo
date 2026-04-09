"""
Unit tests for BaseRun, RunResult, and RunStatus.

Because BaseRun is abstract it cannot be instantiated directly.
All tests use ConcreteRun — a minimal test double defined at the top of this
file that implements the three abstract methods (setup, execute, teardown).

ConcreteRun records every method call in self.call_order so tests can assert
that the template method pattern fires in the correct sequence.
Errors are injected via constructor arguments so each failure scenario can be
tested without any mocking library.

Rules under test
----------------
RunResult:
  1. Initial state is PENDING with no times and no error.
  2. duration_seconds is None when started_at or completed_at is missing.
  3. duration_seconds is computed correctly when both timestamps are set.

BaseRun construction:
  4. status starts as PENDING.
  5. config and fund_config properties return the objects passed in.
  6. result.run_id matches config.run_id.

run() — happy path:
  7. Calls setup → execute → teardown in that exact order.
  8. Status is COMPLETED after a successful run.
  9. result.started_at and result.completed_at are both populated.
  10. result.duration_seconds is non-negative.
  11. result.error is None on success.
  12. run() returns the RunResult.

run() — failure paths:
  13. Exception in validate_config: FAILED, error captured, exception re-raised.
  14. Exception in setup:           FAILED, error captured, exception re-raised.
  15. Exception in execute:         FAILED, error captured, exception re-raised.
  16. Exception in teardown:        FAILED, error captured, exception re-raised.
  17. When setup raises, execute and teardown are NOT called.
  18. When execute raises, teardown is NOT called.

Progress callback:
  19. Callback is called at fraction 0.0 (start) and 1.0 (completion).
  20. No error when no callback is provided.
  21. Callback receives a non-empty message string at each call.

validate_config() default:
  22. Default implementation is a no-op — no exception raised.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

import pytest

from engine.config.fund_config import FundConfig
from engine.config.run_config import RunConfig
from engine.run_modes.base_run import BaseRun, RunResult, RunStatus


# ---------------------------------------------------------------------------
# ConcreteRun — minimal test double
# ---------------------------------------------------------------------------

class ConcreteRun(BaseRun):
    """
    Concrete subclass of BaseRun used only in tests.

    call_order records each method name as it is called, allowing tests to
    assert that the template method calls setup → execute → teardown in order.

    Errors are injected at construction time. Passing an exception instance
    for setup_error, execute_error, or teardown_error causes that method to
    raise when called.
    """

    def __init__(
        self,
        config:            RunConfig,
        fund_config:       FundConfig,
        *,
        validate_error:    Optional[Exception] = None,
        setup_error:       Optional[Exception] = None,
        execute_error:     Optional[Exception] = None,
        teardown_error:    Optional[Exception] = None,
        **kwargs,
    ) -> None:
        super().__init__(config, fund_config, **kwargs)
        self._validate_error = validate_error
        self._setup_error    = setup_error
        self._execute_error  = execute_error
        self._teardown_error = teardown_error
        self.call_order: list[str] = []

    def validate_config(self) -> None:
        self.call_order.append("validate_config")
        if self._validate_error:
            raise self._validate_error

    def setup(self) -> None:
        self.call_order.append("setup")
        if self._setup_error:
            raise self._setup_error

    def execute(self) -> None:
        self.call_order.append("execute")
        if self._execute_error:
            raise self._execute_error

    def teardown(self) -> None:
        self.call_order.append("teardown")
        if self._teardown_error:
            raise self._teardown_error


# ---------------------------------------------------------------------------
# RunResult
# ---------------------------------------------------------------------------

class TestRunResult:
    def test_initial_state(self):
        result = RunResult(run_id="run-001")
        assert result.run_id       == "run-001"
        assert result.status       == RunStatus.PENDING
        assert result.started_at   is None
        assert result.completed_at is None
        assert result.error        is None

    def test_duration_none_when_not_started(self):
        result = RunResult(run_id="run-001")
        assert result.duration_seconds is None

    def test_duration_none_when_started_but_not_completed(self):
        result = RunResult(run_id="run-001")
        result.started_at = datetime.now()
        assert result.duration_seconds is None

    def test_duration_none_when_completed_but_not_started(self):
        # Edge case: completed_at set without started_at — still None.
        result = RunResult(run_id="run-001")
        result.completed_at = datetime.now()
        assert result.duration_seconds is None

    def test_duration_computed_correctly(self):
        result = RunResult(run_id="run-001")
        result.started_at   = datetime(2025, 1, 1, 12, 0, 0)
        result.completed_at = datetime(2025, 1, 1, 12, 0, 45)  # 45 seconds later
        assert result.duration_seconds == pytest.approx(45.0)

    def test_duration_sub_second(self):
        result = RunResult(run_id="run-001")
        result.started_at   = datetime(2025, 1, 1, 12, 0, 0, 0)
        result.completed_at = datetime(2025, 1, 1, 12, 0, 0, 500_000)  # 0.5s
        assert result.duration_seconds == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# BaseRun construction
# ---------------------------------------------------------------------------

class TestBaseRunConstruction:
    def test_status_starts_pending(self, run_config, fund_config):
        run = ConcreteRun(run_config, fund_config)
        assert run.status == RunStatus.PENDING

    def test_config_property_returns_injected_config(self, run_config, fund_config):
        run = ConcreteRun(run_config, fund_config)
        assert run.config is run_config

    def test_fund_config_property_returns_injected_fund_config(self, run_config, fund_config):
        run = ConcreteRun(run_config, fund_config)
        assert run.fund_config is fund_config

    def test_result_run_id_matches_config(self, run_config, fund_config):
        run = ConcreteRun(run_config, fund_config)
        assert run.result.run_id == run_config.run_id

    def test_result_status_starts_pending(self, run_config, fund_config):
        run = ConcreteRun(run_config, fund_config)
        assert run.result.status == RunStatus.PENDING


# ---------------------------------------------------------------------------
# run() — happy path
# ---------------------------------------------------------------------------

class TestRunHappyPath:
    def test_calls_setup_execute_teardown_in_order(self, run_config, fund_config):
        run = ConcreteRun(run_config, fund_config)
        run.run()
        # validate_config is also called — expect it before setup
        assert run.call_order == ["validate_config", "setup", "execute", "teardown"]

    def test_status_is_completed(self, run_config, fund_config):
        run = ConcreteRun(run_config, fund_config)
        run.run()
        assert run.status == RunStatus.COMPLETED

    def test_result_status_is_completed(self, run_config, fund_config):
        run = ConcreteRun(run_config, fund_config)
        run.run()
        assert run.result.status == RunStatus.COMPLETED

    def test_result_started_at_is_populated(self, run_config, fund_config):
        run = ConcreteRun(run_config, fund_config)
        run.run()
        assert run.result.started_at is not None

    def test_result_completed_at_is_populated(self, run_config, fund_config):
        run = ConcreteRun(run_config, fund_config)
        run.run()
        assert run.result.completed_at is not None

    def test_result_duration_is_non_negative(self, run_config, fund_config):
        run = ConcreteRun(run_config, fund_config)
        run.run()
        assert run.result.duration_seconds >= 0.0

    def test_result_error_is_none(self, run_config, fund_config):
        run = ConcreteRun(run_config, fund_config)
        run.run()
        assert run.result.error is None

    def test_run_returns_run_result(self, run_config, fund_config):
        run = ConcreteRun(run_config, fund_config)
        returned = run.run()
        assert isinstance(returned, RunResult)
        assert returned is run.result

    def test_completed_at_is_after_started_at(self, run_config, fund_config):
        run = ConcreteRun(run_config, fund_config)
        run.run()
        assert run.result.completed_at >= run.result.started_at


# ---------------------------------------------------------------------------
# run() — failure paths
# ---------------------------------------------------------------------------

class TestRunFailurePaths:
    @pytest.mark.parametrize("failure_stage,expected_calls", [
        # When validate_config raises: setup/execute/teardown not called
        ("validate_error",  ["validate_config"]),
        # When setup raises: execute and teardown not called
        ("setup_error",     ["validate_config", "setup"]),
        # When execute raises: teardown not called
        ("execute_error",   ["validate_config", "setup", "execute"]),
        # When teardown raises: all three were called
        ("teardown_error",  ["validate_config", "setup", "execute", "teardown"]),
    ])
    def test_failure_sets_failed_status(
        self, run_config, fund_config, failure_stage, expected_calls
    ):
        error = ValueError(f"deliberate {failure_stage}")
        run = ConcreteRun(run_config, fund_config, **{failure_stage: error})
        with pytest.raises(ValueError):
            run.run()
        assert run.status == RunStatus.FAILED

    @pytest.mark.parametrize("failure_stage,expected_calls", [
        ("validate_error",  ["validate_config"]),
        ("setup_error",     ["validate_config", "setup"]),
        ("execute_error",   ["validate_config", "setup", "execute"]),
        ("teardown_error",  ["validate_config", "setup", "execute", "teardown"]),
    ])
    def test_failure_records_error_message(
        self, run_config, fund_config, failure_stage, expected_calls
    ):
        error = ValueError(f"deliberate {failure_stage}")
        run = ConcreteRun(run_config, fund_config, **{failure_stage: error})
        with pytest.raises(ValueError):
            run.run()
        assert f"deliberate {failure_stage}" in run.result.error

    @pytest.mark.parametrize("failure_stage,expected_calls", [
        ("validate_error",  ["validate_config"]),
        ("setup_error",     ["validate_config", "setup"]),
        ("execute_error",   ["validate_config", "setup", "execute"]),
        ("teardown_error",  ["validate_config", "setup", "execute", "teardown"]),
    ])
    def test_failure_only_calls_expected_methods(
        self, run_config, fund_config, failure_stage, expected_calls
    ):
        error = ValueError("deliberate error")
        run = ConcreteRun(run_config, fund_config, **{failure_stage: error})
        with pytest.raises(ValueError):
            run.run()
        assert run.call_order == expected_calls

    @pytest.mark.parametrize("failure_stage", [
        "validate_error", "setup_error", "execute_error", "teardown_error",
    ])
    def test_failure_re_raises_original_exception(
        self, run_config, fund_config, failure_stage
    ):
        error = ValueError("deliberate error")
        run = ConcreteRun(run_config, fund_config, **{failure_stage: error})
        with pytest.raises(ValueError, match="deliberate error"):
            run.run()

    @pytest.mark.parametrize("failure_stage", [
        "validate_error", "setup_error", "execute_error", "teardown_error",
    ])
    def test_failure_populates_completed_at(
        self, run_config, fund_config, failure_stage
    ):
        # completed_at is set even on failure, so duration_seconds is available.
        error = ValueError("deliberate error")
        run = ConcreteRun(run_config, fund_config, **{failure_stage: error})
        with pytest.raises(ValueError):
            run.run()
        assert run.result.completed_at is not None
        assert run.result.duration_seconds is not None


# ---------------------------------------------------------------------------
# Progress callback
# ---------------------------------------------------------------------------

class TestProgressCallback:
    def test_callback_called_at_start_and_completion(self, run_config, fund_config):
        calls: list[tuple[float, str]] = []
        run = ConcreteRun(
            run_config, fund_config,
            progress_callback=lambda f, m: calls.append((f, m)),
        )
        run.run()
        fractions = [f for f, _ in calls]
        assert 0.0 in fractions
        assert 1.0 in fractions

    def test_callback_receives_non_empty_message(self, run_config, fund_config):
        calls: list[tuple[float, str]] = []
        run = ConcreteRun(
            run_config, fund_config,
            progress_callback=lambda f, m: calls.append((f, m)),
        )
        run.run()
        for _, message in calls:
            assert len(message) > 0

    def test_no_callback_does_not_raise(self, run_config, fund_config):
        # progress_callback=None (default) — no AttributeError or TypeError raised.
        run = ConcreteRun(run_config, fund_config)
        run.run()   # must not raise

    def test_fractions_are_non_decreasing(self, run_config, fund_config):
        calls: list[tuple[float, str]] = []
        run = ConcreteRun(
            run_config, fund_config,
            progress_callback=lambda f, m: calls.append((f, m)),
        )
        run.run()
        fractions = [f for f, _ in calls]
        assert fractions == sorted(fractions)

    def test_fractions_within_bounds(self, run_config, fund_config):
        calls: list[tuple[float, str]] = []
        run = ConcreteRun(
            run_config, fund_config,
            progress_callback=lambda f, m: calls.append((f, m)),
        )
        run.run()
        for fraction, _ in calls:
            assert 0.0 <= fraction <= 1.0


# ---------------------------------------------------------------------------
# validate_config() default
# ---------------------------------------------------------------------------

class TestValidateConfigDefault:
    def test_default_is_noop(self, run_config, fund_config):
        # BaseRun.validate_config() must not raise when not overridden.
        # ConcreteRun overrides it, so we test via a subclass that does NOT.
        class MinimalRun(BaseRun):
            def setup(self):    pass
            def execute(self):  pass
            def teardown(self): pass

        run = MinimalRun(run_config, fund_config)
        run.validate_config()   # must not raise
