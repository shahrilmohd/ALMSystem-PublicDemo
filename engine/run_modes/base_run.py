"""
Abstract base class for all ALM run modes.

All three run types (LiabilityOnlyRun, DeterministicRun, StochasticRun) inherit
from BaseRun and implement the three abstract methods:

    setup()     — load data, instantiate all model components
    execute()   — run the projection (the model loop)
    teardown()  — write outputs, release resources

The concrete run() method is the template method: it calls these three in the
correct order, handles timing, captures errors, and drives the progress callback.
No subclass should override run() itself.

Design notes:
- BaseRun holds RunConfig and FundConfig — the fully validated inputs.
- ResultStore is NOT held here. Subclasses create and hold it in setup(), because
  ResultStore does not exist until Step 4 of the build order.
- The progress_callback is optional. When provided, it is called at key milestones
  with a fraction in [0.0, 1.0] and a human-readable message. This is how the
  Worker layer reports live progress to the frontend without polling the engine.
- validate_config() is a no-op hook. Subclasses override it for run-type-specific
  pre-flight checks beyond what Pydantic already enforces at config construction.
  It is called before setup() so errors surface before any I/O starts.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Callable, Optional

from engine.config.fund_config import FundConfig
from engine.config.run_config import RunConfig


# ---------------------------------------------------------------------------
# Progress callback type alias
# ---------------------------------------------------------------------------

# Signature: callback(fraction: float, message: str) -> None
# fraction: 0.0 (not started) to 1.0 (complete)
# message:  short human-readable description of the current stage
ProgressCallback = Callable[[float, str], None]


# ---------------------------------------------------------------------------
# RunStatus
# ---------------------------------------------------------------------------

class RunStatus(str, Enum):
    """
    Lifecycle status of a single model run.

    PENDING:    Run has been created but not started (before run() is called).
    RUNNING:    run() has been called and is in progress.
    COMPLETED:  run() finished without error.
    FAILED:     run() raised an exception. See RunResult.error for details.
    """
    PENDING   = "pending"
    RUNNING   = "running"
    COMPLETED = "completed"
    FAILED    = "failed"


# ---------------------------------------------------------------------------
# RunResult
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    """
    Outcome record for one model run. Populated by BaseRun.run().

    run_id:
        Copied from RunConfig.run_id. Ties this result to the config.

    status:
        Final lifecycle status. Always COMPLETED or FAILED after run() returns.

    started_at:
        UTC datetime when run() was called. None if run() was never called.

    completed_at:
        UTC datetime when run() finished (either completed or failed).
        None if still running.

    error:
        Exception message if status is FAILED. None otherwise.

    duration_seconds:
        Computed property: elapsed time in seconds. None if run not finished.
    """
    run_id:       str
    status:       RunStatus            = RunStatus.PENDING
    started_at:   Optional[datetime]   = field(default=None)
    completed_at: Optional[datetime]   = field(default=None)
    error:        Optional[str]        = field(default=None)

    @property
    def duration_seconds(self) -> Optional[float]:
        """Elapsed wall-clock time in seconds. None if run is not yet finished."""
        if self.started_at is not None and self.completed_at is not None:
            return (self.completed_at - self.started_at).total_seconds()
        return None


# ---------------------------------------------------------------------------
# BaseRun
# ---------------------------------------------------------------------------

class BaseRun(ABC):
    """
    Abstract base class for all ALM run modes.

    Subclasses must implement:
        setup()    — load data, build model components
        execute()  — run the projection
        teardown() — write results, release resources

    The run() method is the template method and must not be overridden.
    """

    def __init__(
        self,
        config:            RunConfig,
        fund_config:       FundConfig,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> None:
        """
        Args:
            config:            Fully validated master run configuration.
            fund_config:       Fully validated fund configuration.
            progress_callback: Optional callable(fraction, message). Called by
                               report_progress() at key milestones during run().
                               Fraction is in [0.0, 1.0]. Ignored if None.
        """
        self._config            = config
        self._fund_config       = fund_config
        self._progress_callback = progress_callback
        self._status            = RunStatus.PENDING
        self._result            = RunResult(run_id=config.run_id)
        self._logger            = logging.getLogger(self.__class__.__name__)

    # -----------------------------------------------------------------------
    # Public read-only properties
    # -----------------------------------------------------------------------

    @property
    def config(self) -> RunConfig:
        """The validated master run configuration."""
        return self._config

    @property
    def fund_config(self) -> FundConfig:
        """The validated fund configuration."""
        return self._fund_config

    @property
    def status(self) -> RunStatus:
        """Current lifecycle status of this run."""
        return self._status

    @property
    def result(self) -> RunResult:
        """
        The RunResult for this run.
        status, started_at, completed_at and error are populated by run().
        """
        return self._result

    # -----------------------------------------------------------------------
    # Abstract interface — subclasses must implement these
    # -----------------------------------------------------------------------

    @abstractmethod
    def setup(self) -> None:
        """
        Load all data and instantiate all model components.

        Called once by run() before execute(). Subclasses should:
        - Load model points from the source defined in config.input_sources
        - Load assumption tables
        - Load asset data (DETERMINISTIC and STOCHASTIC only)
        - Load ESG scenarios (STOCHASTIC only)
        - Instantiate liability models, asset models, strategies
        - Instantiate ResultStore

        Raises:
            Any exception aborts the run and sets status to FAILED.
        """

    @abstractmethod
    def execute(self) -> None:
        """
        Run the projection.

        Called by run() after setup() succeeds. Subclasses should:
        - Run the time step loop (via engine/core/top.py when available)
        - Write results to ResultStore at each timestep

        Raises:
            Any exception aborts the run and sets status to FAILED.
        """

    @abstractmethod
    def teardown(self) -> None:
        """
        Write outputs and release resources.

        Called by run() after execute() succeeds. Subclasses should:
        - Serialise ResultStore outputs to disk (respecting output.output_dir,
          output.result_format, output.compress_outputs)
        - Apply output filtering (output.output_timestep, output.output_horizon_years)
        - Release any open file handles or database connections

        Raises:
            Any exception sets status to FAILED. Results may be partially written.
        """

    # -----------------------------------------------------------------------
    # Override hook — optional for subclasses
    # -----------------------------------------------------------------------

    def validate_config(self) -> None:
        """
        Run-type-specific pre-flight checks beyond Pydantic validation.

        Called by run() before setup(). Default implementation is a no-op.
        Subclasses override this to add checks that cannot be expressed in
        Pydantic validators — typically because they require file I/O or
        cross-referencing multiple data sources.

        Example uses:
        - Verify the scenario file contains at least num_scenarios scenarios
        - Verify the model point file has all required columns
        - Verify asset and liability fund_ids are consistent

        Raises:
            ValueError: If a pre-flight check fails.
        """

    # -----------------------------------------------------------------------
    # Template method — do not override
    # -----------------------------------------------------------------------

    def run(self) -> RunResult:
        """
        Execute the full run lifecycle: validate → setup → execute → teardown.

        This method must not be overridden by subclasses. All run-type-specific
        logic belongs in setup(), execute(), and teardown().

        Returns:
            RunResult with status COMPLETED and timing populated on success.

        Raises:
            Re-raises any exception from validate_config(), setup(), execute(),
            or teardown() after recording it in RunResult and setting status
            to FAILED.
        """
        self._result.started_at = datetime.now()
        self._status            = RunStatus.RUNNING
        self._result.status     = RunStatus.RUNNING

        self._logger.info(
            "Run starting: %s (%s)", self._config.run_id, self._config.run_type.value
        )

        try:
            self.validate_config()

            self.report_progress(0.0, "Setup: loading data and building components")
            self.setup()

            self.report_progress(0.05, "Projection: running model")
            self.execute()

            self.report_progress(0.95, "Teardown: writing outputs")
            self.teardown()

            self._status            = RunStatus.COMPLETED
            self._result.status     = RunStatus.COMPLETED
            self._result.completed_at = datetime.now()

            self.report_progress(1.0, "Run completed")
            self._logger.info(
                "Run completed: %s (%.1fs)",
                self._config.run_id,
                self._result.duration_seconds,
            )

        except Exception as exc:
            self._status              = RunStatus.FAILED
            self._result.status       = RunStatus.FAILED
            self._result.completed_at = datetime.now()
            self._result.error        = str(exc)

            self._logger.exception("Run failed: %s", self._config.run_id)
            raise

        return self._result

    # -----------------------------------------------------------------------
    # Progress reporting
    # -----------------------------------------------------------------------

    def _output_stem(self, suffix: str) -> str:
        """
        Build a human-readable output filename stem.

        Format: {run_name}_{run_type}_{suffix}
        e.g. "Q1_2026_BEL_Validation_deterministic_results"

        run_name is sanitised: whitespace collapsed to underscores, all
        characters that are not alphanumeric, hyphen, or underscore are
        removed, and the result is truncated to 60 characters to keep
        paths manageable on all operating systems.
        """
        import re
        safe_name = re.sub(r"\s+", "_", self._config.run_name.strip())
        safe_name = re.sub(r"[^\w\-]", "", safe_name)[:60]
        run_type  = self._config.run_type.value
        return f"{safe_name}_{run_type}_{suffix}"

    def report_progress(self, fraction: float, message: str = "") -> None:
        """
        Notify the caller of current progress.

        Args:
            fraction: Progress as a value in [0.0, 1.0].
                      0.0 = not started, 1.0 = complete.
            message:  Short human-readable description of the current stage.
                      Displayed in the UI progress panel and written to logs.
        """
        if self._progress_callback is not None:
            self._progress_callback(fraction, message)
