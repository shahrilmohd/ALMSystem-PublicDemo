"""
Unit tests for worker/progress.py.

Coverage
--------
report()
    - outside a job context (get_current_job returns None) → silent no-op
    - inside a job context → writes progress and message into job.meta
    - fraction is clamped to [0.0, 1.0]
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from worker.progress import report


class TestReport:
    def test_no_op_outside_job_context(self):
        # get_current_job() returns None when called outside an RQ worker.
        # report() must not raise in this case.
        with patch("worker.progress.get_current_job", return_value=None):
            report(0.5, "halfway")  # should not raise

    def test_writes_fraction_and_message_to_job_meta(self):
        mock_job = MagicMock()
        mock_job.meta = {}

        with patch("worker.progress.get_current_job", return_value=mock_job):
            report(0.42, "running scenario 42")

        assert mock_job.meta["progress"] == pytest.approx(0.42)
        assert mock_job.meta["message"] == "running scenario 42"
        mock_job.save_meta.assert_called_once()

    def test_fraction_clamped_below_zero(self):
        mock_job = MagicMock()
        mock_job.meta = {}

        with patch("worker.progress.get_current_job", return_value=mock_job):
            report(-0.5, "negative")

        assert mock_job.meta["progress"] == 0.0

    def test_fraction_clamped_above_one(self):
        mock_job = MagicMock()
        mock_job.meta = {}

        with patch("worker.progress.get_current_job", return_value=mock_job):
            report(1.5, "over 100%")

        assert mock_job.meta["progress"] == 1.0

    def test_fraction_at_boundaries_not_clamped(self):
        mock_job = MagicMock()
        mock_job.meta = {}

        with patch("worker.progress.get_current_job", return_value=mock_job):
            report(0.0, "start")
        assert mock_job.meta["progress"] == 0.0

        mock_job.meta = {}
        with patch("worker.progress.get_current_job", return_value=mock_job):
            report(1.0, "end")
        assert mock_job.meta["progress"] == 1.0


import pytest
