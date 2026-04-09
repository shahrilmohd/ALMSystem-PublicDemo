"""
ProgressPanel — live run status widget.

Polls GET /runs/{run_id} every 2 seconds while the run is PENDING or RUNNING.
Stops automatically when the run reaches COMPLETED or FAILED.

HTTP calls are made in a background HttpThread so the Qt main-thread event
loop is never blocked by network I/O.  A new poll is skipped if the previous
one is still in-flight.

Signals
-------
run_completed(run_id: str)
    Emitted once when the run transitions to COMPLETED.
run_failed(run_id: str, error: str)
    Emitted once when the run transitions to FAILED.
    `error` is the error_message from the API, or "" if none.

Usage
-----
    panel = ProgressPanel(api_client)
    panel.track(run_id)         # start polling
    panel.run_completed.connect(lambda rid: self._load_results(rid))
    panel.run_failed.connect(lambda rid, err: QMessageBox.warning(...))
"""
from __future__ import annotations

from PyQt6.QtCore import QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QGroupBox,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
)

from frontend.desktop.api_client import ALMApiClient, APIError, RunStatus
from frontend.desktop.http_thread import HttpThread


_POLL_INTERVAL_MS = 2_000   # 2 seconds
_TERMINAL_STATUSES = {"COMPLETED", "FAILED"}

_STATUS_COLOURS = {
    "PENDING":   "#888888",
    "RUNNING":   "#1a6fcc",
    "COMPLETED": "#2e7d32",
    "FAILED":    "#c62828",
}


class ProgressPanel(QGroupBox):
    """
    Displays the live status of a single projection run.

    The panel is hidden by default. Call `track(run_id)` to show it
    and begin polling.
    """

    run_completed = pyqtSignal(str)        # run_id
    run_failed    = pyqtSignal(str, str)   # run_id, error_message

    def __init__(self, client: ALMApiClient, parent=None) -> None:
        super().__init__("Run Progress", parent)
        self._client   = client
        self._run_id:  str | None = None
        self._fetch_thread: HttpThread | None = None

        self._timer = QTimer(self)
        self._timer.setInterval(_POLL_INTERVAL_MS)
        self._timer.timeout.connect(self._poll)

        layout = QVBoxLayout()

        self._run_id_label  = QLabel("Run: —")
        self._status_label  = QLabel("Status: —")
        self._elapsed_label = QLabel("")

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 0)   # indeterminate (busy) animation
        self._progress_bar.setVisible(False)

        self._stop_btn = QPushButton("Cancel polling")
        self._stop_btn.clicked.connect(self._stop_polling)
        self._stop_btn.setVisible(False)

        layout.addWidget(self._run_id_label)
        layout.addWidget(self._status_label)
        layout.addWidget(self._elapsed_label)
        layout.addWidget(self._progress_bar)
        layout.addWidget(self._stop_btn)

        self.setLayout(layout)
        self.setVisible(False)

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def track(self, run_id: str) -> None:
        """
        Begin tracking a run.  Shows the panel and starts polling.
        Any previously tracked run is forgotten.
        """
        self._run_id = run_id
        self._run_id_label.setText(f"Run: {run_id}")
        self._set_status("PENDING")
        self._progress_bar.setVisible(True)
        self._stop_btn.setVisible(True)
        self.setVisible(True)
        self._timer.start()

    def stop(self) -> None:
        """Stop polling without emitting any signal."""
        self._stop_polling()

    # -----------------------------------------------------------------------
    # Polling — HTTP call runs in a background thread
    # -----------------------------------------------------------------------

    def _poll(self) -> None:
        """Start a background poll of GET /runs/{run_id} (skip if in-flight)."""
        if self._run_id is None:
            self._timer.stop()
            return
        if self._fetch_thread is not None and self._fetch_thread.isRunning():
            return
        self._fetch_thread = HttpThread(
            lambda: self._client.get_run(self._run_id),  # type: ignore[arg-type]
            parent=self,
        )
        self._fetch_thread.result_ready.connect(self._on_poll_ready)
        self._fetch_thread.error_raised.connect(self._on_poll_error)
        self._fetch_thread.start()

    def _on_poll_ready(self, run: RunStatus) -> None:
        """Callback — runs on main thread after a successful status response."""
        self._set_status(run.status)

        if run.duration_seconds is not None:
            self._elapsed_label.setText(f"Elapsed: {run.duration_seconds:.1f}s")

        if run.status in _TERMINAL_STATUSES:
            self._timer.stop()
            self._progress_bar.setVisible(False)
            self._stop_btn.setVisible(False)
            if run.status == "COMPLETED":
                self.run_completed.emit(self._run_id)
            else:
                self.run_failed.emit(self._run_id, run.error_message or "")

    def _on_poll_error(self, exc: Exception) -> None:
        """Callback — runs on main thread when the HTTP call raises."""
        if isinstance(exc, APIError):
            self._status_label.setText("Status: (connection error — retrying…)")
        else:
            self._status_label.setText(f"Status: (error — {exc})")

    # -----------------------------------------------------------------------
    # Internal
    # -----------------------------------------------------------------------

    def _set_status(self, status: str) -> None:
        colour = _STATUS_COLOURS.get(status, "#000000")
        self._status_label.setText(
            f'Status: <span style="color:{colour}; font-weight:bold;">{status}</span>'
        )
        self._status_label.setTextFormat(
            self._status_label.textFormat()
        )  # keep Qt's default (rich text auto-detected)

    def _stop_polling(self) -> None:
        self._timer.stop()
        self._progress_bar.setVisible(False)
        self._stop_btn.setVisible(False)
