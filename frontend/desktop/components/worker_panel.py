"""
WorkerPanel — worker count control and live status display.

Responsibilities
----------------
1. Let the user choose how many RQ worker processes to run (QSpinBox, 1–8).
2. On "Apply", spawn or kill local `rq worker` subprocesses via subprocess.Popen
   so the live count matches the desired count.
3. Display the current worker status by polling GET /workers every 5 seconds.

Architecture note
-----------------
Worker processes are managed here (desktop side) via subprocess.Popen.
The API endpoint GET /workers is read-only — it reflects what Redis/RQ reports.
There is intentional eventual consistency: a newly spawned worker appears in the
table within ~2 seconds (RQ worker startup + next poll cycle).

HTTP calls are made in a background HttpThread so the Qt main-thread event loop
is never blocked by network I/O.  If a poll is still in-flight when the timer
fires again, the new poll is skipped.

Worker command
--------------
Each worker is spawned as:
    uv run rq worker --url redis://localhost:6379 alm

The `alm` queue name must match the queue name used in worker/job_queue.py.
Workers are tracked in self._procs (list of Popen objects).
On "Apply down", the oldest excess workers are terminated with SIGTERM.
"""
from __future__ import annotations

import subprocess
import sys

from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from frontend.desktop.api_client import ALMApiClient, APIError, WorkerList
from frontend.desktop.http_thread import HttpThread


_POLL_INTERVAL_MS = 5_000   # 5 seconds
_WORKER_QUEUE     = "alm"
_REDIS_URL        = "redis://localhost:6379"

_STATE_COLOURS = {
    "idle": "#2e7d32",
    "busy": "#1a6fcc",
}


class WorkerPanel(QGroupBox):
    """
    Worker management and status panel.

    Shows a spinbox for desired worker count, an Apply button to reconcile
    the live subprocess count, and a table showing worker status from the API.
    """

    def __init__(self, client: ALMApiClient, parent=None) -> None:
        super().__init__("Workers", parent)
        self._client = client
        self._procs: list[subprocess.Popen] = []
        self._fetch_thread: HttpThread | None = None

        # ---------------------------------------------------------------
        # Control row: desired count + Apply
        # ---------------------------------------------------------------
        control_layout = QHBoxLayout()
        control_layout.addWidget(QLabel("Desired workers:"))

        self._spin = QSpinBox()
        self._spin.setRange(1, 8)
        self._spin.setValue(1)
        self._spin.setFixedWidth(60)
        control_layout.addWidget(self._spin)

        self._apply_btn = QPushButton("Apply")
        self._apply_btn.setFixedWidth(70)
        self._apply_btn.clicked.connect(self._apply_worker_count)
        control_layout.addWidget(self._apply_btn)

        control_layout.addStretch()

        self._summary_label = QLabel("API: —")
        control_layout.addWidget(self._summary_label)

        # ---------------------------------------------------------------
        # Status table
        # ---------------------------------------------------------------
        self._table = QTableWidget(0, 4)
        self._table.setHorizontalHeaderLabels(["Name", "State", "Current Job", "Queues"])
        self._table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self._table.setAlternatingRowColors(True)
        self._table.verticalHeader().setVisible(False)

        # ---------------------------------------------------------------
        # Layout
        # ---------------------------------------------------------------
        layout = QVBoxLayout()
        layout.addLayout(control_layout)
        layout.addWidget(self._table)
        self.setLayout(layout)

        # ---------------------------------------------------------------
        # Auto-refresh timer — fires every 5 s, HTTP call runs off-thread.
        # ---------------------------------------------------------------
        self._timer = QTimer(self)
        self._timer.setInterval(_POLL_INTERVAL_MS)
        self._timer.timeout.connect(self._refresh)
        self._timer.start()

        # Defer the first poll by 1 s so the window finishes rendering first.
        QTimer.singleShot(1_000, self._refresh)

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def shutdown_all_workers(self) -> None:
        """
        Terminate all worker subprocesses managed by this panel.
        Call this before the application exits.
        """
        for proc in self._procs:
            if proc.poll() is None:
                proc.terminate()
        self._procs.clear()

    # -----------------------------------------------------------------------
    # Worker subprocess management
    # -----------------------------------------------------------------------

    def _apply_worker_count(self) -> None:
        """Spawn or kill processes to reach the desired count."""
        # Prune any processes that have already exited on their own.
        self._procs = [p for p in self._procs if p.poll() is None]

        desired = self._spin.value()
        current = len(self._procs)

        if desired > current:
            for _ in range(desired - current):
                self._spawn_worker()
        elif desired < current:
            for _ in range(current - desired):
                proc = self._procs.pop()
                proc.terminate()

        # Trigger an immediate refresh so the table updates quickly.
        QTimer.singleShot(1_500, self._refresh)

    def _spawn_worker(self) -> None:
        """Launch one RQ worker subprocess."""
        cmd = [
            "uv", "run", "rq", "worker",
            "--url", _REDIS_URL,
            _WORKER_QUEUE,
        ]
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                # On Windows, CREATE_NEW_PROCESS_GROUP allows sending Ctrl-C
                # to the child without affecting the parent process.
                creationflags=(
                    subprocess.CREATE_NEW_PROCESS_GROUP
                    if sys.platform == "win32"
                    else 0
                ),
            )
            self._procs.append(proc)
        except FileNotFoundError:
            # uv not on PATH — surface in summary label, don't crash.
            self._summary_label.setText("Error: 'uv' not found on PATH")

    # -----------------------------------------------------------------------
    # Status polling — HTTP call runs in a background thread
    # -----------------------------------------------------------------------

    def _refresh(self) -> None:
        """Start a background poll of GET /workers (skip if one is in-flight)."""
        if self._fetch_thread is not None and self._fetch_thread.isRunning():
            return
        self._fetch_thread = HttpThread(self._client.list_workers, parent=self)
        self._fetch_thread.result_ready.connect(self._on_workers_ready)
        self._fetch_thread.error_raised.connect(self._on_workers_error)
        self._fetch_thread.start()

    def _on_workers_ready(self, worker_list: WorkerList) -> None:
        """Callback — runs on main thread after successful HTTP response."""
        self._summary_label.setText(
            f"API: {worker_list.total_workers} workers  "
            f"({worker_list.idle_workers} idle, {worker_list.busy_workers} busy)"
        )

        workers = worker_list.workers
        self._table.setRowCount(len(workers))

        for row, w in enumerate(workers):
            colour = _STATE_COLOURS.get(w.state, "#000000")

            name_item  = QTableWidgetItem(w.name)
            state_item = QTableWidgetItem(w.state)
            job_item   = QTableWidgetItem(w.current_job_id or "—")
            queue_item = QTableWidgetItem(", ".join(w.queues))

            state_item.setForeground(QColor(colour))

            self._table.setItem(row, 0, name_item)
            self._table.setItem(row, 1, state_item)
            self._table.setItem(row, 2, job_item)
            self._table.setItem(row, 3, queue_item)

    def _on_workers_error(self, exc: Exception) -> None:
        """Callback — runs on main thread when the HTTP call raises."""
        if isinstance(exc, APIError):
            self._summary_label.setText("API: unreachable")
        else:
            self._summary_label.setText(f"Error: {exc}")
