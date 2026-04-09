"""
MainWindow — top-level application window.

Layout
------
  QMainWindow
    │
    ├── Menu bar
    │     File → Exit
    │     Help → About
    │
    ├── QTabWidget (central widget)
    │     Tab 0: Run        → RunConfigWindow
    │     Tab 1: Results    → ResultsWindow
    │     Tab 2: Workers    → WorkerPanel
    │
    └── Status bar
          API connectivity indicator (checked every 10 s)

All three tabs share a single ALMApiClient instance that is constructed here
and passed into each child widget.

API health checks run in a background HttpThread so the status bar update
never blocks the main-thread event loop.
"""
from __future__ import annotations

from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QApplication,
    QLabel,
    QMainWindow,
    QMessageBox,
    QTabWidget,
)

from frontend.desktop.api_client import ALMApiClient
from frontend.desktop.components.worker_panel import WorkerPanel
from frontend.desktop.http_thread import HttpThread
from frontend.desktop.windows.ai_window import AIWindow
from frontend.desktop.windows.results_window import ResultsWindow
from frontend.desktop.windows.run_config_window import RunConfigWindow


_API_CHECK_INTERVAL_MS = 10_000   # 10 seconds


class MainWindow(QMainWindow):
    """
    Main application window.

    Parameters
    ----------
    client : ALMApiClient
        Shared HTTP client constructed in app.py.
    """

    def __init__(self, client: ALMApiClient, parent=None) -> None:
        super().__init__(parent)
        self._client = client
        self._health_thread: HttpThread | None = None

        self.setWindowTitle("ALM System")
        self.resize(1100, 780)

        # ---------------------------------------------------------------
        # Menu bar
        # ---------------------------------------------------------------
        self._build_menu()

        # ---------------------------------------------------------------
        # Central tab widget
        # ---------------------------------------------------------------
        self._tabs = QTabWidget()

        self._run_tab     = RunConfigWindow(client)
        self._results_tab = ResultsWindow(client)
        self._worker_tab  = WorkerPanel(client)
        self._ai_tab      = AIWindow(client)

        self._tabs.addTab(self._run_tab,     "Run")
        self._tabs.addTab(self._results_tab, "Results")
        self._tabs.addTab(self._worker_tab,  "Workers")
        self._tabs.addTab(self._ai_tab,      "AI Assistant")

        # Refresh the run list whenever the user switches to the Results tab.
        self._tabs.currentChanged.connect(self._on_tab_changed)

        self.setCentralWidget(self._tabs)

        # ---------------------------------------------------------------
        # Status bar — API health (checked off-thread every 10 s)
        # ---------------------------------------------------------------
        self._api_status_label = QLabel("API: checking…")
        self.statusBar().addPermanentWidget(self._api_status_label)

        self._api_timer = QTimer(self)
        self._api_timer.setInterval(_API_CHECK_INTERVAL_MS)
        self._api_timer.timeout.connect(self._check_api)
        self._api_timer.start()
        # Defer first check so the window renders before any network call.
        QTimer.singleShot(500, self._check_api)

    # -----------------------------------------------------------------------
    # Menu construction
    # -----------------------------------------------------------------------

    def _build_menu(self) -> None:
        menu_bar = self.menuBar()

        # File menu
        file_menu = menu_bar.addMenu("File")
        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Help menu
        help_menu = menu_bar.addMenu("Help")
        about_action = QAction("About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    # -----------------------------------------------------------------------
    # Tab change
    # -----------------------------------------------------------------------

    def _on_tab_changed(self, index: int) -> None:
        widget = self._tabs.widget(index)
        if widget is self._results_tab:
            self._results_tab.refresh_run_list()
        elif widget is self._ai_tab:
            self._ai_tab.refresh_run_list()

    # -----------------------------------------------------------------------
    # API health check — HTTP call runs in a background thread
    # -----------------------------------------------------------------------

    def _check_api(self) -> None:
        """Start a background health check (skip if one is in-flight)."""
        if self._health_thread is not None and self._health_thread.isRunning():
            return
        self._health_thread = HttpThread(self._client.is_reachable, parent=self)
        self._health_thread.result_ready.connect(self._on_health_ready)
        self._health_thread.error_raised.connect(
            lambda _: self._on_health_ready(False)
        )
        self._health_thread.start()

    def _on_health_ready(self, reachable: bool) -> None:
        """Callback — runs on main thread with the health-check result."""
        if reachable:
            self._api_status_label.setText(
                f'API: <span style="color:#2e7d32;">connected</span>'
                f' ({self._client._base})'
            )
        else:
            self._api_status_label.setText(
                f'API: <span style="color:#c62828;">unreachable</span>'
                f' ({self._client._base})'
            )
        self._api_status_label.setTextFormat(
            self._api_status_label.textFormat()
        )

    # -----------------------------------------------------------------------
    # Dialogs
    # -----------------------------------------------------------------------

    def _show_about(self) -> None:
        QMessageBox.about(
            self,
            "About ALM System",
            "<b>ALM System</b><br>"
            "Segregated fund Asset and Liability Model<br><br>"
            "Phase 2 — Desktop Frontend (Step 14)<br>"
            "Built with PyQt6 + FastAPI + RQ",
        )

    # -----------------------------------------------------------------------
    # Clean shutdown
    # -----------------------------------------------------------------------

    def closeEvent(self, event) -> None:
        """Terminate managed worker subprocesses before the window closes."""
        self._worker_tab.shutdown_all_workers()
        super().closeEvent(event)
