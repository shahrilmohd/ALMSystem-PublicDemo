"""
ResultsWindow — run results viewer.

Layout
------
  ── Run Selector ──
    [QComboBox — populated from GET /runs]   [Refresh list]

  ── Run Summary ──
    run_id | run_type | status | n_scenarios | n_timesteps | final BEL | final TMV

  ── Results Table ──
    [QTableWidget — loaded on demand, first 500 rows]
    [Load Results]   [Export CSV…]

  ── Progress re-check ──
    [ProgressPanel — can re-attach to a PENDING/RUNNING run]
"""
from __future__ import annotations

import os

from PyQt6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
    QFileDialog,
    QComboBox,
)

from frontend.desktop.api_client import ALMApiClient, APIError, APIResponseError
from frontend.desktop.components.progress_panel import ProgressPanel


_MAX_DISPLAY_ROWS = 500


class ResultsWindow(QWidget):
    """
    Results viewer tab.  Intended to be used as a tab in MainWindow.
    """

    def __init__(self, client: ALMApiClient, parent=None) -> None:
        super().__init__(parent)
        self._client   = client
        self._run_id:  str | None = None
        self._runs:    list       = []   # cached list of RunStatus objects

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        layout.addWidget(self._build_selector_section())
        layout.addWidget(self._build_summary_section())
        layout.addWidget(self._build_table_section())
        layout.addWidget(self._build_results_summary_section())

        # Re-check progress for a selected run that is still in-flight.
        self._progress_panel = ProgressPanel(self._client)
        self._progress_panel.run_completed.connect(self._on_run_completed)
        layout.addWidget(self._progress_panel)

        layout.addStretch()

    # ===================================================================
    # Section builders
    # ===================================================================

    def _build_selector_section(self) -> QGroupBox:
        grp = QGroupBox("Select Run")
        row = QHBoxLayout()

        self._run_combo = QComboBox()
        self._run_combo.setMinimumWidth(420)
        self._run_combo.currentIndexChanged.connect(self._on_run_selected)
        row.addWidget(self._run_combo)

        btn_refresh = QPushButton("Refresh list")
        btn_refresh.clicked.connect(self.refresh_run_list)
        row.addWidget(btn_refresh)
        row.addStretch()

        grp.setLayout(row)
        return grp

    def _build_summary_section(self) -> QGroupBox:
        grp = QGroupBox("Run Summary")
        layout = QVBoxLayout()

        self._lbl_run_name  = QLabel("Run Name: —")
        self._lbl_run_id    = QLabel("Run ID: —")
        self._lbl_type      = QLabel("Type: —")
        self._lbl_status    = QLabel("Status: —")
        self._lbl_scenarios = QLabel("Scenarios: —")
        self._lbl_run_time  = QLabel("Total Run Time: —")

        # Error summary — shown only for FAILED runs.
        # TODO (Phase 3 AI layer): this error_message is also available via
        # GET /runs/{run_id} → RunStatusResponse.error_message.  The AI assistant
        # can fetch it to explain what went wrong and suggest config corrections.
        self._lbl_error = QLabel("")
        self._lbl_error.setWordWrap(True)
        self._lbl_error.setStyleSheet("color: #c62828;")  # red

        for lbl in (
            self._lbl_run_name, self._lbl_run_id, self._lbl_type,
            self._lbl_status, self._lbl_scenarios, self._lbl_run_time,
            self._lbl_error,
        ):
            layout.addWidget(lbl)

        grp.setLayout(layout)
        return grp
    
    def _build_results_summary_section(self) -> QGroupBox:
        grp = QGroupBox("Results Summary")
        layout = QVBoxLayout()

        self._lbl_opening_bel = QLabel("Opening BEL: —")
        self._lbl_totalmv = QLabel("Total Market Value: —")
        self._lbl_totalif = QLabel("Total Number of In-Force Policies: —")

        for lbl in (
            self._lbl_opening_bel,
            self._lbl_totalmv,
            self._lbl_totalif,
        ):
            layout.addWidget(lbl)

        grp.setLayout(layout)
        return grp

    def _build_table_section(self) -> QGroupBox:
        grp = QGroupBox(f"Result Rows (first {_MAX_DISPLAY_ROWS} rows)")
        layout = QVBoxLayout()

        self._table = QTableWidget(0, 0)
        self._table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setAlternatingRowColors(True)
        self._table.verticalHeader().setVisible(False)
        layout.addWidget(self._table)

        btn_row = QHBoxLayout()
        btn_load = QPushButton("Load Results")
        btn_load.clicked.connect(self._load_results)
        btn_row.addWidget(btn_load)

        btn_export = QPushButton("Export CSV…")
        btn_export.clicked.connect(self._export_csv)
        btn_row.addWidget(btn_export)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        grp.setLayout(layout)
        return grp

    # ===================================================================
    # Public API
    # ===================================================================

    def refresh_run_list(self) -> None:
        """Reload the run list from the API and repopulate the combo box."""
        try:
            self._runs = self._client.list_runs()
        except APIError as exc:
            QMessageBox.warning(self, "API Unreachable", str(exc))
            return

        self._run_combo.blockSignals(True)
        self._run_combo.clear()
        for run in self._runs:
            name_part = run.run_name or f"{run.run_id[:8]}…"
            label = (
                f"[{run.status}]  {name_part}  |  "
                f"{run.run_type}  {run.created_at.strftime('%Y-%m-%d %H:%M')}"
            )
            self._run_combo.addItem(label, run.run_id)
        self._run_combo.blockSignals(False)

        if self._runs:
            self._run_combo.setCurrentIndex(0)
            self._on_run_selected(0)

    # ===================================================================
    # Internal
    # ===================================================================

    def _on_run_selected(self, index: int) -> None:
        if index < 0 or index >= len(self._runs):
            return
        run = self._runs[index]
        self._run_id = run.run_id

        self._lbl_run_name.setText(f"Run Name: {run.run_name or '—'}")
        self._lbl_run_id.setText(f"Run ID: {run.run_id}")
        self._lbl_type.setText(f"Type: {run.run_type}")
        self._lbl_status.setText(f"Status: {run.status}")
        self._lbl_scenarios.setText(
            f"Scenarios: {run.n_scenarios:,}" if run.n_scenarios else "Scenarios: —"
        )

        if run.duration_seconds is not None:
            mins, secs = divmod(int(run.duration_seconds), 60)
            self._lbl_run_time.setText(
                f"Total Run Time: {mins}m {secs}s  ({run.duration_seconds:.1f}s)"
            )
        else:
            self._lbl_run_time.setText("Total Run Time: —")

        if run.status == "FAILED" and run.error_message:
            self._lbl_error.setText(f"Error: {run.error_message}")
        else:
            self._lbl_error.setText("")

        # self._lbl_timesteps.setText(f"Timesteps: {run.n_timesteps or '—'}")  # removed per Step 14 UX review
        # self._lbl_bel.setText("Final BEL: —")                                # removed — shown in results table instead
        # self._lbl_tmv.setText("Final Total MV: —")                           # removed — shown in results table instead

        self._table.clearContents()
        self._table.setRowCount(0)
        self._table.setColumnCount(0)

        self._lbl_opening_bel.setText("Opening BEL: —")
        self._lbl_totalmv.setText("Total Market Value: —")
        self._lbl_totalif.setText("Total Number of In-Force Policies: —")

        # If the run is complete, fetch the summary immediately.
        # if run.status == "COMPLETED":                                         # removed — summary no longer needs API call
        #     self._load_summary()
        # If still in flight, offer progress tracking.
        if run.status in ("PENDING", "RUNNING"):
            self._progress_panel.track(run.run_id)

    def _load_summary(self) -> None:
        # Summary panel now populated directly from RunStatus in _on_run_selected —
        # no extra API call needed.  This method is kept for reference in case
        # BEL / TMV are reinstated in the panel later.
        #
        # TODO (Phase 3 AI layer): GET /results/{run_id}/summary still exists on the
        # API and can be called by the AI assistant to retrieve BEL and total market
        # value for result explanation.
        #
        # if not self._run_id:
        #     return
        # try:
        #     summary = self._client.get_results_summary(self._run_id)
        # except (APIError, APIResponseError):
        #     return
        # self._lbl_scenarios.setText(f"Scenarios: {summary.n_scenarios}")
        # self._lbl_timesteps.setText(f"Timesteps: {summary.n_timesteps}")       # removed from panel
        # self._lbl_bel.setText(                                                  # removed from panel
        #     f"Final BEL: {summary.final_bel:,.2f}"
        #     if summary.final_bel is not None else "Final BEL: —"
        # )
        # self._lbl_tmv.setText(                                                  # removed from panel
        #     f"Final Total MV: {summary.final_total_market_value:,.2f}"
        #     if summary.final_total_market_value is not None
        #     else "Final Total MV: —"
        # )
        pass

    def _load_results(self) -> None:
        if not self._run_id:
            QMessageBox.information(self, "No Run Selected", "Select a run from the list first.")
            return

        try:
            csv_bytes = self._client.get_results_csv(self._run_id)
        except APIResponseError as exc:
            QMessageBox.warning(self, "Load Failed", exc.detail)
            return
        except APIError as exc:
            QMessageBox.critical(self, "API Unreachable", str(exc))
            return

        csv_text = csv_bytes.decode("utf-8", errors="replace")
        self._populate_table_from_csv(csv_bytes)
        self._update_results_summary(csv_text)

    def _populate_table_from_csv(self, csv_bytes: bytes) -> None:
        import csv
        import io

        text = csv_bytes.decode("utf-8", errors="replace")
        reader = csv.reader(io.StringIO(text))
        rows = list(reader)

        if not rows:
            self._table.clearContents()
            return

        headers = rows[0]
        data_rows = rows[1 : _MAX_DISPLAY_ROWS + 1]

        self._table.setColumnCount(len(headers))
        self._table.setHorizontalHeaderLabels(headers)
        self._table.setRowCount(len(data_rows))

        for r, row in enumerate(data_rows):
            for c, cell in enumerate(row):
                self._table.setItem(r, c, QTableWidgetItem(cell))

    def _update_results_summary(self, csv_text: str) -> None:
        import csv
        import io

        reader = csv.DictReader(io.StringIO(csv_text))
        rows = list(reader)

        if not rows:
            return
        
        # All three summary values are as at the valuation date (t=0, first row)
        first_row = rows[0]

        def fmt(value: str) -> str:
            """Format numeric string with comas or return '-' if blank"""
            try:
                return f"{float(value):,.0f}"
            except (ValueError, TypeError):
                return value
            
        self._lbl_opening_bel.setText(f"Opening BEL: {fmt(first_row.get('bel', '—'))}")
        self._lbl_totalmv.setText(f"Total Market Value: {fmt(first_row.get('total_market_value', '—'))}")
        self._lbl_totalif.setText(f"Total Number of In-Force Policies: {fmt(first_row.get('in_force_start', '—'))}")
    
    def _export_csv(self) -> None:
        if not self._run_id:
            QMessageBox.information(self, "No Run Selected", "Select a run from the list first.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Results as CSV",
            f"results_{self._run_id[:8]}.csv",
            "CSV files (*.csv)",
        )
        if not path:
            return

        try:
            csv_bytes = self._client.get_results_csv(self._run_id)
        except APIResponseError as exc:
            QMessageBox.warning(self, "Export Failed", exc.detail)
            return
        except APIError as exc:
            QMessageBox.critical(self, "API Unreachable", str(exc))
            return

        with open(path, "wb") as f:
            f.write(csv_bytes)
        QMessageBox.information(self, "Exported", f"Results saved to:\n{path}")

    def _on_run_completed(self, run_id: str) -> None:
        # Refresh the list — this repopulates the combo and re-selects the run,
        # which triggers _on_run_selected and fills in duration + final status.
        self.refresh_run_list()
