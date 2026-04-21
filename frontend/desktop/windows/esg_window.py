"""
EsgWindow — ESG Scenario Generator tab.

Wraps data/tools/generate_esg_scenarios.py in a PyQt6 form so analysts can
generate ESG scenario CSV files without leaving the desktop application.

Layout
------
  QScrollArea
    Basic parameters   — scenarios, months, seed
    Output             — directory (FolderPicker) + auto-suggested filename
    Advanced params    — Vasicek level-factor + equity GBM (collapsible)
    Actions            — Generate button + Open Folder button
    Progress           — indeterminate bar + status label
    Log                — read-only plain-text output with sanity stats

Background execution
--------------------
Generation runs in _GeneratorWorker (QThread) so the Qt event loop is never
blocked.  The worker imports generate_scenarios() from data.tools directly;
no HTTP call or API round-trip is required.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from frontend.desktop.components.file_picker import FolderPicker


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------

class _GeneratorWorker(QThread):
    """
    Runs generate_scenarios() + CSV write in a background thread.

    Signals
    -------
    finished(out_path, summary)
        Emitted on success with the written file path and a
        human-readable summary of t=0 statistics.
    failed(error_message)
        Emitted if an exception is raised during generation or file I/O.
    """

    finished: pyqtSignal = pyqtSignal(str, str)
    failed:   pyqtSignal = pyqtSignal(str)

    def __init__(
        self,
        n_scenarios:       int,
        n_months:          int,
        seed:              int,
        out_path:          Path,
        kappa:             float,
        sigma_level:       float,
        initial_shock_std: float,
        mu_equity:         float,
        sigma_equity:      float,
        parent:            QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._n_scenarios       = n_scenarios
        self._n_months          = n_months
        self._seed              = seed
        self._out_path          = out_path
        self._kappa             = kappa
        self._sigma_level       = sigma_level
        self._initial_shock_std = initial_shock_std
        self._mu_equity         = mu_equity
        self._sigma_equity      = sigma_equity

    def run(self) -> None:
        try:
            import csv
            from data.tools.generate_esg_scenarios import (
                generate_scenarios,
                _RATE_COLS,
            )

            rows = generate_scenarios(
                n_scenarios       = self._n_scenarios,
                n_months          = self._n_months,
                seed              = self._seed,
                kappa             = self._kappa,
                sigma_level       = self._sigma_level,
                initial_shock_std = self._initial_shock_std,
                mu_equity         = self._mu_equity,
                sigma_equity      = self._sigma_equity,
            )

            self._out_path.parent.mkdir(parents=True, exist_ok=True)
            fieldnames = ["scenario_id", "timestep"] + _RATE_COLS + ["equity_return_yr"]
            with self._out_path.open("w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

            summary = _build_summary(rows, self._n_scenarios, self._n_months, self._out_path)
            self.finished.emit(str(self._out_path), summary)

        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))


def _build_summary(
    rows: list[dict],
    n_scenarios: int,
    n_months: int,
    out_path: Path,
) -> str:
    t0 = [r for r in rows if r["timestep"] == 0]
    r12  = [r["r_12m"]          for r in t0]
    eq   = [r["equity_return_yr"] for r in t0]
    size = out_path.stat().st_size / 1024
    return (
        f"Generation complete\n"
        f"  File   : {out_path}\n"
        f"  Size   : {size:.1f} KB\n"
        f"  Rows   : {n_scenarios * n_months:,}  ({n_scenarios} scenarios x {n_months} months)\n"
        f"\n"
        f"Sanity check (t=0 across {n_scenarios} scenarios)\n"
        f"  r_12m  :  mean={np.mean(r12):.4f}  "
        f"min={np.min(r12):.4f}  max={np.max(r12):.4f}\n"
        f"  equity :  mean={np.mean(eq):.4f}  "
        f"min={np.min(eq):.4f}  max={np.max(eq):.4f}\n"
    )


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class EsgWindow(QWidget):
    """
    ESG Scenario Generator tab.

    Does not require a live API connection — generation runs locally using
    data/tools/generate_esg_scenarios.py via a background QThread.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._worker: _GeneratorWorker | None = None
        self._last_out_path: str = ""
        self._build_ui()
        self._connect_signals()
        self._update_filename()

    # -----------------------------------------------------------------------
    # UI construction
    # -----------------------------------------------------------------------

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        outer.addWidget(scroll)

        content = QWidget()
        scroll.setWidget(content)

        layout = QVBoxLayout(content)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        layout.addWidget(self._build_basic_params())
        layout.addWidget(self._build_output_section())
        layout.addWidget(self._build_advanced_params())
        layout.addLayout(self._build_actions())
        layout.addWidget(self._build_progress_section())
        layout.addWidget(self._build_log_section())
        layout.addStretch()

    # --- Basic parameters ---------------------------------------------------

    def _build_basic_params(self) -> QGroupBox:
        box = QGroupBox("Parameters")
        form = QFormLayout(box)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        self._n_scenarios = QSpinBox()
        self._n_scenarios.setRange(1, 10_000)
        self._n_scenarios.setValue(100)
        self._n_scenarios.setSingleStep(50)
        self._n_scenarios.setToolTip("Number of independent ESG scenarios to generate.")
        form.addRow("Scenarios:", self._n_scenarios)

        self._n_months = QSpinBox()
        self._n_months.setRange(12, 1_200)
        self._n_months.setValue(240)
        self._n_months.setSingleStep(12)
        self._n_months.setToolTip(
            "Projection length in months (= projection_term_years x 12).\n"
            "240 = 20 years — covers most conventional policy terms."
        )
        form.addRow("Projection months:", self._n_months)

        self._seed = QSpinBox()
        self._seed.setRange(0, 999_999)
        self._seed.setValue(42)
        self._seed.setToolTip(
            "NumPy random seed.  Fix the seed to get reproducible scenarios;\n"
            "change it to generate a different draw from the same distribution."
        )
        form.addRow("Random seed:", self._seed)

        return box

    # --- Output section -----------------------------------------------------

    def _build_output_section(self) -> QGroupBox:
        box = QGroupBox("Output")
        form = QFormLayout(box)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        self._out_dir = FolderPicker(
            label="",
            placeholder="Select output directory …",
        )
        # Pre-fill with project-relative default
        default_dir = Path("tests/sample_data/q42025/esg").resolve()
        if default_dir.parent.exists():
            self._out_dir.set_path(str(default_dir))
        form.addRow("Output directory:", self._out_dir)

        self._filename_edit = QLineEdit()
        self._filename_edit.setPlaceholderText("esg_scenarios_100_s42.csv")
        self._filename_edit.setToolTip(
            "Output filename.  Auto-suggested from scenario count + seed;\n"
            "edit freely if you need a custom name."
        )
        form.addRow("Filename:", self._filename_edit)

        return box

    # --- Advanced parameters ------------------------------------------------

    def _build_advanced_params(self) -> QGroupBox:
        self._advanced_box = QGroupBox("Advanced Parameters")
        self._advanced_box.setCheckable(True)
        self._advanced_box.setChecked(False)   # collapsed by default

        outer = QVBoxLayout(self._advanced_box)

        # --- Vasicek (rate curve) ---
        rate_box = QGroupBox("Rate Curve — Vasicek Level-Factor")
        rate_form = QFormLayout(rate_box)
        rate_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        self._kappa = self._make_double_spinbox(
            value=0.10, lo=0.001, hi=2.0, step=0.01, decimals=3,
            tip=(
                "Mean-reversion speed κ (per year).  Higher values pull the level\n"
                "factor back to its long-run mean faster.\n"
                "Typical range: 0.05 (slow) – 0.30 (fast)."
            ),
        )
        rate_form.addRow("Mean-reversion speed (κ):", self._kappa)

        self._sigma_level = self._make_double_spinbox(
            value=0.005, lo=0.0001, hi=0.10, step=0.001, decimals=4,
            tip=(
                "Annual volatility of the level shift factor.\n"
                "0.005 = ~50 bps per year.  Increase for more scenario dispersion."
            ),
        )
        rate_form.addRow("Level-factor vol (σ):", self._sigma_level)

        self._initial_shock_std = self._make_double_spinbox(
            value=0.010, lo=0.0001, hi=0.10, step=0.001, decimals=4,
            tip=(
                "Cross-sectional std of the initial level shock at t=0.\n"
                "0.010 = ~100 bps spread across scenarios at inception."
            ),
        )
        rate_form.addRow("Initial shock std (σ₀):", self._initial_shock_std)

        outer.addWidget(rate_box)

        # --- Equity GBM ---
        eq_box = QGroupBox("Equity — GBM (Lognormal)")
        eq_form = QFormLayout(eq_box)
        eq_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        self._mu_equity = self._make_double_spinbox(
            value=0.07, lo=-0.20, hi=0.50, step=0.01, decimals=4,
            tip="Annual equity drift μ (e.g. 0.07 = 7%).  Used in GBM log-return formula.",
        )
        eq_form.addRow("Annual drift (μ):", self._mu_equity)

        self._sigma_equity = self._make_double_spinbox(
            value=0.15, lo=0.01, hi=1.0, step=0.01, decimals=4,
            tip="Annual equity volatility σ (e.g. 0.15 = 15%).  Drives scenario dispersion.",
        )
        eq_form.addRow("Annual vol (σ):", self._sigma_equity)

        outer.addWidget(eq_box)

        # Note about base curve
        note = QLabel(
            "Base yield curve: UK risk-free spot rates from Q4 2025 sample data "
            "(1yr=3.54%, 5yr=3.67%, 10yr=4.05%, 20yr=4.54%, 30yr=4.59%).  "
            "All scenarios shift this curve in parallel by the Vasicek level factor."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: #555; font-style: italic;")
        outer.addWidget(note)

        return self._advanced_box

    # --- Action buttons -----------------------------------------------------

    def _build_actions(self) -> QHBoxLayout:
        row = QHBoxLayout()

        self._generate_btn = QPushButton("Generate")
        self._generate_btn.setFixedWidth(120)
        self._generate_btn.setToolTip("Run the ESG scenario generator in the background.")
        row.addWidget(self._generate_btn)

        self._open_folder_btn = QPushButton("Open Output Folder")
        self._open_folder_btn.setEnabled(False)
        self._open_folder_btn.setToolTip("Open the output directory in Windows Explorer.")
        row.addWidget(self._open_folder_btn)

        row.addStretch()
        return row

    # --- Progress section ---------------------------------------------------

    def _build_progress_section(self) -> QGroupBox:
        box = QGroupBox("Progress")
        vbox = QVBoxLayout(box)

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 0)   # indeterminate by default
        self._progress_bar.setValue(0)
        self._progress_bar.setVisible(False)
        vbox.addWidget(self._progress_bar)

        self._status_label = QLabel("Ready")
        vbox.addWidget(self._status_label)

        return box

    # --- Log section --------------------------------------------------------

    def _build_log_section(self) -> QGroupBox:
        box = QGroupBox("Output Log")
        vbox = QVBoxLayout(box)

        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setFixedHeight(180)
        self._log.setPlaceholderText(
            "Generation output and sanity statistics will appear here …"
        )
        vbox.addWidget(self._log)

        return box

    # -----------------------------------------------------------------------
    # Signal wiring
    # -----------------------------------------------------------------------

    def _connect_signals(self) -> None:
        self._generate_btn.clicked.connect(self._on_generate)
        self._open_folder_btn.clicked.connect(self._on_open_folder)

        # Auto-update filename suggestion when scenarios or seed changes
        self._n_scenarios.valueChanged.connect(self._update_filename)
        self._seed.valueChanged.connect(self._update_filename)

        # Toggle advanced content visibility
        self._advanced_box.toggled.connect(self._on_advanced_toggled)

    # -----------------------------------------------------------------------
    # Slot handlers
    # -----------------------------------------------------------------------

    def _on_advanced_toggled(self, checked: bool) -> None:
        """Show/hide the inner widgets of the Advanced box when toggled."""
        for i in range(self._advanced_box.layout().count()):
            item = self._advanced_box.layout().itemAt(i)
            if item and item.widget():
                item.widget().setVisible(checked)

    def _update_filename(self) -> None:
        """Suggest a filename based on current scenario count + seed."""
        n = self._n_scenarios.value()
        s = self._seed.value()
        self._filename_edit.setText(f"esg_scenarios_{n}_s{s}.csv")

    def _on_generate(self) -> None:
        """Validate inputs then start the background worker."""
        out_dir = self._out_dir.path().strip()
        if not out_dir:
            self._set_status("Please select an output directory.", error=True)
            return

        filename = self._filename_edit.text().strip()
        if not filename:
            self._set_status("Please enter a filename.", error=True)
            return
        if not filename.endswith(".csv"):
            filename += ".csv"

        out_path = Path(out_dir) / filename

        self._log.clear()
        self._set_status("Generating scenarios …")
        self._progress_bar.setVisible(True)
        self._generate_btn.setEnabled(False)
        self._open_folder_btn.setEnabled(False)
        self._last_out_path = ""

        self._worker = _GeneratorWorker(
            n_scenarios       = self._n_scenarios.value(),
            n_months          = self._n_months.value(),
            seed              = self._seed.value(),
            out_path          = out_path,
            kappa             = self._kappa.value(),
            sigma_level       = self._sigma_level.value(),
            initial_shock_std = self._initial_shock_std.value(),
            mu_equity         = self._mu_equity.value(),
            sigma_equity      = self._sigma_equity.value(),
            parent            = self,
        )
        self._worker.finished.connect(self._on_generation_done)
        self._worker.failed.connect(self._on_generation_failed)
        self._worker.start()

    def _on_generation_done(self, out_path: str, summary: str) -> None:
        self._progress_bar.setVisible(False)
        self._generate_btn.setEnabled(True)
        self._last_out_path = out_path
        self._open_folder_btn.setEnabled(True)
        self._set_status(f"Done — {out_path}")
        self._log.setPlainText(summary)

    def _on_generation_failed(self, error: str) -> None:
        self._progress_bar.setVisible(False)
        self._generate_btn.setEnabled(True)
        self._set_status("Generation failed — see log for details.", error=True)
        self._log.setPlainText(f"ERROR\n\n{error}")

    def _on_open_folder(self) -> None:
        if self._last_out_path:
            folder = str(Path(self._last_out_path).parent)
            os.startfile(folder)  # Windows Explorer

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _set_status(self, message: str, error: bool = False) -> None:
        colour = "#c62828" if error else "#2e7d32"
        self._status_label.setText(
            f'<span style="color:{colour};">{message}</span>'
        )

    @staticmethod
    def _make_double_spinbox(
        value: float,
        lo: float,
        hi: float,
        step: float,
        decimals: int,
        tip: str = "",
    ) -> QDoubleSpinBox:
        sb = QDoubleSpinBox()
        sb.setRange(lo, hi)
        sb.setValue(value)
        sb.setSingleStep(step)
        sb.setDecimals(decimals)
        if tip:
            sb.setToolTip(tip)
        return sb
